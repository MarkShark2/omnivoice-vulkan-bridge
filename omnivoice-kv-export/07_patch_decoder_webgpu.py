#!/usr/bin/env python3
"""Patch the OmniVoice decoder for ONNX Runtime WebGPU.

This is the production decoder pipeline step. It combines the two graph fixes
that were developed separately:

1. Replace unsupported/incorrect ConvTranspose output-padding behavior with
   explicit Pad ops, and replace block.4/conv_t1 with an equivalent
   dilate-plus-Conv sequence.
2. Fuse the eight codebook projection branches into one precomputed Gather so
   newer ORT-Web WebGPU builds avoid the fragile small Gather/MatMul stack.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DECODER_ONNX, DECODER_WEBGPU_ONNX  # noqa: E402

def _make_pads_init(g: onnx.GraphProto, name: str, pads: list[int]) -> None:
    g.initializer.append(numpy_helper.from_array(np.array(pads, dtype=np.int64), name=name))


def _make_int64_init(g: onnx.GraphProto, name: str, values: list[int]) -> None:
    g.initializer.append(numpy_helper.from_array(np.array(values, dtype=np.int64), name=name))


def _replace_convt_with_dilate_conv(
    g: onnx.GraphProto,
    node: onnx.NodeProto,
    op_pad: int,
) -> list[onnx.NodeProto]:
    """
    Replace a broken 1-D ConvTranspose(kernel=K, stride=S, pads=[P,P])
    with the equivalent: dilate-by-S -> Conv(K, s=1, pads=[K-1-P, K-1-P]).

    If op_pad>0 also append a trailing Pad op to preserve output_padding.
    Returns the new nodes list.
    """
    # Parse attributes
    def _attr_ints(name: str, default: list[int]) -> list[int]:
        a = next((a for a in node.attribute if a.name == name), None)
        return list(a.ints) if a else default

    strides = _attr_ints("strides", [1])
    kernel = _attr_ints("kernel_shape", [1])
    pads = _attr_ints("pads", [0, 0])
    dilations = _attr_ints("dilations", [1])
    group_attr = next((a for a in node.attribute if a.name == "group"), None)
    group = group_attr.i if group_attr else 1

    if len(strides) != 1 or len(kernel) != 1 or len(pads) != 2:
        raise NotImplementedError(f"need 1-D ConvTranspose, got {node.name}")
    if dilations != [1]:
        raise NotImplementedError(f"dilation != 1 unsupported here ({node.name})")
    if group != 1:
        raise NotImplementedError(f"group != 1 unsupported here ({node.name})")

    S = strides[0]
    K = kernel[0]
    P = pads[0]
    if pads[1] != P:
        raise NotImplementedError(f"asymmetric pads unsupported here ({node.name})")

    # Look up weight + (optional) bias initializers
    assert len(node.input) >= 2, f"ConvTranspose {node.name} has no weight input"
    data_name = node.input[0]
    w_name = node.input[1]
    b_name = node.input[2] if len(node.input) >= 3 else None

    w_init = next((i for i in g.initializer if i.name == w_name), None)
    if w_init is None:
        raise RuntimeError(f"weight initializer not found: {w_name}")
    w_arr = numpy_helper.to_array(w_init)
    # ConvTranspose weight shape: (C_in, C_out / group, K) ; for group=1 -> (C_in, C_out, K)
    assert w_arr.ndim == 3, f"unexpected weight rank {w_arr.ndim}"
    C_in, C_out, Kw = w_arr.shape
    assert Kw == K, f"kernel mismatch: attr={K} weight={Kw}"

    # For equivalent Conv we need Conv-style weight (C_out, C_in, K) with axis 2 flipped.
    w_conv = np.ascontiguousarray(np.flip(w_arr.transpose(1, 0, 2), axis=2))
    w_conv_name = w_name + "__as_conv"
    # Avoid collision if script is run twice.
    if not any(i.name == w_conv_name for i in g.initializer):
        g.initializer.append(numpy_helper.from_array(w_conv, name=w_conv_name))

    out_name = node.output[0]
    base = node.name or out_name
    new_nodes: list[onnx.NodeProto] = []

    # Step 1: Reshape (N, C, L) -> (N, C, L, 1)
    shape4_name = f"{base}__shape_unsq"
    _make_int64_init(g, shape4_name, [0, 0, -1, 1])  # 0-copy for first two, -1 infers L, last is 1
    t_unsq = f"{base}__dilate_unsq"
    new_nodes.append(helper.make_node(
        "Reshape", inputs=[data_name, shape4_name], outputs=[t_unsq],
        name=f"{base}__reshape_unsq", allowzero=0,
    ))

    # Step 2: Pad last dim end by (S-1) zeros -> (N, C, L, S)
    pad_name = f"{base}__dilate_pad_values"
    _make_pads_init(g, pad_name, [0, 0, 0, 0, 0, 0, 0, S - 1])
    t_padded = f"{base}__dilate_padded"
    new_nodes.append(helper.make_node(
        "Pad", inputs=[t_unsq, pad_name], outputs=[t_padded],
        name=f"{base}__pad_stride", mode="constant",
    ))

    # Step 3: Reshape back (N, C, L, S) -> (N, C, L*S)
    shape3_name = f"{base}__shape_interleave"
    _make_int64_init(g, shape3_name, [0, 0, -1])
    t_interleaved = f"{base}__dilate_interleaved"
    new_nodes.append(helper.make_node(
        "Reshape", inputs=[t_padded, shape3_name], outputs=[t_interleaved],
        name=f"{base}__reshape_interleave", allowzero=0,
    ))

    # Step 4: Slice last dim to length (L-1)*S + 1 = L*S - (S-1)  (drop last S-1 zeros)
    # Use end index L*S - (S-1) via negative index: end=-(S-1), axis=2
    sl_starts = f"{base}__slice_starts"
    sl_ends = f"{base}__slice_ends"
    sl_axes = f"{base}__slice_axes"
    sl_steps = f"{base}__slice_steps"
    _make_int64_init(g, sl_starts, [0])
    _make_int64_init(g, sl_ends, [-(S - 1)] if S > 1 else [2**31 - 1])
    _make_int64_init(g, sl_axes, [2])
    _make_int64_init(g, sl_steps, [1])
    t_dilated = f"{base}__dilated"
    if S > 1:
        new_nodes.append(helper.make_node(
            "Slice",
            inputs=[t_interleaved, sl_starts, sl_ends, sl_axes, sl_steps],
            outputs=[t_dilated],
            name=f"{base}__slice_dilate",
        ))
    else:
        # S==1 (no dilation needed), just Identity
        new_nodes.append(helper.make_node(
            "Identity", inputs=[t_interleaved], outputs=[t_dilated],
            name=f"{base}__identity_dilate",
        ))

    # Step 5: Conv with flipped weight, stride 1, pad = K-1-P
    P_prime = K - 1 - P
    conv_out = out_name if op_pad == 0 else out_name + "__pre_pad_op"
    conv_inputs = [t_dilated, w_conv_name]
    if b_name is not None:
        conv_inputs.append(b_name)
    new_nodes.append(helper.make_node(
        "Conv",
        inputs=conv_inputs,
        outputs=[conv_out],
        name=f"{base}__conv_emul",
        kernel_shape=[K],
        strides=[1],
        dilations=[1],
        pads=[P_prime, P_prime],
        group=1,
    ))

    # Step 6 (optional): if original had output_padding=1, add trailing Pad.
    if op_pad > 0:
        op_pad_values = f"{base}__convt_op_pad_values"
        _make_pads_init(g, op_pad_values, [0, 0, 0, 0, 0, op_pad])
        new_nodes.append(helper.make_node(
            "Pad", inputs=[conv_out, op_pad_values], outputs=[out_name],
            name=f"{base}__conv_emul_op_pad", mode="constant",
        ))

    return new_nodes


def rewrite(
    in_path: str,
    out_path: str,
    replace_names: list[str],
) -> dict:
    model = onnx.load(in_path)
    g = model.graph

    # Ensure Pad-13 is usable: opset >= 13 uses `pads` as an input tensor, not attr.
    opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 13)
    if opset < 11:
        raise RuntimeError(f"model opset {opset} too old; need >= 11 for Pad with pads-input")

    new_nodes: list[onnx.NodeProto] = []
    n_op_pad_fix = 0
    n_full_rewrite = 0
    for node in g.node:
        if node.op_type != "ConvTranspose":
            new_nodes.append(node)
            continue

        op_pad_attr = next((a for a in node.attribute if a.name == "output_padding"), None)
        op_pad_val = op_pad_attr.ints[0] if (op_pad_attr and len(op_pad_attr.ints) == 1) else 0

        should_full_rewrite = any(sub and sub in node.name for sub in replace_names)
        if should_full_rewrite:
            replacement = _replace_convt_with_dilate_conv(g, node, op_pad_val)
            new_nodes.extend(replacement)
            n_full_rewrite += 1
            extra = f", output_padding={op_pad_val}" if op_pad_val else ""
            print(f"  replaced ConvTranspose -> dilate+Conv: {node.name}{extra}")
            continue

        if op_pad_val == 0:
            new_nodes.append(node)
            continue

        # Only the output_padding fix remains for this node.
        if len(op_pad_attr.ints) != 1:
            raise NotImplementedError(
                f"need 1-D output_padding, got {list(op_pad_attr.ints)} on {node.name}"
            )
        n_right = op_pad_val

        # Clone the ConvTranspose but drop output_padding, and give it a new
        # intermediate output name.
        new_convt = onnx.NodeProto()
        new_convt.CopyFrom(node)
        # strip output_padding
        del new_convt.attribute[:]
        new_convt.attribute.extend([a for a in node.attribute if a.name != "output_padding"])
        orig_out = node.output[0]
        intermediate = orig_out + "__pre_pad"
        new_convt.output[0] = intermediate

        # Pad op: inputs (data, pads[, constant_value]). pads shape is [2*rank],
        # formatted as [x1_begin, x2_begin, ..., x1_end, x2_end, ...]. For the
        # decoder, data is (N, C, L) → rank=3 → pads length 6.
        # We pad zero on all but the last axis's end.
        pads_name = f"{node.name}__pad_values" if node.name else f"{intermediate}__pad_values"
        pads_init = numpy_helper.from_array(
            np.array([0, 0, 0, 0, 0, n_right], dtype=np.int64), name=pads_name
        )
        g.initializer.append(pads_init)

        pad_name = (node.name or intermediate) + "__rightpad"
        pad_node = helper.make_node(
            "Pad",
            inputs=[intermediate, pads_name],
            outputs=[orig_out],
            name=pad_name,
            mode="constant",
        )
        new_nodes.append(new_convt)
        new_nodes.append(pad_node)
        n_op_pad_fix += 1
        print(f"  rewrote {node.name}: output_padding=[{n_right}] -> trailing Pad(n={n_right})")

    del g.node[:]
    g.node.extend(new_nodes)

    # Re-run shape inference to keep things tidy (non-fatal).
    try:
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False, check_type=False)
    except Exception as e:
        print(f"  warning: shape inference failed post-rewrite: {e}")

    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, out_path)
    return {"op_pad_fix": n_op_pad_fix, "full_rewrite": n_full_rewrite}



def _initializer_map(g: onnx.GraphProto) -> dict[str, onnx.TensorProto]:
    return {init.name: init for init in g.initializer}


def _add_or_replace_initializer(
    g: onnx.GraphProto,
    name: str,
    value: np.ndarray,
) -> None:
    kept = [init for init in g.initializer if init.name != name]
    del g.initializer[:]
    g.initializer.extend(kept)
    g.initializer.append(numpy_helper.from_array(np.ascontiguousarray(value), name=name))


def rewrite_codebook_project(in_path: str, out_path: str) -> int:
    model = onnx.load(in_path)
    g = model.graph
    inits = _initializer_map(g)

    fused_tables: list[np.ndarray] = []

    for q in range(8):
        prefix = f"audio_tokenizer.quantizer.quantizers.{q}"
        embed_name = f"{prefix}.codebook.embed"
        bias_name = f"{prefix}.project_out.bias"
        fused_name = f"{prefix}.project_out.fused_table"

        if fused_name in inits:
            fused = numpy_helper.to_array(inits[fused_name]).astype(np.float32, copy=False)
            fused_tables.append(fused)
            continue

        matmul_name = f"onnx::MatMul_{1901 + q}"
        if embed_name not in inits or matmul_name not in inits or bias_name not in inits:
            raise RuntimeError(f"missing initializer(s) for quantizer {q}")
        embed = numpy_helper.to_array(inits[embed_name]).astype(np.float32, copy=False)
        weight = numpy_helper.to_array(inits[matmul_name]).astype(np.float32, copy=False)
        bias = numpy_helper.to_array(inits[bias_name]).astype(np.float32, copy=False)
        if embed.ndim != 2 or weight.ndim != 2 or bias.ndim != 1:
            raise RuntimeError(
                f"unexpected projection initializer ranks for quantizer {q}: "
                f"embed={embed.shape}, weight={weight.shape}, bias={bias.shape}"
            )
        if embed.shape[1] != weight.shape[0] or weight.shape[1] != bias.shape[0]:
            raise RuntimeError(
                f"unexpected projection shapes for quantizer {q}: "
                f"embed={embed.shape}, weight={weight.shape}, bias={bias.shape}"
            )

        fused = embed @ weight
        fused += bias
        fused_tables.append(fused.astype(np.float32, copy=False))

    flat_table = np.concatenate(fused_tables, axis=0)
    if flat_table.shape != (8192, 1024):
        raise RuntimeError(f"unexpected fused table shape: {flat_table.shape}")

    _add_or_replace_initializer(g, "audio_tokenizer.quantizer.project_out.flat_fused_table", flat_table)
    _add_or_replace_initializer(
        g,
        "audio_tokenizer.quantizer.project_out.flat_offsets",
        (np.arange(8, dtype=np.int64).reshape(8, 1, 1) * 1024),
    )
    _add_or_replace_initializer(
        g,
        "audio_tokenizer.quantizer.project_out.flat_shape",
        np.array([-1], dtype=np.int64),
    )
    _add_or_replace_initializer(
        g,
        "audio_tokenizer.quantizer.project_out.unflat_shape",
        np.array([8, 1, -1, 1024], dtype=np.int64),
    )
    _add_or_replace_initializer(
        g,
        "audio_tokenizer.quantizer.project_out.reduce_axes",
        np.array([0], dtype=np.int64),
    )

    fused_nodes = [
        helper.make_node(
            "Add",
            inputs=["/Transpose_output_0", "audio_tokenizer.quantizer.project_out.flat_offsets"],
            outputs=["/project_out_fused/AddOffsets_output_0"],
            name="/project_out_fused/AddOffsets",
        ),
        helper.make_node(
            "Reshape",
            inputs=[
                "/project_out_fused/AddOffsets_output_0",
                "audio_tokenizer.quantizer.project_out.flat_shape",
            ],
            outputs=["/project_out_fused/FlatCodes_output_0"],
            name="/project_out_fused/FlatCodes",
            allowzero=0,
        ),
        helper.make_node(
            "Gather",
            inputs=[
                "audio_tokenizer.quantizer.project_out.flat_fused_table",
                "/project_out_fused/FlatCodes_output_0",
            ],
            outputs=["/project_out_fused/Gather_output_0"],
            name="/project_out_fused/Gather",
            axis=0,
        ),
        helper.make_node(
            "Reshape",
            inputs=[
                "/project_out_fused/Gather_output_0",
                "audio_tokenizer.quantizer.project_out.unflat_shape",
            ],
            outputs=["/project_out_fused/Unflat_output_0"],
            name="/project_out_fused/Unflat",
            allowzero=0,
        ),
        helper.make_node(
            "ReduceSum",
            inputs=[
                "/project_out_fused/Unflat_output_0",
                "audio_tokenizer.quantizer.project_out.reduce_axes",
            ],
            outputs=["/project_out_fused/Sum_output_0"],
            name="/project_out_fused/Sum",
            keepdims=0,
        ),
        helper.make_node(
            "Transpose",
            inputs=["/project_out_fused/Sum_output_0"],
            outputs=["/Add_7_output_0"],
            name="/project_out_fused/ToChannelsFirst",
            perm=[0, 2, 1],
        ),
    ]

    new_nodes: list[onnx.NodeProto] = []
    for node in g.node:
        if node.name == "/Constant":
            new_nodes.extend(fused_nodes)
        if node.name == "/Constant" or node.name.startswith("/Split") or node.name.startswith("/Squeeze"):
            continue
        if node.name.startswith("/project_out"):
            continue
        if node.name.startswith("/Gather") or node.name.startswith("/Cast"):
            node_idx = int(node.name.rsplit("_", 1)[1]) if "_" in node.name else 0
            if node.name in ("/Gather", "/Cast") or 1 <= node_idx <= 15:
                continue
        if node.name.startswith("/Transpose_") or node.name.startswith("/Add"):
            node_idx = int(node.name.rsplit("_", 1)[1]) if "_" in node.name else 0
            if 1 <= node_idx <= 8 or node.name == "/Add":
                continue
        if node.name.startswith("/Constant_"):
            node_idx = int(node.name.rsplit("_", 1)[1])
            if 1 <= node_idx <= 9:
                continue
        new_nodes.append(node)

    del g.node[:]
    g.node.extend(new_nodes)

    used_initializers = {name for node in g.node for name in node.input}
    kept_initializers = [init for init in g.initializer if init.name in used_initializers]
    del g.initializer[:]
    g.initializer.extend(kept_initializers)

    try:
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False, check_type=False)
    except Exception as e:
        print(f"  warning: shape inference failed post-rewrite: {e}")

    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, out_path)
    return 8



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default=str(DECODER_ONNX))
    ap.add_argument("--out", dest="outp", default=str(DECODER_WEBGPU_ONNX))
    ap.add_argument(
        "--replace",
        default="block.4/conv_t1",
        help="comma-separated ConvTranspose node-name substrings to fully replace",
    )
    args = ap.parse_args()

    replace_names = [s.strip() for s in args.replace.split(",") if s.strip()]
    in_real = os.path.realpath(args.inp)
    if not os.path.isfile(in_real) and os.path.isfile(args.outp):
        print(f"raw decoder not found at {in_real}")
        print(f"keeping existing WebGPU decoder: {args.outp}")
        return
    print(f"reading  {in_real}")
    print(f"replacing ConvTranspose->dilate+Conv for: {replace_names}")
    r = rewrite(in_real, args.outp, replace_names)
    print(f"wrote    {args.outp} ({os.path.getsize(args.outp) / 1e6:.1f} MB)")
    print(f"  {r['op_pad_fix']} nodes: output_padding->Pad")
    print(f"  {r['full_rewrite']} nodes: full dilate+Conv replacement")

    fused_count = rewrite_codebook_project(os.path.realpath(args.outp), args.outp)
    print(f"wrote    {args.outp} ({os.path.getsize(args.outp) / 1e6:.1f} MB)")
    print(f"  {fused_count} branches: Gather+MatMul+Add stack -> single fused Gather")


if __name__ == "__main__":
    main()
