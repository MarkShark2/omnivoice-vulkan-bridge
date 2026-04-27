#!/usr/bin/env python3
"""
Shared fp16 conversion helpers for the production B=1 KV graph. op_block_list keeps softmax / layernorm
etc. in fp32, and we repair Cast boundaries post-conversion so ORT doesn't
reject the session with "Type parameter bound to different types".

These helpers are imported by 05_convert_kv_to_fp16_b1.py.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import onnx
from onnx import TensorProto

from onnxconverter_common import float16  # noqa: E402


DEFAULT_OP_BLOCK_LIST = [
    "LayerNormalization",
    "SimplifiedLayerNormalization",
    "RMSNormalization",
    "Softmax",
    "LogSoftmax",
    "Pow",
    "ReduceMean",
    "Sqrt",
]

SIBLING_FILES = [
    "omnivoice-config.json",
    "omnivoice-decoder.onnx",
    "omnivoice-encoder-fixed.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
]


def resolve_snapshot(source: Path) -> Path:
    """Accept either an export source snapshot or a Hugging Face cache root."""
    source = source.resolve()
    if (source / "omnivoice-main-split.onnx").is_file():
        return source

    snapshots = source / "snapshots"
    if snapshots.is_dir():
        refs_main = source / "refs" / "main"
        if refs_main.is_file():
            commit = refs_main.read_text().strip()
            candidate = snapshots / commit
            if (candidate / "omnivoice-main-split.onnx").is_file():
                return candidate
        for entry in sorted(snapshots.iterdir()):
            if (entry / "omnivoice-main-split.onnx").is_file():
                return entry

    raise FileNotFoundError(
        f"Could not locate omnivoice-main-split.onnx under {source}. "
        "Pass --source pointing at either a snapshot dir or the HF cache root."
    )


def symlink_or_copy(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def stage_source_by_copy(snapshot: Path, out_dir: Path) -> Path:
    """Copy the original ONNX shards into a plain scratch directory."""
    stage = out_dir / ".convert-stage"
    stage.mkdir(parents=True, exist_ok=True)
    needed = [
        "omnivoice-main-split.onnx",
        "omnivoice-main.onnx_data_00",
        "omnivoice-main.onnx_data_01",
        "omnivoice-main.onnx_data_02",
        "omnivoice-main.onnx_data_03",
        "omnivoice-main.onnx_data_04",
    ]
    for name in needed:
        src = snapshot / name
        if not src.exists():
            continue
        dst = stage / name
        real = Path(os.path.realpath(src))
        if dst.exists() and dst.stat().st_size == real.stat().st_size:
            continue
        print(f"[convert]   copying {name} ({real.stat().st_size/1e6:.0f} MB)...")
        shutil.copyfile(real, dst)
    return stage


_DTYPE_ITEMSIZE = {
    onnx.TensorProto.FLOAT: 4,
    onnx.TensorProto.FLOAT16: 2,
    onnx.TensorProto.BFLOAT16: 2,
    onnx.TensorProto.DOUBLE: 8,
    onnx.TensorProto.INT8: 1,
    onnx.TensorProto.UINT8: 1,
    onnx.TensorProto.INT16: 2,
    onnx.TensorProto.UINT16: 2,
    onnx.TensorProto.INT32: 4,
    onnx.TensorProto.UINT32: 4,
    onnx.TensorProto.INT64: 8,
    onnx.TensorProto.UINT64: 8,
    onnx.TensorProto.BOOL: 1,
}


def initializer_stats(model: onnx.ModelProto) -> dict:
    """Return per-dtype counts and total bytes across all graph initializers."""
    stats: dict = {}

    def _walk(graph):
        for init in graph.initializer:
            dt_name = onnx.TensorProto.DataType.Name(init.data_type)
            itemsize = _DTYPE_ITEMSIZE.get(init.data_type, 0)
            numel = 1
            for dim in init.dims:
                numel *= dim
            entry = stats.setdefault(dt_name, {"count": 0, "bytes": 0})
            entry["count"] += 1
            entry["bytes"] += numel * itemsize
        for node in graph.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    _walk(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for subgraph in attr.graphs:
                        _walk(subgraph)

    _walk(model.graph)
    return stats


def print_initializer_stats(label: str, model: onnx.ModelProto) -> int:
    stats = initializer_stats(model)
    total = sum(value["bytes"] for value in stats.values())
    print(f"[convert] === initializer stats: {label} ===")
    for dtype, value in sorted(stats.items(), key=lambda item: -item[1]["bytes"]):
        print(f"[convert]   {dtype:<10} count={value['count']:>5}  bytes={value['bytes']/1e6:>9.1f} MB")
    print(f"[convert]   TOTAL                   bytes={total/1e6:>9.1f} MB")
    return total


_FLOAT_TYPES = {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16}

_NON_FLOAT_INPUT_INDICES = {
    "ConstantOfShape": {0},
    "Expand": {1},
    "Gather": {1},
    "Pad": {1},
    "Reshape": {1},
    "Slice": {1, 2, 3, 4},
    "Squeeze": {1},
    "Unsqueeze": {1},
}


def repair_fp16_cast_boundaries(model: onnx.ModelProto, blocked_op_types: set[str]) -> int:
    """Insert Cast nodes wherever fp16 conversion leaves mismatched float inputs."""
    dtypes: dict[str, int] = {}

    for init in model.graph.initializer:
        dtypes[init.name] = init.data_type
    for inp in model.graph.input:
        dtypes[inp.name] = inp.type.tensor_type.elem_type
    for value_info in model.graph.value_info:
        if value_info.type.tensor_type.elem_type:
            dtypes[value_info.name] = value_info.type.tensor_type.elem_type
    for out in model.graph.output:
        dtypes[out.name] = out.type.tensor_type.elem_type

    new_nodes: list[onnx.NodeProto] = []
    cast_count = 0
    existing_names = {node.name for node in model.graph.node if node.name}

    def _fresh_name(base: str) -> str:
        if base not in existing_names:
            existing_names.add(base)
            return base
        suffix = 0
        while True:
            name = f"{base}_u{suffix}"
            if name not in existing_names:
                existing_names.add(name)
                return name
            suffix += 1

    for node in model.graph.node:
        float_inputs: list[tuple[int, str, int]] = []
        for input_index, name in enumerate(node.input):
            if input_index in _NON_FLOAT_INPUT_INDICES.get(node.op_type, set()):
                continue
            if not name:
                continue
            dtype = dtypes.get(name)
            if dtype in _FLOAT_TYPES:
                float_inputs.append((input_index, name, dtype))

        repaired_target: int | None = None

        if len(float_inputs) >= 2:
            present = {dtype for _, _, dtype in float_inputs}
            if len(present) > 1:
                target = onnx.TensorProto.FLOAT if node.op_type in blocked_op_types else onnx.TensorProto.FLOAT16
                repaired_target = target
                for input_index, name, dtype in float_inputs:
                    if dtype == target:
                        continue
                    cast_count += 1
                    cast_out = f"{name}__fixcast_{cast_count}"
                    cast_node = onnx.helper.make_node(
                        "Cast",
                        inputs=[name],
                        outputs=[cast_out],
                        to=target,
                        name=_fresh_name(f"FixCast_{cast_count}"),
                    )
                    new_nodes.append(cast_node)
                    node.input[input_index] = cast_out
                    dtypes[cast_out] = target

        new_nodes.append(node)

        if node.op_type == "Cast":
            to_dtype = None
            for attr in node.attribute:
                if attr.name == "to":
                    to_dtype = attr.i
                    break
            if to_dtype is not None:
                for out in node.output:
                    if out:
                        dtypes[out] = to_dtype
        elif node.op_type == "Constant":
            const_dtype = None
            for attr in node.attribute:
                if attr.name == "value" and attr.t is not None:
                    const_dtype = attr.t.data_type
                    break
                if attr.name == "value_float":
                    const_dtype = onnx.TensorProto.FLOAT
                    break
                if attr.name in ("value_int", "value_ints"):
                    const_dtype = onnx.TensorProto.INT64
                    break
            if const_dtype is not None:
                for out in node.output:
                    if out:
                        dtypes[out] = const_dtype
        else:
            target_out: int | None = repaired_target
            if target_out is None:
                for _, _, dtype in float_inputs:
                    target_out = dtype
                    break
            if target_out is not None:
                for out in node.output:
                    if out:
                        dtypes[out] = target_out

    if cast_count:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)

    return cast_count

def _promote_kv_io_to_fp16(model: onnx.ModelProto) -> int:
    """Retype past_*/present_* AND audio_logits graph I/O from FLOAT to
    FLOAT16 and drop the Cast nodes that `convert_float_to_float16(
    keep_io_types=True)` put at the graph boundary. Returns the number of
    Cast nodes removed.

    This is safe to call only AFTER `repair_fp16_cast_boundaries` — that
    pass inserts its own Casts at op-block-list boundaries deep in the graph,
    which we don't touch here.

    audio_logits rationale
    ----------------------
    The GPU post-processor in tts-engine.js reads audio_logits once per
    diffusion step and performs a 5-pass log-softmax / CFG fusion / argmax
    over the vocab dim. Keeping the output fp32 at the graph boundary forces
    an extra Cast(fp16→fp32) inside the LLM head plus doubles the bytes the
    post-proc shader streams from VRAM. Promoting the output to fp16 halves
    the post-proc bandwidth (the compute shader now issues f16 loads via
    unpack2x16float) and removes the trailing Cast from the graph.
    """
    graph = model.graph

    def _is_promotable_name(name: str) -> bool:
        return (
            name.startswith("past_key_")
            or name.startswith("past_value_")
            or name.startswith("present_key_")
            or name.startswith("present_value_")
            or name == "audio_logits"
        )

    # --- Retype inputs and outputs ---------------------------------------
    for vi in list(graph.input) + list(graph.output):
        if _is_promotable_name(vi.name):
            vi.type.tensor_type.elem_type = TensorProto.FLOAT16

    input_names = {i.name for i in graph.input if _is_promotable_name(i.name)}
    output_names = {o.name for o in graph.output if _is_promotable_name(o.name)}

    # --- Drop input-side Cast(fp32→fp16) ---------------------------------
    # Pattern: past_key_i → Cast(to=FLOAT16) → cast_out. Multiple distinct
    # Cast nodes may share past_key_i as input (different downstream users).
    # We rewrite every downstream consumer to read past_key_i directly, then
    # drop the Cast.
    nodes = list(graph.node)
    to_remove: list[onnx.NodeProto] = []
    remap: dict[str, str] = {}
    for n in nodes:
        if n.op_type != "Cast":
            continue
        if len(n.input) != 1 or n.input[0] not in input_names:
            continue
        to_type = next((a.i for a in n.attribute if a.name == "to"), None)
        if to_type != TensorProto.FLOAT16:
            continue
        # cast_out may be consumed by multiple nodes; we'll remap them below.
        remap[n.output[0]] = n.input[0]
        to_remove.append(n)

    # --- Drop output-side Cast(fp16→fp32) --------------------------------
    # Pattern: producer → cast_in → Cast(to=FLOAT) → present_key_i
    # We rewire the producer to emit present_key_i directly.
    producer_rename: dict[str, str] = {}
    for n in nodes:
        if n.op_type != "Cast":
            continue
        if len(n.output) != 1 or n.output[0] not in output_names:
            continue
        to_type = next((a.i for a in n.attribute if a.name == "to"), None)
        if to_type != TensorProto.FLOAT:
            continue
        producer_rename[n.input[0]] = n.output[0]
        to_remove.append(n)

    # Apply producer renames. Any existing reference to the scatter output
    # (whether on a .output[] slot of the scatter node itself or on a
    # .input[] slot of a sibling consumer like attention's V-matmul) must be
    # rewritten to the present_* name so the consumer still reads the same
    # tensor. The fp16 scatter output is now the graph output directly.
    if producer_rename:
        for n in graph.node:
            for i, o in enumerate(n.output):
                if o in producer_rename:
                    n.output[i] = producer_rename[o]
            for i, inp in enumerate(n.input):
                if inp in producer_rename:
                    n.input[i] = producer_rename[inp]

    # Apply consumer input remaps (consumer_node.input[i] : cast_out → past_*).
    if remap:
        for n in graph.node:
            for i, inp in enumerate(n.input):
                if inp in remap:
                    n.input[i] = remap[inp]

    # Actually delete the Cast nodes.
    remove_ids = {id(n) for n in to_remove}
    kept = [n for n in graph.node if id(n) not in remove_ids]
    del graph.node[:]
    graph.node.extend(kept)

    return len(to_remove)

