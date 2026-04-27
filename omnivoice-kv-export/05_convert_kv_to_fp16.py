#!/usr/bin/env python3
"""
Fp16-quantize omnivoice-main-kv.onnx the same way convert_to_fp16.py does
for the original monolithic graph: op_block_list keeps softmax / layernorm
etc. in fp32, and we repair Cast boundaries post-conversion so ORT doesn't
reject the session with "Type parameter bound to different types".

Writes:
    $OMNIVOICE_KV_FP16_DIR/
        omnivoice-main-kv-fp16.onnx
        omnivoice-main-kv-fp16.onnx_data         (single external blob)
        omnivoice-main-kv-fp16-manifest.json     (list of external shards)

The JS side reads the manifest and pre-downloads every shard into memory
before handing them to the ORT session, matching what convert_to_fp16.py
produces for the monolithic model.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import onnx
from onnx import TensorProto

from paths import FP16_KV_DIR, FP32_KV_DIR

# Reuse the shared fp16 helpers copied into this folder so the export bundle is
# self-contained inside the bridge repo.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_to_fp16 import (  # type: ignore
    DEFAULT_OP_BLOCK_LIST,
    initializer_stats,
    print_initializer_stats,
    repair_fp16_cast_boundaries,
)
from onnxconverter_common import float16  # noqa: E402

SRC_DIR = FP32_KV_DIR
SRC_ONNX = SRC_DIR / "omnivoice-main-kv.onnx"
SRC_DATA = SRC_DIR / "omnivoice-main-kv.onnx_data"  # single external blob

OUT_DIR = FP16_KV_DIR
OUT_ONNX = OUT_DIR / "omnivoice-main-kv-fp16.onnx"
OUT_DATA_NAME = "omnivoice-main-kv-fp16.onnx_data"
MANIFEST_NAME = "omnivoice-main-kv-fp16-manifest.json"
SAFE_SINGLE_BYTES = int(1.9 * 1024 * 1024 * 1024)  # V8 ArrayBuffer limit safety


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


def main():
    if not SRC_ONNX.is_file() or not SRC_DATA.is_file():
        print(f"[kv-fp16] missing {SRC_ONNX} or {SRC_DATA}", file=sys.stderr)
        sys.exit(2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in OUT_DIR.glob("omnivoice-main-kv-fp16.onnx*"):
        if p.is_file():
            p.unlink()

    # onnx.load() auto-resolves external data next to the .onnx.
    print(f"[kv-fp16] loading {SRC_ONNX}")
    t0 = time.time()
    model = onnx.load(str(SRC_ONNX))
    print(f"[kv-fp16]   loaded in {time.time()-t0:.1f}s")
    print_initializer_stats("fp32 source", model)

    print("[kv-fp16] converting to fp16...")
    print(f"[kv-fp16] op_block_list = {DEFAULT_OP_BLOCK_LIST}")
    t0 = time.time()
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,            # graph I/O stays fp32
        disable_shape_infer=True,      # protobuf >2 GB breaks shape_inference
        op_block_list=DEFAULT_OP_BLOCK_LIST,
    )
    print(f"[kv-fp16]   convert done in {time.time()-t0:.1f}s")
    del model

    fp16_total = print_initializer_stats("post-fp16", model_fp16)

    print("[kv-fp16] repairing fp32/fp16 Cast boundaries...")
    t0 = time.time()
    inserted = repair_fp16_cast_boundaries(model_fp16, set(DEFAULT_OP_BLOCK_LIST))
    print(f"[kv-fp16]   inserted {inserted} Cast node(s) in {time.time()-t0:.1f}s")

    # --- Surgically promote past_*/present_*/audio_logits I/O to fp16 -------
    #
    # convert_float_to_float16(keep_io_types=True) wraps every fp32 input with
    # a Cast(fp32→fp16) and every fp32 output with a Cast(fp16→fp32). Two
    # reasons we strip those at the boundary:
    #
    #  1. past_key_i / past_value_i / present_key_i / present_value_i: per
    #     diffusion step we re-cast ~1.1 GB of KV state (28 layers × 2 × 20 MB)
    #     twice — once in, once out — which pegs the iGPU memory bus on the
    #     AMD BC-250. The JS side only ever hands ORT fp16 GPU buffers, so
    #     fp16 at the graph boundary is the native layout.
    #
    #  2. audio_logits: the GPU post-processor reads this tensor directly
    #     every step and does a 5-pass log-softmax / CFG / argmax over V.
    #     Keeping the graph output fp32 forced an extra Cast in the LLM head
    #     and doubled the bytes the post-proc shader streams. After promotion
    #     the shader reads fp16 via unpack2x16float and the head's final
    #     cast op is gone.
    #
    # Strategy: walk the graph, delete each boundary Cast, rewire
    # consumers/producers to the I/O tensor directly, retype the I/O to fp16.
    print("[kv-fp16] retyping past_*/present_*/audio_logits I/O to fp16 (skip boundary Casts)...")
    promoted = _promote_kv_io_to_fp16(model_fp16)
    print(f"[kv-fp16]   removed {promoted} boundary Cast node(s)")

    # Second repair pass: promoting the scatter outputs to the graph boundary
    # exposes a dtype mismatch that the Cast(fp16→fp32) was previously
    # hiding — e.g. `Softmax(fp32) → MatMul(·, scatter_fp16)`, where the
    # fp32 softmax output needs to be cast down before the MatMul. We run
    # the repair utility again so it inserts any missing Casts.
    print("[kv-fp16] re-repairing cast boundaries exposed by KV promotion...")
    t0 = time.time()
    more = repair_fp16_cast_boundaries(model_fp16, set(DEFAULT_OP_BLOCK_LIST))
    print(f"[kv-fp16]   inserted {more} additional Cast node(s) in {time.time()-t0:.1f}s")

    # Decide single-file vs per-tensor external data. Same rule as the
    # monolithic script: if we exceed 1.9 GB of weight bytes we can't ship
    # the single-blob layout because V8 caps a single ArrayBuffer at ~2 GB.
    single_file = fp16_total <= SAFE_SINGLE_BYTES

    t0 = time.time()
    if single_file:
        print(f"[kv-fp16] writing single external blob {OUT_DATA_NAME}...")
        onnx.save_model(
            model_fp16, str(OUT_ONNX),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=OUT_DATA_NAME,
            size_threshold=1024,
            convert_attribute=False,
        )
        shard_files = [OUT_DATA_NAME]
    else:
        print(
            f"[kv-fp16] fp16 total {fp16_total/1e9:.2f} GB > safety limit — "
            "writing per-tensor external files"
        )
        onnx.save_model(
            model_fp16, str(OUT_ONNX),
            save_as_external_data=True,
            all_tensors_to_one_file=False,
            size_threshold=1024,
            convert_attribute=False,
        )
        shard_files = sorted(
            p.name for p in OUT_DIR.iterdir()
            if p.is_file()
            and p.name != OUT_ONNX.name
            and p.name != MANIFEST_NAME
            and not p.name.endswith(".onnx")
            and not p.name.endswith(".json")
        )
    print(f"[kv-fp16]   saved in {time.time()-t0:.1f}s")

    total_bytes = sum((OUT_DIR / f).stat().st_size for f in shard_files)
    print(
        f"[kv-fp16] graph proto: {OUT_ONNX.name} ({OUT_ONNX.stat().st_size/1e6:.1f} MB)"
    )
    print(
        f"[kv-fp16] external shards: {len(shard_files)} file(s), "
        f"total {total_bytes/1e6:.0f} MB"
    )

    (OUT_DIR / MANIFEST_NAME).write_text(json.dumps(shard_files))
    print(f"[kv-fp16] wrote manifest {MANIFEST_NAME} -> {shard_files}")


if __name__ == "__main__":
    main()
