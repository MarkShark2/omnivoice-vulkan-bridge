#!/usr/bin/env python3
"""
Convert the VocoLoco OmniVoice main diffusion model from fp32 to fp16.

Strategy (from ORT WebGPU experience):
  - Only the main model is converted; encoder (~654 MB) and decoder (~87 MB)
    stay fp32 because they're already small and the decoder is pinned to WASM
    due to BC-250 Vulkan bugs in Encodec.
  - keep_io_types=True so inputs/outputs remain fp32 — no JS changes needed.
  - op_block_list keeps numerically sensitive ops in fp32:
        LayerNormalization / SimplifiedLayerNormalization
        Softmax / LogSoftmax
        ReduceMean
        RMSNormalization
    Converting these in fp16 is the canonical cause of NaN logits on diffusion
    models. Keeping them fp32 costs almost nothing in throughput.
  - disable_shape_infer=True because `onnx.shape_inference.infer_shapes()`
    serializes the model to a single protobuf and dies on models >2 GB.

Output layout (drop-in replacement for the HF snapshot dir):

    ~/omnivoice/omnivoice-fp16/
        omnivoice-main-fp16.onnx
        omnivoice-main-fp16.onnx_data            (single ~1.2 GB external blob)
        omnivoice-main-fp16-manifest.json        (list of external data files)
        omnivoice-config.json                    (symlinked from source)
        omnivoice-decoder.onnx                   (symlinked from source)
        omnivoice-encoder-fixed.onnx             (symlinked from source)
        tokenizer.json                           (symlinked from source)
        tokenizer_config.json                    (symlinked from source)

The manifest file is read by tts-engine.js to decide which external-data
shards to pre-download into memory before handing them to the ORT session.
After fp16 conversion the total external data is roughly half of fp32
(~1.22 GB vs ~2.45 GB), so a single shard fits under the V8 2 GB ArrayBuffer
limit. If that ever changes (larger future model), set --max-shard-bytes to
force multi-file external data.

Usage:
    python convert_to_fp16.py
    python convert_to_fp16.py --source ./models--Gigsu--vocoloco-onnx --out ./omnivoice-fp16
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import onnx
from onnxconverter_common import float16


# Ops that are numerically unsafe in fp16 on diffusion / transformer models.
# Blocking them globally keeps the variance/normalization path and the
# attention softmax in fp32 while letting every MatMul, Add, Mul, Sub, etc.
# convert to fp16. This is what the onnxconverter_common float16 pass was
# actually designed for, and its op-type-based Cast insertion is the
# well-tested code path — node-name blocking breaks Cast insertion at
# boundaries and produces "Type parameter (T) bound to different types"
# session errors at InferenceSession creation.
#
# Rationale per entry:
#   LayerNormalization / SimplifiedLayerNormalization / RMSNormalization
#     The fused ops — present in some re-exports. Zero-cost to include.
#   Softmax / LogSoftmax
#     Attention softmax always overflows in fp16 without max-subtract
#     protection; the fused op does this in fp32 internally so keep it fp32.
#   Pow
#     Squaring activations for variance computation — the activations are
#     *already* near the upper edge of fp16 in transformer blocks.
#   ReduceMean
#     Denominator of variance. Sums of large-abs values overflow fp16.
#   Sqrt
#     std-dev computation. Also keeps reciprocal-sqrt (1/sqrt) numerically
#     stable since most fp16 fallbacks of Sqrt are inaccurate near zero.
#
# Sub, Mul, Add, Div etc. deliberately STAY fp16 — they're the ops that make
# up the bulk of compute and they're numerically safe.
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
    """Accept either a plain model directory or an HF cache directory."""
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
        # Resolve to the actual blob file so the symlink is stable even if the
        # HF cache gets garbage collected in the future.
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def stage_source_by_copy(snapshot: Path, out_dir: Path) -> Path:
    """
    Copy the 6 source files out of the HF cache (which is all symlinks) into
    a plain scratch directory so onnx.load() can read them without tripping
    the symlink / hardlink security checks added in onnx ≥1.17.

    ~2.5 GB one-time disk cost. Faster and simpler than any bypass trick.
    """
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
            # Already copied in a previous run — skip the I/O.
            continue
        print(f"[convert]   copying {name} ({real.stat().st_size/1e6:.0f} MB)...")
        shutil.copyfile(real, dst)
    return stage


# ── Diagnostics ─────────────────────────────────────────────────────────────

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
            for d in init.dims:
                numel *= d
            bytes_used = numel * itemsize
            entry = stats.setdefault(dt_name, {"count": 0, "bytes": 0})
            entry["count"] += 1
            entry["bytes"] += bytes_used
        for node in graph.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    _walk(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for sg in attr.graphs:
                        _walk(sg)

    _walk(model.graph)
    return stats


def print_initializer_stats(label: str, model: onnx.ModelProto) -> int:
    stats = initializer_stats(model)
    total = sum(v["bytes"] for v in stats.values())
    print(f"[convert] === initializer stats: {label} ===")
    for dt, v in sorted(stats.items(), key=lambda kv: -kv[1]["bytes"]):
        print(f"[convert]   {dt:<10} count={v['count']:>5}  bytes={v['bytes']/1e6:>9.1f} MB")
    print(f"[convert]   TOTAL                   bytes={total/1e6:>9.1f} MB")
    return total


# ── Cast-insertion repair pass ───────────────────────────────────────────────

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
    """
    Post-process an fp16-converted model and insert Cast nodes wherever a node
    receives inputs of mismatched float dtype.

    Background
    ----------
    `onnxconverter_common.float16.convert_float_to_float16` has a long-standing
    bug: when a node is in `op_block_list` (kept fp32) and its output feeds a
    non-blocked node (converted to fp16) whose other input is an initializer
    or constant already converted to fp16, the Cast(fp32→fp16) between them
    gets missed. The resulting graph looks like

        (fp32 from blocked ReduceMean) ─┐
                                        ├─► Add (fp16)  ← session creation fails
        (fp16 epsilon initializer)    ──┘

    ORT rejects this at InferenceSession creation time with
        "Type parameter (T) of Optype (Add) bound to different types".

    Strategy
    --------
    Walk the graph in topological order, tracking each tensor's dtype as we
    go (onnx guarantees graph.node is topo-sorted). For every node:
      1. Collect its inputs that we know are float (fp32 or fp16).
      2. If they disagree, decide on a target dtype:
           - If this node's op_type is in `blocked_op_types` → fp32
           - Otherwise → fp16
         and insert a Cast for every input that doesn't already match.
      3. Propagate the node's output dtype forward for downstream lookups
         (Cast uses `to` attribute, Constant uses its tensor's data_type,
         every other op inherits the effective target dtype).

    Returns the number of Cast nodes inserted.
    """
    dtypes: dict[str, int] = {}

    for init in model.graph.initializer:
        dtypes[init.name] = init.data_type
    for inp in model.graph.input:
        dtypes[inp.name] = inp.type.tensor_type.elem_type
    for vi in model.graph.value_info:
        if vi.type.tensor_type.elem_type:
            dtypes[vi.name] = vi.type.tensor_type.elem_type
    for out in model.graph.output:
        dtypes[out.name] = out.type.tensor_type.elem_type

    new_nodes: list[onnx.NodeProto] = []
    cast_count = 0
    existing_names = {n.name for n in model.graph.node if n.name}

    def _fresh_name(base: str) -> str:
        """Pick a node name that doesn't collide with existing graph names.

        Essential when this function is called more than once on the same
        model (e.g. from a re-repair loop): a second call starts its counter
        at 0 again, so without uniquifying we end up with two nodes named
        `FixCast_1` and ORT rejects the graph at session creation.
        """
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
        # Detect mismatched float inputs -------------------------------------
        float_inputs: list[tuple[int, str, int]] = []
        for i, name in enumerate(node.input):
            if i in _NON_FLOAT_INPUT_INDICES.get(node.op_type, set()):
                continue
            if not name:
                continue
            dt = dtypes.get(name)
            if dt in _FLOAT_TYPES:
                float_inputs.append((i, name, dt))

        repaired_target: int | None = None

        if len(float_inputs) >= 2:
            present = {dt for _, _, dt in float_inputs}
            if len(present) > 1:
                target = (
                    onnx.TensorProto.FLOAT
                    if node.op_type in blocked_op_types
                    else onnx.TensorProto.FLOAT16
                )
                repaired_target = target
                for idx, name, dt in float_inputs:
                    if dt == target:
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
                    node.input[idx] = cast_out
                    dtypes[cast_out] = target

        new_nodes.append(node)

        # Propagate output dtype ---------------------------------------------
        # IMPORTANT: if we just repaired this node by casting inputs to
        # `repaired_target`, the node's output dtype is now `repaired_target`,
        # not whatever the first float input USED to be. Using the old dtype
        # here makes downstream mismatches invisible to this same pass.
        if node.op_type == "Cast":
            to_dt = None
            for attr in node.attribute:
                if attr.name == "to":
                    to_dt = attr.i
                    break
            if to_dt is not None:
                for out in node.output:
                    if out:
                        dtypes[out] = to_dt
        elif node.op_type == "Constant":
            const_dt = None
            for attr in node.attribute:
                if attr.name == "value" and attr.t is not None:
                    const_dt = attr.t.data_type
                    break
                if attr.name == "value_float":
                    const_dt = onnx.TensorProto.FLOAT
                    break
                if attr.name == "value_int" or attr.name == "value_ints":
                    const_dt = onnx.TensorProto.INT64
                    break
            if const_dt is not None:
                for out in node.output:
                    if out:
                        dtypes[out] = const_dt
        else:
            target_out: int | None = repaired_target
            if target_out is None:
                for _, _, dt in float_inputs:
                    target_out = dt
                    break
            if target_out is not None:
                for out in node.output:
                    if out:
                        dtypes[out] = target_out

    if cast_count:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)

    return cast_count


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--source",
        default=str(Path.home() / "omnivoice" / "models--Gigsu--vocoloco-onnx"),
        help="Path to HF cache dir or snapshot dir for Gigsu/vocoloco-onnx",
    )
    ap.add_argument(
        "--out",
        default=str(Path.home() / "omnivoice" / "omnivoice-fp16"),
        help="Output directory for the fp16 model",
    )
    ap.add_argument(
        "--max-shard-bytes",
        type=int,
        default=0,
        help=(
            "If >0, split external data into shards of at most this many bytes. "
            "Leave 0 (default) to write a single external-data file, which works "
            "as long as the fp16 blob stays under the ~2 GB V8 ArrayBuffer limit."
        ),
    )
    args = ap.parse_args()

    source = resolve_snapshot(Path(args.source))
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_onnx = out_dir / "omnivoice-main-fp16.onnx"
    out_data_name = "omnivoice-main-fp16.onnx_data"
    manifest_name = "omnivoice-main-fp16-manifest.json"

    # Nuke any leftover output from a previous failed run so we don't confuse
    # ourselves about which .onnx_data file is current.
    for p in out_dir.glob("omnivoice-main-fp16.onnx*"):
        if p.is_file() or p.is_symlink():
            p.unlink()
    for p in out_dir.glob("omnivoice-main-fp16-*.data"):
        if p.is_file() or p.is_symlink():
            p.unlink()

    print(f"[convert] source snapshot: {source}")
    print(f"[convert] output dir:      {out_dir}")

    # Copy the HF cache blobs (symlinks) into a plain scratch dir so onnx's
    # security checker is happy. ~2.5 GB of disk, cached between runs.
    print("[convert] staging source files by copy (one-time 2.5 GB)...")
    stage = stage_source_by_copy(source, out_dir)
    src_onnx = stage / "omnivoice-main-split.onnx"
    print(f"[convert] staging dir:     {stage}")

    # ── Step 1: load fp32 model (external data auto-resolved next to the .onnx)
    t0 = time.time()
    print("[convert] loading fp32 model with external data...")
    model = onnx.load(str(src_onnx))
    print(f"[convert] loaded in {time.time() - t0:.1f}s")
    print_initializer_stats("fp32 source", model)

    # ── Step 2: fp16 conversion
    print("[convert] converting to fp16...")
    print(f"[convert] op_block_list = {DEFAULT_OP_BLOCK_LIST}")
    t0 = time.time()
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=True,
        op_block_list=DEFAULT_OP_BLOCK_LIST,
    )
    print(f"[convert] fp16 conversion done in {time.time() - t0:.1f}s")

    # Free the fp32 reference before writing (halves memory high-water mark).
    del model

    # Diagnostic: how much of the model is actually fp16 now? If FLOAT still
    # dominates, conversion effectively did nothing and we'd be wasting bytes.
    fp16_total = print_initializer_stats("post-fp16", model_fp16)

    # ── Step 2.5: patch up missing Cast nodes at fp32/fp16 boundaries.
    # See repair_fp16_cast_boundaries() for rationale — this fixes the known
    # cast-insertion bug in onnxconverter_common that makes ORT reject the
    # session with "Type parameter (T) of Optype (X) bound to different types".
    print("[convert] repairing fp32/fp16 Cast boundaries...")
    t0 = time.time()
    inserted = repair_fp16_cast_boundaries(model_fp16, set(DEFAULT_OP_BLOCK_LIST))
    print(f"[convert]   inserted {inserted} Cast node(s) in {time.time() - t0:.1f}s")

    # ── Step 3: decide single-file vs per-tensor external storage.
    # V8 caps a single ArrayBuffer at ~2 GB, so each shard fetched by the JS
    # must be < 2 GB. If fp16 total sits below a safety threshold, one file is
    # dramatically simpler. Otherwise fall back to per-tensor files.
    SAFE_SINGLE_BYTES = int(1.9 * 1024 * 1024 * 1024)
    forced_split = args.max_shard_bytes > 0 and fp16_total > args.max_shard_bytes
    auto_split = fp16_total > SAFE_SINGLE_BYTES
    single_file = not (forced_split or auto_split)

    t0 = time.time()
    if single_file:
        print(f"[convert] packing weights into single blob {out_data_name}...")
        # Use onnx.save — this is the blessed API, and it handles clearing any
        # stale external_data entries on the in-memory tensors before writing.
        onnx.save_model(
            model_fp16,
            str(out_onnx),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=out_data_name,
            size_threshold=1024,
            convert_attribute=False,
        )
        shard_files = [out_data_name]
    else:
        print(
            f"[convert] fp16 total ({fp16_total/1e9:.2f} GB) exceeds single-shard "
            f"safety limit — writing per-tensor external files..."
        )
        # One file per big tensor. Files are named after the tensor, which
        # means we must enumerate the output dir after save to build the
        # manifest. The JS loader doesn't care about filenames, only bytes.
        onnx.save_model(
            model_fp16,
            str(out_onnx),
            save_as_external_data=True,
            all_tensors_to_one_file=False,
            size_threshold=1024,
            convert_attribute=False,
        )
        # Everything that isn't the graph proto, the manifest, or a sibling
        # model/json file is a data shard produced by the save above.
        reserved = {out_onnx.name, manifest_name, *SIBLING_FILES}
        shard_files = sorted(
            p.name
            for p in out_dir.iterdir()
            if p.is_file() and p.name not in reserved
            and not p.name.endswith(".onnx")
            and not p.name.endswith(".json")
        )
    print(f"[convert] wrote model in {time.time() - t0:.1f}s")

    total_bytes = sum((out_dir / f).stat().st_size for f in shard_files)
    print(f"[convert] graph proto:     {out_onnx.name} ({out_onnx.stat().st_size/1e6:.1f} MB)")
    print(f"[convert] external shards: {len(shard_files)} file(s), total {total_bytes/1e6:.0f} MB")

    # ── Step 6: manifest of external-data files (JS reads this and preloads each)
    manifest_path = out_dir / manifest_name
    manifest_path.write_text(json.dumps(shard_files))
    print(f"[convert] wrote manifest:  {manifest_path.name} -> {shard_files}")

    # ── Step 7: link sibling files (config, encoder, decoder, tokenizer) so the
    # output directory is self-contained and can be passed as --model-dir.
    print("[convert] linking sibling files...")
    for fname in SIBLING_FILES:
        src = source / fname
        if not src.exists():
            print(f"[convert]   WARNING: {fname} not found in source, skipping")
            continue
        dst = out_dir / fname
        symlink_or_copy(src, dst)
        print(f"[convert]   {fname} -> {os.readlink(dst) if dst.is_symlink() else '(copied)'}")

    print("\n[convert] done.")
    print(f"[convert] point --model-dir at: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
