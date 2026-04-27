#!/usr/bin/env python3
"""
Fp16-quantize the B=1 KV graph exported by 04_export_onnx_b1.py.

Writes:
    $OMNIVOICE_KV_FP16_B1_DIR/
        omnivoice-main-kv-fp16-b1.onnx
        omnivoice-main-kv-fp16-b1.onnx_data          (single external blob)
        omnivoice-main-kv-fp16-b1-manifest.json      (list of external shards)

This uses the same conservative fp16 policy and boundary-Cast repair helpers
as the original fp16 conversion, but keeps only the production B=1 KV path.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import onnx

from paths import B1_FP16_KV_DIR, B1_KV_DIR

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from onnxconverter_common import float16  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "fp16_kv_utils", str(HERE / "fp16_kv_utils.py")
)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
DEFAULT_OP_BLOCK_LIST = _mod.DEFAULT_OP_BLOCK_LIST
print_initializer_stats = _mod.print_initializer_stats
repair_fp16_cast_boundaries = _mod.repair_fp16_cast_boundaries
_promote_kv_io_to_fp16 = _mod._promote_kv_io_to_fp16

SRC_DIR = B1_KV_DIR
SRC_ONNX = SRC_DIR / "omnivoice-main-kv-b1.onnx"
SRC_DATA = SRC_DIR / "omnivoice-main-kv-b1.onnx_data"

OUT_DIR = B1_FP16_KV_DIR
OUT_ONNX = OUT_DIR / "omnivoice-main-kv-fp16-b1.onnx"
OUT_DATA_NAME = "omnivoice-main-kv-fp16-b1.onnx_data"
MANIFEST_NAME = "omnivoice-main-kv-fp16-b1-manifest.json"
SAFE_SINGLE_BYTES = int(1.9 * 1024 * 1024 * 1024)


def main():
    if not SRC_ONNX.is_file() or not SRC_DATA.is_file():
        print(f"[kv-fp16-b1] missing {SRC_ONNX} or {SRC_DATA}", file=sys.stderr)
        sys.exit(2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in OUT_DIR.glob("omnivoice-main-kv-fp16-b1.onnx*"):
        if p.is_file():
            p.unlink()

    print(f"[kv-fp16-b1] loading {SRC_ONNX}")
    t0 = time.time()
    model = onnx.load(str(SRC_ONNX))
    print(f"[kv-fp16-b1]   loaded in {time.time()-t0:.1f}s")
    print_initializer_stats("fp32 source", model)

    print("[kv-fp16-b1] converting to fp16...")
    print(f"[kv-fp16-b1] op_block_list = {DEFAULT_OP_BLOCK_LIST}")
    t0 = time.time()
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=True,
        op_block_list=DEFAULT_OP_BLOCK_LIST,
    )
    print(f"[kv-fp16-b1]   convert done in {time.time()-t0:.1f}s")
    del model

    fp16_total = print_initializer_stats("post-fp16", model_fp16)

    print("[kv-fp16-b1] repairing fp32/fp16 Cast boundaries...")
    t0 = time.time()
    inserted = repair_fp16_cast_boundaries(model_fp16, set(DEFAULT_OP_BLOCK_LIST))
    print(f"[kv-fp16-b1]   inserted {inserted} Cast node(s) in {time.time()-t0:.1f}s")

    print("[kv-fp16-b1] retyping past_*/present_*/audio_logits I/O to fp16...")
    promoted = _promote_kv_io_to_fp16(model_fp16)
    print(f"[kv-fp16-b1]   removed {promoted} boundary Cast node(s)")

    print("[kv-fp16-b1] re-repairing cast boundaries exposed by KV promotion...")
    t0 = time.time()
    more = repair_fp16_cast_boundaries(model_fp16, set(DEFAULT_OP_BLOCK_LIST))
    print(f"[kv-fp16-b1]   inserted {more} additional Cast node(s) in {time.time()-t0:.1f}s")

    single_file = fp16_total <= SAFE_SINGLE_BYTES

    t0 = time.time()
    if single_file:
        print(f"[kv-fp16-b1] writing single external blob {OUT_DATA_NAME}...")
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
            f"[kv-fp16-b1] fp16 total {fp16_total/1e9:.2f} GB > safety limit — "
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
    print(f"[kv-fp16-b1]   saved in {time.time()-t0:.1f}s")

    total_bytes = sum((OUT_DIR / f).stat().st_size for f in shard_files)
    print(
        f"[kv-fp16-b1] graph proto: {OUT_ONNX.name} ({OUT_ONNX.stat().st_size/1e6:.1f} MB)"
    )
    print(
        f"[kv-fp16-b1] external shards: {len(shard_files)} file(s), "
        f"total {total_bytes/1e6:.0f} MB"
    )

    (OUT_DIR / MANIFEST_NAME).write_text(json.dumps(shard_files))
    print(f"[kv-fp16-b1] wrote manifest {MANIFEST_NAME} -> {shard_files}")


if __name__ == "__main__":
    main()
