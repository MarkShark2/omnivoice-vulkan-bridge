#!/usr/bin/env python3
"""Package the fp16 B=1 KV export into the bridge-ready model bundle."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from paths import (
    B1_FP16_KV_DIR,
    FP16_BUNDLE_DIR,
    resolve_hf_snapshot,
    resolve_template_bundle_snapshot,
)

MAIN_FILES = (
    "omnivoice-main-kv-fp16-b1.onnx",
    "omnivoice-main-kv-fp16-b1.onnx_data",
    "omnivoice-main-kv-fp16-b1-manifest.json",
)

TEMPLATE_FILES = (
    ".gitattributes",
    "README.md",
    "omnivoice-decoder.onnx",
    "omnivoice-decoder-webgpu.onnx",
    "omnivoice-encoder-fixed.onnx",
)

SNAPSHOT_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
)

CONFIG_DEFAULTS = {
    "sampling_rate": 24000,
    "frame_rate": 75,
}


def link_or_copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    print(f"[package-b1] staged {destination.name} <- {source}")


def write_runtime_config(source_snapshot: Path, template_snapshot: Path, destination: Path) -> None:
    template_config = template_snapshot / "omnivoice-config.json"
    if template_config.is_file():
        link_or_copy(template_config, destination)
        return

    full_config = json.loads((source_snapshot / "config.json").read_text(encoding="utf-8"))
    runtime_config = {}
    for key in ("audio_vocab_size", "audio_mask_id", "num_audio_codebook", "audio_codebook_weights"):
        runtime_config[key] = full_config[key]
    for key, value in CONFIG_DEFAULTS.items():
        runtime_config.setdefault(key, full_config.get(key, value))

    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.write_text(json.dumps(runtime_config, indent=2) + "\n", encoding="utf-8")
    print(f"[package-b1] wrote {destination.name}")


def main() -> int:
    missing = [str(B1_FP16_KV_DIR / name) for name in MAIN_FILES if not (B1_FP16_KV_DIR / name).is_file()]
    if missing:
        print("[package-b1] missing main export file(s):", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        return 2

    snapshot = resolve_hf_snapshot()
    template_snapshot = resolve_template_bundle_snapshot()
    missing = [str(snapshot / name) for name in SNAPSHOT_FILES if not (snapshot / name).is_file()]
    if missing:
        print("[package-b1] missing Hugging Face snapshot file(s):", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        return 2

    FP16_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    for name in MAIN_FILES:
        link_or_copy(B1_FP16_KV_DIR / name, FP16_BUNDLE_DIR / name)

    write_runtime_config(snapshot, template_snapshot, FP16_BUNDLE_DIR / "omnivoice-config.json")
    for name in SNAPSHOT_FILES:
        link_or_copy(snapshot / name, FP16_BUNDLE_DIR / name)

    for name in TEMPLATE_FILES:
        source = template_snapshot / name
        if source.is_file():
            link_or_copy(source, FP16_BUNDLE_DIR / name)

    required = (
        "omnivoice-config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "omnivoice-main-kv-fp16-b1.onnx",
        "omnivoice-main-kv-fp16-b1.onnx_data",
        "omnivoice-main-kv-fp16-b1-manifest.json",
        "omnivoice-decoder-webgpu.onnx",
        "omnivoice-encoder-fixed.onnx",
    )
    missing = [str(FP16_BUNDLE_DIR / name) for name in required if not (FP16_BUNDLE_DIR / name).is_file()]
    if missing:
        print("[package-b1] bundle is missing required runtime file(s):", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        return 2

    print(f"[package-b1] bundle ready: {FP16_BUNDLE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())