"""Portable path defaults for the KVB1 export scripts.

The bridge repo often lives next to the OmniVoice source checkout, the
Hugging Face cache snapshot, and the generated ONNX output directories. These
defaults follow that layout but every important path can be overridden with an
environment variable for other machines.
"""

from __future__ import annotations

import os
from pathlib import Path


EXPORT_SCRIPT_DIR = Path(__file__).resolve().parent
BRIDGE_DIR = EXPORT_SCRIPT_DIR.parent
WORKSPACE_ROOT = BRIDGE_DIR.parent
OMNIVOICE_HF_REPO_ID = os.environ.get("OMNIVOICE_HF_REPO_ID", "k2-fsa/OmniVoice")


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


OMNIVOICE_SRC = env_path("OMNIVOICE_SRC_DIR", WORKSPACE_ROOT / "OmniVoice")
OMNIVOICE_CACHE_ROOT = env_path(
    "OMNIVOICE_CACHE_ROOT",
    WORKSPACE_ROOT / "models--k2-fsa--OmniVoice",
)
EXPORT_ROOT = env_path("OMNIVOICE_EXPORT_ROOT", EXPORT_SCRIPT_DIR)

B1_KV_DIR = env_path("OMNIVOICE_KV_B1_DIR", EXPORT_ROOT / "omnivoice-main-kv-b1")
B1_FP16_KV_DIR = env_path(
    "OMNIVOICE_KV_FP16_B1_DIR",
    EXPORT_ROOT / "omnivoice-main-kv-fp16-b1",
)

FP16_BUNDLE_DIR = env_path(
    "OMNIVOICE_FP16_BUNDLE_DIR",
    EXPORT_ROOT / "omnivoice-onnx-kv-b1-fp16",
)
TEMPLATE_BUNDLE_DIR = env_path(
    "OMNIVOICE_TEMPLATE_BUNDLE_DIR",
    WORKSPACE_ROOT / "omnivoice-onnx-kv-b1-fp16",
)

DECODER_ONNX = env_path("OMNIVOICE_DECODER_ONNX", FP16_BUNDLE_DIR / "omnivoice-decoder.onnx")
DECODER_WEBGPU_ONNX = env_path("OMNIVOICE_DECODER_WEBGPU_ONNX", FP16_BUNDLE_DIR / "omnivoice-decoder-webgpu.onnx")
ENCODER_ONNX = env_path("OMNIVOICE_ENCODER_ONNX", FP16_BUNDLE_DIR / "omnivoice-encoder-fixed.onnx")


def _snapshot_from_cache(cache_root: Path) -> Path | None:
    if (cache_root / "config.json").is_file():
        return cache_root

    refs = cache_root / "refs" / "main"
    snapshots = cache_root / "snapshots"
    if refs.is_file():
        candidate = snapshots / refs.read_text(encoding="utf-8").strip()
        if (candidate / "config.json").is_file():
            return candidate

    if snapshots.is_dir():
        for entry in sorted(snapshots.iterdir()):
            if (entry / "config.json").is_file():
                return entry

    return None


def resolve_hf_snapshot(cache_root: Path = OMNIVOICE_CACHE_ROOT) -> Path:
    """Resolve k2-fsa/OmniVoice locally, downloading it with huggingface_hub if needed."""
    snapshot = _snapshot_from_cache(cache_root)
    if snapshot is not None:
        return snapshot

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Could not find a local k2-fsa/OmniVoice snapshot and huggingface_hub "
            "is not installed. Install huggingface_hub or set OMNIVOICE_CACHE_ROOT "
            "to a valid local snapshot/cache directory."
        ) from exc

    downloaded = Path(
        snapshot_download(
            repo_id=OMNIVOICE_HF_REPO_ID,
            cache_dir=str(cache_root.parent),
            local_files_only=False,
        )
    ).resolve()
    if not (downloaded / "config.json").is_file():
        raise FileNotFoundError(f"Downloaded snapshot is missing config.json: {downloaded}")
    return downloaded