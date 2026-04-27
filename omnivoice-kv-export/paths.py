"""Portable path defaults for the KVB1 export scripts.

Model files are resolved through Hugging Face with snapshot_download(); local
workspace cache directories are intentionally not part of the export contract.
Only source-code and generated-output paths are configurable here.
"""

from __future__ import annotations

import os
from pathlib import Path


EXPORT_SCRIPT_DIR = Path(__file__).resolve().parent
BRIDGE_DIR = EXPORT_SCRIPT_DIR.parent
WORKSPACE_ROOT = BRIDGE_DIR.parent
OMNIVOICE_HF_REPO_ID = os.environ.get("OMNIVOICE_HF_REPO_ID", "k2-fsa/OmniVoice")
OMNIVOICE_TEMPLATE_BUNDLE_REPO_ID = os.environ.get(
    "OMNIVOICE_TEMPLATE_BUNDLE_REPO_ID",
    "MarkShark2/omnivoice-onnx-kv-b1-fp16",
)


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


OMNIVOICE_SRC = env_path("OMNIVOICE_SRC_DIR", WORKSPACE_ROOT / "OmniVoice")
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
DECODER_ONNX = env_path("OMNIVOICE_DECODER_ONNX", FP16_BUNDLE_DIR / "omnivoice-decoder.onnx")
DECODER_WEBGPU_ONNX = env_path("OMNIVOICE_DECODER_WEBGPU_ONNX", FP16_BUNDLE_DIR / "omnivoice-decoder-webgpu.onnx")
ENCODER_ONNX = env_path("OMNIVOICE_ENCODER_ONNX", FP16_BUNDLE_DIR / "omnivoice-encoder-fixed.onnx")


def _snapshot_download(repo_id: str, required_file: str) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for the OmniVoice export pipeline. "
            "Install it with `pip install huggingface_hub`."
        ) from exc

    downloaded = Path(snapshot_download(repo_id=repo_id, local_files_only=False)).resolve()
    if not (downloaded / required_file).is_file():
        raise FileNotFoundError(f"Downloaded snapshot for {repo_id} is missing {required_file}: {downloaded}")
    return downloaded


def resolve_hf_snapshot() -> Path:
    """Resolve k2-fsa/OmniVoice through the official Hugging Face cache."""
    return _snapshot_download(OMNIVOICE_HF_REPO_ID, "config.json")


def resolve_template_bundle_snapshot() -> Path:
    """Resolve the production bundle template through the official Hugging Face cache."""
    return _snapshot_download(OMNIVOICE_TEMPLATE_BUNDLE_REPO_ID, "omnivoice-config.json")
