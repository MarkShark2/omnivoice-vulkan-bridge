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


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


OMNIVOICE_SRC = env_path("OMNIVOICE_SRC_DIR", WORKSPACE_ROOT / "OmniVoice")
OMNIVOICE_CACHE_ROOT = env_path(
    "OMNIVOICE_CACHE_ROOT",
    WORKSPACE_ROOT / "models--k2-fsa--OmniVoice",
)
EXPORT_ROOT = env_path("OMNIVOICE_EXPORT_ROOT", WORKSPACE_ROOT)

FP32_KV_DIR = env_path("OMNIVOICE_KV_FP32_DIR", EXPORT_ROOT / "omnivoice-main-kv")
B1_KV_DIR = env_path("OMNIVOICE_KV_B1_DIR", EXPORT_ROOT / "omnivoice-main-kv-b1")
FP16_KV_DIR = env_path("OMNIVOICE_KV_FP16_DIR", EXPORT_ROOT / "omnivoice-main-kv-fp16")
B1_FP16_KV_DIR = env_path(
    "OMNIVOICE_KV_FP16_B1_DIR",
    EXPORT_ROOT / "omnivoice-main-kv-fp16-b1",
)