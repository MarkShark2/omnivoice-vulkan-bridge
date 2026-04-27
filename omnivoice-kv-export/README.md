# OmniVoice KVB1 Export Tools

This folder contains the trimmed export path for the production `kv-fp16-b1` main model used by the Vulkan bridge. It is not runtime code; it records the scripts needed to reproduce the B=1 KV ONNX graph and convert it to fp16 for ONNX Runtime WebGPU.

## What Is Included

- `01_load_and_sanity.py`: verifies the PyTorch checkpoint and KV-cache idea on CPU.
- `02_kv_plumbing.py`: measures the cache plumbing and approximation behavior before export.
- `03_wrapper_parity.py`: checks `OmniVoiceKvWrapper` against the original model forward path.
- `kv_wrapper.py`: wraps OmniVoice with fixed-length scatter-style KV cache inputs/outputs.
- `04b_export_onnx_b1.py`: exports the B=1 KV fp32 ONNX graph.
- `05b_convert_kv_to_fp16_b1.py`: converts the B=1 graph to the fp16 production graph.
- `05_convert_kv_to_fp16.py` and `convert_to_fp16.py`: shared fp16 conversion helpers used by the B=1 converter.

Quantization, portable-model experiments, B=2 exports, and packaging scripts are intentionally omitted because they do not directly produce the current bridge model.

## Portable Paths

Path defaults are defined in `paths.py`. By default, the scripts assume this layout:

```text
workspace/
  OmniVoice/
  models--k2-fsa--OmniVoice/
  omnivoice-main-kv-b1/
  omnivoice-main-kv-fp16-b1/
  omnivoice-vulkan-bridge/
```

Override paths with environment variables when your checkout is different:

```bash
export OMNIVOICE_SRC_DIR=/path/to/OmniVoice
export OMNIVOICE_CACHE_ROOT=/path/to/models--k2-fsa--OmniVoice
export OMNIVOICE_EXPORT_ROOT=/path/for/generated-models
```

More specific output overrides are also available: `OMNIVOICE_KV_B1_DIR`, `OMNIVOICE_KV_FP16_B1_DIR`, `OMNIVOICE_KV_FP32_DIR`, and `OMNIVOICE_KV_FP16_DIR`.

## Reproducing The Current Model

Run from this directory after installing the OmniVoice export dependencies:

```bash
python 01_load_and_sanity.py
python 02_kv_plumbing.py
python 03_wrapper_parity.py
python 04b_export_onnx_b1.py
python 05b_convert_kv_to_fp16_b1.py
```

The output expected by the bridge is `omnivoice-main-kv-fp16-b1.onnx`, its external data file, and `omnivoice-main-kv-fp16-b1-manifest.json` in `OMNIVOICE_KV_FP16_B1_DIR`.