# OmniVoice KVB1 Export Tools

This folder contains the production export path for the OmniVoice Vulkan bridge. It is not runtime code; it records the scripts needed to reproduce the B=1 KV fp16 main model and the WebGPU-ready encoder/decoder artifacts used by the API.

## What Is Included

- `01_load_and_sanity.py`: verifies the PyTorch checkpoint and KV-cache idea on CPU.
- `02_kv_plumbing.py`: measures the cache plumbing and approximation behavior before export.
- `03_wrapper_parity.py`: checks `OmniVoiceKvWrapper` against the original model forward path.
- `kv_wrapper.py`: wraps OmniVoice with fixed-length scatter-style KV cache inputs/outputs.
- `04_export_onnx_b1.py`: exports the B=1 KV fp32 ONNX graph.
- `05_convert_kv_to_fp16_b1.py`: converts the B=1 graph to the fp16 production graph.
- `06_package_bundle.py`: stages the bridge-ready `omnivoice-onnx-kv-b1-fp16` bundle.
- `fp16_kv_utils.py`: shared fp16 conversion and KV I/O promotion helpers used by the B=1 converter.
- `07_patch_decoder_webgpu.py`: patches the decoder graph for ORT-Web WebGPU by applying the ConvTranspose fixes and fused codebook projection table.
- `08_validate_encoder_webgpu.py`: validates the production encoder on ORT-Web WebGPU at realistic reference-audio sizes.

Failed INT8/Q4 quantization experiments, portable-model experiments, B=2 exports, and non-production packaging scripts are intentionally omitted because they do not directly produce the current bridge model.

## Portable Paths

Path defaults are defined in `paths.py`. The export scripts use `huggingface_hub` to resolve `k2-fsa/OmniVoice`, downloading it into the Hugging Face cache if the configured cache root is missing. By default, the scripts assume this layout:

```text
workspace/
  OmniVoice/
  models--k2-fsa--OmniVoice/
  omnivoice-vulkan-bridge/
    omnivoice-kv-export/
      omnivoice-main-kv-b1/
      omnivoice-main-kv-fp16-b1/
      omnivoice-onnx-kv-b1-fp16/
```

Override paths with environment variables when your checkout is different:

```bash
export OMNIVOICE_SRC_DIR=/path/to/OmniVoice
export OMNIVOICE_CACHE_ROOT=/path/to/models--k2-fsa--OmniVoice
export OMNIVOICE_EXPORT_ROOT=/path/for/generated-models
```

More specific overrides are also available: `OMNIVOICE_HF_REPO_ID`, `OMNIVOICE_KV_B1_DIR`, `OMNIVOICE_KV_FP16_B1_DIR`, `OMNIVOICE_FP16_BUNDLE_DIR`, `OMNIVOICE_TEMPLATE_BUNDLE_DIR`, `OMNIVOICE_DECODER_ONNX`, `OMNIVOICE_DECODER_WEBGPU_ONNX`, and `OMNIVOICE_ENCODER_ONNX`.

## Reproducing The Current Model

Run from this directory after installing the OmniVoice export dependencies:

```bash
python 01_load_and_sanity.py
python 02_kv_plumbing.py
python 03_wrapper_parity.py
python 04_export_onnx_b1.py
python 05_convert_kv_to_fp16_b1.py
python 06_package_bundle.py
python 07_patch_decoder_webgpu.py
python 08_validate_encoder_webgpu.py --samples 72000
```

The bridge-ready output is `omnivoice-onnx-kv-b1-fp16/` inside this folder by default. It contains the fp16 B=1 main model, tokenizer/config files, `omnivoice-decoder-webgpu.onnx`, and `omnivoice-encoder-fixed.onnx`.