# OmniVoice Vulkan Bridge

High-performance OmniVoice text-to-speech on WebGPU/Vulkan. This project runs the ONNX model in Chromium through ONNX Runtime WebGPU, which makes it useful on machines that have Vulkan support but no official CUDA or ROCm stack.

The production model is hosted on Hugging Face:

https://huggingface.co/MarkShark2/omnivoice-onnx-kv-b1-fp16

## What Runs Where

- Web UI: the user's browser downloads the tokenizer and ONNX files directly from Hugging Face and runs synthesis locally with WebGPU.
- API: Python starts FastAPI plus a headless Chromium WebGPU context. Chromium downloads the model from Hugging Face by default and returns OpenAI-style WAV responses.
- CLI: the CLI sends a request to the API and writes the WAV response. It does not load model files itself.

## Browser Web UI

```bash
python omnivoice_webui.py
```

Open the printed `https://<lan-ip>:7860/webui.html` URL. The LAN server uses HTTPS by default because Chrome requires a trustworthy origin for WebGPU and cross-origin isolation. It creates a self-signed development certificate under `.cert/`; accept the browser warning once.

## API Server

```bash
python omnivoice_api.py
```

The API listens on `0.0.0.0:8000` by default and exposes:

- `POST /v1/audio/speech`
- `GET /v1/audio/models`
- `GET /v1/audio/voices`

Example:

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"omnivoice","input":"Hello from OmniVoice.","voice":null}' \
  --output out.wav
```

## CLI

Start the API first, then run:

```bash
python omnivoice_cli.py --text "Hello from OmniVoice." --output out.wav
```

The CLI accepts compatibility flags used by earlier local workflows, but model loading is handled by the API.

## Voice Cloning

Put reference files in `voices/` using the voice id as the filename, for example:

```text
voices/sj_short.mp3
```

Then pass that id to the API or CLI as `voice="sj_short"`. Voice metadata is generated lazily and cached in memory for the life of the API process.

## Working Optimizations

- FP16 production graph: the main diffusion model is exported as fp16 with numerically sensitive ops kept stable for WebGPU/Vulkan hardware.
- KV pre-cache: the fixed prompt/reference prefix is computed once and reused through the diffusion loop.
- B=1 conditional/unconditional split: classifier-free guidance runs as two B=1 passes so the unconditional branch avoids wasted prefix attention work.
- Zero-copy GPU post-processing: `audio_logits` stay on the GPU and a WGSL shader performs guidance fusion and argmax without CPU bounce traffic.
- WebGPU decoder: the decoder graph is patched for ONNX Runtime WebGPU compatibility and runs on the GPU.
- WebGPU encoder: reference-audio encoding runs on WebGPU with a WASM fallback.
- Binary PCM transfer: API synthesis results are posted from Chromium to Python as raw Float32 bytes instead of large JSON arrays.
- Long-form chunk plumbing: text is chunked, generated audio is cross-faded, and generic long-form requests can condition later chunks on the first chunk's audio tokens.