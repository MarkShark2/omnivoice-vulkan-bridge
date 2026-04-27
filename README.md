# OmniVoice Vulkan Bridge

High-performance OmniVoice text-to-speech on WebGPU/Vulkan. This project runs the ONNX model in Chromium through ONNX Runtime WebGPU, which makes it useful on machines that have Vulkan support but no official CUDA or ROCm stack.

I created this project specifically to work on the AMD BC-250, which does not have ROCm/CUDA support.

The production model is hosted on Hugging Face:

https://huggingface.co/MarkShark2/omnivoice-onnx-kv-b1-fp16

## What Runs Where

- API: Python starts FastAPI plus a headless Chromium WebGPU context on the BC-250. Chromium downloads or serves the B=1 split model bundle and returns OpenAI-style WAV responses.
- CLI: the CLI sends a request to the API and writes the WAV response. It does not load model files itself.

## AMD BC-250 Benchmarks

Benchmarks were run on the AMD BC-250 through Chromium WebGPU/Vulkan with the fp16 B=1 KV model at 24 diffusion steps. Longer requests settle around 1.5x real time, measured as generated audio seconds per wall-clock synthesis second.

| Case | Text | Audio | Synthesis | Speed |
| --- | ---: | ---: | ---: | ---: |
| Generic voice | 43 chars | 2.9s | 4.4s | 0.65x |
| Generic voice | 425 chars | 26.0s | 16.8s | 1.55x |
| Cloned voice, warm | 76 chars | 6.0s | 4.3s | 1.39x |
| Cloned voice, warm | 425 chars | 31.5s | 20.8s | 1.52x |

The first cloned-voice request for `sj_short` also generated voice metadata, which added 8.7s once for that API process.

## API Server

```bash
python omnivoice_api.py
```

The API listens on `0.0.0.0:8000` by default and exposes:

- `POST /v1/audio/speech`
- `GET /v1/audio/models`
- `GET /v1/audio/voices`

It also serves a simple API-backed web UI at:

```text
http://<lan-ip>:8000/webui
```

That page runs no model code in the browser; it sends requests to the API and plays the returned WAV.

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
