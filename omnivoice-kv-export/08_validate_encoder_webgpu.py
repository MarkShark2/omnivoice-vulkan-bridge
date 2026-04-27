#!/usr/bin/env python3
"""
encoder_gpu_smoketest.py — Run the STOCK (uninstrumented) encoder on WebGPU
at realistic reference-audio sizes and compare to the CPU reference.

Why we need this in addition to encoder_gpu_bisect.py:
  The bisect tool exposes ~225 intermediate tensors as graph outputs. At
  ~9600 input samples that works fine, but the WebGPU EP keeps every live
  graph-output on the GPU, and at realistic sizes (~3-10s of audio, i.e.
  72000-240000 samples) that balloons to gigabytes of GPU allocations and
  hangs the Vulkan driver on shared-memory APUs like the BC-250. The
  uninstrumented encoder has only one graph output (`audio_codes`) so GPU
  memory stays bounded.

Usage:
    python encoder_gpu_smoketest.py --samples 72000
    python encoder_gpu_smoketest.py --samples 240000  # ~10s at 24kHz
    python encoder_gpu_smoketest.py --ref-audio /path/to/voice.wav
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import struct
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from playwright.async_api import async_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ENCODER_ONNX  # noqa: E402

DEFAULT_ENCODER = str(ENCODER_ONNX)
ORT_WEB_VERSION = "1.20.1"
HOP_LENGTH = 960
SR = 24000


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body><pre id="log"></pre>
<script type="module">
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@__ORT_VERSION__/dist/ort.all.mjs';
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@__ORT_VERSION__/dist/';
ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
ort.env.wasm.simd = true;

const logEl = document.getElementById('log');
function log(m) { logEl.textContent += m + '\n'; console.log(m); }

const BACKEND  = '__BACKEND__';
const MODEL    = '/model.onnx';
const INPUT    = '/input.bin';
const POST_URL = '/result';

window.go = async function go() {
  try {
    log('backend=' + BACKEND + ' ort=' + JSON.stringify(ort.env.versions));
    const ibuf = await (await fetch(INPUT)).arrayBuffer();
    const jsonLen = new Uint32Array(ibuf, 0, 1)[0];
    const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(ibuf, 4, jsonLen)));
    const payloadStart = 4 + jsonLen;
    const fData = new Float32Array(ibuf, payloadStart, meta.numel);
    log('input shape=' + JSON.stringify(meta.shape) + ' numel=' + meta.numel);

    const mbuf = await (await fetch(MODEL)).arrayBuffer();
    log('model bytes=' + mbuf.byteLength);

    const opts = {
      graphOptimizationLevel: 'all',
      enableMemPattern: false,
      enableCpuMemArena: false,
      logSeverityLevel: 2,
    };
    if (BACKEND === 'webgpu') {
      const adapter = await navigator.gpu.requestAdapter();
      const info = adapter.info || {};
      log('adapter: ' + (info.vendor || '?') + ' / ' + (info.device || info.architecture || '?'));
      opts.executionProviders = ['webgpu'];
    } else {
      opts.executionProviders = ['wasm'];
    }
    const t0 = performance.now();
    const sess = await ort.InferenceSession.create(new Uint8Array(mbuf), opts);
    const t1 = performance.now();
    log('session created in ' + (t1 - t0).toFixed(0) + ' ms');

    const inT = new ort.Tensor('float32', fData, meta.shape);
    // Warm-up
    const warm = await sess.run({ input_values: inT });
    for (const k in warm) warm[k].dispose?.();

    const N = 3;
    const times = [];
    let codes = null;
    for (let i = 0; i < N; i++) {
      const ts = performance.now();
      const r = await sess.run({ input_values: inT });
      const te = performance.now();
      times.push(te - ts);
      if (i === N - 1) {
        const t = r.audio_codes;
        codes = new Float32Array(t.data.length);
        for (let j = 0; j < t.data.length; j++) codes[j] = Number(t.data[j]);
        log('audio_codes shape=' + JSON.stringify(t.dims));
      }
      for (const k in r) r[k].dispose?.();
    }
    log('per-run ms: ' + times.map(t => t.toFixed(1)).join(', '));

    const meta2 = { shape: [1, 8, codes.length / 8], per_run_ms: times };
    let hb = new TextEncoder().encode(JSON.stringify(meta2));
    const pad = (4 - ((4 + hb.length) % 4)) % 4;
    if (pad) {
      const p = new Uint8Array(hb.length + pad);
      p.set(hb);
      for (let i = 0; i < pad; i++) p[hb.length + i] = 0x20;
      hb = p;
    }
    const out = new ArrayBuffer(4 + hb.length + codes.byteLength);
    new Uint32Array(out, 0, 1)[0] = hb.length;
    new Uint8Array(out, 4, hb.length).set(hb);
    new Float32Array(out, 4 + hb.length, codes.length).set(codes);
    await fetch(POST_URL, { method: 'POST', body: out });
    return 'ok';
  } catch (e) {
    const msg = 'ERROR: ' + (e && e.message) + '\n' + (e && e.stack);
    log(msg);
    try { await fetch(POST_URL + '?error=1', { method: 'POST', body: msg }); } catch(_) {}
    return msg;
  }
};
</script>
</body></html>
"""


def build_html(backend: str) -> str:
    return (HTML_TEMPLATE
            .replace("__ORT_VERSION__", ORT_WEB_VERSION)
            .replace("__BACKEND__", backend))


class Server:
    def __init__(self, model_path, input_bytes, html):
        self.model_path = model_path
        self.input_bytes = input_bytes
        self.html = html.encode("utf-8")
        self.result_bytes: Optional[bytes] = None
        self.error: Optional[str] = None
        self.done = threading.Event()
        server = self

        class H(BaseHTTPRequestHandler):
            def log_message(self_, f, *a): pass
            def _h(self_, s=200, ct="application/octet-stream", L=None):
                self_.send_response(s)
                self_.send_header("Content-Type", ct)
                if L is not None:
                    self_.send_header("Content-Length", str(L))
                self_.send_header("Access-Control-Allow-Origin", "*")
                self_.send_header("Cross-Origin-Opener-Policy", "same-origin")
                self_.send_header("Cross-Origin-Embedder-Policy", "require-corp")
                self_.end_headers()

            def do_GET(self_):
                p = self_.path.split("?")[0]
                if p in ("/", "/index.html"):
                    self_._h(ct="text/html", L=len(server.html)); self_.wfile.write(server.html)
                elif p == "/model.onnx":
                    sz = os.path.getsize(server.model_path); self_._h(L=sz)
                    with open(server.model_path, "rb") as f:
                        while True:
                            ch = f.read(1 << 20)
                            if not ch: break
                            try: self_.wfile.write(ch)
                            except BrokenPipeError: return
                elif p == "/input.bin":
                    self_._h(L=len(server.input_bytes)); self_.wfile.write(server.input_bytes)
                else:
                    self_._h(status=404, L=0)

            def do_POST(self_):
                L = int(self_.headers.get("Content-Length", "0"))
                b = self_.rfile.read(L)
                if "error" in self_.path:
                    server.error = b.decode("utf-8", errors="replace")
                else:
                    server.result_bytes = b
                self_._h(L=2); self_.wfile.write(b"ok"); server.done.set()

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self._srv.server_address[1]
        self._t = threading.Thread(target=self._srv.serve_forever, daemon=True); self._t.start()

    def wait(self, to=1200.0): self.done.wait(to)
    def close(self): self._srv.shutdown()


def pack(audio: np.ndarray) -> bytes:
    meta = {"shape": list(audio.shape), "numel": int(audio.size)}
    mj = json.dumps(meta).encode("utf-8")
    pad = (-(4 + len(mj))) % 4
    mj = mj + b" " * pad
    buf = io.BytesIO()
    buf.write(struct.pack("<I", len(mj))); buf.write(mj); buf.write(audio.tobytes(order="C"))
    return buf.getvalue()


def unpack(body: bytes):
    (hl,) = struct.unpack("<I", body[:4])
    meta = json.loads(body[4:4 + hl].decode("utf-8"))
    f = np.frombuffer(body, dtype=np.float32, offset=4 + hl)
    codes = f.reshape(meta["shape"]).astype(np.int64)
    return codes, meta


async def run_browser(url, timeout_ms=1_200_000):
    async with async_playwright() as pw:
        args = [
            "--enable-unsafe-webgpu", "--enable-features=Vulkan",
            "--use-angle=vulkan", "--enable-gpu-rasterization",
            "--ignore-gpu-blocklist", "--disable-gpu-sandbox",
            "--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage",
        ]
        br = await pw.chromium.launch(headless=True, args=args)
        try:
            page = await br.new_page()
            page.set_default_timeout(timeout_ms)
            logs = []
            page.on("console", lambda m: logs.append(f"[console] {m.text}"))
            page.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
            await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            r = await page.evaluate("() => window.go()")
            return {"logs": logs, "r": r}
        finally:
            await br.close()


def wav_read(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        n = w.getnframes(); sr = w.getframerate(); ch = w.getnchannels(); sw = w.getsampwidth()
        raw = w.readframes(n)
    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / (2 ** 31)
    else:
        raise ValueError(f"unsupported sampwidth {sw}")
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        # naive linear resample
        ratio = sr / SR
        new_len = int(len(data) / ratio)
        idx = np.arange(new_len) * ratio
        lo = np.floor(idx).astype(int).clip(0, len(data) - 1)
        hi = np.minimum(lo + 1, len(data) - 1)
        fr = idx - lo
        data = (1 - fr) * data[lo] + fr * data[hi]
    return data.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--samples", type=int, default=72000)
    ap.add_argument("--ref-audio", default=None,
                    help="optional .wav file; overrides --samples")
    ap.add_argument("--backend", choices=["webgpu", "wasm"], default="webgpu")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if args.ref_audio:
        pcm = wav_read(args.ref_audio)
        trunc = len(pcm) - (len(pcm) % HOP_LENGTH)
        pcm = pcm[:trunc]
        audio = pcm.reshape(1, 1, -1).astype(np.float32)
        print(f"loaded {args.ref_audio}: {audio.shape[-1]} samples ({audio.shape[-1] / SR:.2f}s)")
    else:
        s = args.samples - (args.samples % HOP_LENGTH)
        if s != args.samples:
            print(f"rounded {args.samples} -> {s} (multiple of {HOP_LENGTH})")
        rng = np.random.default_rng(args.seed)
        audio = (rng.standard_normal((1, 1, s)) * 0.1).astype(np.float32)
        print(f"generated random audio: {audio.shape[-1]} samples ({audio.shape[-1] / SR:.2f}s)")

    enc_path = os.path.realpath(args.encoder)
    print(f"running CPU reference on {enc_path} ...")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(enc_path, so, providers=["CPUExecutionProvider"])
    import time
    t0 = time.perf_counter()
    ref = sess.run(["audio_codes"], {"input_values": audio})[0]
    t1 = time.perf_counter()
    print(f"  CPU ref: {ref.shape} in {(t1 - t0) * 1000:.0f} ms")

    input_bytes = pack(audio)
    html = build_html(args.backend)
    srv = Server(enc_path, input_bytes, html)
    url = f"http://127.0.0.1:{srv.port}/"
    print(f"launching browser backend={args.backend} at {url}")
    try:
        info = asyncio.run(run_browser(url))
    finally:
        srv.wait(30); srv.close()

    for ln in info["logs"]:
        print(f"  {ln}")

    if srv.error:
        print("\nBROWSER ERROR:\n" + srv.error); sys.exit(2)
    if not srv.result_bytes:
        print("no result bytes"); sys.exit(3)

    gpu, meta = unpack(srv.result_bytes)
    print(f"\n{args.backend} codes: {gpu.shape}")
    print(f"per-run ms (3 iters): {meta.get('per_run_ms')}")

    if gpu.shape != ref.shape:
        print(f"SHAPE MISMATCH cpu={ref.shape} gpu={gpu.shape}"); sys.exit(4)
    equal = np.array_equal(ref, gpu)
    mism = int((ref != gpu).sum())
    print(f"\naudio_codes match: {equal}  ({mism} / {ref.size} mismatched tokens)")

    if not equal and ref.size < 500:
        print("CPU codes (first 8 per codebook):")
        for cb in range(ref.shape[1]):
            print(f"  cb{cb} cpu: {ref[0, cb, :16].tolist()}")
            print(f"  cb{cb} gpu: {gpu[0, cb, :16].tolist()}")


if __name__ == "__main__":
    main()
