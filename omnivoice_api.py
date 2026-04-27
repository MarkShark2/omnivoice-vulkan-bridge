import os
import re
import sys
import struct
import math
import logging
import uuid
from logging.handlers import RotatingFileHandler

# Force unbuffered output so logs appear in real-time when redirected to a file
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import time
import asyncio
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
import uvicorn
from pydantic import BaseModel
from contextlib import asynccontextmanager

from playwright.async_api import async_playwright
from server import (
    DEFAULT_MODEL_REPO_ID,
    ensure_hf_model_snapshot_dir,
    start_server,
    register_pcm_waiter,
    await_pcm_result,
    cleanup_pcm_waiter,
)
from omnivoice_cli import read_audio_ffmpeg, write_wav

_LOG_DIR = Path(__file__).resolve().parent
_LOG_FILE = _LOG_DIR / "server.log"
_RUNTIME_DIR = _LOG_DIR / ".runtime"


def _configure_logging() -> logging.Logger:
    lg = logging.getLogger("omnivoice")
    if lg.handlers:
        return lg
    lg.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        fh = RotatingFileHandler(
            _LOG_FILE,
            maxBytes=50 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        lg.addHandler(fh)
    except OSError as e:
        sys.stderr.write(f"[omnivoice] Could not open {_LOG_FILE}: {e}\n")

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    return lg


logger = _configure_logging()


def _browser_console_level(msg_type: str, text: str) -> int:
    if "[W:onnxruntime" in text or re.search(r"\[W:[^\]]*onnxruntime", text):
        return logging.WARNING
    if msg_type == "warning":
        return logging.WARNING
    if msg_type == "error":
        return logging.ERROR
    return logging.DEBUG


API_WEBUI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>OmniVoice API Web UI</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #657181;
      --line: #d9e0ea;
      --accent: #0f766e;
      --accent-strong: #0b5f59;
      --warn: #9a3412;
      --shadow: 0 12px 30px rgba(23, 32, 42, 0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font: 15px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(1120px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }
    h1 { margin: 0; font-size: 28px; font-weight: 720; letter-spacing: 0; }
    .source { color: var(--muted); font-size: 13px; text-align: right; overflow-wrap: anywhere; }
    .workspace {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr);
      gap: 18px;
    }
    section, aside {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .editor, .controls { padding: 18px; }
    label {
      display: block;
      margin: 0 0 7px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }
    textarea, input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }
    textarea {
      min-height: 330px;
      resize: vertical;
      padding: 13px 14px;
    }
    input, select { padding: 10px 11px; }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 13px;
    }
    .field { margin-bottom: 14px; }
    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 16px;
    }
    button, a.download {
      min-height: 40px;
      border: 1px solid transparent;
      border-radius: 6px;
      padding: 0 14px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      font: inherit;
      font-weight: 700;
      text-decoration: none;
      cursor: pointer;
    }
    button.primary { background: var(--accent); color: #fff; }
    button.primary:hover { background: var(--accent-strong); }
    button.secondary, a.download {
      background: #eef2f7;
      color: var(--ink);
      border-color: var(--line);
    }
    button:disabled, a.download[aria-disabled="true"] {
      opacity: 0.55;
      cursor: not-allowed;
      pointer-events: none;
    }
    .status {
      min-height: 42px;
      margin: 0;
      padding: 12px 14px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    .status[data-tone="warn"] { color: var(--warn); }
    audio { width: 100%; margin-top: 14px; }
    @media (max-width: 860px) {
      header { align-items: flex-start; flex-direction: column; }
      .source { text-align: left; }
      .workspace { grid-template-columns: 1fr; }
      textarea { min-height: 250px; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>OmniVoice</h1>
      <div class="source">API-backed synthesis on this server</div>
    </header>
    <div class="workspace">
      <section class="editor">
        <label for="text">Text</label>
        <textarea id="text" spellcheck="true">OmniVoice is running on the API server.</textarea>
      </section>
      <aside>
        <div class="controls">
          <div class="grid">
            <div class="field">
              <label for="steps">Steps</label>
              <input id="steps" type="number" min="4" max="32" step="1" value="24">
            </div>
            <div class="field">
              <label for="speed">Speed</label>
              <input id="speed" type="number" min="0.5" max="2" step="0.05" value="1.0">
            </div>
            <div class="field">
              <label for="guidance">Guidance</label>
              <input id="guidance" type="number" min="0" max="8" step="0.1" value="2.0">
            </div>
            <div class="field">
              <label for="seed">Seed</label>
              <input id="seed" type="number" step="1" value="42">
            </div>
          </div>
          <div class="field">
            <label for="voice">Voice</label>
            <select id="voice"><option value="">Default voice</option></select>
          </div>
          <div class="field">
            <label for="refText">Reference Transcript</label>
            <input id="refText" type="text" placeholder="Optional transcript for selected voice">
          </div>
          <div class="field">
            <label for="instruct">Instruction</label>
            <input id="instruct" type="text" placeholder="calm, clear narration">
          </div>
          <div class="actions">
            <button id="synthesize" class="primary" type="button">Synthesize</button>
            <button id="abort" class="secondary" type="button" disabled>Abort</button>
            <a id="download" class="download" href="#" download="omnivoice.wav" aria-disabled="true">Download WAV</a>
          </div>
          <audio id="audio" controls></audio>
        </div>
        <p id="status" class="status">Ready.</p>
      </aside>
    </div>
  </main>
  <script>
    const els = {
      text: document.getElementById('text'),
      steps: document.getElementById('steps'),
      speed: document.getElementById('speed'),
      guidance: document.getElementById('guidance'),
      seed: document.getElementById('seed'),
      voice: document.getElementById('voice'),
      refText: document.getElementById('refText'),
      instruct: document.getElementById('instruct'),
      synthesize: document.getElementById('synthesize'),
      abort: document.getElementById('abort'),
      download: document.getElementById('download'),
      audio: document.getElementById('audio'),
      status: document.getElementById('status'),
    };
    let currentObjectUrl = null;
    let currentController = null;

    function setStatus(text, tone = '') {
      els.status.textContent = text;
      els.status.dataset.tone = tone;
    }

    async function loadVoices() {
      try {
        const response = await fetch('/v1/audio/voices');
        if (!response.ok) throw new Error(`voices HTTP ${response.status}`);
        const data = await response.json();
        for (const voice of data.voices || []) {
          const option = document.createElement('option');
          option.value = voice.id;
          option.textContent = voice.name || voice.id;
          els.voice.appendChild(option);
        }
      } catch (err) {
        setStatus(`Could not load voices: ${err.message}`, 'warn');
      }
    }

    function resetAudio() {
      if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
      currentObjectUrl = null;
      els.download.href = '#';
      els.download.setAttribute('aria-disabled', 'true');
      els.audio.removeAttribute('src');
    }

    els.synthesize.addEventListener('click', async () => {
      const text = els.text.value.trim();
      if (!text) {
        setStatus('Enter text first.', 'warn');
        return;
      }

      resetAudio();
      els.synthesize.disabled = true;
      els.abort.disabled = false;
      currentController = new AbortController();
      setStatus('Synthesis running on the API server...');

      const payload = {
        model: 'omnivoice',
        input: text,
        voice: els.voice.value || null,
        response_format: 'wav',
        speed: Number(els.speed.value),
        ref_text: els.refText.value.trim() || null,
        instruct: els.instruct.value.trim() || null,
        num_step: Number(els.steps.value),
        guidance_scale: Number(els.guidance.value),
        t_shift: 0.1,
        denoise: true,
      };
      if (els.seed.value !== '') payload.seed = Number(els.seed.value);

      try {
        const started = performance.now();
        const response = await fetch('/v1/audio/speech', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Accept: 'audio/wav' },
          body: JSON.stringify(payload),
          signal: currentController.signal,
        });
        if (!response.ok) {
          const body = await response.text();
          throw new Error(`HTTP ${response.status}: ${body.slice(0, 300)}`);
        }
        const blob = await response.blob();
        currentObjectUrl = URL.createObjectURL(blob);
        els.audio.src = currentObjectUrl;
        els.download.href = currentObjectUrl;
        els.download.removeAttribute('aria-disabled');
        setStatus(`Generated ${Math.round(blob.size / 1024)} KB WAV in ${((performance.now() - started) / 1000).toFixed(1)}s.`);
      } catch (err) {
        if (err.name === 'AbortError') {
          setStatus('Request aborted.', 'warn');
        } else {
          setStatus(err.message, 'warn');
        }
      } finally {
        currentController = null;
        els.synthesize.disabled = false;
        els.abort.disabled = true;
      }
    });

    els.abort.addEventListener('click', () => {
      if (currentController) currentController.abort();
      setStatus('Abort requested.');
    });

    loadVoices();
  </script>
</body>
</html>
"""


def _text_preview(text: str, max_len: int = 160) -> str:
    t = text.replace("\n", "\\n")
    if len(t) <= max_len:
        return repr(t)
    return repr(t[:max_len]) + f"... ({len(text)} chars)"


# Global Playwright state
browser = None
page = None
playwright_context = None
model_base_url = None
synthesis_lock = asyncio.Lock()
_voice_cache = {}  # voice_name -> dict(pcm, refText, tokens)
_file_server_port = None  # set during lifespan, used for binary PCM transfer

class SpeechRequest(BaseModel):
    model: str = "omnivoice"
    input: str
    voice: Optional[str] = None
    response_format: str = "wav"
    speed: float = 1.0
    lang: Optional[str] = None
    ref_text: Optional[str] = None
    instruct: Optional[str] = None
    num_step: int = 24
    guidance_scale: float = 2.0
    t_shift: float = 0.1
    seed: int = 42
    denoise: bool = True

def chunk_text(text, max_chars=250):
    chunks = []
    sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', text) if s.strip()]
    current = ""
    for s in sentences:
        if len(current) + len(s) < max_chars:
            current += (" " + s if current else s)
        else:
            if current: chunks.append(current.strip())
            current = s
    if current: chunks.append(current.strip())
    return chunks if chunks else [text]


def cross_fade_chunks(chunk_arrays: list[list[float]], sample_rate: int,
                      silence_sec: float = 0.3) -> list[float]:
    """Merge raw decoded PCM chunks with stock-style boundary fades."""
    if len(chunk_arrays) <= 1:
        return list(chunk_arrays[0]) if chunk_arrays else []

    total_n = int(silence_sec * sample_rate)
    fade_n = max(1, total_n // 3)
    silence_n = fade_n

    merged = list(chunk_arrays[0])  # copy first chunk

    for chunk in chunk_arrays[1:]:
        # Fade-out tail of merged
        fout_n = min(fade_n, len(merged))
        for i in range(fout_n):
            w = 1.0 - i / fout_n  # 1 → 0
            merged[len(merged) - fout_n + i] *= w

        # Silence gap
        merged.extend([0.0] * silence_n)

        # Fade-in head of next chunk
        chunk_copy = list(chunk)
        fin_n = min(fade_n, len(chunk_copy))
        for i in range(fin_n):
            w = i / fin_n  # 0 → 1
            chunk_copy[i] *= w

        merged.extend(chunk_copy)

    return merged


def pcm_rms(pcm: list[float]) -> float:
    if not pcm:
        return 0.0
    return math.sqrt(sum(float(v) * float(v) for v in pcm) / len(pcm))


def normalize_pcm_peak(pcm: list[float], target_peak: float = 0.5) -> list[float]:
    peak = 0.0
    for v in pcm:
        a = abs(v)
        if a > peak:
            peak = a
    if peak < 1e-6:
        return pcm
    scale = target_peak / peak
    return [v * scale for v in pcm]


def remove_silence_stockish(
    pcm: list[float],
    sample_rate: int,
    mid_sil_ms: int = 500,
    lead_sil_ms: int = 100,
    trail_sil_ms: int = 100,
    silence_threshold_db: float = -50.0,
) -> list[float]:
    """Approximate stock OmniVoice remove_silence() directly on PCM.

    Stock uses pydub's split_on_silence at -50 dBFS, then trims edges while
    keeping short leading/trailing margins. This local version avoids adding
    pydub as a runtime dependency for the bridge.
    """
    if not pcm or sample_rate <= 0:
        return pcm

    threshold = 10.0 ** (silence_threshold_db / 20.0)
    seek_n = max(1, int(sample_rate * 0.010))
    loud = []
    for start in range(0, len(pcm), seek_n):
        frame = pcm[start:start + seek_n]
        if not frame:
            break
        loud.append(pcm_rms(frame) > threshold)

    loud_indices = [i for i, is_loud in enumerate(loud) if is_loud]
    if not loud_indices:
        return pcm

    lead_n = int(sample_rate * lead_sil_ms / 1000.0)
    trail_n = int(sample_rate * trail_sil_ms / 1000.0)
    start_sample = max(0, loud_indices[0] * seek_n - lead_n)
    end_sample = min(len(pcm), (loud_indices[-1] + 1) * seek_n + trail_n)
    trimmed = pcm[start_sample:end_sample]

    mid_sil_n = int(sample_rate * mid_sil_ms / 1000.0)
    if mid_sil_n <= 0 or len(trimmed) <= mid_sil_n:
        return trimmed

    threshold_n = max(1, mid_sil_n // seek_n)
    capped = []
    silence_run_start = None
    frame_count = math.ceil(len(trimmed) / seek_n)

    def frame_is_silent(frame_idx: int) -> bool:
        start = frame_idx * seek_n
        frame = trimmed[start:start + seek_n]
        return pcm_rms(frame) <= threshold

    i = 0
    while i < frame_count:
        if not frame_is_silent(i):
            if silence_run_start is not None:
                run_start_sample = silence_run_start * seek_n
                run_end_sample = i * seek_n
                run_len = i - silence_run_start
                keep_samples = min(run_end_sample - run_start_sample, mid_sil_n)
                if run_len >= threshold_n:
                    capped.extend(trimmed[run_start_sample:run_start_sample + keep_samples])
                else:
                    capped.extend(trimmed[run_start_sample:run_end_sample])
                silence_run_start = None
            capped.extend(trimmed[i * seek_n:min(len(trimmed), (i + 1) * seek_n)])
        elif silence_run_start is None:
            silence_run_start = i
        i += 1

    if silence_run_start is not None:
        run_start_sample = silence_run_start * seek_n
        run_end_sample = len(trimmed)
        run_len = frame_count - silence_run_start
        keep_samples = min(run_end_sample - run_start_sample, mid_sil_n)
        if run_len >= threshold_n:
            capped.extend(trimmed[run_start_sample:run_start_sample + keep_samples])
        else:
            capped.extend(trimmed[run_start_sample:run_end_sample])

    return capped


def fade_and_pad_audio(
    pcm: list[float],
    sample_rate: int,
    pad_sec: float = 0.1,
    fade_sec: float = 0.1,
) -> list[float]:
    if not pcm or sample_rate <= 0:
        return pcm

    out = list(pcm)
    fade_n = min(int(fade_sec * sample_rate), len(out) // 2)
    if fade_n > 0:
        denom = max(1, fade_n - 1)
        for i in range(fade_n):
            out[i] *= i / denom
            out[len(out) - fade_n + i] *= 1.0 - (i / denom)

    pad_n = int(pad_sec * sample_rate)
    if pad_n > 0:
        silence = [0.0] * pad_n
        out = silence + out + silence
    return out


def post_process_assembled_pcm(
    pcm: list[float],
    sample_rate: int,
    ref_rms: Optional[float] = None,
    postprocess_output: bool = True,
) -> list[float]:
    """Stock-style final output pass after all chunks are assembled."""
    out = list(pcm)
    if postprocess_output:
        out = remove_silence_stockish(
            out,
            sample_rate,
            mid_sil_ms=500,
            lead_sil_ms=100,
            trail_sil_ms=100,
        )

    if ref_rms is not None and ref_rms < 0.1:
        out = [v * ref_rms / 0.1 for v in out]
    elif ref_rms is None:
        out = normalize_pcm_peak(out, target_peak=0.5)

    return fade_and_pad_audio(out, sample_rate)

def find_voice_file(voice_name):
    """Scan the voices/ hot folder for a matching filename."""
    if not voice_name:
        return None
    voices_dir = Path(__file__).parent / "voices"
    if not voices_dir.exists():
        return None
    # match extensions
    for ext in ['.mp3', '.wav', '.flac']:
        path = voices_dir / f"{voice_name}{ext}"
        if path.exists():
            return str(path)
    return None


def list_available_voice_names() -> list[str]:
    voices_dir = Path(__file__).parent / "voices"
    if not voices_dir.exists():
        return []

    names = []
    for file in sorted(voices_dir.iterdir()):
        if file.suffix.lower() in {".mp3", ".wav", ".flac"} and not file.stem.startswith("cli_"):
            names.append(file.stem)
    return names


def list_available_models() -> list[dict[str, str]]:
    return [
        {
            "id": "omnivoice",
            "name": "OmniVoice",
        }
    ]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global browser, page, playwright_context, model_base_url, _file_server_port
    
    # 1. Start HTTP server for static files, binary PCM transfer, and the
    # production model bundle from the Hugging Face cache.
    static_dir = str(Path(__file__).parent)
    model_dir = ensure_hf_model_snapshot_dir(DEFAULT_MODEL_REPO_ID)
    logger.info("Mounting Hugging Face cached model bundle: %s", model_dir)
    file_server_thread, port = start_server(static_dir=static_dir, model_dir=model_dir, port=0)
    model_base_url = f"http://127.0.0.1:{port}/models"
    logger.info("Using model base URL: %s", model_base_url)
    _file_server_port = port

    # 2. Launch Playwright
    logger.info("Booting Playwright WebGPU context")
    playwright_context = await async_playwright().start()
    browser_args = [
        "--enable-unsafe-webgpu",
        "--enable-features=Vulkan",
        "--use-angle=vulkan",
        "--enable-gpu-rasterization",
        "--ignore-gpu-blocklist",
        "--disable-gpu-sandbox",
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-dev-shm-usage",
        '--js-flags="--expose-gc --max-old-space-size=4096"',
    ]
    browser = await playwright_context.chromium.launch(headless=True, args=browser_args)
    page = await browser.new_page()
    page.set_default_timeout(600000)
    def _console_log(msg):
        logger.log(_browser_console_level(msg.type, msg.text), "[browser %s] %s", msg.type, msg.text)
    page.on("console", _console_log)
    page.on("pageerror", lambda err: logger.error("[browser] %s", err))
    
    await page.goto(f"http://127.0.0.1:{port}/inference.html", wait_until="networkidle", timeout=30000)

    try:
        await page.wait_for_function("() => window.ttsEngine && typeof window.ttsEngine.init === 'function'", timeout=30000)
    except Exception as e:
        logger.error("tts-engine.js did not finish loading within 30s: %s", e)
        raise

    logger.info("Initializing TTS engine models (may take several minutes on first run)")
    try:
        init_result = await page.evaluate("(url) => window.ttsEngine.init(url)", model_base_url)
        logger.info("Engine ready. Backend: %s", init_result.get("backend", "unknown"))
    except Exception as e:
        logger.exception("Engine init failed: %s", e)
        raise

    logger.info("Voice metadata cache is in-memory and lazy-loaded on first use")

    yield

    logger.info("Shutting down")
    await browser.close()
    await playwright_context.stop()
    file_server_thread.shutdown()


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def api_webui_root():
    return HTMLResponse(API_WEBUI_HTML)


@app.get("/webui", response_class=HTMLResponse)
async def api_webui():
    return HTMLResponse(API_WEBUI_HTML)


@app.get("/v1/audio/models")
async def get_models():
    return {"models": list_available_models()}


@app.get("/v1/audio/voices")
async def get_voices():
    return {
        "voices": [
            {
                "id": name,
                "name": name,
            }
            for name in list_available_voice_names()
        ]
    }

@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    client_port = request.client.port if request.client else "0"
    rid = uuid.uuid4().hex[:12]

    logger.info(
        "speech_request rid=%s ip=%s:%s chars=%s voice=%r speed=%s model=%r",
        rid,
        client_ip,
        client_port,
        len(req.input or ""),
        req.voice,
        req.speed,
        req.model,
    )
    logger.debug("speech_input rid=%s full=%s", rid, repr(req.input))
    logger.debug("speech_input_preview rid=%s %s", rid, _text_preview(req.input or ""))

    ref_audio_pcm = None
    precalculatedRefTokens = None
    ref_text = None
    ref_rms = None
    
    if req.voice:
        if req.voice in _voice_cache:
            logger.info("rid=%s using cached in-memory voice metadata for %r", rid, req.voice)
            vc = _voice_cache[req.voice]
            ref_audio_pcm = vc["pcm"]
            ref_text = req.ref_text or vc["refText"]
            precalculatedRefTokens = vc["tokens"]
            ref_rms = vc.get("refRms")
        else:
            voice_path = find_voice_file(req.voice)
            if voice_path:
                logger.info("rid=%s mapped voice %r -> %s", rid, req.voice, voice_path)
                ref_audio_pcm = read_audio_ffmpeg(voice_path)
                ref_rms = pcm_rms(ref_audio_pcm)
                t_voice_start = time.perf_counter()
                try:
                    generated = await page.evaluate(
                        "(args) => window.ttsEngine.generateVoiceData(args.pcm, args.hint)",
                        {"pcm": ref_audio_pcm, "hint": req.ref_text or ""},
                    )
                except Exception:
                    logger.exception("rid=%s failed to generate voice metadata for %r", rid, req.voice)
                    raise
                t_voice_elapsed = time.perf_counter() - t_voice_start
                logger.info("rid=%s voice metadata generated for %r in %.2f sec", rid, req.voice, t_voice_elapsed)
                ref_text = generated.get("refText") or ""
                precalculatedRefTokens = generated.get("tokens")
                _voice_cache[req.voice] = {
                    "pcm": ref_audio_pcm,
                    "refText": ref_text,
                    "tokens": precalculatedRefTokens,
                    "refRms": ref_rms,
                    "load_time": t_voice_elapsed
                }
            else:
                logger.warning(
                    "rid=%s voice %r not found under voices/; proceeding with generic voice",
                    rid,
                    req.voice,
                )
        
    synth_params = {
        "text": req.input,
        "lang": req.lang,
        "refText": ref_text,
        "instruct": req.instruct,
        "numStep": req.num_step,
        "guidanceScale": req.guidance_scale,
        "tShift": req.t_shift,
        "speed": req.speed,
        "seed": req.seed,
        "denoise": req.denoise,
        "precalculatedRefTokens": precalculatedRefTokens
    }
    # Once voice metadata has been generated or restored from cache, JS only
    # needs the compact audio-token rows. Sending the raw PCM through CDP for
    # every chunk adds seconds of dead serialization work and is ignored by
    # tts-engine.js when precalculatedRefTokens is present.
    if ref_audio_pcm is not None and precalculatedRefTokens is None:
        synth_params["refAudio"] = ref_audio_pcm

    text_chunks = chunk_text(req.input)
    use_generated_chunk_reference = (
        len(text_chunks) > 1
        and ref_audio_pcm is None
        and precalculatedRefTokens is None
    )
    first_chunk_ref_text = None
    first_chunk_ref_tokens = None
    missing_anchor_logged = False
    for i in range(1, len(text_chunks)):
        if text_chunks[i] == text_chunks[i - 1]:
            logger.warning(
                "rid=%s duplicate consecutive text_chunks at %s (len=%s): %s",
                rid,
                i,
                len(text_chunks[i]),
                _text_preview(text_chunks[i]),
            )
    chunk_fps = [
        hashlib.sha256(c.encode("utf-8")).hexdigest()[:16] for c in text_chunks
    ]
    logger.info(
        "rid=%s chunk_plan count=%s lens=%s fps=%s",
        rid,
        len(text_chunks),
        [len(c) for c in text_chunks],
        chunk_fps,
    )
    if use_generated_chunk_reference:
        logger.info(
            "rid=%s generic long-form mode: chunk 1 audio tokens will condition later chunks",
            rid,
        )

    chunk_pcm_arrays = []  # list of per-chunk PCM arrays (not flat)
    gen_sample_rate = 24000
    abort_event = asyncio.Event()

    async def poll_disconnect():
        while not abort_event.is_set():
            if await request.is_disconnected():
                logger.warning("rid=%s client disconnected; aborting WebGPU synthesis", rid)
                abort_event.set()
                await page.evaluate("() => window.ttsEngine.abortGeneration()")
                break
            await asyncio.sleep(0.2)

    poller_task = asyncio.create_task(poll_disconnect())
    
    try:
        async with synthesis_lock:
            for idx, chunk in enumerate(text_chunks):
                if abort_event.is_set():
                    break

                synth_params["text"] = chunk
                synth_params["returnAudioTokens"] = False
                synth_params["refText"] = ref_text
                synth_params["precalculatedRefTokens"] = precalculatedRefTokens
                if use_generated_chunk_reference:
                    synth_params.pop("refAudio", None)
                    if idx == 0:
                        synth_params["refText"] = None
                        synth_params["precalculatedRefTokens"] = None
                        synth_params["returnAudioTokens"] = True
                    elif first_chunk_ref_tokens is not None:
                        synth_params["refText"] = first_chunk_ref_text
                        synth_params["precalculatedRefTokens"] = first_chunk_ref_tokens
                    else:
                        synth_params["refText"] = None
                        synth_params["precalculatedRefTokens"] = None
                        if not missing_anchor_logged:
                            logger.warning(
                                "rid=%s generic chunk reference unavailable; later chunks use standalone generic conditioning",
                                rid,
                            )
                            missing_anchor_logged = True


                t0 = time.perf_counter()
                logger.info(
                    "rid=%s chunk %s/%s SYNTH start fp=%s chars=%s preview=%s",
                    rid,
                    idx + 1,
                    len(text_chunks),
                    chunk_fps[idx],
                    len(chunk),
                    _text_preview(chunk),
                )

                # Binary PCM transfer: register a waiter, pass resultUrl to
                # the browser so it POSTs raw Float32 bytes to the local HTTP
                # server instead of JSON-serializing them through CDP.
                chunk_rid = f"{rid}_{idx}"
                register_pcm_waiter(chunk_rid)
                synth_params["resultUrl"] = f"http://127.0.0.1:{_file_server_port}/_results/{chunk_rid}"

                try:
                    result = await page.evaluate("(params) => window.ttsEngine.synthesize(params)", synth_params)
                    elapsed = time.perf_counter() - t0

                    # Retrieve raw PCM bytes from the HTTP server (already
                    # deposited by the browser's POST to /_results/<chunk_rid>)
                    pcm_bytes = await await_pcm_result(chunk_rid)
                    pcm = list(struct.unpack(f"<{len(pcm_bytes) // 4}f", pcm_bytes))
                except Exception:
                    cleanup_pcm_waiter(chunk_rid)
                    raise

                if use_generated_chunk_reference and idx == 0:
                    first_chunk_ref_tokens = result.get("audioTokens")
                    if first_chunk_ref_tokens:
                        first_chunk_ref_text = chunk
                        logger.info(
                            "rid=%s generic chunk reference captured fp=%s tokens=%s",
                            rid,
                            chunk_fps[idx],
                            len(first_chunk_ref_tokens[0]) if first_chunk_ref_tokens else 0,
                        )
                    else:
                        logger.warning(
                            "rid=%s generic chunk reference requested but browser returned no audioTokens",
                            rid,
                        )
                chunk_pcm_arrays.append(pcm)
                gen_sample_rate = result["sampleRate"]
                logger.info(
                    "rid=%s chunk %s/%s SYNTH done fp=%s samples=%s rate=%s sec=%.2f",
                    rid,
                    idx + 1,
                    len(text_chunks),
                    chunk_fps[idx],
                    len(pcm),
                    result["sampleRate"],
                    elapsed,
                )
    except Exception as e:
        if "ABORTED_BY_CLIENT" in str(e):
            logger.info("rid=%s synthesis aborted by client (499)", rid)
            return Response(status_code=499)
        raise
    finally:
        abort_event.set()
        poller_task.cancel()
        
    if not chunk_pcm_arrays:
        logger.warning("rid=%s empty PCM after synthesis (499)", rid)
        return Response(status_code=499)

    # Cross-fade raw decoded chunks, then apply the stock-style output
    # processing once to the assembled utterance.
    all_pcm = cross_fade_chunks(chunk_pcm_arrays, gen_sample_rate)
    all_pcm = post_process_assembled_pcm(
        all_pcm,
        gen_sample_rate,
        ref_rms=ref_rms,
        postprocess_output=True,
    )

    duration_s = len(all_pcm) / float(gen_sample_rate) if gen_sample_rate else 0.0
    logger.info(
        "rid=%s response wav chunks=%s samples=%s rate=%s duration=%.2fs bytes_est=%s",
        rid,
        len(chunk_pcm_arrays),
        len(all_pcm),
        gen_sample_rate,
        duration_s,
        len(all_pcm) * 2 + 44,
    )

    # Write to a local runtime file, then read as bytes (OpenAI returns raw WAV payload).
    import tempfile

    _RUNTIME_DIR.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", dir=_RUNTIME_DIR, delete=False) as tmp:
        tmp_path = tmp.name
    write_wav(tmp_path, all_pcm, gen_sample_rate)

    with open(tmp_path, "rb") as f:
        wav_bytes = f.read()
    os.remove(tmp_path)

    return Response(content=wav_bytes, media_type="audio/wav")

if __name__ == "__main__":
    logger.info("Starting uvicorn; app log file: %s", _LOG_FILE)
    port = int(os.environ.get("OMNIVOICE_API_PORT", "8000"))
    uvicorn.run("omnivoice_api:app", host="0.0.0.0", port=port)
