"""
Local HTTP server for OmniVoice WebGPU CLI.

Serves static files (inference HTML/JS) and ONNX model files from the
HuggingFace cache. Adds required CORS and COOP/COEP headers for
cross-origin isolation (needed for SharedArrayBuffer / multi-threaded WASM).

Also provides a binary PCM result endpoint (/_results/<rid>) so the browser
can POST raw Float32 audio bytes instead of returning them through CDP JSON.
"""

import asyncio
import mimetypes
import os
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

REQUIRED_MODEL_FILES = (
    "omnivoice-config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "omnivoice-main-kv-fp16-b1.onnx",
    "omnivoice-main-kv-fp16-b1.onnx_data",
    "omnivoice-main-kv-fp16-b1-manifest.json",
    "omnivoice-decoder-webgpu.onnx",
    "omnivoice-encoder-fixed.onnx",
)


def _has_required_model_files(model_dir):
    return all(os.path.isfile(os.path.join(model_dir, name)) for name in REQUIRED_MODEL_FILES)

# ─── Binary PCM result coordination ──────────────────────────────────────────
# The browser POSTs raw Float32 PCM bytes to /_results/<rid>.  The API layer
# registers a waiter *before* dispatching page.evaluate, then awaits the result
# after the evaluate returns.  Threading primitives bridge the stdlib HTTP
# server thread and the asyncio event loop in omnivoice_api.py.

_pcm_results: dict[str, bytes] = {}
_pcm_events: dict[str, threading.Event] = {}
_pcm_lock = threading.Lock()


def register_pcm_waiter(rid: str) -> None:
    """Pre-register a slot so the POST handler can deposit bytes into it."""
    with _pcm_lock:
        _pcm_events[rid] = threading.Event()


async def await_pcm_result(rid: str, timeout: float = 120.0) -> bytes:
    """Await the binary PCM bytes that the browser POSTs to /_results/<rid>.

    Runs the blocking wait in a thread so the asyncio loop stays responsive.
    """
    loop = asyncio.get_running_loop()
    evt = _pcm_events.get(rid)
    if evt is None:
        raise KeyError(f"No PCM waiter registered for rid={rid}")

    got_it = await loop.run_in_executor(None, evt.wait, timeout)
    if not got_it:
        cleanup_pcm_waiter(rid)
        raise TimeoutError(f"Timed out waiting for PCM result rid={rid}")

    with _pcm_lock:
        data = _pcm_results.pop(rid, b"")
    return data


def cleanup_pcm_waiter(rid: str) -> None:
    """Remove a waiter slot (idempotent)."""
    with _pcm_lock:
        _pcm_events.pop(rid, None)
        _pcm_results.pop(rid, None)


class OmniVoiceHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves from multiple roots and adds required headers."""

    def __init__(self, *args, static_dir, model_dir, ref_audio_path=None, **kwargs):
        self.static_dir = static_dir
        self.model_dir = model_dir
        self.ref_audio_path = ref_audio_path
        super().__init__(*args, **kwargs)

    def end_headers(self):
        # CORS headers for onnxruntime-web
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        # Cross-origin isolation for SharedArrayBuffer (multi-threaded WASM)
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        """Handle POST /_results/<rid> — receive raw PCM bytes from browser."""
        if self.path.startswith("/_results/"):
            rid = self.path[len("/_results/"):]
            # Strip query string if present
            if "?" in rid:
                rid = rid.split("?")[0]

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b""

            with _pcm_lock:
                evt = _pcm_events.get(rid)
                if evt is None:
                    self.send_error(404, f"No waiter for rid={rid}")
                    return
                _pcm_results[rid] = body
                evt.set()

            self.send_response(204)
            self.end_headers()
            return

        self.send_error(404, "POST not supported for this path")

    def do_GET(self):
        # Route: /models/<filename> → serve from model directory
        if self.path.startswith("/models/"):
            if not self.model_dir:
                self.send_error(404, "No local model directory is configured")
                return
            filename = self.path[len("/models/"):]
            # Strip query string
            if "?" in filename:
                filename = filename.split("?")[0]
            filepath = os.path.join(self.model_dir, filename)
            self._serve_file(filepath)
            return

        # Route: /ref-audio → serve the reference audio file
        if self.path == "/ref-audio" and self.ref_audio_path:
            self._serve_file(self.ref_audio_path)
            return

        # Route: everything else → serve from static directory
        # Map / to /inference.html
        path = self.path
        if "?" in path:
            path = path.split("?")[0]
        if path == "/":
            path = "/inference.html"

        filepath = os.path.join(self.static_dir, path.lstrip("/"))
        self._serve_file(filepath)

    def _serve_file(self, filepath):
        """Serve a file with proper content type and range request support."""
        # Resolve symlinks (HuggingFace cache uses symlinks to blobs)
        filepath = os.path.realpath(filepath)

        if not os.path.isfile(filepath):
            self.send_error(404, f"File not found: {filepath}")
            return

        file_size = os.path.getsize(filepath)
        content_type = self._guess_type(filepath)

        # Handle range requests (important for large model files)
        range_header = self.headers.get("Range")
        if range_header:
            self._serve_range(filepath, file_size, content_type, range_header)
        else:
            self._serve_full(filepath, file_size, content_type)

    def _serve_full(self, filepath, file_size, content_type):
        """Serve the entire file."""
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

        with open(filepath, "rb") as f:
            # Stream in 1MB chunks to avoid loading huge files into memory
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except BrokenPipeError:
                    break

    def _serve_range(self, filepath, file_size, content_type, range_header):
        """Serve a byte range of the file."""
        try:
            range_spec = range_header.replace("bytes=", "")
            start_str, end_str = range_spec.split("-")
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1
        except (ValueError, IndexError):
            self.send_error(416, "Invalid range")
            return

        self.send_response(206)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

        with open(filepath, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk_size = min(1024 * 1024, remaining)
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except BrokenPipeError:
                    break
                remaining -= len(chunk)

    def _guess_type(self, filepath):
        """Guess MIME type from file extension."""
        ext = os.path.splitext(filepath)[1].lower()
        type_map = {
            ".html": "text/html",
            ".js": "application/javascript",
            ".mjs": "application/javascript",
            ".json": "application/json",
            ".onnx": "application/octet-stream",
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".css": "text/css",
        }
        return type_map.get(ext, "application/octet-stream")

    def log_message(self, format, *args):
        """Suppress request logging to keep CLI output clean."""
        pass


def find_model_snapshot_dir(model_dir):
    """
    Resolve the HuggingFace cache model directory to the production snapshot.
    
    Supports:
    - Direct path to a directory containing model files
    - HuggingFace cache structure (models--org--repo/)
    """
    model_dir = os.path.realpath(model_dir)
    
    # Check if this already has model files directly
    if _has_required_model_files(model_dir):
        return model_dir
    
    # Check HuggingFace cache structure
    snapshots_dir = os.path.join(model_dir, "snapshots")
    if os.path.isdir(snapshots_dir):
        # Use refs/main to find the right snapshot, or take the first one
        refs_main = os.path.join(model_dir, "refs", "main")
        if os.path.isfile(refs_main):
            with open(refs_main) as f:
                commit_hash = f.read().strip()
            snapshot = os.path.join(snapshots_dir, commit_hash)
            if os.path.isdir(snapshot) and _has_required_model_files(snapshot):
                return snapshot
        
        # Fallback: first snapshot directory
        for entry in os.listdir(snapshots_dir):
            candidate = os.path.join(snapshots_dir, entry)
            if os.path.isdir(candidate) and _has_required_model_files(candidate):
                return candidate
    
    raise FileNotFoundError(
        f"Could not find the OmniVoice fp16 KV B=1 model bundle in: {model_dir}\n"
        f"Expected either model files directly or a HuggingFace cache structure containing:\n"
        + "\n".join(f"  - {name}" for name in REQUIRED_MODEL_FILES)
    )


def start_server(static_dir, model_dir=None, ref_audio_path=None, port=0):
    """
    Start the HTTP server in a background thread.
    
    Args:
        static_dir: Path to directory containing inference.html and tts-engine.js
        model_dir: Path to directory containing ONNX model files
        ref_audio_path: Optional path to reference audio file for voice cloning
        port: Port to listen on (0 = auto-assign)
    
    Returns:
        (server, port) tuple. Server is running in a daemon thread.
    """
    handler = partial(
        OmniVoiceHandler,
        static_dir=static_dir,
        model_dir=model_dir,
        ref_audio_path=ref_audio_path,
    )
    
    server = HTTPServer(("127.0.0.1", port), handler)
    actual_port = server.server_address[1]
    
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    
    return server, actual_port
