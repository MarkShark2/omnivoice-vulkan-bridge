#!/usr/bin/env python3
"""
OmniVoice CLI — thin wrapper around the local OmniVoice API.

This script no longer drives Playwright or WebGPU directly. It only prepares a
request, calls the HTTP API, and writes the returned WAV bytes to disk.
"""

import argparse
import array
import json
import os
import shutil
import time
import struct
import hashlib
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path

from server import DEFAULT_MODEL_REPO_ID, ensure_hf_model_snapshot_dir

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def parse_args():
    """Parse CLI arguments for the API wrapper."""
    parser = argparse.ArgumentParser(
        prog="omnivoice-cli",
        description="OmniVoice TTS via the local HTTP API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python omnivoice_cli.py --text "Hello, world" --output out.wav
  python omnivoice_cli.py --text "Hello" --output out.wav --api-url http://127.0.0.1:8000
  python omnivoice_cli.py --text "Hello" --output out.wav --ref-audio ref.wav --ref-text "Reference."
        """,
    )

    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output", required=True, help="Output WAV file path")

    parser.add_argument(
        "--api-url",
        default=None,
        help="Base URL of the OmniVoice API (default: http://127.0.0.1:<port>)",
    )
    parser.add_argument("--port", type=int, default=8000, help="API port used when --api-url is omitted")

    # Compatibility options retained so existing scripts keep working.
    parser.add_argument("--model-dir", "--model", default=None, help="Legacy compatibility option")
    parser.add_argument("--language", default=None, help="Language code (e.g., en, zh, ja)")
    parser.add_argument("--ref-audio", "--ref_audio", default=None, help="Reference audio for voice cloning")
    parser.add_argument("--ref-text", "--ref_text", default=None, help="Transcript of reference audio")
    parser.add_argument("--instruct", default=None, help="Style instruction text")

    parser.add_argument("--speed", type=float, default=1.0, help="Speed factor (default: 1.0)")
    parser.add_argument("--num-step", "--num_step", type=int, default=20, help="Diffusion steps (default: 20)")
    parser.add_argument(
        "--guidance-scale", "--guidance_scale",
        type=float, default=4.0,
        help="Classifier-free guidance scale (default: 4.0)",
    )
    parser.add_argument("--t-shift", "--t_shift", type=float, default=0.05, help="Time shift (default: 0.05)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic output")
    parser.add_argument("--denoise", type=str, default="true", help="Enable denoising (default: true)")
    parser.add_argument("--pcm-cache", "--pcm_cache", type=str, default="false", help="Legacy compatibility option")
    parser.add_argument("--device", default="webgpu", help="Legacy compatibility option")
    parser.add_argument("--dtype", default="auto", help="Legacy compatibility option")
    parser.add_argument("--duration", type=float, default=None, help="Legacy compatibility option")
    parser.add_argument("--asr-model", default=None, help="Legacy compatibility option")
    parser.add_argument(
        "--layer-penalty-factor", "--layer_penalty_factor",
        type=float, default=5.0, help="Legacy compatibility option",
    )
    parser.add_argument(
        "--position-temperature", "--position_temperature",
        type=float, default=5.0, help="Legacy compatibility option",
    )
    parser.add_argument(
        "--class-temperature", "--class_temperature",
        type=float, default=0.0, help="Legacy compatibility option",
    )
    parser.add_argument(
        "--preprocess-prompt", "--preprocess_prompt",
        type=str, default="true", help="Legacy compatibility option",
    )
    parser.add_argument(
        "--postprocess-output", "--postprocess_output",
        type=str, default="true", help="Legacy compatibility option",
    )
    parser.add_argument(
        "--audio-chunk-duration", "--audio_chunk_duration",
        type=float, default=15.0, help="Legacy compatibility option",
    )
    parser.add_argument(
        "--audio-chunk-threshold", "--audio_chunk_threshold",
        type=float, default=30.0, help="Legacy compatibility option",
    )
    parser.add_argument("--headless", type=str, default="true", help="Legacy compatibility option")

    args = parser.parse_args()
    args.api_url = args.api_url or f"http://127.0.0.1:{args.port}"

    args.denoise = _parse_bool(args.denoise)
    args.pcm_cache = _parse_bool(args.pcm_cache)
    args.headless = _parse_bool(args.headless)
    args.preprocess_prompt = _parse_bool(args.preprocess_prompt)
    args.postprocess_output = _parse_bool(args.postprocess_output)

    return args


def find_default_model_dir():
    """
    Auto-detect or download the production ONNX model in the Hugging Face cache.

    Kept for compatibility with older local tooling; the API no longer calls
    this unless a user explicitly chooses cache-backed model serving.
    """
    return ensure_hf_model_snapshot_dir(DEFAULT_MODEL_REPO_ID)


def read_audio_ffmpeg(path, target_sr=24000):
    """Read any audio file using FFmpeg and return float32 PCM samples."""
    cmd = [
        "ffmpeg", "-i", path,
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(target_sr),
        "-v", "error",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg decoding failed: {e.stderr.decode()}", file=sys.stderr)
        sys.exit(1)

    raw = proc.stdout
    fmt = f"<{len(raw) // 4}f"
    return list(struct.unpack(fmt, raw))


def write_wav(path, pcm, sample_rate):
    """Write float32 PCM data to a 16-bit WAV file."""
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    int_samples = array.array("h")
    for sample in pcm:
        clamped = max(-1.0, min(1.0, float(sample)))
        int_samples.append(int(clamped * 32767))

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())


def _prepare_temp_voice(args):
    """Stage reference audio with a stable voice name so API cache keys persist."""
    if not args.ref_audio:
        return None, []

    source_path = Path(args.ref_audio).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Reference audio not found: {source_path}")

    voices_dir = Path(__file__).parent / "voices"
    voices_dir.mkdir(exist_ok=True)

    # If audio is already in voices/, use its existing stable stem directly.
    if source_path.parent == voices_dir:
        return source_path.stem, []

    # Stable voice id: preserve original filename signal and add path fingerprint.
    # This avoids a new random voice id on every CLI call, which prevents cache misses.
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", source_path.stem).strip("_") or "voice"
    fingerprint = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
    voice_name = f"cli_{safe_name}_{fingerprint}"

    out_suffix = source_path.suffix or ".wav"
    staged_audio_path = voices_dir / f"{voice_name}{out_suffix}"
    shutil.copyfile(source_path, staged_audio_path)

    # Keep staged file in voices/ to survive across calls and restarts.
    return voice_name, []


def _post_json(api_url, payload):
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/v1/audio/speech",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "audio/wav"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60 * 60) as response:
            return response.read(), response.headers
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API request failed ({error.code}): {body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach OmniVoice API at {api_url}: {error.reason}") from error


def _is_api_healthy(api_url, timeout=1.0):
    health_url = f"{api_url.rstrip('/')}/openapi.json"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def _can_autostart_api(api_url):
    parsed = urllib.parse.urlparse(api_url)
    if parsed.scheme != "http":
        return False
    return parsed.hostname in ("127.0.0.1", "localhost")


def _get_api_port(api_url):
    parsed = urllib.parse.urlparse(api_url)
    if parsed.port is not None:
        return parsed.port
    if parsed.scheme == "http":
        return 80
    raise ValueError(f"Cannot infer API port from URL: {api_url}")


def _select_api_python(script_dir):
    venv_python = script_dir.parent / "omnivoice-env" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _start_local_api(api_url):
    if not _can_autostart_api(api_url):
        raise RuntimeError(
            f"API is not reachable at {api_url}, and auto-start is only supported for local http URLs."
        )

    script_dir = Path(__file__).parent
    api_script = script_dir / "omnivoice_api.py"
    python_bin = _select_api_python(script_dir)
    port = _get_api_port(api_url)

    print(f"API not running. Starting local API on port {port}...")
    env = os.environ.copy()
    env["OMNIVOICE_API_PORT"] = str(port)
    proc = subprocess.Popen(
        [python_bin, str(api_script)],
        cwd=str(script_dir),
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
    )

    deadline = time.time() + 180
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Auto-started API exited early with code {proc.returncode}. "
                "Start omnivoice_api.py manually to inspect logs."
            )
        if _is_api_healthy(api_url, timeout=1.5):
            print("API startup complete.")
            return proc
        time.sleep(0.5)

    proc.terminate()
    raise RuntimeError("Timed out waiting for API startup.")


def _stop_local_api(proc):
    if not proc:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_synthesis(args):
    """Send one synthesis request to the API and write the returned WAV bytes."""
    print(f"API URL: {args.api_url}")

    api_proc = None
    if _is_api_healthy(args.api_url):
        print("Using existing API process.")
    else:
        api_proc = _start_local_api(args.api_url)

    voice_name = None
    cleanup_paths = []
    if args.ref_audio:
        voice_name, cleanup_paths = _prepare_temp_voice(args)
        print(f"Reference audio staged as voice: {voice_name}")

    payload = {
        "model": "omnivoice",
        "input": args.text,
        "voice": voice_name,
        "response_format": "wav",
        "speed": args.speed,
        "lang": args.language,
        "ref_text": args.ref_text,
        "instruct": args.instruct,
        "num_step": args.num_step,
        "guidance_scale": args.guidance_scale,
        "t_shift": args.t_shift,
        "denoise": args.denoise,
    }
    if args.seed is not None:
        payload["seed"] = args.seed

    try:
        wav_bytes, headers = _post_json(args.api_url, payload)
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "wb") as output_file:
            output_file.write(wav_bytes)
        print(f"Wrote {len(wav_bytes)} bytes to {args.output}")
        print(f"Content-Type: {headers.get('Content-Type', 'unknown')}")
    finally:
        for path in cleanup_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        _stop_local_api(api_proc)


def main():
    args = parse_args()
    try:
        run_synthesis(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as error:
        print(f"\nError: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
