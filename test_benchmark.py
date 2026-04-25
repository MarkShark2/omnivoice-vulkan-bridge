#!/usr/bin/env python3
"""
OmniVoice Benchmark Test Suite

Tests performance with various text lengths and voice cloning configurations.
Requires the OmniVoice server to be running:
  python omnivoice_api.py

Example test matrix:
- 1 sentence (~20-30 chars) without voice cloning
- 2 sentences (~80-100 chars) without voice cloning  
- 3 sentences (~150-200 chars) without voice cloning
- Same tests repeated with voice cloning enabled
- Tests both with fresh voice encoding and cached voices

Results are logged to server.log and can be analyzed with the parse_results() function.
"""

import subprocess
import sys
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# Test cases: (name, text, voice_file_optional)
TEST_CASES = [
    # ===== Without voice cloning (generic voice) =====
    ("generic_1sent", 
     "OmniVoice is a high-performance TTS system.",
     None),
    
    ("generic_2sent",
     "OmniVoice is a high-performance TTS system. It uses WebGPU for acceleration.",
     None),
    
    ("generic_3sent",
     "OmniVoice is a high-performance TTS system. It uses WebGPU for acceleration. This ensures stability on fringe hardware.",
     None),

    ("generic_5sent",
     "OmniVoice is a high-performance TTS system. It uses WebGPU for acceleration. This ensures stability on fringe hardware. "
     "By utilizing Vulkan translation, it runs on many devices. The performance is impressive even on mining cards.",
     None),

    ("generic_7sent",
     "OmniVoice is a high-performance TTS system. It uses WebGPU for acceleration. This ensures stability on fringe hardware. "
     "By utilizing Vulkan translation, it runs on many devices. The performance is impressive even on mining cards. "
     "It supports custom voice cloning with low latency. This makes it ideal for real-time applications.",
     None),

    ("generic_9sent",
     "OmniVoice is a high-performance TTS system. It uses WebGPU for acceleration. This ensures stability on fringe hardware. "
     "By utilizing Vulkan translation, it runs on many devices. The performance is impressive even on mining cards. "
     "It supports custom voice cloning with low latency. This makes it ideal for real-time applications. "
     "Users can easily integrate it via a simple API. The project is open source and community driven.",
     None),
    
    # ===== With voice cloning (sj_short voice) =====
    ("cloned_1sent",
     "OmniVoice is a high-performance TTS system.",
     "sj_short"),
    
    ("cloned_2sent",
     "OmniVoice is a high-performance TTS system. It uses WebGPU for acceleration.",
     "sj_short"),
    
    ("cloned_3sent",
     "OmniVoice is a high-performance TTS system. It uses WebGPU for acceleration. This ensures stability on fringe hardware.",
     "sj_short"),

    ("cloned_5sent",
     "OmniVoice is a high-performance TTS system. It uses WebGPU for acceleration. This ensures stability on fringe hardware. "
     "By utilizing Vulkan translation, it runs on many devices. The performance is impressive even on mining cards.",
     "sj_short"),

    ("cloned_7sent",
     "OmniVoice is a high-performance TTS system. It uses WebGPU for acceleration. This ensures stability on fringe hardware. "
     "By utilizing Vulkan translation, it runs on many devices. The performance is impressive even on mining cards. "
     "It supports custom voice cloning with low latency. This makes it ideal for real-time applications.",
     "sj_short"),

    ("cloned_9sent",
     "OmniVoice is a high-performance TTS system. It uses WebGPU for acceleration. This ensures stability on fringe hardware. "
     "By utilizing Vulkan translation, it runs on many devices. The performance is impressive even on mining cards. "
     "It supports custom voice cloning with low latency. This makes it ideal for real-time applications. "
     "Users can easily integrate it via a simple API. The project is open source and community driven.",
     "sj_short"),
]


def get_api_status(api_url: str = "http://127.0.0.1:8000", timeout: float = 2.0) -> bool:
    """Check if the API server is running and ready."""
    try:
        result = subprocess.run(
            ["curl", "-s", "-f", f"{api_url}/openapi.json"],
            capture_output=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except Exception:
        return False


def run_test(test_name: str, text: str, voice: Optional[str], api_url: str) -> Dict:
    """Run a single test case and return result dict with execution details."""
    output_file = Path(__file__).parent / f"test_output_{test_name}.wav"
    
    # Build CLI command
    cmd = [
        sys.executable,
        "omnivoice_cli.py",
        "--text", text,
        "--output", str(output_file),
        "--api-url", api_url,
        "--pcm-cache", "false",
        "--num-step", "24",
    ]
    
    if voice:
        cmd.extend(["--ref-audio", f"voices/{voice}.mp3"])
    
    print(f"  {test_name:30s}", end=" ", flush=True)
    start = time.time()
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            timeout=300, 
            cwd=Path(__file__).parent
        )
        elapsed = time.time() - start
        
        if result.returncode != 0:
            stderr = result.stderr.decode()[:80]
            print(f"✗ FAILED ({elapsed:.1f}s): {stderr}")
            return {"status": "failed", "elapsed": elapsed}
        
        if output_file.exists():
            file_size = output_file.stat().st_size
            output_file.unlink()  # Clean up
            print(f"✓ OK ({elapsed:.1f}s, {file_size:,} bytes)")
            return {
                "status": "ok",
                "elapsed": elapsed,
                "file_size": file_size,
                "text_len": len(text),
                "voice": voice or "<generic>",
            }
        else:
            print(f"✗ Output file not created")
            return {"status": "failed", "elapsed": elapsed}
            
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT after 300s")
        return {"status": "timeout"}
    except Exception as e:
        print(f"✗ ERROR: {str(e)[:60]}")
        return {"status": "error"}


def parse_logs_for_metrics(server_log_path: Path, start_offset: int = 0) -> List[Dict]:
    """
    Parse server logs to extract timing metrics for each request.
    Returns list of dicts with: chars, load_time, synth_time, total_time, duration, rtf, voice
    """
    if not server_log_path.exists():
        return []
    
    with open(server_log_path) as f:
        if start_offset > 0:
            f.seek(start_offset)
        content = f.read()
    
    results = []
    
    request_re = re.compile(r"speech_request rid=(\w+).*?chars=(\d+)\s+voice=(None|'([^']*)')")
    response_re = re.compile(r"rid=(\w+) response wav.*?duration=([\d.]+)s")
    synth_re = re.compile(r"rid=(\w+) chunk \d+/\d+ SYNTH done .*? sec=([\d.]+)")
    load_re = re.compile(r"rid=(\w+) voice metadata generated for .*? in ([\d.]+) sec")

    request_map: Dict[str, Dict] = {}
    for line in content.splitlines():
        req_m = request_re.search(line)
        if req_m:
            rid = req_m.group(1)
            chars = int(req_m.group(2))
            voice = req_m.group(4) if req_m.group(4) else "<generic>"
            request_map[rid] = {
                "chars": chars,
                "voice": voice,
                "load_time": 0.0,
                "synth_time": 0.0,
                "duration": None,
            }
            continue

        load_m = load_re.search(line)
        if load_m:
            rid = load_m.group(1)
            if rid in request_map:
                request_map[rid]["load_time"] = float(load_m.group(2))
            continue

        synth_m = synth_re.search(line)
        if synth_m:
            rid = synth_m.group(1)
            if rid in request_map:
                request_map[rid]["synth_time"] += float(synth_m.group(2))
            continue

        resp_m = response_re.search(line)
        if resp_m:
            rid = resp_m.group(1)
            if rid in request_map:
                request_map[rid]["duration"] = float(resp_m.group(2))

    for rid, row in request_map.items():
        if row["duration"] is None or row["synth_time"] <= 0:
            continue
        duration = row["duration"]
        synth_time = row["synth_time"]
        load_time = row["load_time"]
        total_time = load_time + synth_time
        
        # RTF based on synthesis time
        rtf = duration / synth_time if synth_time > 0 else 0
        results.append(
            {
                "chars": row["chars"],
                "load_time": load_time,
                "synth_time": synth_time,
                "total_time": total_time,
                "duration": duration,
                "rtf": rtf,
                "voice": row["voice"],
            }
        )
    
    return results


def categorize_results(metrics: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Categorize metrics into generic voice and voice-cloned results."""
    generic = [m for m in metrics if m['voice'] == '<generic>']
    cloned = [m for m in metrics if m['voice'] != '<generic>']
    return generic, cloned


def print_results_table(title: str, results: List[Dict]):
    """Print a formatted results table."""
    print(f"\n{'=' * 90}")
    print(f"{title}")
    print(f"{'=' * 90}")
    
    if not results:
        print("(no results)")
        return
    
    is_cloned = any(r.get('voice') != '<generic>' for r in results)
    
    if not is_cloned:
        # Generic voice table
        print("Chars | Synth (s) | Audio (s) | RTF   ")
        print("------|-----------|----------|-------")
        for r in results:
            print(f"{r['chars']:5d} | {r['synth_time']:9.2f} | {r['duration']:8.2f} | {r['rtf']:5.2f}x")
    else:
        # Voice cloning table  
        print("Chars | Load (s) | Synth (s) | Total (incl load) | Audio (s) | RTF   | Voice")
        print("------|----------|-----------|-------------------|----------|-------|----------------------")
        for r in results:
            voice_short = r['voice'][:20] if len(r['voice']) > 20 else r['voice']
            print(f"{r['chars']:5d} | {r['load_time']:8.2f} | {r['synth_time']:9.2f} | {r['total_time']:17.2f} | {r['duration']:8.2f} | {r['rtf']:5.2f}x | {voice_short}")
    
    if results:
        avg_rtf = sum(r['rtf'] for r in results) / len(results)
        print(f"\nAverage RTF: {avg_rtf:.2f}x (higher is better, > 1.0 is faster than real-time)")


def main():
    api_url = "http://127.0.0.1:8000"
    script_dir = Path(__file__).parent
    server_log_path = script_dir / "server.log"
    
    print("\n" + "=" * 75)
    print("OmniVoice Performance Benchmark Suite")
    print("=" * 75)
    
    # Check API
    print("\nChecking API server status...")
    if not get_api_status(api_url):
        print(f"\n✗ ERROR: API server not responding at {api_url}")
        print("\nPlease start the server in another terminal:")
        print(f"  cd {script_dir}")
        print("  python omnivoice_api.py")
        sys.exit(1)
    print("✓ API server is ready\n")
    
    # Record log size before tests
    log_size_before = server_log_path.stat().st_size if server_log_path.exists() else 0
    
    # Run tests
    print(f"Running {len(TEST_CASES)} test cases...")
    print("-" * 75)
    
    results = {}
    for test_name, text, voice in TEST_CASES:
        result = run_test(test_name, text, voice, api_url)
        results[test_name] = result
        time.sleep(0.5)  # Small delay between tests
    
    # Parse logs to extract metrics
    print("\n" + "-" * 75)
    print("Parsing server logs for timing metrics...")
    time.sleep(2)  # Give server time to write logs
    
    metrics = parse_logs_for_metrics(server_log_path, start_offset=log_size_before)
    
    if not metrics:
        print("⚠ WARNING: No metrics found in server logs")
        print("This may indicate tests failed or logs weren't updated")
        print(f"Log file: {server_log_path}")
        return
    
    # Categorize and display results
    generic_results, cloned_results = categorize_results(metrics)
    
    print_results_table("Results: Without Voice Cloning (Generic Voice)", generic_results)
    print_results_table("Results: With Voice Cloning (sj_short)", cloned_results)
    
    # Export detailed results as JSON
    export_file = script_dir / "benchmark_results.json"
    with open(export_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generic_voice": generic_results,
            "cloned_voice": cloned_results,
        }, f, indent=2)
    print(f"\n✓ Results exported to: {export_file}\n")


if __name__ == "__main__":
    main()
