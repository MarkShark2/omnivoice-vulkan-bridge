#!/usr/bin/env python3
"""Serve the browser-only OmniVoice WebGPU UI on the local network."""

import os
import socket
import ssl
import subprocess
from functools import partial
from http.server import HTTPServer
from pathlib import Path

from server import OmniVoiceHandler

CERT_DIR = Path(__file__).resolve().parent / ".cert"


def _best_effort_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _ensure_dev_cert(lan_ip: str) -> tuple[Path, Path]:
    CERT_DIR.mkdir(exist_ok=True)
    cert_path = CERT_DIR / "omnivoice-webui-cert.pem"
    key_path = CERT_DIR / "omnivoice-webui-key.pem"
    if cert_path.is_file() and key_path.is_file():
        return cert_path, key_path

    san = f"DNS:localhost,IP:127.0.0.1,IP:{lan_ip}"
    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", str(key_path),
        "-out", str(cert_path),
        "-days", "3650",
        "-subj", "/CN=OmniVoice WebUI",
        "-addext", f"subjectAltName={san}",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return cert_path, key_path


def main() -> None:
    static_dir = str(Path(__file__).resolve().parent)
    host = os.environ.get("OMNIVOICE_WEBUI_HOST", "0.0.0.0")
    port = int(os.environ.get("OMNIVOICE_WEBUI_PORT", "7860"))
    lan_ip = _best_effort_lan_ip()
    use_https = _parse_bool(os.environ.get("OMNIVOICE_WEBUI_HTTPS", "true"))
    handler = partial(
        OmniVoiceHandler,
        static_dir=static_dir,
        model_dir=None,
        ref_audio_path=None,
    )
    httpd = HTTPServer((host, port), handler)
    if use_https:
        cert_path, key_path = _ensure_dev_cert(lan_ip)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=cert_path, keyfile=key_path)
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

    display_host = lan_ip if host == "0.0.0.0" else host
    scheme = "https" if use_https else "http"
    print(f"OmniVoice WebGPU UI: {scheme}://{display_host}:{port}/webui.html", flush=True)
    print(f"Local URL: {scheme}://localhost:{port}/webui.html", flush=True)
    if host == "0.0.0.0":
        print(
            "If another device times out, open the port with: "
            f"sudo firewall-cmd --add-port={port}/tcp --permanent && sudo firewall-cmd --reload",
            flush=True,
        )
    if use_https:
        print("Using a local self-signed certificate from .cert/; accept the browser warning once.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()