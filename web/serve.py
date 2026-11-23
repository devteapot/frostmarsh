#!/usr/bin/env python3
"""Lightweight local HTTP server for the Frostmarsh WebGPU Browser Client.

Serves web/ on http://127.0.0.1:8080 with proper .wasm MIME types and
Cross-Origin-Opener-Policy / Cross-Origin-Embedder-Policy headers.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import mimetypes
import os
import socketserver
from pathlib import Path


class FrostmarshHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".js": "application/javascript; charset=utf-8",
        ".mjs": "application/javascript; charset=utf-8",
        ".wgsl": "text/plain; charset=utf-8",
        ".bin": "application/octet-stream",
        ".json": "application/json; charset=utf-8",
        ".html": "text/html; charset=utf-8",
    }

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


class ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Frostmarsh WebGPU browser client")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    args = parser.parse_args()

    mimetypes.add_type("application/wasm", ".wasm")
    mimetypes.add_type("application/javascript", ".js")

    web_dir = Path(__file__).resolve().parent
    handler = functools.partial(FrostmarshHTTPRequestHandler, directory=str(web_dir))

    with ReusableTCPServer((args.host, args.port), handler) as httpd:
        url = f"http://{args.host}:{args.port}"
        print(f"[Frostmarsh WebGPU] Serving {web_dir} at {url}")
        print("[Frostmarsh WebGPU] COOP/COEP + application/wasm enabled. Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[Frostmarsh WebGPU] Server stopped.")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
