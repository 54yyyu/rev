"""`rev serve` - Jev's HTTP endpoint, on this machine.

    rev serve                      # http://127.0.0.1:8421
    curl -s localhost:8421/v1/systemone -d '{"state": "...", "questions": {...}}'

The route, request and response are TypeSafe's, so a client written for Jev
works by changing its base URL; the Authorization header and `model` field are
accepted and ignored. One process holds the weights, so every tool on the
machine shares one copy instead of loading its own.

Requests are answered one at a time. MLX runs one forward pass at a time on
the GPU anyway, and serialising keeps each answer independent of what else
arrived, which is what `tests/test_determinism.py` guarantees.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_PORT = 8421
MAX_BODY = 4 * 1024 * 1024


def _port_taken(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def make_handler(rev, lock: threading.Lock, orders: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "rev"

        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") in ("", "/health"):
                return self._send(200, {"ok": True, "model": rev.name, "capacity": rev.capacity})
            self._send(404, {"error": f"no route {self.path}"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                return self._send(404, {"error": f"no route {self.path}; POST /v1/systemone"})
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._send(413, {"error": "request body too large"})
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if "state" not in body:
                    raise ValueError("missing state")
                with lock:
                    started = time.perf_counter()
                    out = rev.ask(body["state"], body.get("questions"), orders=orders)
                out["seconds"] = round(time.perf_counter() - started, 4)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                return self._send(400, {"error": str(e)})
            self._send(200, out)

        def log_message(self, fmt, *args):
            sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

    return Handler


def serve(argv=None) -> int:
    p = argparse.ArgumentParser(prog="rev serve", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--bits", type=int, default=8, choices=(4, 8, 0))
    p.add_argument("--host", default="127.0.0.1",
                   help="default 127.0.0.1; 0.0.0.0 exposes it to the network")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--orders", default="auto", choices=("one", "two", "auto"))
    a = p.parse_args(argv)

    if _port_taken("127.0.0.1" if a.host == "0.0.0.0" else a.host, a.port):
        p.error(f"port {a.port} is already in use; pick another with --port")

    from .decide import Rev

    print(f"loading {a.model}...", file=sys.stderr)
    rev = Rev(a.model, bits=None if a.bits == 0 else a.bits)
    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(rev, threading.Lock(), a.orders))
    print(f"rev serving {rev.name} on http://{a.host}:{a.port}/v1/systemone", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0
