"""`rev ask` / `rev health` against a stub server: batch order, retries on a
down model, local images, exit codes. No model needed. With REV_URL set, also
one real request and one real image against that server."""
import base64, io, json, os, socket, struct, subprocess, sys, tempfile, threading, time, zlib
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rev.ask import ask, health


def png(rgb, size=64):
    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


seen, down = [], {"left": 0}


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Length", str(len(data))); self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(200, {"ok": True, "model": "stub", "mode": "stub"})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        seen.append(req)
        if down["left"] > 0:
            down["left"] -= 1
            return self._send(502, {"error": "upstream: down"})
        state = req["state"]
        if isinstance(state, dict) and "delay" in state:
            time.sleep(state["delay"])
        answers = {}
        for qid, q in req["questions"].items():
            if q["type"] == "choice":
                k = list(q["criteria"])[0]
                answers[qid] = {"type": "choice", "choice": k, "probabilities": {k: 1.0}, "confidence": 1.0}
            elif q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": 0.25}
            else:
                answers[qid] = {"type": "score", "score": 1.4, "confidence": 0.6,
                                "legend": {"0": "low", "1": "mid", "2": "high"}, "probabilities": {}}
        self._send(200, {"model": "stub", "answers": answers, "seconds": 0.0})


srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{srv.server_address[1]}"
fail = 0


def run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = ask(argv)
    return rc, buf.getvalue()


def check(cond, what):
    global fail
    print(("  ok    " if cond else "  FAIL  ") + what)
    fail += not cond


tmp = Path(tempfile.mkdtemp())
qs = {"t": {"type": "choice", "instructions": "Which?", "criteria": {"a": "A", "b": "B"}},
      "y": {"type": "noul", "instructions": "Yes?"},
      "s": {"type": "score", "instructions": "How much?", "criteria": ["low", "mid", "high"]}}
(tmp / "q.json").write_text(json.dumps(qs))

# Brief output: choice, probability of yes, nearest level.
rc, out = run(["--url", url, "--questions", str(tmp / "q.json"), "--state", "x", "--brief"])
b = json.loads(out)
check(rc == 0 and b == {"t": {"choice": "a", "confidence": 1.0}, "y": {"noul": 0.25},
                        "s": {"score": 1.4, "level": "mid", "confidence": 0.6}}, "brief answers")

# Batch keeps input order although later lines finish first.
lines = [{"id": i, "state": {"delay": 0.3 - 0.05 * i}} for i in range(6)]
(tmp / "in.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))
rc, out = run(["--url", url, "--jsonl", str(tmp / "in.jsonl"), "--questions", str(tmp / "q.json"), "-j", "6"])
ids = [json.loads(l)["id"] for l in out.splitlines()]
check(rc == 0 and ids == list(range(6)), f"batch order under concurrency {ids}")

# A model that is down for one request is retried; down for good is exit 3.
down["left"] = 1
rc, out = run(["--url", url, "--questions", str(tmp / "q.json"), "--state", "x", "--retries", "1"])
check(rc == 0 and "answers" in json.loads(out), "one 502 then an answer")
down["left"] = 99
rc, _ = run(["--url", url, "--questions", str(tmp / "q.json"), "--state", "x", "--retries", "0"])
check(rc == 3, f"502 with no retries left exits 3 (got {rc})")
down["left"] = 0

rc, _ = run(["--url", "http://127.0.0.1:1", "--questions", str(tmp / "q.json"), "--state", "x", "--retries", "0"])
check(rc == 3, f"unreachable exits 3 (got {rc})")

# Local image paths become data URIs; --image adds one to a text state.
(tmp / "red.png").write_bytes(png((255, 0, 0)))
seen.clear()
rc, _ = run(["--url", url, "--questions", str(tmp / "q.json"), "--state", "look", "--image", str(tmp / "red.png")])
st = seen[-1]["state"]
check(rc == 0 and st[0] == {"type": "text", "text": "look"}
      and st[1]["image_url"]["url"].startswith("data:image/png;base64,"), "--image on a text state")
(tmp / "req.json").write_text(json.dumps({"questions": qs, "state": {"shot": {
    "type": "image_url", "image_url": {"url": str(tmp / "red.png")}}}}))
rc, _ = run(["--url", url, str(tmp / "req.json")])
check(rc == 0 and seen[-1]["state"]["shot"]["image_url"]["url"].startswith("data:image/png"),
      "a local path inside the state")
rc, _ = run(["--url", url, "--questions", str(tmp / "q.json"), "--state", "x", "--image", str(tmp / "nope.png")])
check(rc == 2, f"a missing image file exits 2 (got {rc})")

# Health.
with redirect_stdout(io.StringIO()):
    check(health(["--url", url]) == 0, "health up")
    check(health(["--url", "http://127.0.0.1:1", "--timeout", "2"]) == 3, "health down exits 3")

# The real thing, when there is one.
real = os.environ.get("REV_URL")
if real:
    print(f"live: {real}")
    rc, out = run(["--url", real, "--brief", "--state", "My card was charged twice for one order.",
                   "--questions", str(tmp / "q.json")])
    check(rc == 0 and "t" in json.loads(out), f"live text {out.strip()}")
    (tmp / "red.png").write_bytes(png((220, 20, 20), 128))
    (tmp / "c.json").write_text(json.dumps({"c": {"type": "choice", "instructions": "What colour fills Picture 1?",
                                                  "criteria": {"red": "Red", "green": "Green", "blue": "Blue"}}}))
    rc, out = run(["--url", real, "--brief", "--state", "A picture.", "--image", str(tmp / "red.png"),
                   "--questions", str(tmp / "c.json")])
    check(rc == 0 and json.loads(out)["c"]["choice"] == "red", f"live image {out.strip()}")

srv.shutdown()
print("PASS" if not fail else f"{fail} FAILED")
sys.exit(1 if fail else 0)
