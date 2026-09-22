"""The remote engine reads a served model, and `rev serve --upstream` speaks
Jev's protocol in front of it. Needs an sglang endpoint: REV_UPSTREAM, default
http://localhost:30002. Skips (exit 0) when none answers."""
import json, os, socket, subprocess, sys, time, urllib.request
from rev import Client

UP = os.environ.get("REV_UPSTREAM", "http://localhost:30002")
try:
    urllib.request.urlopen(f"{UP}/get_server_info", timeout=5)
except Exception as e:
    print(f"SKIP: no sglang at {UP} ({e})"); raise SystemExit(0)

from rev.remote import Remote

fail = 0
h = Remote(UP)
print(h.name, "capacity", h.capacity)
state = "Help! My payouts have been failing for 3 days."
opts = {"billing": "Payments, invoicing, refunds",
        "technical": "Bugs, outages, integrations",
        "sales": "Pricing, upgrades, new accounts"}
d = h.decide(state, "Which team should handle this?", opts)
print(f"mode={h.mode}  {d.choice} p={d.confidence:.3f}  {d.seconds:.2f}s  {d.input_tokens} tokens")
if d.choice not in ("billing", "technical"):
    print("  FAIL: choice"); fail += 1
if abs(sum(d.probabilities.values()) - 1) > 1e-6:
    print("  FAIL: probabilities do not sum to 1"); fail += 1

# Reversed options give the same answer: the option id is decoupled from the letter.
d2 = h.decide(state, "Which team should handle this?", dict(reversed(list(opts.items()))))
if d2.choice != d.choice:
    print(f"  FAIL: order changed the answer ({d.choice} -> {d2.choice})"); fail += 1

# Images reach the model: a solid colour read from a picture, not from the text.
import base64, struct, zlib
def png(rgb, w=64, h=64):
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    ck = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    return "data:image/png;base64," + base64.b64encode(
        b"\x89PNG\r\n\x1a\n" + ck(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + ck(b"IDAT", zlib.compress(raw)) + ck(b"IEND", b"")).decode()
colours = {"red": "Red", "green": "Green", "blue": "Blue"}
for want, rgb in (("red", (220, 20, 20)), ("green", (20, 170, 40)), ("blue", (20, 40, 220))):
    img = [{"type": "text", "text": "A photo of a wall."},
           {"type": "image_url", "image_url": {"url": png(rgb)}}]
    di = h.decide(img, "What colour is the wall?", colours)
    print(f"image {want}: {di.choice} p={di.confidence:.3f} {di.seconds:.2f}s {di.input_tokens} tokens")
    if di.choice != want:
        print(f"  FAIL: image read as {di.choice}"); fail += 1

# Thinking is off for this request only: the rendered prompt ends in an empty
# reasoning block, and the server's own default is not touched.
from rev.prompt import render
tail = render(h.tokenizer, state, "q?", [("A", "x"), ("B", "y")])[-40:]
if "<think>" in tail and "</think>" not in tail:
    print("  FAIL: prompt leaves the reasoning block open"); fail += 1

# The server in front of it.
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]
port = free_port()
proc = subprocess.Popen([sys.executable, "-m", "rev.cli", "serve", "--port", str(port),
                         "--upstream", UP], stderr=subprocess.PIPE, text=True)
try:
    c = Client(f"http://127.0.0.1:{port}")
    for _ in range(120):
        try:
            c.ask("x", {"q": {"type": "noul", "instructions": "ok?"}}); break
        except RuntimeError:
            time.sleep(1)
    out = c.ask(state, {
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": opts},
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                        "criteria": ["Calm", "Frustrated", "Very angry"]}})
    a = out["answers"]
    print({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()
               if kk != "probabilities"} for k, v in a.items()}, out["usage"], f"{out['seconds']}s")
    if a["department"]["choice"] not in ("billing", "technical"):
        print("  FAIL: department"); fail += 1
    if not a["is_urgent"]["noul"] > 0.5:
        print("  FAIL: urgency"); fail += 1
    if not 0 <= a["frustration"]["score"] <= 2:
        print("  FAIL: score range"); fail += 1
    out = c.ask([{"type": "image_url", "image_url": {"url": png((20, 40, 220))}}],
                {"c": {"type": "choice", "instructions": "What colour is Picture 1?",
                       "criteria": colours}})
    print("served image:", out["answers"]["c"]["choice"], out["usage"])
    if out["answers"]["c"]["choice"] != "blue":
        print("  FAIL: image through rev serve"); fail += 1
    health = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health"))
    print("health:", health)
    if health.get("mode") not in ("logprob",) and "samples" not in str(health.get("mode")):
        print("  FAIL: health does not report the read mode"); fail += 1
finally:
    proc.terminate(); proc.wait(10)
print("FAILURES:", fail)
raise SystemExit(1 if fail else 0)
