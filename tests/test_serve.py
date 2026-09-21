"""`rev serve` speaks Jev's protocol, and a Jev-shaped client gets the same
answer from it as from the in-process API."""
import socket, subprocess, sys, time
from rev import Client, Rev

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]

port = free_port()
proc = subprocess.Popen([sys.executable, "-m", "rev.cli", "serve", "--port", str(port)],
                        stderr=subprocess.PIPE, text=True)
fail = 0
try:
    c = Client(f"http://127.0.0.1:{port}")
    for _ in range(120):
        try:
            c.ask("x", {"q": {"type": "noul", "instructions": "ok?"}}); break
        except RuntimeError:
            time.sleep(1)
    state = "Help! My payouts have been failing for 3 days."
    out = c.ask(state, {
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"billing": "Payments, invoicing, refunds",
                                    "technical": "Bugs, outages, integrations",
                                    "sales": "Pricing, upgrades, new accounts"}},
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                        "criteria": ["Calm", "Frustrated", "Very angry"]}})
    a = out["answers"]
    print({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()
               if kk != "probabilities"} for k, v in a.items()}, out["usage"])
    if a["department"]["choice"] not in ("billing", "technical"):
        print("  FAIL: department"); fail += 1
    if not a["is_urgent"]["noul"] > 0.5:
        print("  FAIL: urgency"); fail += 1
    if not 0 <= a["frustration"]["score"] <= 2:
        print("  FAIL: score range"); fail += 1

    local = Rev().ask(state, {"d": {"type": "choice", "instructions": "Which team should handle this?",
                                    "criteria": {"billing": "Payments, invoicing, refunds",
                                                 "technical": "Bugs, outages, integrations",
                                                 "sales": "Pricing, upgrades, new accounts"}}})
    delta = max(abs(local["answers"]["d"]["probabilities"][k] - a["department"]["probabilities"][k])
                for k in a["department"]["probabilities"])
    print(f"server vs in-process: max probability delta {delta:.6f}")
    if delta > 1e-6:
        print("  FAIL: server and library disagree"); fail += 1

    try:
        c.ask("x", {"q": {"type": "choice", "instructions": "x", "criteria": {"a": "one"}}})
        print("  FAIL: bad question accepted"); fail += 1
    except RuntimeError as e:
        if "400" not in str(e):
            print("  FAIL:", e); fail += 1
finally:
    proc.terminate(); proc.wait(10)
print("FAILURES:", fail)
raise SystemExit(1 if fail else 0)
