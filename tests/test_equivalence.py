"""rev must agree with SemIf's reference implementation where both can run."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/Users/yiyu/Documents/misc/semif/src")
import numpy as np
from rev import Rev

items = Path("/Users/yiyu/Documents/misc/system-one-boundary/jb_items.jsonl")
if not items.exists():
    print("reference items not present; skipping"); sys.exit(0)
from semif_phase1 import mlx_backend

rows = [json.loads(l) for l in items.read_text().splitlines()][:40]
model, tok, meta = mlx_backend.load_model(
    "Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc", bits=8)
h = Rev.__new__(Rev)
h.model, h.tokenizer, h.max_tokens, h.temperature, h.offsets, h._slots = model, tok, 8192, 1.0, {}, None
h.orders, h.auto_threshold = "one", 0.9
# The reference path: one full forward, as SemIf computes it. The fast path
# (last-position head, cached prefixes) is compared separately below.
h._split, h._states = None, None

same, worst = 0, 0.0
for r in rows:
    ours = h.decide(r["state"], r["question"], r["options"], orders="one")   # the reference reads one order
    theirs = mlx_backend.score(model, tok, r, meta, max_tokens=8192)
    tp = dict(zip(theirs["option_ids"], theirs["probabilities"]))
    same += ours.choice == max(tp, key=tp.get)
    worst = max(worst, max(abs(ours.probabilities[k] - tp[k]) for k in tp))
print(f"{len(rows)} items: same choice {same}/{len(rows)}, max probability delta {worst:.6f}")
ok = same == len(rows) and worst < 1e-6

# The fast path computes the same function in pieces, so bf16 rounding moves
# the last bits. It must still agree on the answer almost everywhere.
from rev.engine import last_position_head
from rev.prefix import StateCache
ref = {r["id"]: h.decide(r["state"], r["question"], r["options"], orders="one") for r in rows}
h._split = last_position_head(model)
h._states = StateCache(model, h._split[0], tok)
agree, drift = 0, 0.0
for r in rows:
    fast = h.decide(r["state"], r["question"], r["options"], orders="one")
    agree += fast.choice == ref[r["id"]].choice
    drift = max(drift, max(abs(fast.probabilities[k] - ref[r["id"]].probabilities[k]) for k in fast.probabilities))
print(f"fast path: same choice {agree}/{len(rows)}, max probability delta {drift:.4f}")
ok = ok and agree >= len(rows) - 1
sys.exit(0 if ok else 1)
