"""The same question must answer the same way wherever it sits in a batch.

MLX picks kernels partly from allocator state, and on prompts of a few thousand
tokens that changes the last bits of a logit. Left alone it moves answers: the
reference implementation's CLI and its own in-process API disagree on two of
111 hard items for exactly this reason. Bounding the allocator cache and
synchronising around each forward pass is what makes this test pass.
"""
import json, sys
import numpy as np
from rev import Rev
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from bench.jevbench import fetch, options_of

rows = fetch("hard")
target = sorted(rows, key=lambda r: -len(str(r["state"])))[0]
r = Rev("Qwen/Qwen3.5-2B", bits=8)


def ask(row):
    state = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], ensure_ascii=False)
    return r.decide(state, row["question"]["instructions"], options_of(row), orders="one").logits


first = ask(target)
for row in rows[:40]:
    ask(row)
later = ask(target)

delta = max(abs(first[k] - later[k]) for k in first)
print(f"{len(first)} options, {len(str(target['state']))//4} tokens: max drift {delta:.6f}")
sys.exit(0 if delta == 0.0 else 1)
