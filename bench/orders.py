"""Recompute the reading-policy table in DECISIONS.md from saved logits.

    python bench/orders.py

bench/results/orders-2026-09-21.json holds, for every item, the slot logits of
one reading in the given option order (z1) and one in reverse (z2). Policies
are compared offline on exactly the same forward passes. Calendar rows were left
out of the saved file because their ids carry personal event titles.
"""

import json
from pathlib import Path

import numpy as np

RISKY = {"send", "delete", "edit_other"}
TASKS = ["paste", "risk", "jb-easy", "jb-standard", "jb-hard"]


def softmax(z):
    z = np.asarray(z, float) - max(z)
    e = np.exp(z)
    return e / e.sum()


def read(row, policy, threshold=0.9):
    """(probabilities, confidence) under a policy; score items never reverse."""
    p1 = softmax(row["z1"])
    if policy == "one" or row["id"].endswith(":score") or (policy == "auto" and p1.max() >= threshold):
        return p1, p1.max()
    p = softmax((np.array(row["z1"]) + np.array(row["z2"])) / 2)
    best = int(np.argmax(p))
    return p, min(p1[best], softmax(row["z2"])[best])      # the shipped rule


def correct(row, p):
    choice = row["ids"][int(np.argmax(p))]
    if row["task"] == "risk":
        return (choice in RISKY) == (row["gold"] == "RISKY")
    return choice == row["gold"]


def main():
    rows = json.loads((Path(__file__).parent / "results/orders-2026-09-21.json").read_text())["rows"]
    print(f"{'task':12s} {'n':>4s}   one    two   auto   second pass   errors at 0.9 (one / auto)")
    for task in TASKS:
        sub = [r for r in rows if r["task"] == task]
        acc, errs = {}, {}
        for policy in ("one", "two", "auto"):
            res = [(correct(r, p), c) for r in sub for p, c in [read(r, policy)]]
            acc[policy] = np.mean([ok for ok, _ in res])
            errs[policy] = sum(1 for ok, c in res if c >= 0.9 and not ok)
        second = np.mean([softmax(r["z1"]).max() < 0.9 and not r["id"].endswith(":score") for r in sub])
        print(f"{task:12s} {len(sub):4d}  {acc['one']:.3f}  {acc['two']:.3f}  {acc['auto']:.3f}   {second:10.0%}    {errs['one']} / {errs['auto']}")


if __name__ == "__main__":
    main()
