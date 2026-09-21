"""Reproduce the published numbers on JevBench's public items.

    python bench/jevbench.py --tier hard

Downloads the items from the benchmark repository (MIT) on first run.
"""

from __future__ import annotations

import argparse, json, urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

BASE = "https://raw.githubusercontent.com/fstandhartinger/jevbench/main/datasets/public"
HERE = Path(__file__).parent / "fixtures"


def fetch(tier: str) -> list[dict]:
    HERE.mkdir(exist_ok=True)
    local = HERE / f"{tier}.jsonl"
    if not local.exists():
        name = "original" if tier == "standard" else tier
        urllib.request.urlretrieve(f"{BASE}/{name}.jsonl", local)
    return [json.loads(l) for l in local.read_text().splitlines() if l.strip()]


def options_of(row: dict):
    q = row["question"]; c = q["criteria"]
    if q["type"] == "noul":
        return {"no": c["false"], "yes": c["true"]}
    if q["type"] == "score":
        return {str(i): d for i, d in enumerate(c)}
    return {k: c[k] for k in row["labels"]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tier", default="hard", choices=("easy", "standard", "hard"))
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--bits", type=int, default=8)
    p.add_argument("--orders", default="auto", choices=("one", "two", "auto"))
    a = p.parse_args()

    from rev import Rev
    from rev.calibrate import coverage_curve, ece

    rows = fetch(a.tier)
    h = Rev(a.model, bits=a.bits)
    probs, confs, gold, by_family = [], [], [], {}
    for r in rows:
        state = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False)
        d = h.decide(state, r["question"]["instructions"], options_of(r), orders=a.orders,
                     ordered=r["question"]["type"] == "score")
        probs.append(d.probabilities); confs.append(d.confidence); gold.append(str(r["expected"]))
        by_family.setdefault(r["family"], []).append(d.choice == str(r["expected"]))

    hit = [max(p, key=p.get) == g for p, g in zip(probs, gold)]
    print(f"\n{a.tier}  n={len(rows)}  model={a.model} {a.bits}-bit"
          f" orders={a.orders}")
    print(f"  accuracy {np.mean(hit):.3f}   ECE {ece(probs, gold):.3f}")
    print("  by family:")
    for fam, v in sorted(by_family.items()):
        print(f"    {fam:18s} {len(v):3d}  {np.mean(v):.3f}")
    print("  gate (threshold / coverage / selective accuracy):")
    for t, cov, sel in coverage_curve(probs, gold, confidences=confs):
        print(f"    {t:4.2f}  {cov:5.2f}  {sel:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
