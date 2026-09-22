"""Reproduce the published numbers on JevBench's public items.

    python bench/jevbench.py --tier hard                       # the MLX engine
    python bench/jevbench.py --tier hard --upstream URL        # a served model
    python bench/jevbench.py --tier hard --jev                 # Jev itself, key from ~/typesafe.txt
    python bench/jevbench.py --tier hard --upstream URL --out bench/results/jevbench-hard-27b.json

Downloads the items from the benchmark repository (MIT) on first run. `--out`
saves one row per item (family, gold, choice, probabilities, confidence,
seconds) so bench/plot_readme.py can draw the README's charts without a rerun.
"""

from __future__ import annotations

import argparse, json, time, urllib.request
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
    p.add_argument("--jev", action="store_true", help="measure api.typesafe.ai instead of rev")
    p.add_argument("--key-file", default="~/typesafe.txt")
    p.add_argument("--out", type=Path, help="write per-item results here (JSON)")
    from rev.remote import add_engine_args, engine_from_args
    add_engine_args(p)
    a = p.parse_args()

    from rev.calibrate import coverage_curve, ece

    rows = fetch(a.tier)
    if a.jev:
        from rev import Client
        client = Client("https://api.typesafe.ai", key=Path(a.key_file).expanduser().read_text().strip())
        name, mode = "Jev (api.typesafe.ai)", "hosted"
        def decide(r):
            q = dict(r["question"])                    # Jev's own shape, sent as is
            t0 = time.perf_counter()
            ans = client.ask(r["state"], {"q": q})["answers"]["q"]
            dt = time.perf_counter() - t0
            if q["type"] == "noul":
                pr = {"yes": ans["noul"], "no": 1 - ans["noul"]}
            else:
                pr = {str(k): v for k, v in ans["probabilities"].items()}
            choice = max(pr, key=pr.get)
            return choice, pr, float(ans.get("confidence", pr[choice])), dt
    else:
        h = engine_from_args(a)
        name, mode = h.name, getattr(h, "mode", f"{a.bits}-bit")
        def decide(r):
            state = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False)
            d = h.decide(state, r["question"]["instructions"], options_of(r), orders=a.orders,
                         ordered=r["question"]["type"] == "score")
            return d.choice, d.probabilities, d.confidence, d.seconds

    probs, confs, gold, by_family, items = [], [], [], {}, []
    t0 = time.perf_counter()
    for r in rows:
        choice, pr, conf, dt = decide(r)
        probs.append(pr); confs.append(conf); gold.append(str(r["expected"]))
        by_family.setdefault(r["family"], []).append(choice == str(r["expected"]))
        items.append({"id": r.get("id"), "family": r["family"], "type": r["question"]["type"],
                      "gold": str(r["expected"]), "choice": choice, "probabilities": pr,
                      "confidence": conf, "seconds": dt})

    hit = [max(p, key=p.get) == g for p, g in zip(probs, gold)]
    print(f"\n{a.tier}  n={len(rows)}  model={name} mode={mode} orders={a.orders}"
          f"  {time.perf_counter() - t0:.0f}s")
    print(f"  accuracy {np.mean(hit):.3f}   ECE {ece(probs, gold):.3f}")
    print("  by family:")
    for fam, v in sorted(by_family.items()):
        print(f"    {fam:18s} {len(v):3d}  {np.mean(v):.3f}")
    print("  gate (threshold / coverage / selective accuracy):")
    for t, cov, sel in coverage_curve(probs, gold, confidences=confs):
        print(f"    {t:4.2f}  {cov:5.2f}  {sel:.3f}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({
            "date": time.strftime("%Y-%m-%d"), "tier": a.tier, "model": name, "mode": mode,
            "orders": a.orders, "accuracy": float(np.mean(hit)), "ece": float(ece(probs, gold)),
            "by_family": {f: float(np.mean(v)) for f, v in sorted(by_family.items())},
            "n_by_family": {f: len(v) for f, v in sorted(by_family.items())},
            "items": items}, ensure_ascii=False, indent=1))
        print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
