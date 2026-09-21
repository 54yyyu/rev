"""Latency and throughput of rev against Jev: the same client and the same items,
only the URL different.

    rev serve &                                   # in another terminal
    python bench/speed.py                         # the local server
    python bench/speed.py --jev                   # TypeSafe, key from ~/typesafe.txt

Only synthetic and public items are sent: clipboard paste, and JevBench's
public standard and hard tiers. Writes bench/results/speed-<label>.json.

Measured 2026-09-21 on an M2 Pro (README has the table): rev is faster on short
and medium questions, slower on 1-4k-token documents, and does not scale with
concurrent requests; Jev holds ~330 ms at any length.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import cases  # noqa: E402
from jevbench import fetch, options_of  # noqa: E402


def question(criterion, options, kind="choice"):
    if kind == "noul":
        return {"type": "noul", "instructions": criterion,
                "criteria": {"false": options["no"], "true": options["yes"]}}
    if kind == "score":
        return {"type": "score", "instructions": criterion,
                "criteria": [options[k] for k in sorted(options, key=int)]}
    return {"type": "choice", "instructions": criterion, "criteria": options}


def request_sets():
    sets = {"short: clipboard paste": [
        (s, {"q": question(c, o)}) for s, c, o, _ in cases.paste_cases(sizes=(8,))]}
    for tier, name in (("standard", "medium: JevBench standard"), ("hard", "long: JevBench hard")):
        rows = []
        for row in fetch(tier):
            st = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], ensure_ascii=False)
            rows.append((st, {"q": question(row["question"]["instructions"], options_of(row),
                                            row["question"]["type"])}))
        sets[name] = rows
    # The same long documents with three questions in one request.
    sets["long: three questions per request"] = [
        (st, {**qs,
              "deadline": {"type": "noul", "instructions": "Does the document mention a deadline or time limit?"},
              "complexity": {"type": "score", "instructions": "How complex is this document?",
                             "criteria": ["Simple", "Moderate", "Complex"]}})
        for st, qs in sets["long: JevBench hard"][:40]]
    return sets


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8421")
    p.add_argument("--jev", action="store_true", help="measure api.typesafe.ai instead")
    p.add_argument("--key-file", default="~/typesafe.txt")
    p.add_argument("--parallel", type=int, default=8)
    a = p.parse_args()

    from rev import Client
    if a.jev:
        client, label = Client("https://api.typesafe.ai",
                               key=Path(a.key_file).expanduser().read_text().strip()), "jev"
    else:
        client, label = Client(a.url), "rev"

    sets = request_sets()
    client.ask(*sets["short: clipboard paste"][0])            # warm-up, not counted
    results = {}
    for name, items in sets.items():
        latency, tokens = [], []
        for state, qs in items:
            started = time.perf_counter()
            out = client.ask(state, qs)
            latency.append(time.perf_counter() - started)
            tokens.append(out["usage"]["input_tokens"])
        started = time.perf_counter()
        with ThreadPoolExecutor(a.parallel) as pool:
            list(pool.map(lambda item: client.ask(*item), items))
        parallel = len(items) / (time.perf_counter() - started)
        latency.sort()
        results[name] = {"n": len(items), "p50_ms": round(statistics.median(latency) * 1000),
                         "p95_ms": round(latency[int(0.95 * len(latency))] * 1000),
                         "mean_input_tokens": round(sum(tokens) / len(tokens)),
                         "sequential_per_s": round(len(items) / sum(latency), 2),
                         f"parallel{a.parallel}_per_s": round(parallel, 2)}
        r = results[name]
        print(f"{label:4s} {name:36s} p50 {r['p50_ms']:5d} ms  p95 {r['p95_ms']:5d} ms  "
              f"~{r['mean_input_tokens']:5d} tok  {a.parallel} in flight: {parallel:5.2f}/s", flush=True)
    out = HERE / "results" / f"speed-{label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"date": date.today().isoformat(), "target": label, "results": results}, indent=1))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
