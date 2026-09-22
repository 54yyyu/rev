"""`rev score` - answer a JSONL file of decisions.

The input shape is SemIf's, so files written for either tool run through both:

    {"id": "...", "state": ..., "question": "...",
     "options": [{"id": "...", "description": "..."}, ...]}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def score(argv=None) -> int:
    p = argparse.ArgumentParser(prog="rev score", description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--bits", type=int, default=8, choices=(4, 8, 0),
                   help="in-memory quantization; 0 keeps the checkpoint precision (default: 8)")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--orders", default="auto", choices=("one", "two", "auto"),
                   help="read the options a second time in reverse: never, always, or only "
                        "when the first pass is unsure (default: auto)")
    p.add_argument("--offsets", type=Path, help="JSON map of option id to logit shift")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=8192)
    from .remote import add_engine_args, engine_from_args
    add_engine_args(p)
    a = p.parse_args(argv)

    if a.output.exists():
        p.error(f"{a.output} exists; refusing to overwrite a result")

    offsets = json.loads(a.offsets.read_text()) if a.offsets else None
    h = engine_from_args(a, max_tokens=a.max_tokens, temperature=a.temperature, offsets=offsets)
    print(f"{h.capacity} answer slots available", file=sys.stderr)

    rows = [json.loads(line) for line in a.input.read_text().splitlines() if line.strip()]
    started = time.perf_counter()
    with a.output.open("w") as fh:
        for i, row in enumerate(rows):
            d = h.decide(row["state"], row["question"], row["options"],
                         orders=a.orders)
            fh.write(json.dumps({
                "id": row["id"], "choice": d.choice, "probabilities": d.probabilities,
                "logits": d.logits, "confidence": d.confidence,
                "disagreement": d.disagreement, "input_tokens": d.input_tokens,
                "seconds": d.seconds,
            }, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                rate = (i + 1) / (time.perf_counter() - started)
                print(f"  {i+1}/{len(rows)}  {rate:.1f}/s", file=sys.stderr, flush=True)
    elapsed = time.perf_counter() - started
    print(f"{len(rows)} decisions in {elapsed:.0f}s ({len(rows)/elapsed:.1f}/s) -> {a.output}",
          file=sys.stderr)
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "score":
        return score(argv[1:])
    if argv and argv[0] == "serve":
        from .serve import serve
        return serve(argv[1:])
    print("usage: rev serve [--port 8421] [--model M] [--bits 8] [--upstream URL]\n"
          "       rev score --input FILE --output FILE [--model M] [--bits 8] [--upstream URL]",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
