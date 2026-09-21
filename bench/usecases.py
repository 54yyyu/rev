"""Run rev on the tasks it was built for.

    python bench/usecases.py                     # paste, risk, calendar
    python bench/usecases.py mail --dump         # pull recent inbox into bench/private/
    #   ... fill bench/private/mail_labels.json by the rule in cases.py ...
    python bench/usecases.py mail                # score the labelled mail

Calendar and mail are read from this Mac through pyapple, found at
~/Documents/projects/pyapple-mcp (`uv pip install -e ".[bench]"` adds what it
imports). Nothing personal is written outside bench/private/.

Numbers on 2026-09-21, M2 Pro, defaults: paste 1.000, risk gate 0.965,
calendar 0.986 (216 items then), mail AUC 0.941 with orders="one".
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import cases  # noqa: E402


def gate(rows, threshold=0.9):
    hi = [ok for ok, conf in rows if conf >= threshold]
    return len(hi) / len(rows), (np.mean(hi) if hi else float("nan"))


def report(name, rows, seconds):
    cover, acc = gate(rows)
    print(f"{name:34s} n={len(rows):3d}  accuracy {np.mean([ok for ok, _ in rows]):.3f}  "
          f"p50 {statistics.median(seconds)*1000:4.0f} ms  at 0.9: answers {cover:.0%}, {acc:.3f} right")


def run_paste(h, orders):
    rows, t = [], []
    for state, crit, opts, gold in cases.paste_cases():
        d = h.decide(state, crit, opts, orders=orders)
        rows.append((d.choice == gold, d.confidence)); t.append(d.seconds)
    report("clipboard paste", rows, t)


def run_risk(h, orders):
    from rev.guards import vague_action
    rows, t, misses = [], [], []
    for state, crit, opts, risky, request in cases.risk_cases():
        d = h.decide(state, crit, opts, orders=orders)
        flagged = d.choice in cases.RISKY_KINDS or vague_action(request)
        rows.append((flagged == risky, d.confidence)); t.append(d.seconds)
        if flagged != risky:
            misses.append(f"{request!r} -> {d.choice} p={d.confidence:.2f} ({'risky' if risky else 'safe'})")
    report("write-action gate", rows, t)
    for m in misses:
        print("    miss:", m)


def run_calendar(h, orders):
    titles = cases.calendar_titles()
    items = cases.calendar_cases(titles)
    skipped = [n for _, n, _ in cases.CALENDAR_QUERIES
               if len([x for x in titles if n.lower() in x.lower()]) != 1]
    rows, t, by_lang = [], [], {}
    for state, crit, opts, gold, lang in items:
        d = h.decide(state, crit, opts, orders=orders)
        rows.append((d.choice == gold, d.confidence)); t.append(d.seconds)
        by_lang.setdefault(lang, []).append(d.choice == gold)
    report(f"calendar pick ({len(titles)} events)", rows, t)
    print("    " + "  ".join(f"{k}: {np.mean(v):.3f}" for k, v in by_lang.items())
          + (f"   skipped, not on the calendar once: {', '.join(skipped)}" if skipped else ""))


def auc(scores, y):
    s = np.asarray(scores, float); ranks = np.argsort(np.argsort(s, kind="stable"), kind="stable") + 1
    pos = y == 1
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def run_mail(h, orders):
    data = json.loads((cases.PRIVATE / "mail.json").read_text())
    labels = json.loads((cases.PRIVATE / "mail_labels.json").read_text())
    if not labels["important"]:
        sys.exit("bench/private/mail_labels.json has no important messages; label first")
    keep = [i for i in range(len(data)) if i not in set(labels["ambiguous"])]
    y = np.array([int(i in set(labels["important"])) for i in keep])
    scores, t = [], []
    for i in keep:
        d = h.decide(cases.mail_state(data[i]), "What kind of email is this?",
                     cases.MAIL_CATEGORIES, orders=orders)
        scores.append(sum(d.probabilities[c] for c in cases.IMPORTANT_CATEGORIES)); t.append(d.seconds)
    order = np.argsort(-np.asarray(scores), kind="stable")
    top = lambda f: y[order[: max(1, round(f * len(y)))]].sum() / y.sum()
    print(f"{'email triage':34s} n={len(y):3d}  important {y.sum()}  AUC {auc(scores, y):.3f}  "
          f"recall@20% {top(.2):.2f}  recall@40% {top(.4):.2f}  p50 {statistics.median(t)*1000:.0f} ms")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tasks", nargs="*", choices=["paste", "risk", "calendar", "mail", []],
                   help="default: paste risk calendar")
    p.add_argument("--orders", default=None, choices=("one", "two", "auto"),
                   help="default: auto, except one for mail (a ranking task)")
    p.add_argument("--bits", type=int, default=8, choices=(4, 8, 0))
    p.add_argument("--dump", action="store_true", help="with mail: fetch the inbox and stop")
    a = p.parse_args()

    if a.dump:
        path = cases.dump_mail()
        print(f"wrote {path}; label it in {cases.PRIVATE / 'mail_labels.json'} before scoring")
        return 0

    from rev import Rev
    h = Rev(bits=None if a.bits == 0 else a.bits)
    for task in a.tasks or ["paste", "risk", "calendar"]:
        orders = a.orders or ("one" if task == "mail" else "auto")
        {"paste": run_paste, "risk": run_risk, "calendar": run_calendar, "mail": run_mail}[task](h, orders)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
