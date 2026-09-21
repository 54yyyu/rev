"""Fitting the two things the raw readout does not give you.

**Offsets** correct a standing preference for one option id. On Qwen3.5-2B the
binary questions came back "yes" 32 times out of 38 where the truth was 17 to
21, worth about eight points once removed. The offset is a property of the model
and prompt rather than of the items: one fitted on easy questions transferred to
hard ones. Fit it on your own labelled rows, never on the rows you then report.

**Temperature** makes the numbers usable as thresholds. It does not change which
option wins.

A content-free prior - the same prompt with the state replaced by "N/A" - was
measured and rejected: it made the binary questions worse, not better. What the
model prefers with no evidence is not what it prefers with evidence.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np


def fit_offsets(logits: Sequence[dict[str, float]], gold: Sequence[str],
                min_rows: int = 20) -> dict[str, float]:
    """A per-id logit shift that removes a standing preference.

    For each option id that appears in at least `min_rows` questions, the shift
    is the mean amount by which its logit exceeds the row's mean when it is
    wrong, less that amount when it is right.
    """
    if len(logits) != len(gold):
        raise ValueError("logits and gold must be the same length")
    seen: dict[str, list[float]] = defaultdict(list)
    for row, answer in zip(logits, gold):
        if len(row) < 2:
            continue
        centre = float(np.mean(list(row.values())))
        for oid, value in row.items():
            seen[oid].append((value - centre) * (-1.0 if oid == answer else 1.0))
    return {oid: float(np.mean(v)) for oid, v in seen.items() if len(v) >= min_rows}


def fit_temperature(probabilities: Sequence[dict[str, float]], gold: Sequence[str],
                    grid: Iterable[float] | None = None) -> float:
    """The temperature minimising negative log likelihood of the true option."""
    grid = list(grid) if grid is not None else list(np.linspace(0.3, 4.0, 148))
    best, best_nll = 1.0, float("inf")
    for t in grid:
        total = 0.0
        for row, answer in zip(probabilities, gold):
            ids = list(row)
            z = np.log(np.clip([row[i] for i in ids], 1e-12, None)) / max(t, 1e-6)
            z = z - z.max()
            p = np.exp(z) / np.exp(z).sum()
            total -= float(np.log(max(p[ids.index(answer)], 1e-12)))
        if total < best_nll:
            best, best_nll = float(t), total
    return best


def reliability(probabilities: Sequence[dict[str, float]], gold: Sequence[str],
                bins: int = 10) -> list[tuple[float, float, int, float, float]]:
    """(low, high, n, mean confidence, observed accuracy) per confidence bin."""
    conf = np.array([max(r.values()) for r in probabilities])
    hit = np.array([max(r, key=r.get) == g for r, g in zip(probabilities, gold)], dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        rows.append((float(lo), float(hi), int(m.sum()),
                     float(conf[m].mean()) if m.any() else float("nan"),
                     float(hit[m].mean()) if m.any() else float("nan")))
    return rows


def ece(probabilities: Sequence[dict[str, float]], gold: Sequence[str], bins: int = 10) -> float:
    rows = reliability(probabilities, gold, bins)
    n = len(probabilities)
    return sum(c / n * abs(p - o) for _, _, c, p, o in rows if c > 0)


def coverage_curve(probabilities: Sequence[dict[str, float]], gold: Sequence[str],
                   thresholds: Iterable[float] = (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95),
                   confidences: Sequence[float] | None = None):
    """(threshold, coverage, selective accuracy) - what a gate would actually buy.

    Pass each `Decision.confidence` as `confidences`: after a second reading it
    is lower than the top probability, and it is what a gate should use.
    """
    conf = np.array(confidences if confidences is not None
                    else [max(r.values()) for r in probabilities])
    hit = np.array([max(r, key=r.get) == g for r, g in zip(probabilities, gold)], dtype=float)
    out = []
    for t in thresholds:
        m = conf >= t
        if m.sum() == 0:
            continue
        out.append((float(t), float(m.mean()), float(hit[m].mean())))
    return out
