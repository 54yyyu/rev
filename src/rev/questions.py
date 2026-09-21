"""Jev's question shapes, answered locally.

TypeSafe's API takes one state and a map of named, typed questions:

    {"state": ..., "questions": {
        "department": {"type": "choice", "instructions": "...", "criteria": {"billing": "...", ...}},
        "is_urgent":  {"type": "noul",   "instructions": "...", "criteria": {"true": "...", "false": "..."}},
        "severity":   {"type": "score",  "instructions": "...", "criteria": ["Cosmetic", "Blocking", ...]}}}

and answers each under the same key. This module turns each question into
options for one decision and formats the answer the way Jev does, so a client
written against Jev only has to change its URL.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

KINDS = ("choice", "noul", "score")

# Jev lets a noul question omit its criteria.
DEFAULT_YES = "Yes."
DEFAULT_NO = "No."


def _text(value: Any) -> str:
    """Jev accepts strings, objects or arrays wherever text goes."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def options_for(question: Mapping[str, Any]) -> tuple[str, str, list[tuple[str, str]]]:
    """(kind, criterion, [(option id, description)]) for one Jev question."""
    kind = question.get("type")
    if kind == "boolean":            # the name Vercel's gateway uses for noul
        kind = "noul"
    if kind not in KINDS:
        raise ValueError(f"question type must be one of {KINDS}, got {kind!r}")
    if "instructions" not in question:
        raise ValueError("every question needs instructions")
    criterion = _text(question["instructions"])
    criteria = question.get("criteria")

    if kind == "noul":
        criteria = criteria or {}
        yes = criteria.get("true", criteria.get(True))
        no = criteria.get("false", criteria.get(False))
        return kind, criterion, [("no", DEFAULT_NO if no is None else _text(no)),
                                 ("yes", DEFAULT_YES if yes is None else _text(yes))]

    if kind == "score":
        if not isinstance(criteria, (list, tuple)) or len(criteria) < 2:
            raise ValueError("a score question needs an ordered list of at least two levels")
        return kind, criterion, [(str(i), _text(level)) for i, level in enumerate(criteria)]

    if not isinstance(criteria, Mapping) or len(criteria) < 2:
        raise ValueError("a choice question needs a map of at least two options")
    return kind, criterion, [(str(k), str(k) if v is None else _text(v)) for k, v in criteria.items()]


def answer_for(kind: str, decision, options: list[tuple[str, str]] | None = None) -> dict[str, Any]:
    """Jev's answer shape for one question."""
    p = decision.probabilities
    if kind == "noul":
        return {"type": "noul", "noul": p["yes"]}
    if kind == "score":
        return {"type": "score",
                "score": sum(int(k) * v for k, v in p.items()),
                "confidence": decision.confidence,
                "legend": dict(options or []),
                "probabilities": p}
    return {"type": "choice", "choice": decision.choice,
            "probabilities": p, "confidence": decision.confidence}
