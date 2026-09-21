"""The public API.

    from rev import Rev
    h = Rev("Qwen/Qwen3.5-2B")
    h.decide("My card was charged twice.",
             "Which team should handle this?",
             {"billing": "Payments, refunds and invoices",
              "technical": "Bugs and outages"})
    -> Decision(choice='billing', probabilities={...}, confidence=0.99, ...)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import mlx.core as mx
import numpy as np

from .engine import load
from .labels import check_boundary, verified_slots
from .prompt import render
from .readout import confidence, disagreement, softmax


@dataclass
class Decision:
    choice: str
    probabilities: dict[str, float]
    confidence: float
    logits: dict[str, float] = field(repr=False, default_factory=dict)
    disagreement: float | None = None
    input_tokens: int = 0
    seconds: float = 0.0

    def above(self, threshold: float) -> str | None:
        """The choice, or None when it should go to a person instead."""
        return self.choice if self.confidence >= threshold else None


def _as_options(options) -> list[tuple[str, str]]:
    """Accepts {id: description}, [(id, description)] or a plain [id, ...]."""
    if isinstance(options, Mapping):
        return [(str(k), str(v)) for k, v in options.items()]
    out = []
    for item in options:
        if isinstance(item, Mapping):
            out.append((str(item["id"]), str(item.get("description", item["id"]))))
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            out.append((str(item[0]), str(item[1])))
        else:
            out.append((str(item), str(item)))
    return out


class Rev:
    def __init__(self, model: str = "Qwen/Qwen3.5-2B", bits: int | None = 8,
                 max_tokens: int = 8192, temperature: float = 1.0,
                 offsets: Mapping[str, float] | None = None):
        self.model, self.tokenizer = load(model, bits=bits)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.offsets = dict(offsets or {})
        self._slots: list[tuple[str, int]] | None = None

    @property
    def slots(self) -> list[tuple[str, int]]:
        if self._slots is None:
            self._slots = verified_slots(self.tokenizer)
        return self._slots

    @property
    def capacity(self) -> int:
        """How many options one question may carry on this tokenizer."""
        return len(self.slots)

    def _branch(self, state: Any, criterion: str, options: list[tuple[str, str]]) -> tuple[np.ndarray, int]:
        slots = self.slots[: len(options)]
        prompt = render(self.tokenizer, state, criterion,
                        [(slot[0], text) for slot, (_, text) in zip(slots, options)])
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > self.max_tokens:
            raise ValueError(
                f"{len(ids)} input tokens exceed max_tokens={self.max_tokens}; "
                "shorten the state rather than letting it be truncated")
        check_boundary(self.tokenizer, prompt, ids, slots)
        mx.synchronize()
        # Slice the answer position and widen before gathering, matching the
        # reference implementation exactly. Reading elements one at a time from
        # the native dtype builds a different graph, and on prompts of a few
        # thousand tokens that costs the last bit or two of each logit.
        logits = self.model(mx.array([ids]))[0, -1].astype(mx.float32)
        selected = logits[mx.array([token for _, token in slots])].tolist()
        mx.synchronize()
        return np.array(selected, dtype=float), len(ids)

    def decide(self, state: Any, criterion: str, options, *, both_orders: bool = False) -> Decision:
        opts = _as_options(options)
        if not 2 <= len(opts) <= self.capacity:
            raise ValueError(f"Need 2 to {self.capacity} options, got {len(opts)}")
        started = time.perf_counter()

        z, ntok = self._branch(state, criterion, opts)
        totals = {oid: value for (oid, _), value in zip(opts, z)}
        branches = [softmax(z, self.temperature)]

        if both_orders:
            flipped = list(reversed(opts))
            z2, _ = self._branch(state, criterion, flipped)
            for (oid, _), value in zip(flipped, z2):
                totals[oid] = (totals[oid] + value) / 2
            order = [oid for oid, _ in opts]
            branches.append(softmax(np.array([
                dict(zip((o for o, _ in flipped), softmax(z2, self.temperature)))[o] for o in order]), 1.0))

        for oid, shift in self.offsets.items():
            if oid in totals:
                totals[oid] -= shift

        ids = list(totals)
        probs = softmax(np.array([totals[i] for i in ids]), self.temperature)
        best = ids[int(np.argmax(probs))]
        return Decision(
            choice=best,
            probabilities=dict(zip(ids, probs.tolist())),
            confidence=confidence(probs),
            logits={i: float(totals[i]) for i in ids},
            disagreement=disagreement(branches) if both_orders else None,
            input_tokens=ntok,
            seconds=time.perf_counter() - started,
        )

    def decide_many(self, state: Any, questions: Iterable[tuple[str, Any]], **kw) -> dict[str, Decision]:
        """Several criteria against one state. Independent, so order is free."""
        return {name: self.decide(state, name if isinstance(name, str) else name,
                                  options, **kw)
                for name, options in questions}
