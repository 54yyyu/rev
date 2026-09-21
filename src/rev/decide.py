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
from typing import Any, Iterable, Mapping, Sequence

import mlx.core as mx
import numpy as np

from .decision import Decision
from .engine import last_position_head, load
from .prefix import StateCache
from .labels import check_boundary, verified_slots
from .prompt import render
from .questions import answer_for, options_for
from .readout import disagreement, softmax


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
                 offsets: Mapping[str, float] | None = None,
                 orders: str = "auto", auto_threshold: float = 0.9):
        self.model, self.tokenizer = load(model, bits=bits)
        self._split = last_position_head(self.model)
        self._states = (StateCache(self.model, self._split[0], self.tokenizer)
                        if self._split is not None and hasattr(self.model, "make_cache") else None)
        self.name = f"{model} ({bits}-bit)" if bits else model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.offsets = dict(offsets or {})
        self.orders = orders
        self.auto_threshold = auto_threshold
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
        # Only the answer position is read, so only it goes through the
        # vocabulary projection. Widen before gathering: reading elements one
        # at a time from the native dtype costs the last bit of each logit.
        prefix = self._states.prefix_for(state) if self._states is not None else ()
        if prefix and tuple(ids[: len(prefix)]) == prefix:
            # The state was read once; only this reading's own tokens run here.
            logits = self._split[1](self._states.hidden_last(prefix, ids))[0, -1].astype(mx.float32)
        elif self._split is not None:
            body, head = self._split
            logits = head(body(mx.array([ids]))[:, -1:, :])[0, -1].astype(mx.float32)
        else:
            logits = self.model(mx.array([ids]))[0, -1].astype(mx.float32)
        selected = logits[mx.array([token for _, token in slots])].tolist()
        mx.synchronize()
        return np.array(selected, dtype=float), len(ids)

    def decide(self, state: Any, criterion: str, options, *, orders: str | None = None,
               ordered: bool = False) -> Decision:
        """One choice among `options`, from one or two forward passes.

        `orders` handles position bias: unsure, the model leans toward the first
        letters it sees. Reading the options again in reverse and averaging the
        logits cancels that. "one" never does, "two" always does, and "auto"
        (the default) does only when the first pass is below `auto_threshold`.

        A second reading may change the answer but never raises its confidence:
        `confidence` is the lower of the two readings' probabilities for the
        chosen option. Averaged logits are more accurate but overconfident, and
        what a confidence gate lets through must not get worse. So below the
        gate `confidence` can be less than `max(probabilities)`.

        `ordered=True` marks options whose order carries meaning, such as score
        levels. They are never reversed; doing so was measured to wreck them.
        """
        orders = orders or self.orders
        if orders not in ("one", "two", "auto"):
            raise ValueError(f"orders must be one, two or auto, got {orders!r}")
        opts = _as_options(options)
        if not 2 <= len(opts) <= self.capacity:
            raise ValueError(f"Need 2 to {self.capacity} options, got {len(opts)}")
        ids = [oid for oid, _ in opts]
        shift = np.array([self.offsets.get(oid, 0.0) for oid in ids])
        started = time.perf_counter()

        z, ntok = self._branch(state, criterion, opts)
        z = z - shift
        p1 = softmax(z, self.temperature)
        readings = [p1]

        second = not ordered and (orders == "two" or
                                  (orders == "auto" and p1.max() < self.auto_threshold))
        if second:
            z2, _ = self._branch(state, criterion, list(reversed(opts)))
            z2 = z2[::-1] - shift                    # back into the caller's order
            readings.append(softmax(z2, self.temperature))
            z = (z + z2) / 2

        probs = softmax(z, self.temperature)
        best = int(np.argmax(probs))
        conf = min(r[best] for r in readings)
        return Decision(
            choice=ids[best],
            probabilities=dict(zip(ids, probs.tolist())),
            confidence=float(conf),
            logits=dict(zip(ids, z.tolist())),
            disagreement=disagreement(readings) if second else None,
            input_tokens=ntok,
            seconds=time.perf_counter() - started,
        )

    def ask(self, state: Any, questions: Mapping[str, Mapping[str, Any]], **kw) -> dict[str, Any]:
        """Jev's request, answered locally: one state, a map of named typed questions.

        Returns Jev's response shape. Questions are independent forward passes,
        so their order does not matter.
        """
        if not isinstance(questions, Mapping) or not questions:
            raise ValueError("questions must be a non-empty map of id to question")
        parsed = {qid: options_for(q) for qid, q in questions.items()}   # validate all first
        answers, tokens = {}, 0
        for qid, (kind, criterion, options) in parsed.items():
            d = self.decide(state, criterion, options, ordered=kind == "score", **kw)
            answers[qid] = answer_for(kind, d, options)
            tokens += d.input_tokens
        return {"model": self.name, "answers": answers,
                "usage": {"input_tokens": tokens, "output_tokens": 0}}
