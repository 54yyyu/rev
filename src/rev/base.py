"""What every engine shares: the reading policy, Jev's question shapes, the answer.

An engine is anything that can render the prompt with a tokenizer and return
the logits at the answer slots (`_branch`). `Rev` in `decide.py` runs the model
in-process through MLX; `Remote` in `remote.py` asks an sglang server that is
already serving the same model for something else. Everything above that line
- the second reading when unsure, offsets, temperature, the confidence rule,
Jev's response shape - is here, once, so a policy measured on one engine is
the policy the other one runs.

No MLX import here: a remote engine must load without a GPU.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from .decision import Decision
from .labels import verified_slots
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


class Decider:
    """The reading policy over an engine's `_branch`. Subclasses set `tokenizer`
    and `name` before calling `__init__`."""

    tokenizer: Any
    name: str

    def __init__(self, max_tokens: int = 8192, temperature: float = 1.0,
                 offsets: Mapping[str, float] | None = None,
                 orders: str = "auto", auto_threshold: float = 0.9):
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
        """Logits at the answer slots for one rendering, and the prompt length."""
        raise NotImplementedError

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
