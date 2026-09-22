"""The public API, in-process on Apple Silicon.

    from rev import Rev
    h = Rev("Qwen/Qwen3.5-2B")
    h.decide("My card was charged twice.",
             "Which team should handle this?",
             {"billing": "Payments, refunds and invoices",
              "technical": "Bugs and outages"})
    -> Decision(choice='billing', probabilities={...}, confidence=0.99, ...)

The reading policy (`decide`, `ask`) lives in `base.Decider` and is shared with
the remote engine; this file is only how MLX produces the answer-slot logits.
"""

from __future__ import annotations

from typing import Any, Mapping

import mlx.core as mx
import numpy as np

from .base import Decider
from .engine import last_position_head, load
from .prefix import StateCache
from .labels import check_boundary
from .prompt import render, split_images


class Rev(Decider):
    def __init__(self, model: str = "Qwen/Qwen3.5-2B", bits: int | None = 8,
                 max_tokens: int = 8192, temperature: float = 1.0,
                 offsets: Mapping[str, float] | None = None,
                 orders: str = "auto", auto_threshold: float = 0.9):
        self.model, self.tokenizer = load(model, bits=bits)
        self._split = last_position_head(self.model)
        self._states = (StateCache(self.model, self._split[0], self.tokenizer)
                        if self._split is not None and hasattr(self.model, "make_cache") else None)
        self.name = f"{model} ({bits}-bit)" if bits else model
        super().__init__(max_tokens=max_tokens, temperature=temperature, offsets=offsets,
                         orders=orders, auto_threshold=auto_threshold)

    def _branch(self, state: Any, criterion: str, options: list[tuple[str, str]]) -> tuple[np.ndarray, int]:
        if split_images(state)[1]:
            raise ValueError("this engine reads text only; images need a vision model "
                             "served by sglang (rev serve --upstream URL)")
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
