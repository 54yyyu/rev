"""Read a state once, however many times it is asked about.

The prompt puts the evidence before the criterion and the options, so every
reading of one state - the second, reversed reading, and every other question
in a Jev request - begins with the same tokens. On a 3,000-token document that
shared part is nearly all the work. It is run once, the model's cache is kept
at the end of it, and each reading then runs only its own few dozen tokens.

Where the shared part ends is found from the state alone: the state is rendered
under two fixed placeholder questions and the tokens they share are the prefix.
So a reading is computed the same way whatever else it is asked alongside,
and whether or not its prefix was already cached - which is what keeps answers
independent of batch position (`tests/test_determinism.py`).

The chat template and system prompt, the same for every request, are one more
level down: read once at startup, and every new state continues from them.

A cache holds two kinds of layer state. Attention layers write into a buffer
past the prefix and are trimmed back to it. The linear-attention layers replace
their state arrays rather than writing into them, so keeping references to the
arrays at the end of the prefix is enough to restore them.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any

import mlx.core as mx

from .prompt import render

# Two questions that differ in their first character, with trivial options.
_PROBES = ("Alpha?", "Zulu!")
_PROBE_OPTIONS = [("A", "x"), ("B", "y")]


def _snapshot(cache) -> list:
    return [c.offset if hasattr(c, "offset") else list(c.cache) for c in cache]


def _copy(cache, snapshot, make_cache):
    """A new cache holding the same prefix; extending it leaves the original intact."""
    fresh = make_cache()
    for new, old, saved in zip(fresh, cache, snapshot):
        if isinstance(saved, int):
            k, v = old.keys[..., :saved, :], old.values[..., :saved, :]
            new.keys, new.values, new.offset = k, v, saved
        else:
            new.cache = list(saved)
    return fresh


class StateCache:
    def __init__(self, net, body, tokenizer, capacity: int = 4):
        self.net, self.body, self.tokenizer = net, body, tokenizer
        self.capacity = capacity
        self._splits: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        self._prefills: OrderedDict[tuple[int, ...], tuple[list, list]] = OrderedDict()
        self._template: tuple[int, ...] | None = None
        self._template_cache = None

    @staticmethod
    def _key(state: Any) -> str:
        return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)

    def prefix_for(self, state: Any) -> tuple[int, ...]:
        """The token ids every reading of this state begins with."""
        key = self._key(state)
        if key in self._splits:
            self._splits.move_to_end(key)
            return self._splits[key]
        a, b = (self.tokenizer.encode(render(self.tokenizer, state, q, _PROBE_OPTIONS),
                                      add_special_tokens=False) for q in _PROBES)
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        prefix = tuple(a[:n]) if n > len(self.template()) else ()
        self._splits[key] = prefix
        if len(self._splits) > 64:
            self._splits.popitem(last=False)
        return prefix

    def template(self) -> tuple[int, ...]:
        """The tokens every prompt starts with, whatever the state."""
        if self._template is None:
            a, b = (self.tokenizer.encode(render(self.tokenizer, st, _PROBES[0], _PROBE_OPTIONS),
                                          add_special_tokens=False) for st in ("Alpha state", "Zulu state"))
            n = 0
            while n < min(len(a), len(b)) and a[n] == b[n]:
                n += 1
            self._template = tuple(a[:n])
            cache = self.net.make_cache()
            self.body(mx.array([list(self._template)]), cache=cache)
            mx.eval([c.state for c in cache])
            self._template_cache = (cache, _snapshot(cache))
        return self._template

    def _prefill(self, prefix: tuple[int, ...]):
        if prefix in self._prefills:
            self._prefills.move_to_end(prefix)
            return self._prefills[prefix]
        t = self.template()
        if len(prefix) > len(t) and prefix[: len(t)] == t:
            cache = _copy(*self._template_cache, self.net.make_cache)
            self.body(mx.array([list(prefix[len(t):])]), cache=cache)
        else:
            cache = self.net.make_cache()
            self.body(mx.array([list(prefix)]), cache=cache)
        mx.eval([c.state for c in cache])
        snapshot = _snapshot(cache)
        self._prefills[prefix] = (cache, snapshot)
        if len(self._prefills) > self.capacity:
            self._prefills.popitem(last=False)
        return cache, snapshot

    def hidden_last(self, prefix: tuple[int, ...], ids: list[int]) -> mx.array:
        """Final hidden state at the last position of `ids`, which must start with `prefix`."""
        cache, snapshot = self._prefill(prefix)
        for c, saved in zip(cache, snapshot):
            if isinstance(saved, int):
                c.trim(c.offset - saved)
            else:
                c.cache = list(saved)
        return self.body(mx.array([ids[len(prefix):]]), cache=cache)[:, -1:, :]
