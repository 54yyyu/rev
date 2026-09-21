"""Answer slots: single characters whose token survives a round trip.

The readout works by reading the logit of one token per option at the position
where the model would emit its answer. That only holds if each label is exactly
one token and decodes back to itself, so every candidate is verified against the
live tokenizer rather than assumed.

Upper case comes first so a question with few options is rendered A, B, C - the
familiar layout, and the one every comparable implementation uses.
"""

from __future__ import annotations

import string

PREFERRED = string.ascii_uppercase
FALLBACK = (
    string.ascii_lowercase
    + string.digits
    + "αβγδεζηθικλμνξοπρστυφχψω"
    + "!@#$%^&*+=~?<>|"
)


def verified_slots(tokenizer, limit: int | None = None) -> list[tuple[str, int]]:
    """(character, token id) pairs that are safe to use as answer slots."""
    out: list[tuple[str, int]] = []
    seen: set[int] = set()
    for ch in PREFERRED + FALLBACK:
        encoded = tokenizer.encode(ch, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != ch:
            continue
        if encoded[0] in seen:
            continue
        seen.add(encoded[0])
        out.append((ch, encoded[0]))
        if limit and len(out) >= limit:
            break
    if len(out) < 2:
        raise ValueError("Tokenizer offers fewer than two usable answer slots")
    return out


def check_boundary(tokenizer, prompt: str, prompt_ids: list[int], slots) -> None:
    """Appending a label must not change how the prompt itself is tokenized.

    Without this a label can merge with the preceding character and the logit
    being read is no longer the one the label refers to.
    """
    for ch, token in slots:
        if tokenizer.encode(prompt + ch, add_special_tokens=False) != prompt_ids + [token]:
            raise ValueError(f"Answer boundary shifts tokenization for slot {ch!r}")
