"""The prompt. One state, one criterion, a labelled option list.

The Evidence / Criterion framing is the one both SemIf and reflex arrived at
independently, and reflex measured it as one of only three prompt choices that
transferred. The option id never enters the prompt: the model sees a label and a
description, and the id is only how the caller gets its answer back.
"""

from __future__ import annotations

import json
from typing import Any

SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. "
    "Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


def messages(state: Any, criterion: str, labelled: list[tuple[str, str]]) -> list[dict]:
    payload = {
        "evidence": state,
        "criterion": criterion,
        "options": [{"letter": letter, "description": text} for letter, text in labelled],
    }
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def render(tokenizer, state: Any, criterion: str, labelled: list[tuple[str, str]]) -> str:
    """Chat template up to the point the answer letter would be emitted.

    `enable_thinking=False` is load-bearing. Qwen3.5 otherwise opens a reasoning
    block, the last position is no longer the answer slot, and accuracy collapses
    to chance while every tokenization check still passes.
    """
    return tokenizer.apply_chat_template(
        messages(state, criterion, labelled),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
