"""The prompt. One state, one criterion, a labelled option list.

The Evidence / Criterion framing is the one both SemIf and reflex arrived at
independently, and reflex measured it as one of only three prompt choices that
transferred. The option id never enters the prompt: the model sees a label and a
description, and the id is only how the caller gets its answer back.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. "
    "Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


# Where an image may sit in a state: an OpenAI content part, the shape a
# chat-completions client already sends.
#   {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
URL_SCHEMES = ("data:image/", "http://", "https://")


def _image_url(part: Any) -> str | None:
    if not isinstance(part, Mapping) or part.get("type") != "image_url":
        return None
    ref = part.get("image_url")
    url = ref.get("url") if isinstance(ref, Mapping) else ref
    if not isinstance(url, str) or not url.startswith(URL_SCHEMES):
        # Anything else would be read by the model server as a path on its own disk.
        raise ValueError("an image_url must be a data:image/... URL or an http(s) URL")
    return url


def split_images(state: Any) -> tuple[Any, list[str]]:
    """The state with each image replaced by "Picture N", and the image URLs in order.

    Images may sit anywhere in the state. The model sees them ahead of the text,
    each introduced as "Picture N:", so the evidence can refer to them by that
    name. A state that is itself a list of content parts (text and images, as in
    a chat message) becomes one text with the references in place.
    """
    urls: list[str] = []

    def walk(value: Any) -> Any:
        url = _image_url(value)
        if url is not None:
            urls.append(url)
            return f"Picture {len(urls)}"
        if isinstance(value, Mapping):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [walk(v) for v in value]
        return value

    parts = (isinstance(state, (list, tuple)) and state and
             all(isinstance(p, Mapping) and p.get("type") in ("text", "image_url") for p in state))
    out = walk(state)
    if parts and urls:
        out = "\n\n".join(p["text"] if isinstance(p, Mapping) else p for p in out)
    return out, urls


def messages(state: Any, criterion: str, labelled: list[tuple[str, str]],
             images: int = 0) -> list[dict]:
    payload = {
        "evidence": state,
        "criterion": criterion,
        "options": [{"letter": letter, "description": text} for letter, text in labelled],
    }
    text = json.dumps(payload, ensure_ascii=False)
    user: Any = text
    if images:
        user = [{"type": "image"}] * images + [{"type": "text", "text": text}]
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]


def render(tokenizer, state: Any, criterion: str, labelled: list[tuple[str, str]],
           images: int = 0) -> str:
    """Chat template up to the point the answer letter would be emitted.

    `enable_thinking=False` is load-bearing. Qwen3.5 otherwise opens a reasoning
    block, the last position is no longer the answer slot, and accuracy collapses
    to chance while every tokenization check still passes.

    `images` is how many images `split_images` took out of the state; the
    template puts a placeholder for each ahead of the text, which the model
    server fills from the image data sent alongside the prompt.
    """
    extra = {"add_vision_id": True} if images else {}
    return tokenizer.apply_chat_template(
        messages(state, criterion, labelled, images),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        **extra,
    )
