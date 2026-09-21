"""Checks that belong in code, in front of the model, because the model was
measured failing them confidently.

    from rev.guards import looks_secret, vague_action

- An API key on the clipboard came back as "an email address", "a phone number"
  and "a social profile", each at p = 1.00. Nothing secret should reach a
  question whose answer might paste or send it.
- "Tidy up my calendar for next week" was classified as a harmless local edit at
  p = 0.99, in English and Chinese. Open-ended verbs have to force a confirmation
  before the model is asked anything.

Both are deliberately broad: a false alarm costs one confirmation, a miss costs
a leaked key or a moved inbox.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_SECRET_PATTERNS = re.compile(
    r"""(
        \bsk-[A-Za-z0-9_\-]{8,}              # OpenAI / Anthropic style
      | \bsk_(live|test)_[A-Za-z0-9]{8,}     # Stripe
      | \bgh[pousr]_[A-Za-z0-9]{20,}         # GitHub tokens
      | \bgithub_pat_[A-Za-z0-9_]{20,}
      | \bxox[abprs]-[A-Za-z0-9\-]{10,}      # Slack
      | \bAKIA[0-9A-Z]{16}\b                 # AWS access key id
      | \bAIza[0-9A-Za-z_\-]{30,}            # Google API key
      | \bhf_[A-Za-z0-9]{20,}                # Hugging Face
      | \beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}   # JWT
      | -----BEGIN\ [A-Z ]*PRIVATE\ KEY-----
      | \b(password|passwd|pwd|secret|token|api[_\-]?key)\s*[:=]\s*\S+
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def _entropy(s: str) -> float:
    counts = Counter(s)
    return -sum(c / len(s) * math.log2(c / len(s)) for c in counts.values())


def looks_secret(text: str) -> bool:
    """True for anything that might be a credential. Broad on purpose."""
    if _SECRET_PATTERNS.search(text):
        return True
    # URLs are addresses, not credentials; a path is long and mixed by nature.
    text = re.sub(r"(https?://|www\.)\S+", " ", text)
    if re.search(r"\b[0-9a-fA-F]{32,}\b", text):       # hex keys and hashes
        return True
    # A long unbroken run of mixed letters and digits with high entropy: a
    # token nobody recognised the prefix of.
    for word in re.findall(r"[A-Za-z0-9_\-+=]{24,}", text):
        parts = re.split(r"[_\-]", word)
        if len(parts) >= 3 and max(map(len, parts)) <= 10:
            continue                          # a_file_name-like-this, not a token
        if re.search(r"[A-Za-z]", word) and re.search(r"\d", word) and _entropy(word) > 4.0:
            return True
    return False


VAGUE = re.compile(
    r"tidy|clean\s*up|clear\s*out|organi[sz]e|sort\s*out|declutter|get\s+rid\s+of"
    r"|整理|清理|收拾|清空|处理掉",
    re.IGNORECASE,
)


def vague_action(request: str) -> bool:
    """True when a request names an open-ended cleanup whose scope the model
    cannot be trusted to judge. Route these straight to a confirmation."""
    return bool(VAGUE.search(request))
