"""The same decisions from a model that is already being served elsewhere.

    from rev.remote import Remote
    h = Remote("http://localhost:30002")           # an sglang endpoint (fleet)
    h.decide(state, "Which team should handle this?", {...})

fleet keeps a 27B model up on a cluster for coding agents. Nothing about the
readout needs the model in this process: the prompt is rendered here with the
served model's own tokenizer and chat template (thinking off, for this request
only - the server's default is untouched and every other client keeps theirs),
and the server is asked for the next-token distribution at the answer slots.
The request is one prompt and one output token, so it rides alongside an
agent's long generations without displacing them.

Two ways to read the distribution, chosen at the first request:

- exact: `/generate` with `return_logprob` and `token_ids_logprob`, which
  returns the log-probability of each slot token at the first output position.
  This is the same quantity the MLX engine reads.
- sampled: some speculative decoders refuse to return log-probabilities at all
  (DSpark, on the fleet endpoint as of 2026-09-21: "DSpark speculative decoding
  does not support return_logprob yet"). Then `n` single-token samples are
  drawn at temperature 1 and the slot counts estimate the distribution. The
  prefix is cached server-side, so 32 samples cost about half a second; the
  estimate carries sampling noise of about sqrt(p(1-p)/n), which blurs a
  confidence gate but not the choice. 32 measured the same accuracy as 64 at
  60% of the time, and 16 lost points (DECISIONS.md). When the server stops
  refusing, the exact path is taken again without a restart.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from typing import Any, Mapping

import numpy as np

from .base import Decider
from .labels import check_boundary
from .prompt import render

# The tokenizer of what fleet serves as qwen38-27b. Only tokenizer and template
# files are fetched; the weights are never touched.
DEFAULT_TOKENIZER = "abhishekchohan/Qwen3.8-27B-AWQ-INT4"
DEFAULT_SAMPLES = 32
# After a refusal, try the exact path again this often. The endpoint behind
# the URL is replaced every few hours on a cluster, and the replacement may
# allow what its predecessor refused.
RETRY_EXACT_SECONDS = 300


class RemoteError(RuntimeError):
    def __init__(self, code: int, message: str):
        super().__init__(f"{code}: {message}")
        self.code, self.message = code, message


def load_tokenizer(source: str):
    """A tokenizer whose chat template is set, from a directory or an HF repo id.

    Newer checkpoints ship the template as `chat_template.jinja` next to the
    tokenizer, which transformers does not always pick up; it is attached here
    so `render` (thinking off) works the same as on the MLX engine.
    """
    from transformers import AutoTokenizer

    path = os.path.expanduser(source)
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download
        path = snapshot_download(source, allow_patterns=["tokenizer*", "*.json", "*.jinja"],
                                 ignore_patterns=["*.safetensors*"])
    tok = AutoTokenizer.from_pretrained(path)
    jinja = os.path.join(path, "chat_template.jinja")
    if not getattr(tok, "chat_template", None) and os.path.exists(jinja):
        tok.chat_template = open(jinja, encoding="utf-8").read()
    if not getattr(tok, "chat_template", None):
        raise ValueError(f"{source} has no chat template; pass a tokenizer that does")
    return tok


class Remote(Decider):
    def __init__(self, upstream: str, tokenizer: str = DEFAULT_TOKENIZER,
                 samples: int = DEFAULT_SAMPLES, timeout: float = 120.0,
                 max_tokens: int = 65536, temperature: float = 1.0,
                 offsets: Mapping[str, float] | None = None,
                 orders: str = "auto", auto_threshold: float = 0.9):
        self.upstream = upstream.rstrip("/")
        self.samples = samples
        self.timeout = timeout
        self.tokenizer = load_tokenizer(tokenizer)
        self.exact: bool | None = None          # unknown until the first request
        self._retry_exact_at = 0.0
        self.served = self._served_name()
        self.name = f"{self.served} via {self.upstream}"
        super().__init__(max_tokens=max_tokens, temperature=temperature, offsets=offsets,
                         orders=orders, auto_threshold=auto_threshold)

    @property
    def mode(self) -> str:
        return {None: "undecided", True: "logprob", False: f"{self.samples} samples"}[self.exact]

    def _served_name(self) -> str:
        """What the upstream serves, or a placeholder when it is not up yet.

        Not fatal: on a cluster the endpoint is down for minutes at each
        renewal, and a server in front of it should wait, not die.
        """
        try:
            with urllib.request.urlopen(f"{self.upstream}/get_server_info", timeout=10) as r:
                info = json.load(r)
            name = info.get("served_model_name") or info.get("model_path") or "?"
            spec = info.get("speculative_algorithm")
            return f"{name}{f' +{spec}' if spec else ''}"
        except (urllib.error.URLError, ValueError, OSError) as e:
            print(f"[remote] {self.upstream} not answering yet ({e}); will keep trying",
                  file=sys.stderr)
            return "?"

    def _post(self, body: dict) -> Any:
        req = urllib.request.Request(f"{self.upstream}/generate", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            try:
                detail = json.loads(detail).get("error", {}).get("message", detail)
            except (ValueError, AttributeError):
                pass
            raise RemoteError(e.code, detail) from None
        except urllib.error.URLError as e:
            raise RemoteError(0, f"cannot reach {self.upstream} ({e.reason})") from None

    def _logprobs(self, prompt: str, tids: list[int]) -> np.ndarray:
        out = self._post({"text": prompt,
                          "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                          "return_logprob": True, "token_ids_logprob": tids})
        rows = out["meta_info"]["output_token_ids_logprobs"][0]    # [[logprob, id, text], ...]
        by_id = {int(r[1]): float(r[0]) for r in rows}
        return np.array([by_id[t] for t in tids], dtype=float)

    def _sampled(self, prompt: str, tids: list[int]) -> np.ndarray:
        n = self.samples
        out = self._post({"text": prompt,
                          "sampling_params": {"max_new_tokens": 1, "temperature": 1.0,
                                              "top_p": 1.0, "top_k": -1, "n": n}})
        if isinstance(out, dict):
            out = [out]
        counts = Counter(o["output_ids"][0] for o in out if o.get("output_ids"))
        off = sum(v for k, v in counts.items() if k not in tids)
        if off > n // 4:
            print(f"[remote] {off}/{n} samples were not an answer letter", file=sys.stderr)
        # Additive smoothing so an unseen option is unlikely rather than impossible;
        # softmax of these logs is the smoothed sample fraction.
        return np.log([(counts.get(t, 0) + 0.5) / (n + 0.5 * len(tids)) for t in tids])

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
        tids = [token for _, token in slots]

        if self.served == "?":
            self.served = self._served_name()
            self.name = f"{self.served} via {self.upstream}"
        if self.exact is not False or time.monotonic() >= self._retry_exact_at:
            try:
                z = self._logprobs(prompt, tids)
                if self.exact is not True:
                    print(f"[remote] {self.served}: reading log-probabilities", file=sys.stderr)
                self.exact = True
                return z, len(ids)
            except RemoteError as e:
                if e.code != 400 or "logprob" not in e.message.lower():
                    raise
                if self.exact is not False:
                    print(f"[remote] {self.served} refuses logprobs ({e.message}); "
                          f"estimating from {self.samples} samples per reading, "
                          f"retrying the exact path every {RETRY_EXACT_SECONDS} s", file=sys.stderr)
                self.exact = False
                self._retry_exact_at = time.monotonic() + RETRY_EXACT_SECONDS
        return self._sampled(prompt, tids), len(ids)


def add_engine_args(p) -> None:
    """The flags every entry point shares for choosing an engine."""
    p.add_argument("--upstream", metavar="URL",
                   help="use a running sglang server instead of loading a model here, "
                        "e.g. http://localhost:30002 (fleet) - thinking is off for these "
                        "requests only")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                   help=f"with --upstream: the served model's tokenizer (default {DEFAULT_TOKENIZER})")
    p.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                   help="with --upstream: single-token samples per reading when the server "
                        f"cannot return log-probabilities (default {DEFAULT_SAMPLES})")


def engine_from_args(a, **kw) -> Decider:
    """`Remote` when --upstream is given, else the in-process `Rev`."""
    if getattr(a, "upstream", None):
        print(f"connecting to {a.upstream}...", file=sys.stderr)
        return Remote(a.upstream, tokenizer=a.tokenizer, samples=a.samples, **kw)
    from .decide import Rev
    bits = getattr(a, "bits", 8)
    print(f"loading {a.model} ({'source precision' if not bits else f'{bits}-bit'})...",
          file=sys.stderr)
    return Rev(a.model, bits=bits or None, **kw)
