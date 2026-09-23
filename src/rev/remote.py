"""The same decisions from a model that is already being served elsewhere.

    from rev.remote import Remote
    h = Remote("http://localhost:30002")           # an sglang endpoint
    h.decide(state, "Which team should handle this?", {...})

We keep a 27B model up on a cluster for coding agents. Nothing about the
readout needs the model in this process: the prompt is rendered here with the
served model's own tokenizer and chat template (thinking off, for this request
only - the server's default is untouched and every other client keeps theirs),
and the server is asked for the next-token distribution at the answer slots.
The request is one prompt and one output token, so it rides alongside an
agent's long generations without displacing them.

Images in the state (OpenAI `image_url` content parts, see `prompt.split_images`)
are sent to the server as `image_data`; the template's placeholders mark where
they go, and the server's vision encoder reads them as it would for a chat
request. Only the served model needs to see images; the tokenizer here does not.

Two ways to read the distribution, chosen at the first request:

- exact: `/generate` with `return_logprob` and `token_ids_logprob`, which
  returns the log-probability of each slot token at the first output position.
  This is the same quantity the MLX engine reads.
- sampled: some speculative decoders refuse to return log-probabilities at all
  (DSpark, on our endpoint as of 2026-09-21: "DSpark speculative decoding
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
from .prompt import render, render_free, split_images

# The tokenizer of the model we serve. Only tokenizer and template
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

    def _logprobs(self, prompt: str, tids: list[int], media: dict) -> tuple[np.ndarray, int | None]:
        out = self._post({"text": prompt, **media,
                          "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                          "return_logprob": True, "token_ids_logprob": tids})
        rows = out["meta_info"]["output_token_ids_logprobs"][0]    # [[logprob, id, text], ...]
        by_id = {int(r[1]): float(r[0]) for r in rows}
        return np.array([by_id[t] for t in tids], dtype=float), out["meta_info"].get("prompt_tokens")

    def _sampled(self, prompt: str, tids: list[int], media: dict) -> tuple[np.ndarray, int | None]:
        n = self.samples
        out = self._post({"text": prompt, **media,
                          "sampling_params": {"max_new_tokens": 1, "temperature": 1.0,
                                              "top_p": 1.0, "top_k": -1, "n": n}})
        if isinstance(out, dict):
            out = [out]
        served = out[0].get("meta_info", {}).get("prompt_tokens") if out else None
        counts = Counter(o["output_ids"][0] for o in out if o.get("output_ids"))
        off = sum(v for k, v in counts.items() if k not in tids)
        if off > n // 4:
            print(f"[remote] {off}/{n} samples were not an answer letter", file=sys.stderr)
        # Additive smoothing so an unseen option is unlikely rather than impossible;
        # softmax of these logs is the smoothed sample fraction.
        return np.log([(counts.get(t, 0) + 0.5) / (n + 0.5 * len(tids)) for t in tids]), served

    def _branch(self, state: Any, criterion: str, options: list[tuple[str, str]]) -> tuple[np.ndarray, int]:
        slots = self.slots[: len(options)]
        state, images = split_images(state)
        prompt = render(self.tokenizer, state, criterion,
                        [(slot[0], text) for slot, (_, text) in zip(slots, options)], len(images))
        media = {"image_data": images} if images else {}
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        # The text alone; each image adds its own tokens on the server, which
        # enforces its context length on the whole.
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
                z, served = self._logprobs(prompt, tids, media)
                if self.exact is not True:
                    print(f"[remote] {self.served}: reading log-probabilities", file=sys.stderr)
                self.exact = True
                return z, served or len(ids)
            except RemoteError as e:
                if e.code != 400 or "logprob" not in e.message.lower():
                    raise
                if self.exact is not False:
                    print(f"[remote] {self.served} refuses logprobs ({e.message}); "
                          f"estimating from {self.samples} samples per reading, "
                          f"retrying the exact path every {RETRY_EXACT_SECONDS} s", file=sys.stderr)
                self.exact = False
                self._retry_exact_at = time.monotonic() + RETRY_EXACT_SECONDS
        z, served = self._sampled(prompt, tids, media)
        return z, served or len(ids)


    def read_tokens(self, state: Any, question: str, answers: list[str],
                    system: str = "Answer with a single token and nothing else.") -> tuple[dict[str, float], float]:
        """The model's next-token distribution over `answers`, each one token.

        For answers that are not option letters: a digit on a scale, a word.
        Returns (probabilities renormalised over `answers`, the share of the
        model's whole next-token probability that fell on them); a low share
        means it wanted to say something else. Exact when the server returns
        log-probabilities, estimated from samples otherwise.
        """
        state, images = split_images(state)
        prompt = render_free(self.tokenizer, state, question, system, len(images))
        media = {"image_data": images} if images else {}
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        tids = []
        for a in answers:
            extended = self.tokenizer.encode(prompt + a, add_special_tokens=False)
            if extended[:-1] != ids or len(extended) != len(ids) + 1:
                raise ValueError(f"answer {a!r} is not one token after this prompt")
            tids.append(extended[-1])
        if len(set(tids)) != len(tids):
            raise ValueError("answers share a token")
        z = None
        if self.exact is not False or time.monotonic() >= self._retry_exact_at:
            try:
                z, _ = self._logprobs(prompt, tids, media)
                self.exact = True
            except RemoteError as e:
                if e.code != 400 or "logprob" not in e.message.lower():
                    raise
                self.exact = False
                self._retry_exact_at = time.monotonic() + RETRY_EXACT_SECONDS
        if z is None:
            z, _ = self._sampled(prompt, tids, media)
        mass = float(np.exp(z).sum())
        p = np.exp(z - z.max()); p /= p.sum()
        return dict(zip(answers, p.tolist())), mass

    def generate(self, state: Any, question: str, max_new_tokens: int = 24,
                 system: str = "You are a helpful assistant.") -> str:
        """A short greedy answer, for formats that take several tokens (coordinates)."""
        state, images = split_images(state)
        prompt = render_free(self.tokenizer, state, question, system, len(images))
        media = {"image_data": images} if images else {}
        out = self._post({"text": prompt, **media,
                          "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0}})
        return out["text"] if isinstance(out, dict) else out[0]["text"]


def add_engine_args(p) -> None:
    """The flags every entry point shares for choosing an engine."""
    p.add_argument("--upstream", metavar="URL",
                   help="use a running sglang server instead of loading a model here, "
                        "e.g. http://localhost:30002 - thinking is off for these "
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
