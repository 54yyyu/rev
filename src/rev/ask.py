"""`rev ask` and `rev health` - ask a running `rev serve` (or Jev) from a shell.

    rev ask request.json                   {"state": ..., "questions": {...}}
    rev ask < request.json                 the same, from stdin
    rev ask --questions q.json --state-file notes.txt
    rev ask --questions q.json --state "Charged twice" --image shot.png
    rev ask --jsonl items.jsonl --questions q.json -o out.jsonl

The request is Jev's: `questions` maps an id to a choice / noul / score
question. The answer printed is the server's response, or with `--brief` one
line per request holding only what a caller acts on.

`--jsonl` answers one request per line, `{"id": ..., "state": ..., "questions":
...}`, with `questions` optional when `--questions` gives them to every line.
Output lines come back in input order, `{"id": ..., "answers": ...}` or
`{"id": ..., "error": ...}`; a failed line does not stop the others.

Images: an `image_url` part whose url is a local file path is read here and
sent as a data URI, and `--image PATH` adds one to the state. The server
itself only accepts data: and http(s) URLs.

The server is --url, else $REV_URL, else http://127.0.0.1:8421. $REV_KEY (or
--key) is sent as a bearer token, for Jev.

Exit status: 0 answered; 1 some --jsonl lines failed; 2 a bad request or
arguments; 3 the server could not be reached or its model is down (502/503).
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .client import Client, RevError
from .serve import DEFAULT_PORT

UNREACHABLE = 3
# Down, or between model servers: worth another try. A 400 is not.
TRANSIENT = (None, 502, 503, 504)


def default_url() -> str:
    return os.environ.get("REV_URL") or f"http://127.0.0.1:{DEFAULT_PORT}"


def _data_uri(path: str) -> str:
    p = Path(path).expanduser()
    if not p.is_file():
        raise ValueError(f"image {path!r} is not a file (urls must be data:, http(s) or a local path)")
    mime = mimetypes.guess_type(p.name)[0] or ""
    if not mime.startswith("image/"):
        raise ValueError(f"image {path!r} does not look like an image ({mime or 'unknown type'})")
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"


def _image_part(path: str) -> dict:
    return {"type": "image_url", "image_url": {"url": _data_uri(path)}}


def inline_images(value: Any) -> Any:
    """Every image_url part whose url is a local path, with the file read in."""
    if isinstance(value, dict):
        if value.get("type") == "image_url":
            iu = value.get("image_url")
            url = iu.get("url") if isinstance(iu, dict) else iu
            if isinstance(url, str) and not url.startswith(("data:", "http://", "https://")):
                return {**value, "image_url": {**(iu if isinstance(iu, dict) else {}),
                                               "url": _data_uri(url)}}
            return value
        return {k: inline_images(v) for k, v in value.items()}
    if isinstance(value, list):
        return [inline_images(v) for v in value]
    return value


def add_images(state: Any, paths: list[str]) -> Any:
    """The state with images appended where the server will find them."""
    if not paths:
        return state
    parts = [_image_part(p) for p in paths]
    if state is None:
        return parts
    if isinstance(state, str):
        return [{"type": "text", "text": state}, *parts]
    if (isinstance(state, list) and
            all(isinstance(p, dict) and p.get("type") in ("text", "image_url") for p in state)):
        return [*state, *parts]
    if isinstance(state, dict) and "images" not in state:
        return {**state, "images": parts}
    return {"state": state, "images": parts}


def brief(response: dict) -> dict:
    """Only what a caller acts on: the choice, the probability of yes, the level."""
    out = {}
    for qid, a in response.get("answers", {}).items():
        t = a.get("type")
        if t == "choice":
            out[qid] = {"choice": a["choice"], "confidence": round(a["confidence"], 3)}
        elif t == "noul":
            out[qid] = {"noul": round(a["noul"], 3)}
        elif t == "score":
            legend = a.get("legend") or {}
            nearest = legend.get(str(round(a["score"])))
            out[qid] = {"score": round(a["score"], 3), "level": nearest,
                        "confidence": round(a["confidence"], 3)}
        else:
            out[qid] = a
    return out


def _load_json(text: str, where: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{where} is not JSON: {e}") from None


def _ask(client: Client, state: Any, questions: Any, retries: int) -> dict:
    for attempt in range(retries + 1):
        try:
            return client.ask(state, questions)
        except RevError as e:
            if e.status not in TRANSIENT or attempt == retries:
                raise
            time.sleep((3, 10, 30)[min(attempt, 2)])
    raise AssertionError("unreachable")


def _client(a) -> Client:
    return Client(a.url, key=a.key, timeout=a.timeout)


def _single(a, p) -> int:
    if a.request == "-" or (a.request is None and not sys.stdin.isatty()
                            and a.state is None and a.state_file is None):
        # An agent's shell hands over a non-terminal, empty stdin; that is no request.
        text = sys.stdin.read()
        req = _load_json(text, "stdin") if text.strip() else {}
    elif a.request is not None:
        req = _load_json(Path(a.request).read_text(), a.request)
    else:
        req = {}
    if not isinstance(req, dict):
        p.error("the request must be a JSON object {\"state\": ..., \"questions\": {...}}")
    state = req.get("state")
    if a.state is not None:
        state = a.state
    if a.state_file is not None:
        state = Path(a.state_file).read_text()
    questions = req.get("questions") or a.shared_questions
    if not questions:
        p.error("no questions: give them in the request or with --questions")
    if state is None and not a.image:
        p.error("no state: give it in the request, or with --state / --state-file / --image")
    state = add_images(inline_images(state), a.image)
    out = _ask(_client(a), state, questions, a.retries)
    print(json.dumps(brief(out) if a.brief else out, ensure_ascii=False,
                     indent=None if a.brief else 1))
    return 0


def _batch(a, p) -> int:
    src = sys.stdin if a.jsonl == "-" else open(a.jsonl)
    lines = [(n, line) for n, line in enumerate(src, 1) if line.strip()]
    if a.output and a.output.exists():
        p.error(f"{a.output} exists; refusing to overwrite a result")
    client = _client(a)

    def one(item):
        n, line = item
        rid: Any = n
        try:
            row = _load_json(line, f"line {n}")
            if not isinstance(row, dict):
                raise ValueError(f"line {n} is not a JSON object")
            rid = row.get("id", n)
            questions = row.get("questions") or a.shared_questions
            if not questions:
                raise ValueError(f"line {n} has no questions and --questions was not given")
            state = inline_images(row.get("state"))
            out = _ask(client, state, questions, a.retries)
            res = {"id": rid, "answers": brief(out) if a.brief else out["answers"]}
            if not a.brief:
                res["seconds"] = out.get("seconds")
            return res
        except RevError as e:
            return {"id": rid, "error": str(e), "status": e.status,
                    "_transient": e.status in TRANSIENT}
        except (ValueError, OSError) as e:
            return {"id": rid, "error": str(e)}

    fh = a.output.open("w") if a.output else sys.stdout
    failed = unreachable = 0
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for i, res in enumerate(pool.map(one, lines), 1):   # map keeps input order
                if "error" in res:
                    failed += 1
                    unreachable += res.pop("_transient", False)
                fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 50 == 0:
                    rate = i / (time.perf_counter() - started)
                    print(f"  {i}/{len(lines)}  {rate:.1f}/s", file=sys.stderr, flush=True)
    finally:
        if fh is not sys.stdout:
            fh.close()
    elapsed = time.perf_counter() - started
    print(f"{len(lines)} requests, {failed} failed, in {elapsed:.1f}s"
          + (f" -> {a.output}" if a.output else ""), file=sys.stderr)
    if lines and unreachable == len(lines):
        return UNREACHABLE
    return 1 if failed else 0


def ask(argv=None) -> int:
    p = argparse.ArgumentParser(prog="rev ask", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("request", nargs="?", help="request JSON file, or - for stdin")
    p.add_argument("--questions", type=Path, help="JSON file: the questions map, for every request")
    p.add_argument("--state", help="the state, as text")
    p.add_argument("--state-file", help="the state, as the text of this file")
    p.add_argument("--image", action="append", default=[], metavar="PATH",
                   help="add an image file to the state (repeatable)")
    p.add_argument("--jsonl", metavar="FILE", help="one request per line; - for stdin")
    p.add_argument("-o", "--output", type=Path, help="--jsonl results here instead of stdout")
    p.add_argument("-j", "--jobs", type=int, default=2,
                   help="--jsonl requests in flight (default 2, at most 8: they share the "
                        "server's model with whatever else it serves)")
    p.add_argument("--brief", action="store_true",
                   help="print only choice / noul / score (+ level, confidence)")
    p.add_argument("--url", default=default_url(), help="default: $REV_URL, else the local server")
    p.add_argument("--key", default=os.environ.get("REV_KEY"), help="bearer token (Jev); default $REV_KEY")
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--retries", type=int, default=2,
                   help="retries when the server is unreachable or its model is down (default 2)")
    a = p.parse_args(argv)
    if not 1 <= a.jobs <= 8:
        p.error("--jobs must be 1..8")
    if a.jsonl and (a.request or a.state is not None or a.state_file or a.image):
        p.error("--jsonl takes its states from the file; drop the request / --state / "
                "--state-file / --image (put images in each line's state)")
    try:
        a.shared_questions = (_load_json(a.questions.read_text(), str(a.questions))
                              if a.questions else None)
        return _batch(a, p) if a.jsonl else _single(a, p)
    except RevError as e:
        print(f"rev ask: {e}", file=sys.stderr)
        if e.status in TRANSIENT:
            print("rev ask: the server or its model is down (it may be restarting); "
                  "`rev health` to check", file=sys.stderr)
            return UNREACHABLE
        return 2
    except (ValueError, OSError) as e:
        print(f"rev ask: {e}", file=sys.stderr)
        return 2


def health(argv=None) -> int:
    p = argparse.ArgumentParser(prog="rev health",
                                description="Is the server up, and what does it read? Exit 0 yes, 3 no.")
    p.add_argument("--url", default=default_url(), help="default: $REV_URL, else the local server")
    p.add_argument("--timeout", type=float, default=10)
    a = p.parse_args(argv)
    url = a.url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=a.timeout) as resp:
            body = json.loads(resp.read())
    except Exception as e:  # noqa: BLE001 - any failure here means "not usable"
        print(f"rev health: cannot reach {url} ({getattr(e, 'reason', e)})", file=sys.stderr)
        return UNREACHABLE
    print(json.dumps(body, ensure_ascii=False))
    return 0 if body.get("ok") else UNREACHABLE
