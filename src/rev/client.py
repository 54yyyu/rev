"""Talk to `rev serve` without loading a model. Standard library only.

    from rev import Client
    rev = Client()                       # http://127.0.0.1:8421
    d = rev.decide("My card was charged twice.", "Which team?",
                   {"billing": "Payments and refunds", "technical": "Bugs"})
    d.choice, d.confidence

`Client(url, key)` also talks to TypeSafe's own endpoint, since the protocol is
the same: `Client("https://api.typesafe.ai", key=...)`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Mapping

from .decision import Decision
from .serve import DEFAULT_PORT


class Client:
    def __init__(self, url: str = f"http://127.0.0.1:{DEFAULT_PORT}", key: str | None = None,
                 model: str = "jev-latest", timeout: float = 120):
        self.url = url.rstrip("/") + "/v1/systemone"
        self.key, self.model, self.timeout = key, model, timeout

    def ask(self, state: Any, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        """Jev's request and response, verbatim."""
        body = json.dumps({"model": self.model, "state": state, "questions": questions}).encode()
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(self.url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            raise RuntimeError(f"{e.code} from {self.url}: {detail}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(f"cannot reach {self.url} ({e.reason}); is `rev serve` running?") from None

    def decide(self, state: Any, criterion: str, options: Mapping[str, str]) -> Decision:
        """One choice question, returned the way `Rev.decide` returns it."""
        out = self.ask(state, {"q": {"type": "choice", "instructions": criterion,
                                     "criteria": dict(options)}})
        a = out["answers"]["q"]
        return Decision(choice=a["choice"], probabilities=a["probabilities"],
                        confidence=a["confidence"],
                        input_tokens=out.get("usage", {}).get("input_tokens", 0),
                        seconds=out.get("seconds", 0.0))

    def noul(self, state: Any, question: str, yes: str | None = None, no: str | None = None) -> float:
        """Probability the answer is yes."""
        q: dict[str, Any] = {"type": "noul", "instructions": question}
        if yes or no:
            q["criteria"] = {k: v for k, v in (("true", yes), ("false", no)) if v}
        return self.ask(state, {"q": q})["answers"]["q"]["noul"]
