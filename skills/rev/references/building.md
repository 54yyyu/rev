# Building software that uses rev

Read this when the program you are writing needs a semantic judgment at run
time: routing a request, choosing among candidates, checking a claim, scoring
items for a view. The program owns the workflow; rev supplies the judgment
where ordinary code would need to understand language or a picture. Keep known
rules, calculations, exact lookups and side effects in code.

## Calling it

Python, standard library only, no model loaded in the caller:

```python
import os
from rev import Client            # pip install -e <rev checkout>, or copy client.py

rev = Client(os.environ.get("REV_URL", "http://127.0.0.1:8421"))
out = rev.ask(state, {
    "team":   {"type": "choice", "instructions": "Which team should handle this ticket?",
               "criteria": {"billing": "...", "technical": "...", "none": "..."}},
    "urgent": {"type": "noul", "instructions": "Does the ticket convey urgency?"},
})
out["answers"]["team"]["choice"], out["answers"]["urgent"]["noul"]

rev.decide(state, "Which team?", {"billing": "...", "technical": "..."}).choice   # one choice
rev.noul(state, "Is this spam?")                                                  # p(yes)
```

`Client.ask` raises `rev.client.RevError` with `.status` (None when
unreachable, 502 while the served model is between restarts, 400 for a bad
question). Anything else: `POST $REV_URL/v1/systemone` with
`{"state": ..., "questions": {...}}`; the response is
`{"answers": {id: {...}}, "usage": {...}, "seconds": ...}`.

The same client talks to TypeSafe's hosted Jev:
`Client("https://api.typesafe.ai", key=...)`. Jev takes no images; rev does
when its model is a vision model. Keep keys server-side.

## Shapes that work

- **Route and fill.** One `choice` picks the handler; in the same request, ask
  the branch-specific questions each handler would need (their premise stated
  in the instructions: "If this is a refund request, ..."). Use only the
  answers of the branch taken. One round trip instead of two.
- **Select, do not generate.** Find candidate values or spans in code (regex,
  parser, retrieval), make them the options of a `choice`, and copy the chosen
  one. Include a "none of these" option, and test that the right candidate is
  usually among the options.
- **Rank.** One `score` per item on the same scale ("How relevant is
  `doc` to `query`?"), then sort in code. Several scored dimensions can be
  combined with weights in code, and re-weighted without asking again.
- **Several labels.** One `noul` per label; a `choice` forces exactly one.
- **Verify and escalate.** A `noul` per claim ("Is `claim` supported by
  `source`?"). Accept confident answers; send low-confidence or failing ones
  to a person or a reasoning model.
- **Changing state.** Keep goals and observations in code; ask a fresh
  judgment for each next step, with the current state, and do not reuse an
  answer after the state it was about has changed.

## Getting the judgments right

- One narrow judgment per question; the id is not sent, so the instructions
  carry the full meaning. Criteria describe each option so it stands on its
  own; score levels describe concrete situations, lowest first.
- Give each question the evidence it needs: source text, identities, policy,
  current facts. Named JSON fields beat one long string; refer to them by
  path (`ticket.messages[0].text`).
- Questions in one request do not see each other's answers. When a later
  question needs an earlier answer (to fetch evidence or build new options),
  that is a second request.
- `confidence` measures how concentrated the distribution is, not whether the
  workflow is right. Choose thresholds on real examples of the task and the
  cost of each kind of mistake; do not copy them from here. Low confidence
  across two harmless options is fine; ignore uncertainty on branches you do
  not use.
- Test on representative cases. When one fails, look at the exact state,
  questions, options and answers, and decide whether the evidence was missing,
  the options were wrong, the model was wrong, or the code around it was.

TypeSafe's documentation (https://docs.typesafe.ai/llms.txt, pages as `.md`)
has longer worked examples of these patterns; the protocol is the same, and
rev supports its `choice`, `noul` and `score` question types.
