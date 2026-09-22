---
name: rev
description: >
  Typed judgments from rev, a Jev-compatible decision server: pick one of a set
  of options (choice), the probability a condition holds (noul), or a level on
  a described scale (score), for text, JSON or screenshots, in well under a
  second each. Use to classify, filter, triage or rank many items (log lines,
  files, rows, messages, screenshots) without reading them all into your
  context; to get a calibrated probability to gate an action on; or when
  writing code that needs a semantic decision (routing, picking the right
  candidate value, verifying a claim) through rev or TypeSafe's Jev.
license: MIT
---

# rev: typed decisions as a tool

rev answers TypeSafe's Jev protocol (`POST /v1/systemone`) by reading one
token's log-probabilities from a language model. You give it a **state** (the
evidence) and **questions**; it returns typed answers with probabilities.
Nothing is generated, so the answer is always one of the options you listed.

## Is it there?

```bash
rev health        # exit 0 and {"ok": true, "mode": ...} when usable; exit 3 when not
```

The server is `$REV_URL` (else a local `rev serve` on 127.0.0.1:8421). If
`rev health` exits 3, or `$REV_URL` is unset and nothing is local, rev is not
available here: do the judgment yourself and do not keep retrying. If the
`rev` command is missing but the URL answers, use the curl form below.

## When it is worth a call, and when it is not

rev may be reading **the same model you are running on**, with thinking off.
A single judgment you can already make from what is in your context is not
better for going through rev. Use it when:

- **There are many items.** Twenty or two thousand log lines, files, rows or
  screenshots, each needing the same question. Put them in a JSONL file and
  let `rev ask --jsonl` read them; only the answers come back to you, not the
  items.
- **You need a number to act on.** A probability or confidence you can put a
  threshold on ("only delete if p(unused) > 0.95", "ask the user when
  confidence < 0.8"), and that means the same thing every time.
- **The evidence is a picture.** A screenshot costs rev about a thousand tokens
  and half a second; it does not enter your context at all.
- **You are writing a program** that needs the judgment at run time. See
  [references/building.md](references/building.md).

Do not use it for arithmetic, exact lookups, parsing, or anything plain code
does; do that in code.

## Asking

A question is `{"type": ..., "instructions": ..., "criteria": ...}` under an id
of your choosing. The id is never sent to the model, so the instructions must
say the whole question.

| type | criteria | answer |
|---|---|---|
| `choice` | `{"key": "what this option means", ...}` | `choice` (a key), `probabilities`, `confidence` |
| `noul` | optional `{"true": "...", "false": "..."}` | `noul`: probability of yes |
| `score` | `["lowest level", ..., "highest level"]` | `score` (0..n-1, probability-weighted), `confidence` |

One request, questions inline (all questions about one state go in one
request; each is answered on its own and cannot see the others):

```bash
rev ask --brief <<'EOF'
{"state": "Payouts have failed for 3 days and we are losing money.",
 "questions": {
   "team":   {"type": "choice", "instructions": "Which team should handle this message?",
              "criteria": {"billing": "Payments, payouts, invoices, refunds",
                           "technical": "Bugs, outages, integrations",
                           "none": "Neither team; not a support request"}},
   "urgent": {"type": "noul", "instructions": "Does the message convey urgency?"},
   "mood":   {"type": "score", "instructions": "How frustrated is the writer?",
              "criteria": ["Calm", "Annoyed", "Very angry"]}}}
EOF
# {"team": {"choice": "billing", "confidence": 0.93}, "urgent": {"noul": 1.0},
#  "mood": {"score": 1.05, "level": "Annoyed", "confidence": 0.91}}
```

The state from a file, the questions from another, a screenshot added:

```bash
rev ask --questions q.json --state-file build.log --brief
rev ask --questions q.json --state "The orders page after clicking Refund." --image shot.png --brief
```

Many items: one JSON object per line, `{"id": ..., "state": ...}`, the same
questions for all (or a `questions` field per line):

```bash
rev ask --jsonl items.jsonl --questions q.json --brief -o answers.jsonl
# answers.jsonl: {"id": ..., "answers": {...}} or {"id": ..., "error": ...}, in input order
```

Build `items.jsonl` with a short script (`jq -c`, python), not by typing it
into your reply. Then read answers.jsonl with code: filter, sort, count. `-j`
sets requests in flight (default 2, max 8); the server's model is usually
serving other work too, so leave it low unless the batch is large.

Images: a state can hold OpenAI `image_url` parts anywhere; a local file path
as the url is read and sent for you. With several images, the model sees them
as "Picture 1", "Picture 2", ... in order, so the instructions can refer to
them by that name.

Without the `rev` command:

```bash
curl -s "$REV_URL/v1/systemone" -H 'Content-Type: application/json' -d @request.json
```

## Writing good questions

- **One narrow judgment per question.** Split independent dimensions into
  separate questions in the same request; do not split a relationship you need
  judged as a whole.
- **Describe every option**, in the criteria, so each stands on its own.
  Add a `none` / `other` option when nothing may fit; the model cannot pick an
  answer you did not list.
- **Give it the evidence it needs** in the state: the text itself, the
  relevant policy, the facts. Prefer named JSON fields when the state has
  several parts, and refer to them in the instructions by path, e.g.
  `ticket.messages[0].text`.
- **Several labels may apply?** One `noul` per label, not a `choice`.
- **Ranking?** One `score` per item on the same scale, then sort in code.
- **Selecting a value** (which of these dates is the due date): find the
  candidates in code, make them the options, copy the chosen one. Check that
  the right candidate is among the options at all.

## Reading the answers

- `confidence` (choice, score) is how concentrated the probability is. Below
  about 0.8, treat the answer as unsure: check it yourself, ask the user, or
  report it as uncertain. These are starting thresholds; judge them on the
  task.
- A `noul` near 0.5 means yes and no are about equally likely, not "half
  true". Gate actions on it with a threshold chosen for the cost of a mistake.
- Typed answers guarantee the shape, not the truth. For anything destructive
  or outward-facing, spot-check a few answers against the items before acting
  on all of them, and say in your report that the classification came from
  rev.

## When it fails

`rev ask` exits 0 answered, 1 when some `--jsonl` lines failed (their lines
say why), 2 for a bad request (the message says what; fix and rerun), 3 when
the server or its model is down. A down model is usually restarting and back
in minutes; `rev ask` already retried. Do not loop: carry on without it, or
tell the user it is down.
