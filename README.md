# rev

**Typed decisions from a frozen open model, in one forward pass, on Apple Silicon.**

You give it a state, a criterion and a list of options. It returns a probability
for each option. Nothing is generated, so it cannot answer with something that
was not on your list, and there is no JSON to repair.

```python
from rev import Rev

h = Rev("Qwen/Qwen3.5-2B")          # 8-bit by default, 1.9 GB resident
d = h.decide(
    "My card was charged twice for one order.",
    "Which team should handle this?",
    {"billing": "Payments, refunds and invoices",
     "technical": "Bugs and outages",
     "sales": "Pricing and new accounts"},
)
d.choice          # 'billing'
d.confidence      # 0.965
d.above(0.9)      # 'billing', or None when it should go to a person
```

No training, no adapter, no API key. The model is an ordinary open checkpoint;
everything here is how it is asked and how its logits are read.

## A local Jev

`rev serve` answers TypeSafe's endpoint on this machine: same route, same
request, same response. A client written for Jev works by changing its URL.

```bash
rev serve                                   # http://127.0.0.1:8421, loads once
curl -s localhost:8421/v1/systemone -d '{
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "department":  {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"billing": "Payments, invoicing, refunds",
                                 "technical": "Bugs, outages, integrations"}},
    "is_urgent":   {"type": "noul",   "instructions": "Does this convey urgency?"},
    "frustration": {"type": "score",  "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}}}'
```

From Python, without loading a model in your own process:

```python
from rev import Client
rev = Client()                                    # the local server
jev = Client("https://api.typesafe.ai", key=...)  # the same calls, against Jev
rev.decide(state, "Which team?", {"billing": "...", "technical": "..."}).choice
rev.noul(state, "Does this convey urgency?")      # probability of yes
```

`Rev.ask(state, questions)` is the same thing in-process. One server holds one
copy of the weights for every tool on the machine; it binds 127.0.0.1 unless
told otherwise, and refuses to start on a port that is already taken.

The three question types are Jev's: `choice` (a map of options), `noul` (yes/no,
answer is the probability of yes) and `score` (ordered levels, answer is the
probability-weighted level). Instructions and criteria may be strings, objects
or arrays, as in Jev.

## Why it exists

Three good implementations of this idea already exist — [SemIf], [reflex] and
[decider] — and this one started as a measurement harness to compare them. It
became its own thing when the measurements said the harness mattered more than
the model: a frozen Qwen3.5-2B moved from 0.428 to 0.550 on JevBench's hard
public items without changing a single weight.

[SemIf]: https://github.com/TheoLeeCJ/SemIf
[reflex]: https://github.com/kshetrajna12/reflex
[decider]: https://huggingface.co/Mapika/decider-2b

What each of them contributed, and what was measured here:

| from | what | measured |
|---|---|---|
| SemIf | Evidence / Criterion prompt, letter-slot readout, tokenizer round-trip checks | the reference; `rev` matches its in-process API exactly, probability delta 0 |
| decider | a wide label space instead of a fixed alphabet | 16 slots → **101** on Qwen3.5 |
| reflex | averaging two option orders | fixes position bias: 0.906 → 1.000 on clipboard paste; used only when the first reading is unsure, and never raises confidence |
| here | **8-bit instead of 4-bit** | **about +9 points on hard, no latency cost**; 8-bit and bf16 give identical answers |
| here | a fitted per-option offset | +7.9 points on binary questions, transfers across task tiers |

## Measured

Qwen3.5-2B, 8-bit, on this machine (M-series, 16 GB), JevBench public items:

| | easy (48) | standard (72) | hard (111) |
|---|---:|---:|---:|
| rev, frozen Qwen3.5-2B | 1.000 | 0.764 | **0.550** |
| rev, one reading only (`orders="one"`) | 1.000 | 0.764 | 0.523 |
| decider-2b (fine-tuned, same base) | 1.000 | **0.861** | 0.477 |
| reflex, published, 2B bf16 | 1.000 | — | 0.468 |
| Jev 1.13.0 (commercial) | 1.000 | 0.986 | 0.730 |

Speed against Jev, same client and items, only the URL different (M2 Pro,
`rev serve` with defaults, 2026-09-21):

| request | rev p50 / p95 | Jev p50 / p95 |
|---|---:|---:|
| short: clipboard paste, ~230 tokens | **266 / 473 ms** | 354 / 423 ms |
| medium: JevBench standard, ~160 tokens | **240 / 423 ms** | 356 / 396 ms |
| long: JevBench hard, 1-4k tokens | 821 / 3666 ms | **326 / 437 ms** |
| long, three questions in one request | 1280 / 3862 ms | **328 / 388 ms** |
| 8 requests in flight at once | 0.7-3.9 per s | **21-24 per s** |

Short and medium questions are faster than Jev. Long documents are not and
cannot be on this laptop: reading 3,300 tokens is about 13 TFLOP, two to three
seconds on an M2 Pro whatever the code does, where Jev runs on datacenter GPUs.
What `rev` does instead is read a state once: a second reading or another
question on the same document costs about 150 ms, not another full read. Nor
does it parallelise: one laptop GPU runs one forward pass at a time.

Memory: 1.86 GB of weights at 8-bit, 3.2 GB peak on a 3,300-token prompt.

| | weights | short question | 3.3k-token question |
|---|---:|---:|---:|
| 2B, 8-bit *(default)* | 1.86 GB | 262 ms | 3.4 s |
| 2B, bf16 | 4.3 GB | 212 ms | 2.6 s |
| 2B, 4-bit | 0.99 GB | 266 ms | 3.5 s |

**Use 8-bit.** 4-bit saves 0.87 GB, no time, and costs nine points on hard;
bf16 is 20% faster for 2.4 GB more and gives the same answers.

The tasks this was built for, written the way a user would write them
(`Rev()` defaults, real calendar, 8-bit):

| task | n | accuracy | at confidence ≥ 0.9: answered / correct |
|---|---:|---:|---:|
| clipboard paste, 5–20 items on the clipboard | 160 | **1.000** | 66% / 1.000 |
| pick a calendar event from a request, English and Chinese, 5–27 candidates | 216 | **0.986** | 71% / 0.987 |
| write-action gate: classify the action, decide risk in code, plus `vague_action` | 57 | **0.965** | 75% / 1.000 |

Read through pyapple instead, the same calendar has 97 events in a two-month
window and the pick scores 0.953 on 192 items: lecture and recitation of one
course ("lec" / "rec") are the usual confusion, and one wrong pick came at
p = 0.98, so a write that follows a pick still needs confirming.

Jev scored 1.000 on the first two. Every miss in the gate was a safe request
sent for confirmation, never the other way. Latency: 225-330 ms median.

Email triage on 146 real messages from this inbox (19 important, labelled
before the model ran): asking "what kind of email is this?" over seven
categories and ranking by the action, personal and account probabilities gave
**AUC 0.941** (95% interval 0.89-0.98), with 18 of the 19 in the top 18%. A
sender regex scored 0.375, because the important mail comes from noreply
addresses as often as the junk does. Use `orders="one"` for ranking like this:
the second reading did not help (0.927-0.936, inside the interval) and doubled
the time.

## Gating

The number you act on is not accuracy, it is what a confidence gate buys. On the
standard tier:

| threshold | coverage | accuracy on what it answered |
|---:|---:|---:|
| none | 1.00 | 0.764 |
| 0.80 | 0.61 | 0.886 |
| **0.90** | **0.47** | **0.941** |

Act above the threshold, hand the rest to a person. `Decision.above(t)` returns
`None` below it.

When the first reading is unsure, `rev` reads the options again in reverse and
lets the average pick the answer, but `confidence` stays at the lower of the two
readings. So the gate above is exactly what a single reading gives, and the
answers below it, the ones a person sees, are more often right.

## Limits worth knowing before you build on it

- **Arithmetic, dates and counting do not work.** Nine of nine `temporal_numeric`
  items were wrong, and the commercial Jev scores 0.267 on the same family. This
  is a property of the model class, not of this harness.
- **A plausible question can come back inverted.** "Does this email require the
  recipient to do something?" scored AUC 0.176 — reliably backwards — on real
  mail, while "what kind of email is this?" scored 0.878 on the same messages.
  Measure the AUC of every question you ship; you cannot tell from the output.
- **Ask for a category, not a judgement.** Asking "is this action risky?"
  produced a constant "confirm". Asking "what kind of action is this?" and
  deciding risk in code reached 0.930 on the same items.
- **Never let it classify secrets.** An API key in the option list came back as
  "an email address", "a phone number" and "a social profile", each at p = 1.00.
  Exclude those in code: `rev.guards.looks_secret(text)`.
- **Open-ended cleanups read as harmless.** "Tidy up my calendar" was classified
  as a local note edit at p = 0.99, in English and Chinese. Route them to a
  person before asking the model: `rev.guards.vague_action(request)`.
- Long inputs are slow: a 3.3k-token question costs about 3.4 s the first time
  its document is read. Ask everything you need about a document in one
  request, so it is read once.
- Two of 111 hard items sit close enough that a last-bit difference decides
  them. `rev` answers each question the same way wherever it sits in a batch
  (`tests/test_determinism.py`); the reference implementation's CLI does not,
  which is why its published single-reading hard figure is 0.541 against the
  0.523 here on identical items and identical weights.

## Install

```bash
uv venv && uv pip install -e .
rev serve                                            # the local endpoint
rev score --input decisions.jsonl --output answers.jsonl
```

Measuring it:

```bash
uv pip install -e ".[bench]"
python bench/usecases.py                  # clipboard, write-action gate, this Mac's calendar
python bench/usecases.py mail --dump      # then label bench/private/mail_labels.json, then:
python bench/usecases.py mail
python bench/jevbench.py --tier hard      # public JevBench items
python bench/speed.py && python bench/speed.py --jev
python bench/orders.py                    # the reading-policy table, from saved logits
```

Calendar and mail are read through pyapple; personal data stays in
`bench/private/`, which git ignores. `docs/BOUNDARY.md` is the lab notebook
this grew out of.

The first run downloads the 4.3 GB checkpoint from Hugging Face; after that a
load takes about 4 s.

Apple Silicon and MLX. The JSONL shape is SemIf's, so files written for either
tool run through both.

## Calibration

```python
from rev.calibrate import fit_offsets, fit_temperature, coverage_curve
```

Fit on your own labelled rows — fifty is enough — and never on the rows you then
report. An offset fitted on easy questions transferred to hard ones, so it is a
property of the model and prompt rather than of the items.

A content-free prior (the same prompt with the state replaced by "N/A") was
measured and **rejected**: it made binary questions worse. What the model
prefers with no evidence is not what it prefers with evidence.
