# hinge

**Typed decisions from a frozen open model, in one forward pass, on Apple Silicon.**

You give it a state, a criterion and a list of options. It returns a probability
for each option. Nothing is generated, so it cannot answer with something that
was not on your list, and there is no JSON to repair.

```python
from hinge import Hinge

h = Hinge("Qwen/Qwen3.5-2B")          # 8-bit by default, 1.9 GB resident
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

## Why it exists

Three good implementations of this idea already exist — [SemIf], [reflex] and
[decider] — and this one started as a measurement harness to compare them. It
became its own thing when the measurements said the harness mattered more than
the model: a frozen Qwen3.5-2B moved from 0.450 to 0.550 on JevBench's hard
public items without changing a single weight.

[SemIf]: https://github.com/TheoLeeCJ/SemIf
[reflex]: https://github.com/kshetrajna12/reflex
[decider]: https://huggingface.co/Mapika/decider-2b

What each of them contributed, and what was measured here:

| from | what | measured |
|---|---|---|
| SemIf | Evidence / Criterion prompt, letter-slot readout, tokenizer round-trip checks | the reference; `hinge` matches it exactly, 40/40 items, probability delta 0 |
| decider | a wide label space instead of a fixed alphabet | 16 slots → **101** on Qwen3.5 |
| reflex | averaging two option orders | helps binary questions, hurts multiple choice; off by default |
| here | **8-bit instead of 4-bit** | **+9.1 points on hard, no latency cost** |
| here | a fitted per-option offset | +7.9 points on binary questions, transfers across task tiers |

## Measured

Qwen3.5-2B, 8-bit, on this machine (M-series, 16 GB), JevBench public items:

| | easy (48) | standard (72) | hard (111) |
|---|---:|---:|---:|
| hinge, frozen Qwen3.5-2B | 1.000 | 0.764 | **0.550** |
| decider-2b (fine-tuned, same base) | 1.000 | **0.861** | 0.477 |
| reflex, published, 2B bf16 | 1.000 | — | 0.468 |
| Jev 1.13.0 (commercial) | 1.000 | 0.986 | 0.730 |

Cost of running it:

| | weights | peak | short question | 2.8k-token question |
|---|---:|---:|---:|---:|
| 2B, 8-bit *(default)* | 1.86 GB | 4.36 GB | 275 ms | 3.7 s |
| 2B, 4-bit | 0.99 GB | 3.49 GB | 285 ms | 3.8 s |
| 4B, 4-bit | 2.20 GB | 5.00 GB | 611 ms | 8.6 s |

**Use 8-bit.** It costs 0.87 GB and no time, and buys nine points.

## Gating

The number you act on is not accuracy, it is what a confidence gate buys. On the
standard tier:

| threshold | coverage | accuracy on what it answered |
|---:|---:|---:|
| none | 1.00 | 0.764 |
| 0.80 | 0.68 | 0.857 |
| **0.90** | **0.47** | **0.941** |

Act above the threshold, hand the rest to a person. `Decision.above(t)` returns
`None` below it.

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
  Exclude those in code.
- Long inputs are slow and memory-hungry: a 2.8k-token question costs 3.7 s and
  most of the peak above.

## Install

```bash
uv venv && uv pip install -e .
python -m hinge.cli score --input decisions.jsonl --output answers.jsonl
python bench/jevbench.py --tier standard
```

Apple Silicon and MLX. The JSONL shape is SemIf's, so files written for either
tool run through both.

## Calibration

```python
from hinge.calibrate import fit_offsets, fit_temperature, coverage_curve
```

Fit on your own labelled rows — fifty is enough — and never on the rows you then
report. An offset fitted on easy questions transferred to hard ones, so it is a
property of the model and prompt rather than of the items.

A content-free prior (the same prompt with the state replaced by "N/A") was
measured and **rejected**: it made binary questions worse. What the model
prefers with no evidence is not what it prefers with evidence.
