# Decisions

Why this is the way it is. Each entry is something that was measured, not
argued; the numbers are on JevBench public items with Qwen3.5-2B unless stated.

## 8-bit is the default, not 4-bit — 2026-09-21

4-bit costs **9.1 points** on the hard tier (0.450 against 0.541, both measured
before the numerics were pinned, one reading) and saves **no latency** (285 ms
against 275 ms on a short question). It saves 0.87 GB.

This one choice outweighs every prompt technique tried here, and it also
explains an anomaly: at 4-bit, averaging two option orders *hurt*
(0.450 → 0.387), which contradicted reflex's published result. At 8-bit the
effect goes the way they reported. **The anomaly was quantization noise.**

Anything earlier that concluded "this model cannot do X" at 4-bit deserves
re-testing.

## Labels are verified, not hard-coded — 2026-09-21

SemIf fixes `LETTERS = "ABCDEFGHIJKLMNOP"` and so refuses questions with more
than 16 options; reflex uses A–Z and stops at 26. Both are conservative: on
Qwen3.5's vocabulary **101** single characters are one token and decode back to
themselves. `labels.verified_slots` finds them at load time, so the ceiling
follows the tokenizer rather than a constant.

Verified at 30 options on a real task. decider's approach (one label token per
option beyond ten) would go further still and is the next step if 101 is not
enough.

## A second reading when unsure; it never raises confidence — 2026-09-21

*Supersedes "two option orders are off by default", measured the same day
before the numerics were pinned.*

This started as an anomaly: on clipboard paste (160 items), 4-bit scored 0.988
and 8-bit 0.906. bf16 scored 0.906 too, with all 160 answers identical to 8-bit,
so 8-bit is faithful and the unquantized model itself was wrong. All 15 errors
picked option A, B or C when the answer sat at position 4-17, and 14 of them
had confidence below 0.6. **Position bias under uncertainty.** 4-bit's rounding
happened to break it on this task; the same rounding costs nine points on hard.

Reading the options again in reverse and averaging the logits removes the bias.
Every item was run both ways once and the policies compared offline:

| task | one reading | always two | two when first < 0.9 |
|---|---:|---:|---:|
| clipboard paste (160) | 0.906 | 1.000 | 1.000 |
| calendar pick (216) | 0.972 | 0.991 | 0.986 |
| action kind for risk (57) | 0.930 | 0.912 | 0.912 |
| JevBench easy / standard | 1.000 / 0.764 | 1.000 / 0.764 | 1.000 / 0.764 |
| JevBench hard, score items kept to one reading | 0.523 | — | 0.550 |

Two costs had to be handled:

- **Averaged logits are overconfident.** At the 0.9 gate on standard they let
  7 errors through instead of 2. Averaging probabilities instead let 4 through
  but lost part of the accuracy. The shipped rule: the averaged logits choose,
  `confidence` is the lower of the two readings' probabilities for that choice.
  At 0.9 its coverage and errors equal a single reading's on every task above,
  and the accuracy is the averaged logits'.
- **Score levels must not be reversed.** Their order carries meaning; reversing
  took hard's six score items from 3 right to 1. `ordered=True`, set
  automatically for Jev score questions, keeps them to one reading.

Only unsure items pay for the second pass: 8% on easy, 29-34% on paste and
calendar, 79% on hard. The variant was chosen after seeing these items (four
were tried), which is why the property it is chosen for is a structural one:
the gate cannot get worse, because confidence can only go down.

## The option id never enters the prompt — 2026-09-21

An early version used the option text as its own key. Replacing that with an
opaque label moved a 20-option question from **0.240 to 0.720** on confusable
distractors. The content competes for the option budget twice when it appears
in both the key and the description.

## `enable_thinking=False` — 2026-09-21

Without it Qwen3.5 opens a reasoning block, the final position is no longer the
answer slot, and accuracy falls to chance (0.222 on an 8-option task) while
**every tokenization check still passes**. The bug is in the semantics of the
position, not the tokens.

This is why `tests/test_equivalence.py` exists: comparing against a known-good
implementation item by item is the only thing that catches it.

## Offsets are fitted on labels; content-free priors were rejected — 2026-09-21

The model answered "yes" to 32 of 38 binary questions whose truth was 17 yes and
21 no. Removing a fitted offset:

| offset from | accuracy | leaks? |
|---|---:|---|
| the same items | 0.632 | **yes — discarded** |
| the other half of the same tier | 0.610 | no |
| a different tier entirely | 0.579 | no |

The last is the one to trust: +7.9 points, and it means the offset belongs to
the model and prompt, not the items.

A content-free prior needs no labels and would have been better, so it was
tried: 0.500 → 0.474 and 0.368. **Rejected**, reproducing reflex's own finding.

## Frozen, not fine-tuned — 2026-09-21

decider-2b is the same base model fine-tuned on ~95 decision datasets with a
reinforcement stage. Against it on the same items, same harness:

| | standard | hard | wrong at p > 0.9 |
|---|---:|---:|---:|
| frozen, one reading | 0.764 | **0.523** | **9%** |
| fine-tuned | **0.861** | 0.477 | **27%** |

Fine-tuning buys short rule-application and loses long-document choice. The
third column decides it for a system that gates on confidence: the fine-tune is
confidently wrong three times as often, so a threshold lets three times as many
errors through. If a human reads every answer, prefer the fine-tune.

Its harness was not the reason: running decider's weights through `rev`
*improved* on its own published numbers (standard 0.847 → 0.861).

## A local server speaking Jev's protocol — 2026-09-21

The library alone made every tool load its own 2-4 GB copy of the model, and
kept code written for Jev from running locally. `rev serve` answers
`POST /v1/systemone` with TypeSafe's request and response shapes, verified by
pointing the same `Client` at both (`tests/test_serve.py` checks the server
gives exactly the in-process probabilities). Requests are serialised: MLX runs
one forward pass at a time anyway, and it keeps each answer independent of what
else arrived. It binds 127.0.0.1 and refuses a port that is already in use
rather than taking it.

## Guards live in code, not in the prompt — 2026-09-21

Two failures were confident, not uncertain, so no threshold catches them: a key
classified as an email address at p = 1.00, and "tidy up my calendar" as a local
edit at p = 0.99. `rev.guards` checks for both before the model is asked. They
are deliberately broad because a false alarm costs one confirmation.

## Read the answer position, and each state, once — 2026-09-21

Against Jev on the same client and items, short and medium questions were
already level, but long documents took 0.9 s median and 4.5 s at p95 against
Jev's flat 330 ms. Profiled on an M2 Pro:

- **The vocabulary projection ran over every position** to read the last one:
  3,300 x 248k logits per long prompt. Applying the head to the last position
  alone: -24% latency, peak 4.69 -> 3.16 GB, same answer on all 223 items.
- **Every reading re-read its state.** The evidence comes first in the prompt,
  so all readings of one state share a token prefix. It is prefilled once and
  the cache restored for each reading; the chat template is one more cached
  level below it. A second reading of a long document: 670 -> 150 ms.
- The rest is arithmetic. 8-bit dequantisation is ~20% (bf16: 262 -> 212 ms
  but 2.4 GB more, not taken). Qwen3.5's linear-attention layers run a
  sequential per-token recurrence in MLX, ~22%; a chunked kernel could win part
  of that back and has not been written. A 3,300-token first read cannot get
  near Jev's 330 ms on this hardware.

Computing in pieces moves bf16 rounding: up to 0.6 in a logit, 3 of 111 hard
answers flipped each way on near-ties (hard 0.550 -> 0.568, standard
unchanged, the user's three tasks identical). The split point is found from
the state alone, so an answer is still the same whatever else is asked and
whether its prefix was cached (drift 0). SemIf equivalence is still checked
exactly on the full-forward path.

## Ranking tasks read once — 2026-09-21

On email triage (146 real messages, 19 important, ranked by summed category
probability) the second reading did not help: AUC 0.941 one reading, 0.927
auto, 0.936 always two, all inside one bootstrap interval, at twice the time.
Position bias shows up when the answer is one item among look-alikes; a
fixed list of distinct categories has no look-alikes to confuse. The default
stays `auto`; the README says to pass `orders="one"` for ranking.
