# Decisions

Why this is the way it is. Each entry is something that was measured, not
argued; the numbers are on JevBench public items with Qwen3.5-2B unless stated.

## 8-bit is the default, not 4-bit — 2026-09-21

4-bit costs **9.1 points** on the hard tier (0.450 against 0.541) and saves
**no latency** (285 ms against 275 ms on a short question). It saves 0.87 GB.

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

## Two option orders are off by default — 2026-09-21

reflex measured order-averaging as one of only three things that helped, and it
does help here — but only on binary questions.

| | binary (38) | multiple choice (73) |
|---|---:|---:|
| one order | 0.553 | **0.534** |
| two orders | **0.605** | 0.507 |

Net zero on the hard tier. Choosing per question type would score higher, but
that choice was made by looking at the test items, so it is not shipped.

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
| frozen | 0.764 | **0.550** | **9%** |
| fine-tuned | **0.861** | 0.477 | **27%** |

Fine-tuning buys short rule-application and loses long-document choice. The
third column decides it for a system that gates on confidence: the fine-tune is
confidently wrong three times as often, so a threshold lets three times as many
errors through. If a human reads every answer, prefer the fine-tune.

Its harness was not the reason: running decider's weights through `rev`
*improved* on its own published numbers (standard 0.847 → 0.861).
