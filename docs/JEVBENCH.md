# JevBench, and why rev is not on it (2026-09-22)

`fstandhartinger/jevbench` is the public leaderboard for Jev-class systems.
Looked at in detail on 2026-09-22 (README, RESULTS-v1.2.md, adapters, issues
#1-#30, PRs #17 and #21). What matters for us:

## How it works

- A submission is a GitHub issue `[bench request]: Add <name> (<base>, <wire
  format>, <hosting>)`: readout mechanism, pinned commit + HF weight revision +
  licences, an exact serve command with GPU size, the adapter to use, the
  submitter's own numbers on the 231 public items, a cost basis, and a
  Disclosure of anything tuned on public items.
- The maintainer runs all 534 decisions himself on rented GPUs. 146 judge items
  and 109 hard items are not public. Since v1.2.7 a submitter-operated endpoint
  gets a partial, unranked row; a rank needs public serving code he can run.
- Score = geometric mean of Intelligence, Calibration, Speed, Cost. Self-hosted
  latency is charged at x2 + 0.15 s; self-hosted cost at a hosted tariff for
  the same weights (Qwen3.8-27B ~$0.21/M tokens -> Cost ~35).
- Our wire format is already what `--adapter typesafe` expects (`/v1/systemone`,
  `answers.decision`, `probabilities` keyed by option / level, `usage` passed
  through). No adapter would be needed. Entrants reading option logits are
  allowed and labelled `native`.

## Where we would land

- Qwen3.8-27B zero-shot rows (SimpleJev, NInfer, reflex-27b) score Intelligence
  84-86, above Jev's 85.7 or level with it, and finish #20-29 on Cost 32-40.
  rev on the same weights would land there too: about 66-67 composite.
- The top rows are 4B models with Intelligence ~76: Hopper 75.4, Jev 74.4,
  Jobe 73.4. The composite pays for cheap more than for right.
- Our own 2B / 4B are below Jev on hard by accuracy, not cost (0.55 / 0.60 vs
  0.73 on public hard).

## Decision

Not submitted. What we use rev for is the 27B being more accurate than Jev on
hard and faster on short questions, and the leaderboard's composite cannot
show that. If it is ever wanted: make the repo public with a LICENSE, write a
Slurm-free recipe (`sglang serve Qwen/Qwen3.8-27B` + `rev serve --upstream`;
the DSpark patch is not needed on a plain sglang), run their harness on the
231 public items against `oracle:8421` for numbers in their metric, disclose
that the second-reading rule was chosen on 2B with public items, file the
issue. Their harness is also the way to get our Intelligence / Calibration in
their units without submitting.
