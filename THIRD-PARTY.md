# Third-party material

- `bench/fixtures/easy.jsonl`, `standard.jsonl`, `hard.jsonl`: the public items
  of JevBench v1 (https://github.com/fstandhartinger/jevbench), MIT License,
  downloaded by `bench/jevbench.py` on first run.
- `bench/fixtures/semif_items.jsonl`: items in SemIf's JSONL shape
  (https://github.com/TheoLeeCJ/SemIf), MIT License, used by
  `tests/test_equivalence.py` to pin this implementation to SemIf's
  probabilities.
- The Evidence / Criterion prompt framing follows SemIf and reflex
  (https://github.com/kshetrajna12/reflex); the wide label space follows
  decider (https://huggingface.co/Mapika/decider-2b). README "Why it exists"
  says what was taken from each and what was measured here.
- The `/v1/systemone` request and response shapes are TypeSafe's Jev API,
  reimplemented from its public documentation; no TypeSafe code is included.
