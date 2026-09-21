# rev

A local, open stand-in for TypeSafe's Jev: typed decisions (choice / noul /
score) from a frozen Qwen3.5-2B in one forward pass, on Apple Silicon via MLX.
The user wants it to be as easy to use as Jev on *their own* tasks; judge any
change on those first (`bench/usecases.py`), benchmarks second.

Read before changing anything: `DECISIONS.md` (why the defaults are what they
are, every entry measured). `docs/BOUNDARY.md` is the earlier lab notebook in
Chinese; later sections overturn earlier ones.

## Running

- Venv: `.venv/` (`uv pip install -e ".[bench]"`). No pytest; tests are
  scripts: `for t in tests/test_*.py; do .venv/bin/python $t; done`.
  `test_equivalence.py` needs the SemIf checkout at `~/Documents/misc/semif`.
- `rev serve` listens on 127.0.0.1:8421 and speaks Jev's `/v1/systemone`
  protocol. Check the port is free first and stop the server when done;
  never take :8000, :5173 or :8443 (Parley).
- Jev itself: key in `~/typesafe.txt`; `rev.Client("https://api.typesafe.ai",
  key=...)` is the same client. Only send synthetic or public items unless
  the user says otherwise; their mail has never been sent to Jev.

## Evaluating

- `bench/usecases.py`: the user's tasks. Clipboard paste and the write-action
  gate are synthetic; calendar and mail are read live from this Mac through
  pyapple (`~/Documents/projects/pyapple-mcp`). Mail must be labelled by the
  rule in `bench/cases.py` *before* the model runs; personal data stays in
  `bench/private/` (git-ignored) and should be deleted after use.
- `bench/jevbench.py --tier hard|standard|easy`: public JevBench items.
- `bench/speed.py` (and `--jev`): same client, same items, only the URL differs.
- `bench/orders.py`: recomputes the reading-policy table from saved logits.
- Expected on an M2 Pro, defaults: paste 1.000, gate 0.965, calendar ~0.99,
  mail AUC ~0.94 (`orders="one"`), JevBench easy 1.000 / standard 0.764 /
  hard ~0.55-0.57.

## Things that bit before

- `enable_thinking=False` or the answer slot is wrong and accuracy is chance
  while every token check passes.
- 4-bit once looked more accurate than 8-bit; it was noise breaking a position
  bias. bf16 and 8-bit give identical answers. Check against bf16 before
  believing a quantization result.
- Computing in pieces (last-position head, prefix cache) moves bf16 last bits;
  near-tie answers can flip. Compare choices, not exact probabilities, except
  on the full-forward path that `test_equivalence.py` pins to SemIf.
- Long documents are slow on this laptop (~1k tokens/s prefill) and cannot
  match Jev's ~330 ms; ask all questions about a document in one request.
