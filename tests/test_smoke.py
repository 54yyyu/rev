"""Enough to catch the failures that actually happened while building this."""
import json, sys
import numpy as np
from rev import Rev

MODEL = "Qwen/Qwen3.5-2B"
h = Rev(MODEL, bits=8)
fail = 0

print(f"capacity: {h.capacity} slots")
assert h.capacity >= 26, "expected at least the 26 upper-case slots"

d = h.decide("My card was charged twice for one order.",
             "Which team should handle this?",
             {"billing": "Payments, refunds and invoices",
              "technical": "Bugs and outages",
              "sales": "Pricing and new accounts"})
print(f"routing: {d.choice} p={d.confidence:.3f} ({d.input_tokens} tok, {d.seconds*1000:.0f} ms)")
if d.choice != "billing":
    print("  FAIL: expected billing"); fail += 1
if d.confidence < 0.5:
    print("  FAIL: near-chance confidence suggests the answer slot is misplaced"); fail += 1

# The bug that cost an afternoon: without enable_thinking=False this is chance.
opts = {f"o{i}": t for i, t in enumerate(
    ["A person's name", "An email address", "A phone number", "A city",
     "A company name", "A job title", "A web link", "A snippet of code"])}
d2 = h.decide('Filling in a form. The field is labelled "Email address".',
              "Which clipboard item belongs in this field?", opts)
print(f"8-way pick: {d2.choice} p={d2.confidence:.3f}")
if d2.choice != "o1":
    print("  FAIL: expected the email option"); fail += 1

# Wide label space: more options than a fixed A..P alphabet allows.
wide = {f"x{i}": f"Option number {i}" for i in range(30)}
wide["x7"] = "The capital city of France"
d3 = h.decide("Paris", "Which description matches the state?", wide)
print(f"30-way pick: {d3.choice} p={d3.confidence:.3f}")
if d3.choice != "x7":
    print("  FAIL: expected x7"); fail += 1

d4 = h.decide("The build failed with exit code 1.", "Did the build succeed?",
              {"no": "It did not succeed", "yes": "It succeeded"}, orders="two")
print(f"both orders: {d4.choice} p={d4.confidence:.3f} disagreement={d4.disagreement:.3f}")
if d4.choice != "no":
    print("  FAIL: expected no"); fail += 1

try:
    h.decide("x", "y", {"only": "one option"})
    print("  FAIL: single option should raise"); fail += 1
except ValueError:
    pass

print("FAILURES:", fail)
sys.exit(1 if fail else 0)
