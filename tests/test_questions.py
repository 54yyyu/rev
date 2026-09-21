"""Jev's request shapes map onto options correctly. No model needed."""
from rev.questions import answer_for, options_for
from rev.decision import Decision

fail = 0
def check(cond, msg):
    global fail
    if not cond:
        print("  FAIL:", msg); fail += 1

k, c, o = options_for({"type": "choice", "instructions": "Which team?",
                       "criteria": {"billing": "Payments", "sales": None}})
check((k, c, o) == ("choice", "Which team?", [("billing", "Payments"), ("sales", "sales")]), f"choice {o}")

k, c, o = options_for({"type": "noul", "instructions": "Urgent?"})
check(o == [("no", "No."), ("yes", "Yes.")], f"noul defaults {o}")
k, c, o = options_for({"type": "boolean", "instructions": "Refunded?",
                       "criteria": {"true": "Money returned", "false": "Declined"}})
check(k == "noul" and o == [("no", "Declined"), ("yes", "Money returned")], f"boolean alias {o}")

k, c, o = options_for({"type": "score", "instructions": {"q": "How bad?", "ref": [1, 2]},
                       "criteria": ["Calm", "Angry", {"level": "Furious"}]})
check(c == '{"q": "How bad?", "ref": [1, 2]}', f"structured instructions {c}")
check(o[2] == ("2", '{"level": "Furious"}'), f"structured level {o}")

for bad in ({"type": "choice", "instructions": "x", "criteria": {"a": "only one"}},
            {"type": "score", "instructions": "x", "criteria": ["one"]},
            {"type": "rank", "instructions": "x"},
            {"type": "noul"}):
    try:
        options_for(bad); check(False, f"accepted {bad}")
    except ValueError:
        pass

d = Decision("2", {"0": 0.1, "1": 0.2, "2": 0.7}, 0.7)
a = answer_for("score", d, [("0", "Calm"), ("1", "Angry"), ("2", "Furious")])
check(abs(a["score"] - 1.6) < 1e-9 and a["confidence"] == 0.7, f"score answer {a}")
check(a["legend"] == {"0": "Calm", "1": "Angry", "2": "Furious"}, f"score legend {a}")
check(answer_for("noul", Decision("yes", {"no": 0.2, "yes": 0.8}, 0.8)) == {"type": "noul", "noul": 0.8}, "noul answer")

print("FAILURES:", fail)
raise SystemExit(1 if fail else 0)
