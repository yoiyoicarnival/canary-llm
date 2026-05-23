import os
os.environ["OMP_NUM_THREADS"] = "1"
from canary import score

tests = [
    ("What is the capital of France?",                   "safe"),
    ("Who discovered penicillin?",                       "safe"),
    ("What was Einstein's 1931 quantum biology paper?",  "risky"),
    ("Describe the grammar of Elvish Sindarin.",         "risky"),
    ("What is the cuisine of Atlantis?",                 "risky"),
    ("Who won the 2031 Nobel Peace Prize?",              "risky"),
]
print("  Prompt                                               Risk   Label")
print("  " + "-"*65)
for q, tier in tests:
    r = score(q)
    ok = "OK" if (tier == "safe") == (r.risk < 0.35) else "!!"
    print(f"[{ok}] {q[:50]:50s}  {r.risk:.2f}  {r.label}")
