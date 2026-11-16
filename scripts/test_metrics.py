# Runs the pre-written cases. Exit 1 on any failure. Login node (CPU).
import sys, json
sys.path.insert(0, __import__("os").path.dirname(__file__))
from metric_cases import CASES, POSTPROCESS_CASES
from eval_metrics import fabrication
from common import postprocess
fails = 0
for cid, cat, h, a, ef, en, el in CASES:
    r = fabrication(h, a)
    ok = (r["fabricated"], r["numbers_only"], r["listicle"]) == (ef, en, el)
    fails += not ok
    print(("PASS" if ok else "FAIL"), cid, cat, h, "|", {k: r[k] for k in ("fabricated", "numbers_only", "listicle", "unsup_ent", "unsup_num")})
for raw, exp in POSTPROCESS_CASES:
    got = postprocess(raw); ok = got == exp; fails += not ok
    print(("PASS" if ok else "FAIL"), "postprocess", repr(raw), "->", repr(got))
print(f"TOTAL {len(CASES)+len(POSTPROCESS_CASES)} FAILS {fails}")
sys.exit(1 if fails else 0)
