"""Score BFCL multi_turn results from our 4-protocol runs.

Usage:
    python score_bfcl.py logs_bfcl/sharded_n30.jsonl
"""
import json, sys, re
from pathlib import Path
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
from bfcl_eval.model_handler.utils import convert_to_function_call

PKG = Path("/root/miniconda3/lib/python3.12/site-packages/bfcl_eval")
DATA = PKG / "data" / "BFCL_v4_multi_turn_base.json"
GOLD = PKG / "data" / "possible_answer" / "BFCL_v4_multi_turn_base.json"


def extract_tool_calls(text):
    pattern = r"<tool_call>\n(.*?)\n</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    out = []
    for m in matches:
        try:
            out.append(json.loads(m))
        except Exception:
            pass
    return out


def decode(text):
    """Convert model text into list-of-callable-strings as multi_turn_checker expects."""
    calls = extract_tool_calls(text)
    if not calls:
        return []
    return convert_to_function_call(
        [{c["name"]: c.get("arguments", {})} for c in calls]
    )


def main():
    result_file = sys.argv[1]
    out_per_id = sys.argv[2] if len(sys.argv) > 2 else None
    instances = {json.loads(l)["id"]: json.loads(l) for l in open(DATA)}
    gold = {json.loads(l)["id"]: json.loads(l) for l in open(GOLD)}

    n_total = 0
    n_correct = 0
    per_id = {}
    for line in open(result_file):
        rec = json.loads(line)
        if rec.get("error"):
            n_total += 1
            per_id[rec["id"]] = False
            continue
        tid = rec["id"]
        entry = instances[tid]
        gt = gold[tid]["ground_truth"]

        # Decode model response per turn per step
        decoded = []
        for turn_responses in rec["result"]:
            turn_decoded = []
            for step_text in turn_responses:
                turn_decoded.append(decode(step_text))
            decoded.append(turn_decoded)

        # Pad to match GT length (in case of force_quit)
        while len(decoded) < len(gt):
            decoded.append([[]])

        try:
            r = multi_turn_checker(decoded, gt, entry, "multi_turn_base", "qwen2_5_14b")
            ok = r.get("valid", False)
        except Exception as e:
            ok = False

        n_total += 1
        per_id[tid] = ok
        if ok:
            n_correct += 1

    acc = n_correct / n_total if n_total else 0
    print(f"{result_file}")
    print(f"  N={n_total}, correct={n_correct}, acc={acc:.3f}")
    if out_per_id:
        json.dump({k: bool(v) for k, v in per_id.items()}, open(out_per_id, "w"))
    return acc, per_id


if __name__ == "__main__":
    main()
