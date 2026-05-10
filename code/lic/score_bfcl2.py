"""Score BFCL multi_turn results.

Usage:
    python score_bfcl2.py --result logs_bfcl/sharded_n30.jsonl --category base
    python score_bfcl2.py --result logs_bfcl/sharded_missparam_n30.jsonl --category miss_param
    python score_bfcl2.py --result logs_bfcl/sharded_longctx_n30.jsonl --category long_context
"""
import json, sys, re, argparse
from pathlib import Path
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
from bfcl_eval.model_handler.utils import convert_to_function_call

PKG = Path("/root/miniconda3/lib/python3.12/site-packages/bfcl_eval")

CATEGORY_MAP = {
    "base":         ("BFCL_v4_multi_turn_base.json",         "multi_turn_base"),
    "miss_param":   ("BFCL_v4_multi_turn_miss_param.json",   "multi_turn_miss_param"),
    "long_context": ("BFCL_v4_multi_turn_long_context.json", "multi_turn_long_context"),
}


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
    calls = extract_tool_calls(text)
    if not calls:
        return []
    return convert_to_function_call(
        [{c["name"]: c.get("arguments", {})} for c in calls]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--category", default="base", choices=list(CATEGORY_MAP))
    ap.add_argument("--out", default=None, help="optional per_id JSON output path")
    args = ap.parse_args()

    fname, checker_cat = CATEGORY_MAP[args.category]
    DATA = PKG / "data" / fname
    GOLD = PKG / "data" / "possible_answer" / fname

    instances = {json.loads(l)["id"]: json.loads(l) for l in open(DATA)}
    gold = {json.loads(l)["id"]: json.loads(l) for l in open(GOLD)}

    n_total = 0
    n_correct = 0
    per_id = {}
    for line in open(args.result):
        rec = json.loads(line)
        if rec.get("error"):
            n_total += 1
            per_id[rec["id"]] = False
            continue
        tid = rec["id"]
        if tid not in instances:
            continue
        entry = instances[tid]
        gt = gold[tid]["ground_truth"]

        decoded = []
        for turn_responses in rec["result"]:
            turn_decoded = []
            for step_text in turn_responses:
                turn_decoded.append(decode(step_text))
            decoded.append(turn_decoded)

        while len(decoded) < len(gt):
            decoded.append([[]])

        try:
            r = multi_turn_checker(decoded, gt, entry, checker_cat, "qwen2_5_14b")
            ok = r.get("valid", False)
        except Exception:
            ok = False

        n_total += 1
        per_id[tid] = ok
        if ok:
            n_correct += 1

    acc = n_correct / n_total if n_total else 0
    print(f"{args.result}  [{args.category}]")
    print(f"  N={n_total}, correct={n_correct}, acc={acc:.3f}")
    if args.out:
        json.dump({k: bool(v) for k, v in per_id.items()}, open(args.out, "w"))
    return acc, per_id


if __name__ == "__main__":
    main()
