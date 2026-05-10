"""Per-turn accuracy v3 — caps long traces, skips actions task.

Drop-in replacement for per_turn_v2.py with:
- Skip conversations with >12 assistant turns (avoid actions hell)
- Restrict to math + data2text by default
- Try/except around each grade call so single failures don't crash
"""

import argparse, json, os, re, traceback
from pathlib import Path
from collections import defaultdict
from openai import OpenAI


def build_gold_index(dataset_dir):
    idx = {}
    for f in Path(dataset_dir).glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
            items = data if isinstance(data, list) else []
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        items.extend(v)
            for item in items:
                tid = item.get("task_id")
                if not tid:
                    continue
                task = item.get("task", "")
                if task == "math":
                    ans = item.get("answer", "")
                    if "####" in ans:
                        gold = ans.split("####")[-1].strip()
                    else:
                        gold = ans.strip().split("\n")[-1].strip()
                elif task in ("actions", "database"):
                    gold = json.dumps(item.get("expected_results", item.get("ground_truth",
                           item.get("reference_answer", ""))))
                elif task == "data2text":
                    refs = item.get("references", item.get("reference", []))
                    gold = refs[0] if refs else str(item.get("target", ""))
                else:
                    gold = str(item.get("answer", item.get("reference", "")))
                if gold:
                    idx[tid] = {"gold": gold, "task": task}
        except Exception:
            pass
    print(f"Gold index: {len(idx)} entries", flush=True)
    return idx


def load_convs(log_dir, n_conv, tasks):
    recs = []
    for task in tasks:
        for f in sorted((Path(log_dir) / task).glob("**/*.jsonl")):
            try:
                with open(f) as fp:
                    for line in fp:
                        try:
                            r = json.loads(line.strip())
                            r["_src"] = str(f)
                            recs.append(r)
                        except:
                            pass
            except:
                pass
    import random; random.seed(42); random.shuffle(recs)
    return recs[:n_conv]


def extract_asst_turns(trace):
    turns = []
    asst_idx = 0
    for entry in trace:
        if entry.get("role") == "assistant":
            content = entry.get("content", "")
            if content.strip():
                turns.append({"asst_idx": asst_idx, "content": content})
                asst_idx += 1
    return turns


def grade_math(pred, gold):
    pred = pred.strip().lower()
    gold = gold.strip().lower()
    m = re.search(r"\\boxed\{([^}]+)\}", pred)
    if m:
        pred = m.group(1).strip().lower()
    if "####" in pred:
        pred = pred.split("####")[-1].strip()
    return 1.0 if pred == gold or gold in pred else 0.0


def grade_llm(grader, grader_model, pred, gold, task):
    prompt = (
        f"Task: {task}\nGold: {gold}\nPrediction: {pred}\n"
        f"Is the prediction correct? Reply only JSON: "
        f'{{"correct": true}} or {{"correct": false}}'
    )
    try:
        resp = grader.chat.completions.create(
            model=grader_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32, temperature=0.0, timeout=20
        )
        result = resp.choices[0].message.content or ""
        try:
            return 1.0 if json.loads(result.strip()).get("correct") else 0.0
        except:
            return 1.0 if "true" in result.lower() else 0.0
    except Exception:
        return 0.0


def grade(grader, grader_model, pred, gold, task):
    if task == "math":
        s = grade_math(pred, gold)
        if s > 0:
            return s
    return grade_llm(grader, grader_model, pred, gold, task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--grader_url", default="http://127.0.0.1:8002/v1")
    ap.add_argument("--grader_model", default="qwen2.5-7b")
    ap.add_argument("--n_conv", type=int, default=80)
    ap.add_argument("--max_turns", type=int, default=12,
                    help="Skip conversations with more turns than this")
    ap.add_argument("--tasks", nargs="+", default=["math", "data2text"])
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    grader = OpenAI(api_key="sk-local", base_url=args.grader_url)
    out_path = Path(args.output_dir) / "per_turn_results.jsonl"

    gold_index = build_gold_index(args.dataset_dir)
    recs = load_convs(args.log_dir, args.n_conv * 2, args.tasks)
    print(f"Loaded {len(recs)} candidate conversations", flush=True)

    turn_k_scores = defaultdict(list)
    first_correct_counts = defaultdict(int)
    case_studies = []
    skipped_no_gold = 0
    skipped_too_long = 0
    written = 0

    with open(out_path, "w") as fp:
        for ci, rec in enumerate(recs):
            if written >= args.n_conv:
                break
            try:
                tid = rec.get("task_id", "")
                gi = gold_index.get(tid)
                if not gi:
                    skipped_no_gold += 1
                    continue
                gold, task = gi["gold"], gi["task"]
                turns = extract_asst_turns(rec.get("trace", []))
                if not turns or len(turns) > args.max_turns:
                    skipped_too_long += 1
                    continue

                turn_scores = []
                for t in turns:
                    s = grade(grader, args.grader_model, t["content"], gold, task)
                    turn_scores.append(s)
                    turn_k_scores[t["asst_idx"]].append(s)

                first_correct = next((k for k, s in enumerate(turn_scores) if s >= 0.5), None)
                if first_correct is not None:
                    first_correct_counts[first_correct] += 1

                final_correct = float(rec.get("score", rec.get("is_correct", 0))) > 0.5

                entry = {
                    "ci": ci, "task": task, "task_id": tid,
                    "n_turns": len(turns),
                    "turn_scores": turn_scores,
                    "first_correct_turn": first_correct,
                    "final_correct": final_correct,
                    "gold_snippet": gold[:60],
                }
                fp.write(json.dumps(entry) + "\n")
                fp.flush()
                written += 1

                if (not final_correct and len(turns) >= 3 and turn_scores
                        and turn_scores[0] < 0.5 and len(case_studies) < 5):
                    t0 = turns[0]
                    case_studies.append({
                        "task": task, "task_id": tid,
                        "first_asst_response": t0["content"][:400],
                        "gold": gold[:100],
                        "n_turns": len(turns),
                        "all_turn_scores": turn_scores,
                    })

                if written % 10 == 0:
                    print(f"[{written}/{args.n_conv}] no_gold={skipped_no_gold} too_long={skipped_too_long}",
                          flush=True)
                    for k in sorted(turn_k_scores)[:6]:
                        v = turn_k_scores[k]
                        print(f"  Turn {k}: acc={sum(v)/len(v):.3f} (n={len(v)})", flush=True)
            except Exception as e:
                print(f"  Error on ci={ci}: {e}", flush=True)
                traceback.print_exc()
                continue

    summary = {
        "n_written": written,
        "n_skipped_no_gold": skipped_no_gold,
        "n_skipped_too_long": skipped_too_long,
        "per_turn_accuracy": {
            str(k): {"acc": sum(v)/len(v), "n": len(v)}
            for k, v in sorted(turn_k_scores.items())
        },
        "first_correct_turn_distribution": {
            str(k): v for k, v in sorted(first_correct_counts.items())
        },
        "case_studies": case_studies,
    }
    with open(Path(args.output_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Per-turn accuracy ===", flush=True)
    for k, d in summary["per_turn_accuracy"].items():
        bar = "#" * int(float(d["acc"]) * 20)
        print(f"  Turn {k}: {float(d['acc']):.3f}  {bar}  (n={d['n']})", flush=True)
    print(f"\nResults -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
