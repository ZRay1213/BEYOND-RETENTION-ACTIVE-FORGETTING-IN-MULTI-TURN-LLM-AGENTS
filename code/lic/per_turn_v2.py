"""Per-turn accuracy analysis v2 — loads gold from dataset file.

For each conversation in the sharded logs, scores every assistant turn
against the gold answer. Outputs per-turn accuracy curves and case studies.

Usage:
    python per_turn_v2.py \
        --log_dir /root/autodl-tmp/DCC/data/lost_in_conversation/logs_sharded_stage1 \
        --dataset_dir /root/autodl-tmp/DCC/data/lost_in_conversation/data \
        --grader_url http://127.0.0.1:8002/v1 \
        --grader_model qwen2.5-7b \
        --n_conv 120 \
        --output_dir /root/autodl-tmp/DCC/data/lost_in_conversation/logs_per_turn_v2
"""

import argparse, json, os, re
from pathlib import Path
from collections import defaultdict
from openai import OpenAI


def build_gold_index(dataset_dir):
    """Build {task_id: gold_answer} from all dataset files."""
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
        except Exception as e:
            pass
    print(f"Gold index: {len(idx)} entries from {dataset_dir}")
    return idx


def load_convs(log_dir, n_conv):
    recs = []
    for f in sorted(Path(log_dir).glob("**/*.jsonl")):
        with open(f) as fp:
            for line in fp:
                try:
                    r = json.loads(line.strip())
                    r["_src"] = str(f)
                    recs.append(r)
                except:
                    pass
    import random; random.seed(42); random.shuffle(recs)
    return recs[:n_conv]


def extract_asst_turns_with_context(trace):
    """Return list of (asst_content, messages_so_far, turn_number, response_type)."""
    turns = []
    messages = []
    asst_idx = 0
    pending_veri = None
    for entry in trace:
        role = entry.get("role")
        content = entry.get("content", "")
        if role == "log":
            try:
                c = json.loads(content) if isinstance(content, str) else content
                if isinstance(c, dict) and c.get("type") == "system-verification":
                    pending_veri = c.get("response", {}).get("response_type", "unknown")
            except:
                pass
            continue
        if role in ("system", "user"):
            messages.append({"role": role, "content": content})
        elif role == "assistant":
            turns.append({
                "content": content,
                "messages_before": list(messages),
                "asst_idx": asst_idx,
                "response_type": pending_veri or "unknown",
            })
            messages.append({"role": "assistant", "content": content})
            asst_idx += 1
            pending_veri = None
    return turns


def grade_math(pred, gold):
    pred = pred.strip().lower()
    gold = gold.strip().lower()
    # extract boxed
    m = re.search(r'\\boxed\{([^}]+)\}', pred)
    if m:
        pred = m.group(1).strip().lower()
    # extract #### pattern
    if "####" in pred:
        pred = pred.split("####")[-1].strip()
    return 1.0 if pred == gold or gold in pred else 0.0


def grade_llm(grader, grader_model, pred, gold, task):
    prompt = (
        f"Task: {task}\nGold: {gold}\nPrediction: {pred}\n"
        f"Is the prediction correct? Reply only JSON: {{\"correct\": true}} or {{\"correct\": false}}"
    )
    try:
        resp = grader.chat.completions.create(
            model=grader_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32, temperature=0.0
        )
        result = resp.choices[0].message.content or ""
        try:
            return 1.0 if json.loads(result.strip()).get("correct") else 0.0
        except:
            return 1.0 if "true" in result.lower() else 0.0
    except:
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
    ap.add_argument("--n_conv", type=int, default=120)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    grader = OpenAI(api_key="sk-local", base_url=args.grader_url)
    out_path = Path(args.output_dir) / "per_turn_results.jsonl"

    gold_index = build_gold_index(args.dataset_dir)
    recs = load_convs(args.log_dir, args.n_conv)
    print(f"Loaded {len(recs)} conversations")

    # turn_k_scores[k] = list of scores for convs reaching turn k
    turn_k_scores = defaultdict(list)
    # first_correct_turn[k] = n conversations where first correct response is at turn k
    first_correct_counts = defaultdict(int)
    case_studies = []
    skipped = 0

    with open(out_path, "w") as fp:
        for ci, rec in enumerate(recs):
            task_id = rec.get("task_id", "")
            gi = gold_index.get(task_id)
            if not gi:
                skipped += 1
                continue
            gold = gi["gold"]
            task = gi["task"]

            turns = extract_asst_turns_with_context(rec.get("trace", []))
            if not turns:
                skipped += 1
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
                "ci": ci, "task": task, "task_id": task_id,
                "n_turns": len(turns),
                "turn_scores": turn_scores,
                "first_correct_turn": first_correct,
                "final_correct": final_correct,
                "gold_snippet": gold[:60],
            }
            fp.write(json.dumps(entry) + "\n")
            fp.flush()

            # Collect case study: early wrong commitment, final wrong
            if (not final_correct and len(turns) >= 3 and turn_scores
                    and turn_scores[0] < 0.5 and len(case_studies) < 5):
                t0 = turns[0]
                last_user = next((m["content"] for m in reversed(t0["messages_before"])
                                  if m["role"] == "user"), "")
                case_studies.append({
                    "task": task, "task_id": task_id,
                    "first_user_turn": last_user[:300],
                    "first_asst_response": t0["content"][:400],
                    "gold": gold[:100],
                    "n_turns": len(turns),
                    "all_turn_scores": turn_scores,
                })

            if (ci + 1) % 20 == 0:
                print(f"[{ci+1}/{len(recs)}] skipped={skipped}")
                for k in sorted(turn_k_scores)[:6]:
                    v = turn_k_scores[k]
                    print(f"  Turn {k}: acc={sum(v)/len(v):.3f} (n={len(v)})")

    n_total = sum(len(v) for v in turn_k_scores.values()) // max(len(turn_k_scores), 1)
    summary = {
        "per_turn_accuracy": {
            str(k): {"acc": sum(v)/len(v), "n": len(v)}
            for k, v in sorted(turn_k_scores.items())
        },
        "first_correct_turn_distribution": {
            str(k): v for k, v in sorted(first_correct_counts.items())
        },
        "case_studies": case_studies,
        "n_skipped": skipped,
    }
    with open(Path(args.output_dir) / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    print("\n=== Per-turn accuracy ===")
    for k, d in summary["per_turn_accuracy"].items():
        bar = "█" * int(float(d["acc"]) * 20)
        print(f"  Turn {k}: {float(d['acc']):.3f}  {bar}  (n={d['n']})")
    print(f"\nCase studies collected: {len(case_studies)}")
    print(f"Results → {out_path}")


if __name__ == "__main__":
    main()
