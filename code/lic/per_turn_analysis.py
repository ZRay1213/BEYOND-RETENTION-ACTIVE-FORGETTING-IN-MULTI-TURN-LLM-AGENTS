"""Per-turn degradation analysis.

Loads all sharded Stage 1 logs. For each conversation, reconstructs what
the model knew (and answered) at each turn position, then re-grades each
assistant response against the gold answer using the grader.

Output: per_turn_results.json with:
  - turn_accuracy[k]: fraction of convs where turn-k response is correct
  - commitment_turn[]: which turn the model first commits to wrong answer
  - case_studies[]: concrete before/after examples for the paper

Usage:
    python per_turn_analysis.py \
        --log_dir /root/autodl-tmp/DCC/data/lost_in_conversation/logs_sharded_stage1 \
        --grader_url http://127.0.0.1:8002/v1 \
        --grader_model qwen2.5-7b \
        --n_conv 80 \
        --output_dir /root/autodl-tmp/DCC/data/lost_in_conversation/logs_per_turn
"""

import argparse, json, os, re, copy
from pathlib import Path
from collections import defaultdict
from openai import OpenAI


def load_convs(log_dir, n_conv, tasks=("math", "actions")):
    recs = []
    for task in tasks:
        task_dir = Path(log_dir) / task
        for f in sorted(task_dir.glob("**/*.jsonl")):
            with open(f) as fp:
                for line in fp:
                    try:
                        r = json.loads(line.strip())
                        r["_task"] = task
                        r["_src"] = str(f)
                        recs.append(r)
                    except:
                        pass
    import random; random.seed(42); random.shuffle(recs)
    return recs[:n_conv]


def extract_gold(rec):
    task = rec.get("_task", "math")
    # Try common gold fields
    for key in ("gold", "answer", "reference", "expected"):
        if key in rec:
            v = rec[key]
            if task == "math" and isinstance(v, str) and "####" in v:
                return v.split("####")[-1].strip()
            return str(v)
    # Dig into trace for gold
    for t in rec.get("trace", []):
        if t.get("role") == "log":
            try:
                c = json.loads(t["content"]) if isinstance(t["content"], str) else t["content"]
                if isinstance(c, dict) and "gold" in c:
                    return str(c["gold"])
            except:
                pass
    return None


def extract_assistant_turns(trace):
    """Return list of (turn_idx, content, prior_messages_up_to_this_turn)."""
    turns = []
    messages = []
    asst_count = 0
    for entry in trace:
        role = entry.get("role")
        if role == "log":
            continue
        content = entry.get("content", "")
        if role in ("system", "user"):
            messages.append({"role": role, "content": content})
        elif role == "assistant":
            # Record: (asst_turn_index, content, context_before_this_response)
            turns.append((asst_count, content, copy.deepcopy(messages)))
            messages.append({"role": "assistant", "content": content})
            asst_count += 1
    return turns


def grade(grader, grader_model, pred, gold, task):
    prompt = (
        f"Task: {task}\nGold: {gold}\nPrediction: {pred}\n"
        f"Is the prediction correct? Reply JSON: {{\"correct\": true}} or {{\"correct\": false}}"
    )
    try:
        resp = grader.chat.completions.create(
            model=grader_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32, temperature=0.0
        )
        result = resp.choices[0].message.content or ""
        return 1.0 if json.loads(result.strip()).get("correct") else 0.0
    except:
        return 1.0 if "true" in result.lower() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--grader_url", default="http://127.0.0.1:8002/v1")
    ap.add_argument("--grader_model", default="qwen2.5-7b")
    ap.add_argument("--n_conv", type=int, default=80)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    grader = OpenAI(api_key="sk-local", base_url=args.grader_url)
    out_path = Path(args.output_dir) / "per_turn_results.jsonl"

    recs = load_convs(args.log_dir, args.n_conv)
    print(f"Loaded {len(recs)} conversations")

    # turn_k_scores[k] = list of 0/1 scores for conversations with at least k+1 turns
    turn_k_scores = defaultdict(list)
    case_studies = []

    with open(out_path, "w") as fp:
        for ci, rec in enumerate(recs):
            gold = extract_gold(rec)
            if not gold:
                continue
            task = rec.get("_task", "math")
            turns = extract_assistant_turns(rec.get("trace", []))
            if not turns:
                continue

            turn_scores = []
            for k, (turn_idx, content, context) in enumerate(turns):
                score = grade(grader, args.grader_model, content, gold, task)
                turn_scores.append(score)
                turn_k_scores[k].append(score)

            # Find first wrong commitment (first turn where score=0)
            first_wrong = next((k for k, s in enumerate(turn_scores) if s < 0.5), None)
            final_correct = float(rec.get("score", rec.get("is_correct", 0))) > 0.5

            entry = {
                "ci": ci, "task": task, "task_id": rec.get("task_id", ""),
                "n_turns": len(turns),
                "turn_scores": turn_scores,
                "first_wrong_turn": first_wrong,
                "final_correct": final_correct,
                "gold_snippet": gold[:80],
            }
            fp.write(json.dumps(entry) + "\n")
            fp.flush()

            # Collect case study: failed conv with clear early wrong commitment
            if (not final_correct and first_wrong == 0 and len(turns) >= 3
                    and len(case_studies) < 5):
                # Find the first assistant turn content and last user turn before it
                _, content0, ctx0 = turns[0]
                last_user = next((m["content"] for m in reversed(ctx0) if m["role"] == "user"), "")
                case_studies.append({
                    "task": task,
                    "task_id": rec.get("task_id", ""),
                    "first_user": last_user[:300],
                    "first_asst_wrong": content0[:300],
                    "gold": gold[:100],
                    "n_turns": len(turns),
                })

            if (ci + 1) % 10 == 0:
                print(f"[{ci+1}/{len(recs)}]", {
                    k: f"{sum(v)/len(v):.3f}(n={len(v)})"
                    for k, v in sorted(turn_k_scores.items())
                })

    # Summary
    summary = {
        "per_turn_accuracy": {
            k: {"acc": sum(v)/len(v), "n": len(v)}
            for k, v in sorted(turn_k_scores.items())
        },
        "case_studies": case_studies,
    }
    with open(Path(args.output_dir) / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print("\nPer-turn accuracy:")
    for k, d in summary["per_turn_accuracy"].items():
        print(f"  Turn {k}: {d['acc']:.3f}  (n={d['n']})")
    print(f"\nCase studies: {len(case_studies)}")
    print(f"Results → {out_path}")


if __name__ == "__main__":
    main()
