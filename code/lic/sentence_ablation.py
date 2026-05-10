"""Sentence-level ablation for commitment anchor identification.

For each failed sharded conversation, we split each assistant turn into
sentences and drop them one at a time. A sentence is a "commitment anchor"
if its removal flips the final answer from wrong to correct.

This is the implementation of the Thought Anchors methodology
(arXiv 2506.19143) applied to cross-turn anchoring in multi-turn LLMs.

Output per conversation:
  - anchor_sentence: the specific text that, when removed, fixes the answer
  - anchor_turn: which assistant turn it came from
  - anchor_position: sentence index within that turn
  - flip_rate: fraction of sentences that are anchors

Usage:
    python sentence_ablation.py \
        --log_dir /root/autodl-tmp/DCC/data/lost_in_conversation/logs_sharded_stage1 \
        --n_conv 40 \
        --vllm_url http://127.0.0.1:8102/v1 \
        --model qwen2.5-14b \
        --grader_url http://127.0.0.1:8002/v1 \
        --grader_model qwen2.5-7b \
        --output_dir /root/autodl-tmp/DCC/data/lost_in_conversation/logs_sentence_ablation
"""

import argparse, json, os, re, copy
from pathlib import Path
from openai import OpenAI


def split_sentences(text):
    """Split text into sentences, preserving structure."""
    text = text.strip()
    # Split on sentence-ending punctuation followed by whitespace or end
    parts = re.split(r'(?<=[.!?])\s+', text)
    # Also split on newlines
    result = []
    for p in parts:
        for line in p.split('\n'):
            line = line.strip()
            if len(line) > 10:  # skip very short fragments
                result.append(line)
    return result if result else [text]


def load_failed_convs(log_dir, n_conv, tasks=("math", "actions")):
    failed = []
    for task in tasks:
        task_dir = Path(log_dir) / task
        for f in sorted(task_dir.glob("**/*.jsonl")):
            with open(f) as fp:
                for line in fp:
                    try:
                        r = json.loads(line.strip())
                        if float(r.get("score", r.get("is_correct", 1))) < 0.5:
                            r["_task"] = task
                            r["_src"] = str(f)
                            failed.append(r)
                    except:
                        pass
    import random; random.seed(42); random.shuffle(failed)
    return failed[:n_conv]


def extract_messages(rec):
    for key in ("messages", "conversation", "history", "turns"):
        if key in rec and isinstance(rec[key], list):
            return rec[key]
    # Build from trace
    msgs = []
    for t in rec.get("trace", []):
        role = t.get("role")
        if role in ("system", "user", "assistant"):
            msgs.append({"role": role, "content": t.get("content", "")})
    return msgs if msgs else None


def extract_gold(rec):
    task = rec.get("_task", "math")
    for key in ("gold", "answer", "reference"):
        if key in rec:
            v = rec[key]
            if task == "math" and isinstance(v, str) and "####" in v:
                return v.split("####")[-1].strip()
            return str(v)
    return None


def call_llm(client, model, messages, max_tokens=400):
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=0.0
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            err = str(e).lower()
            if "context length" in err or "maximum context" in err:
                return "[SKIP_TOOLONG]"
            if attempt == 2:
                return f"[ERROR: {e}]"
            import time; time.sleep(2)
    return ""


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
        try:
            return 1.0 if json.loads(result.strip()).get("correct") else 0.0
        except:
            return 1.0 if "true" in result.lower() else 0.0
    except:
        return 0.0


def drop_sentence_from_messages(messages, asst_turn_idx, sent_idx):
    """Return new messages with sentence sent_idx removed from asst turn asst_turn_idx."""
    msgs = copy.deepcopy(messages)
    asst_count = 0
    for m in msgs:
        if m["role"] == "assistant":
            if asst_count == asst_turn_idx:
                sents = split_sentences(m["content"])
                if sent_idx < len(sents):
                    remaining = [s for i, s in enumerate(sents) if i != sent_idx]
                    m["content"] = " ".join(remaining) if remaining else "[removed]"
                break
            asst_count += 1
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--n_conv", type=int, default=40)
    ap.add_argument("--vllm_url", default="http://127.0.0.1:8102/v1")
    ap.add_argument("--model", default="qwen2.5-14b")
    ap.add_argument("--grader_url", default="http://127.0.0.1:8002/v1")
    ap.add_argument("--grader_model", default="qwen2.5-7b")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    client = OpenAI(api_key="sk-local", base_url=args.vllm_url)
    grader = OpenAI(api_key="sk-local", base_url=args.grader_url)
    out_path = Path(args.output_dir) / "sentence_ablation_results.jsonl"

    recs = load_failed_convs(args.log_dir, args.n_conv)
    print(f"Loaded {len(recs)} failed conversations")

    all_anchors = []
    anchor_sentences_corpus = []  # for case study extraction

    with open(out_path, "w") as fp:
        for ci, rec in enumerate(recs):
            msgs = extract_messages(rec)
            if not msgs:
                continue
            gold = extract_gold(rec)
            if not gold:
                continue
            task = rec.get("_task", "math")

            # Get all assistant turns and their sentences
            asst_turns = [(i, m) for i, m in enumerate(msgs) if m["role"] == "assistant"]
            if not asst_turns:
                continue

            # Baseline: original (should be wrong)
            orig_pred = call_llm(client, args.model, msgs)
            if "[SKIP" in orig_pred or "[ERROR" in orig_pred:
                continue
            orig_score = grade(grader, args.grader_model, orig_pred, gold, task)

            conv_anchors = []
            total_sentences = 0

            for turn_idx, (msg_idx, asst_msg) in enumerate(asst_turns[:-1]):
                # Skip last asst turn (that's the output being measured)
                sents = split_sentences(asst_msg["content"])
                total_sentences += len(sents)

                for si, sent in enumerate(sents):
                    if len(sent.strip()) < 15:
                        continue
                    # Drop this sentence
                    modified_msgs = drop_sentence_from_messages(msgs, turn_idx, si)
                    mod_pred = call_llm(client, args.model, modified_msgs)
                    if "[SKIP" in mod_pred or "[ERROR" in mod_pred:
                        continue
                    mod_score = grade(grader, args.grader_model, mod_pred, gold, task)

                    is_anchor = (orig_score < 0.5 and mod_score >= 0.5)
                    conv_anchors.append({
                        "turn_idx": turn_idx,
                        "sent_idx": si,
                        "sentence": sent[:200],
                        "orig_score": orig_score,
                        "mod_score": mod_score,
                        "is_anchor": is_anchor,
                    })

                    if is_anchor:
                        anchor_sentences_corpus.append({
                            "conv_idx": ci,
                            "task": task,
                            "anchor_sentence": sent[:300],
                            "turn_idx": turn_idx,
                            "gold": gold[:80],
                        })

            flip_rate = sum(1 for a in conv_anchors if a["is_anchor"]) / max(len(conv_anchors), 1)
            entry = {
                "ci": ci, "task": task, "task_id": rec.get("task_id", ""),
                "n_asst_turns": len(asst_turns),
                "total_sentences": total_sentences,
                "n_anchors": sum(1 for a in conv_anchors if a["is_anchor"]),
                "flip_rate": flip_rate,
                "orig_score": orig_score,
                "anchors": [a for a in conv_anchors if a["is_anchor"]],
                "all_results": conv_anchors,
            }
            fp.write(json.dumps(entry) + "\n")
            fp.flush()
            all_anchors.append(flip_rate)

            if (ci + 1) % 5 == 0:
                mean_fr = sum(all_anchors) / len(all_anchors)
                n_with_anchor = sum(1 for x in all_anchors if x > 0)
                print(f"  [{ci+1}/{len(recs)}] mean_flip_rate={mean_fr:.3f} "
                      f"convs_with_anchor={n_with_anchor}/{len(all_anchors)}")

    n = len(all_anchors)
    mean_flip = sum(all_anchors) / n if n else 0
    summary = {
        "n_conv": n,
        "mean_sentence_flip_rate": mean_flip,
        "n_conv_with_any_anchor": sum(1 for x in all_anchors if x > 0),
        "anchor_sentences_examples": anchor_sentences_corpus[:10],
    }
    with open(Path(args.output_dir) / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"\nFinal: n={n}, mean_sentence_flip_rate={mean_flip:.3f}")
    print(f"Convs with at least 1 anchor sentence: {summary['n_conv_with_any_anchor']}")
    print("\nSample anchor sentences:")
    for ex in anchor_sentences_corpus[:3]:
        print(f"  [{ex['task']}] \"{ex['anchor_sentence'][:120]}\"")
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
