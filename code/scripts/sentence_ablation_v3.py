"""Sentence-level ablation v3 — caps long traces, math+data2text only.

Drop-in replacement for sentence_ablation_v2.py with:
- Skip conversations with >8 assistant turns
- Cap sentences tested per conversation at 30 (sample if more)
- Skip actions task (huge traces)
- Per-call timeout via openai client
"""

import argparse, json, os, re, copy, time, random, traceback
from pathlib import Path
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
                    gold = ans.split("####")[-1].strip() if "####" in ans else ans.strip()
                elif task in ("actions", "database"):
                    gold = json.dumps(item.get("expected_results",
                           item.get("reference_answer", "")))
                elif task == "data2text":
                    refs = item.get("references", [])
                    gold = refs[0] if refs else str(item.get("target", ""))
                else:
                    gold = str(item.get("answer", ""))
                if gold:
                    idx[tid] = {"gold": gold, "task": task}
        except:
            pass
    print(f"Gold index: {len(idx)} entries", flush=True)
    return idx


def load_failed_convs(log_dir, gold_index, n_conv, tasks, max_asst_turns):
    failed = []
    for task in tasks:
        task_dir = Path(log_dir) / task
        if not task_dir.exists():
            continue
        for f in sorted(task_dir.glob("**/*.jsonl")):
            with open(f) as fp:
                for line in fp:
                    try:
                        r = json.loads(line.strip())
                        if float(r.get("score", r.get("is_correct", 1))) < 0.5:
                            tid = r.get("task_id", "")
                            if tid in gold_index:
                                # Quick filter for short conversations
                                trace = r.get("trace", [])
                                n_asst = sum(1 for t in trace if t.get("role") == "assistant")
                                if 2 <= n_asst <= max_asst_turns:
                                    r["_gold_info"] = gold_index[tid]
                                    failed.append(r)
                    except:
                        pass
    random.seed(42); random.shuffle(failed)
    return failed[:n_conv]


def extract_messages(trace):
    msgs = []
    for t in trace:
        role = t.get("role")
        if role in ("system", "user", "assistant"):
            msgs.append({"role": role, "content": t.get("content", "")})
    return msgs


def split_sentences(text):
    text = text.strip()
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 15]


def call_llm(client, model, messages, max_tokens=400):
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=0.0, timeout=30
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            err = str(e).lower()
            if "context length" in err or "maximum context" in err:
                return "[SKIP_TOOLONG]"
            if attempt == 1:
                return f"[ERROR: {str(e)[:80]}]"
            time.sleep(1)
    return ""


def grade_math(pred, gold):
    pred_c = pred.strip().lower()
    gold_c = gold.strip().lower()
    m = re.search(r"\\boxed\{([^}]+)\}", pred)
    if m:
        pred_c = m.group(1).strip().lower()
    return 1.0 if pred_c == gold_c or gold_c in pred_c else 0.0


def grade(grader, grader_model, pred, gold, task):
    if task == "math":
        s = grade_math(pred, gold)
        if s > 0:
            return s
    prompt = (
        f"Task: {task}\nGold: {gold}\nPrediction: {pred}\n"
        f"Is the prediction correct? Reply JSON: "
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
    except:
        return 0.0


def drop_sentence(messages, asst_turn_idx, sent_idx):
    msgs = copy.deepcopy(messages)
    asst_count = 0
    for m in msgs:
        if m["role"] == "assistant":
            if asst_count == asst_turn_idx:
                sents = split_sentences(m["content"])
                remaining = [s for i, s in enumerate(sents) if i != sent_idx]
                m["content"] = " ".join(remaining) or "[removed]"
                break
            asst_count += 1
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--n_conv", type=int, default=20)
    ap.add_argument("--max_asst_turns", type=int, default=8)
    ap.add_argument("--max_sentences_per_conv", type=int, default=30)
    ap.add_argument("--tasks", nargs="+", default=["math", "data2text"])
    ap.add_argument("--vllm_url", default="http://127.0.0.1:8102/v1")
    ap.add_argument("--model", default="qwen2.5-14b-tool")
    ap.add_argument("--grader_url", default="http://127.0.0.1:8101/v1")
    ap.add_argument("--grader_model", default="qwen2.5-7b-tool")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    client = OpenAI(api_key="sk-local", base_url=args.vllm_url)
    grader = OpenAI(api_key="sk-local", base_url=args.grader_url)
    out_path = Path(args.output_dir) / "results.jsonl"

    gold_index = build_gold_index(args.dataset_dir)
    recs = load_failed_convs(args.log_dir, gold_index, args.n_conv,
                             args.tasks, args.max_asst_turns)
    print(f"Loaded {len(recs)} failed conversations (after length filter)", flush=True)

    all_flip_rates = []
    anchor_examples = []

    with open(out_path, "w") as fp:
        for ci, rec in enumerate(recs):
            try:
                gi = rec["_gold_info"]
                gold, task = gi["gold"], gi["task"]
                task_id = rec.get("task_id", "")
                msgs = extract_messages(rec.get("trace", []))
                if not msgs:
                    continue

                # Baseline
                orig_pred = call_llm(client, args.model, msgs)
                if "[SKIP" in orig_pred or "[ERROR" in orig_pred:
                    print(f"  ci={ci} baseline failed: {orig_pred[:50]}", flush=True)
                    continue
                orig_score = grade(grader, args.grader_model, orig_pred, gold, task)

                asst_turns = [(i, m) for i, m in enumerate(msgs) if m["role"] == "assistant"]
                if len(asst_turns) < 2:
                    continue

                # Build all candidate (turn, sent_idx, sent) triples, then sample
                candidates = []
                for turn_local_idx, (msg_idx, asst_msg) in enumerate(asst_turns[:-1]):
                    sents = split_sentences(asst_msg["content"])
                    for si, sent in enumerate(sents):
                        candidates.append((turn_local_idx, si, sent))

                # Sample if too many
                if len(candidates) > args.max_sentences_per_conv:
                    random.seed(ci)
                    candidates = random.sample(candidates, args.max_sentences_per_conv)

                sentence_results = []
                n_anchors = 0

                for turn_local_idx, si, sent in candidates:
                    mod_msgs = drop_sentence(msgs, turn_local_idx, si)
                    mod_pred = call_llm(client, args.model, mod_msgs)
                    if "[SKIP" in mod_pred or "[ERROR" in mod_pred:
                        continue
                    mod_score = grade(grader, args.grader_model, mod_pred, gold, task)
                    is_anchor = (orig_score < 0.5 and mod_score >= 0.5)
                    if is_anchor:
                        n_anchors += 1
                    sentence_results.append({
                        "turn_idx": turn_local_idx,
                        "sent_idx": si,
                        "sentence": sent[:250],
                        "orig_score": orig_score,
                        "mod_score": mod_score,
                        "is_anchor": is_anchor,
                    })
                    if is_anchor and len(anchor_examples) < 8:
                        anchor_examples.append({
                            "conv_idx": ci, "task": task, "task_id": task_id,
                            "anchor_sentence": sent,
                            "turn_idx": turn_local_idx,
                            "gold": gold[:80],
                            "orig_pred_snippet": orig_pred[:150],
                            "fixed_pred_snippet": mod_pred[:150],
                        })

                flip_rate = n_anchors / max(len(sentence_results), 1)
                all_flip_rates.append(flip_rate)

                entry = {
                    "ci": ci, "task": task, "task_id": task_id,
                    "n_asst_turns": len(asst_turns),
                    "n_sentences_tested": len(sentence_results),
                    "n_anchors": n_anchors,
                    "flip_rate": flip_rate,
                    "orig_score": orig_score,
                    "anchor_sentences": [r for r in sentence_results if r["is_anchor"]],
                }
                fp.write(json.dumps(entry) + "\n")
                fp.flush()

                if (ci + 1) % 2 == 0:
                    mean_fr = sum(all_flip_rates) / len(all_flip_rates)
                    n_with = sum(1 for x in all_flip_rates if x > 0)
                    print(f"  [{ci+1}/{len(recs)}] mean_flip={mean_fr:.3f} "
                          f"convs_with_anchor={n_with}/{len(all_flip_rates)} "
                          f"anchors_total={len(anchor_examples)}", flush=True)
            except Exception as e:
                print(f"  Error on ci={ci}: {e}", flush=True)
                traceback.print_exc()
                continue

    n = len(all_flip_rates)
    summary = {
        "n_conv": n,
        "mean_sentence_flip_rate": sum(all_flip_rates)/n if n else 0,
        "n_conv_with_anchor": sum(1 for x in all_flip_rates if x > 0),
        "anchor_examples": anchor_examples,
    }
    with open(Path(args.output_dir) / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"\nn={n}, mean_flip={summary['mean_sentence_flip_rate']:.3f}, "
          f"convs_with_anchor={summary['n_conv_with_anchor']}", flush=True)
    print("\nAnchor sentence examples:", flush=True)
    for ex in anchor_examples[:3]:
        print(f"  [{ex['task']}] Turn {ex['turn_idx']}: {ex['anchor_sentence'][:100]}", flush=True)
    print(f"\nResults -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
