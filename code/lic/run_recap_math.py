"""
Run Recap protocol on math task to close the dagger footnote in paper Table 1.
Reads existing sharded math conversations, appends a Recap user message,
calls vLLM 14B for the final answer, then scores with system_agent + evaluator_function.

Usage:
    cd /root/autodl-tmp/DCC/data/lost_in_conversation
    OPENAI_API_KEY=EMPTY OPENAI_BASE_URL=http://localhost:8002/v1 \\
      OPENAI_BASE_URL_14B=http://localhost:8002/v1 \\
      OPENAI_BASE_URL_7B=http://localhost:8001/v1 \\
      python3 run_recap_math.py --N 2
"""
import os, sys, json, uuid, copy, argparse, time, datetime, traceback
sys.path.insert(0, "/root/autodl-tmp/DCC/data/lost_in_conversation")

from tasks.tasks import get_task
from system_agent import SystemAgent
from model_openai import generate as model_generate


def date_str():
    return datetime.datetime.now().isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=2, help="reps per task")
    ap.add_argument("--source_log", type=str,
        default="/root/autodl-tmp/DCC/data/lost_in_conversation/logs_stage1/math/sharded/sharded_math_qwen2.5-14b.jsonl")
    ap.add_argument("--out_dir", type=str,
        default="/root/autodl-tmp/DCC/data/lost_in_conversation/logs_recap_multi/math/recap_multi")
    ap.add_argument("--multi_heldout", type=str,
        default="/root/autodl-tmp/DCC/data/lost_in_conversation/data/multi_heldout48.json")
    ap.add_argument("--assistant_model", type=str, default="qwen2.5-14b-tool")
    ap.add_argument("--system_model", type=str, default="qwen2.5-7b")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_fn = os.path.join(args.out_dir, "recap_multi_math_qwen2.5-14b.jsonl")

    done_source_conv_ids = set()
    if os.path.exists(out_fn):
        with open(out_fn) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_source_conv_ids.add(rec.get("source_conv_id"))
                except Exception:
                    continue
        print(f"[resume] {len(done_source_conv_ids)} done", flush=True)

    with open(args.multi_heldout) as f:
        held = json.load(f)
    math_samples = [s for s in held if s["task"] == "math"]
    sample_by_id = {s["task_id"]: s for s in math_samples}
    print(f"[heldout] {len(math_samples)} math samples", flush=True)

    source_convs = []
    with open(args.source_log) as f:
        for line in f:
            try:
                conv = json.loads(line)
            except Exception:
                continue
            if conv.get("task_id") in sample_by_id and conv.get("is_correct") is not None:
                source_convs.append(conv)
    print(f"[source] {len(source_convs)} sharded math convs in heldout", flush=True)

    by_task = {}
    for c in source_convs:
        by_task.setdefault(c["task_id"], []).append(c)
    selected = []
    for tid, lst in sorted(by_task.items()):
        for c in lst[:args.N]:
            if c["conv_id"] not in done_source_conv_ids:
                selected.append(c)
    print(f"[selected] {len(selected)} convs to run", flush=True)

    task = get_task("math")
    n_correct = 0
    n_total = 0
    fout = open(out_fn, "a")
    for i, src in enumerate(selected):
        try:
            tid = src["task_id"]
            sample = sample_by_id[tid]
            recap_text = task.populate_concat_prompt(sample)

            trace = copy.deepcopy(src["trace"])
            trace.append({
                "role": "user",
                "content": (
                    "Let me consolidate everything I've shared as one complete request:\n\n" + recap_text +
                    "\n\nPlease provide your final answer."
                ),
                "timestamp": date_str(),
            })

            msgs = []
            for m in trace:
                if m.get("role") in ("system", "user", "assistant"):
                    c = m["content"]
                    if isinstance(c, str):
                        msgs.append({"role": m["role"], "content": c})

            resp = model_generate(model=args.assistant_model, messages=msgs, temperature=1.0, max_tokens=2048)
            content = resp if isinstance(resp, str) else resp.get("content", str(resp))
            trace.append({"role": "assistant", "content": content, "timestamp": date_str()})

            sa = SystemAgent("math", args.system_model, sample=sample)
            try:
                extracted = sa.extract_answer(trace)
            except Exception as e:
                extracted = content
                print(f"[extract_warn] {tid}: {e}", flush=True)

            try:
                ev = task.evaluator_function(extracted, sample)
                if isinstance(ev, dict):
                    score = ev.get("score")
                    if "is_correct" in ev:
                        is_correct = bool(ev["is_correct"])
                    elif score is not None:
                        is_correct = float(score) >= 1.0
                    else:
                        is_correct = False
                else:
                    is_correct = bool(ev[0]) if isinstance(ev, tuple) else bool(ev)
                    score = None
            except Exception as e:
                print(f"[eval_err] {tid}: {e}", flush=True)
                is_correct = False
                score = None
                ev = {"error": str(e)}

            trace.append({
                "role": "log",
                "content": {"type": "answer-evaluation", "extracted_answer": str(extracted), "is_correct": is_correct, "score": score},
                "timestamp": date_str(),
            })

            rec = {
                "conv_id": uuid.uuid4().hex[:24],
                "conv_type": "recap_multi",
                "task": "math",
                "task_id": tid,
                "dataset_fn": "multi_heldout48.json",
                "assistant_model": args.assistant_model,
                "system_model": args.system_model,
                "user_model": src.get("user_model", args.system_model),
                "trace": trace,
                "is_correct": is_correct,
                "score": score,
                "source_conv_id": src["conv_id"],
                "source_is_correct": src["is_correct"],
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_total += 1
            if is_correct:
                n_correct += 1
            print(f"[{i+1}/{len(selected)}] {tid} -> {'OK' if is_correct else 'X'} (running acc {n_correct}/{n_total} = {n_correct/max(1,n_total):.3f})", flush=True)
        except Exception as e:
            print(f"[err] {src.get('task_id')}: {e}", flush=True)
            traceback.print_exc()
            time.sleep(2)
            continue
    fout.close()
    print(f"=== DONE === {n_correct}/{n_total} = {n_correct/max(1,n_total):.3f}", flush=True)


if __name__ == "__main__":
    main()
