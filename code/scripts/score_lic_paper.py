#!/usr/bin/env python3
"""Score LiC experiment logs, report paper-comparable (math/code/actions) + all tasks.
Usage: python3 score_lic_paper.py <log_folder1> [<log_folder2> ...]
"""
import json, sys
from pathlib import Path
from collections import defaultdict

PAPER_TASKS = {'math', 'code', 'actions'}

def score_folder(folder):
    folder = Path(folder)
    if not folder.exists():
        return None, {}
    by_task = defaultdict(list)
    for jsonl in folder.rglob("*.jsonl"):
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    correct = bool(r.get("is_correct", False))
                    task = r.get("task", "unknown")
                    by_task[task].append(correct)
                except Exception:
                    pass
    return by_task

def fmt_task_acc(by_task, task):
    v = by_task.get(task, [])
    if not v:
        return "---"
    return f"{sum(v)/len(v):.3f}(n={len(v)})"

print(f"{'Protocol':<35} {'Math':>10} {'Code':>10} {'Actions':>10} {'Avg(3)':>9} {'N':>5}")
print("-" * 85)

for folder in sys.argv[1:]:
    by_task = score_folder(folder)
    name = Path(folder).name
    if by_task is None:
        print(f"{name:<35}  (no data)")
        continue

    # Paper 3 tasks
    math_v = by_task.get('math', [])
    code_v = by_task.get('code', [])
    act_v = by_task.get('actions', [])
    paper_vals = [v for lst in [math_v, code_v, act_v] for v in lst]

    math_acc = sum(math_v)/len(math_v) if math_v else float('nan')
    code_acc = sum(code_v)/len(code_v) if code_v else float('nan')
    act_acc = sum(act_v)/len(act_v) if act_v else float('nan')

    # Average of the 3 (by task, then average tasks)
    task_avgs = []
    for v in [math_v, code_v, act_v]:
        if v:
            task_avgs.append(sum(v)/len(v))
    avg3 = sum(task_avgs)/len(task_avgs) if task_avgs else float('nan')

    n_paper = len(paper_vals)

    print(f"{name:<35} {math_acc:>10.3f} {code_acc:>10.3f} {act_acc:>10.3f} {avg3:>9.3f} {n_paper:>5}")

    # Also show other tasks
    other_tasks = [(t, v) for t, v in sorted(by_task.items()) if t not in PAPER_TASKS]
    if other_tasks:
        other_str = "  other: " + "  ".join(f"{t}={sum(v)/len(v):.3f}(n={len(v)})" for t,v in other_tasks)
        print(f"  {other_str}")
