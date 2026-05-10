#!/usr/bin/env python3
"""Score LiC experiment logs. Usage: python3 score_lic.py logs_sharded_llama_mh logs_fresh_last_llama_mh logs_fresh_every_llama_mh"""
import json, sys, os
from pathlib import Path
from collections import defaultdict

def score_folder(folder):
    folder = Path(folder)
    if not folder.exists():
        return None, {}
    records = []
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
                    records.append(correct)
                    by_task[task].append(correct)
                except Exception:
                    pass
    if not records:
        return None, {}
    acc = sum(records) / len(records)
    task_acc = {t: (sum(v)/len(v), len(v)) for t, v in by_task.items()}
    return acc, task_acc

print(f"{'Protocol':<30} {'Acc':>6} {'N':>5}  {'By task type'}")
print("-" * 80)
for folder in sys.argv[1:]:
    acc, by_task = score_folder(folder)
    name = Path(folder).name
    if acc is None:
        print(f"{name:<30}  (no data)")
        continue
    n_total = sum(v[1] for v in by_task.values())
    task_str = "  ".join(f"{t}={v[0]:.3f}(n={v[1]})" for t, v in sorted(by_task.items()))
    print(f"{name:<30} {acc:>6.3f} {n_total:>5}  {task_str}")
