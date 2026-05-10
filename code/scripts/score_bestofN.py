"""Score LiC log folder using best-of-N per task_id (vs default mean-of-N).
Lets us check: does Concat with N attempts (best) match Fresh-Every?
"""
import json, sys
from pathlib import Path
from collections import defaultdict

PAPER_TASKS = {'math', 'code', 'actions'}


def score(folder, mode='mean'):
    folder = Path(folder)
    by_task_id = defaultdict(list)
    by_class = defaultdict(list)
    for jsonl in folder.rglob("*.jsonl"):
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("task") not in PAPER_TASKS:
                        continue
                    tid = r.get("task_id")
                    correct = bool(r.get("is_correct", False))
                    by_task_id[tid].append((r.get("task"), int(correct)))
                except Exception:
                    pass
    for tid, rows in by_task_id.items():
        task_class = rows[0][0]
        outcomes = [c for _, c in rows]
        if mode == 'mean':
            agg = sum(outcomes) / len(outcomes)
        elif mode == 'best':
            agg = 1 if any(outcomes) else 0
        elif mode == 'worst':
            agg = 1 if all(outcomes) else 0
        by_class[task_class].append(agg)
    avg_per_class = {c: sum(v) / len(v) for c, v in by_class.items() if v}
    avg3 = sum(avg_per_class.values()) / len(avg_per_class) if avg_per_class else float('nan')
    n_tasks = sum(len(v) for v in by_class.values())
    return avg_per_class, avg3, n_tasks


if __name__ == "__main__":
    print(f"{'Folder':<45} {'Mode':<8} {'Math':>7} {'Code':>7} {'Actions':>8} {'Avg':>7} {'N':>5}")
    print("-" * 90)
    for folder in sys.argv[1:]:
        for mode in ['mean', 'best']:
            apc, avg3, n = score(folder, mode)
            m = apc.get('math', float('nan'))
            c = apc.get('code', float('nan'))
            a = apc.get('actions', float('nan'))
            label = Path(folder).name
            print(f"{label:<45} {mode:<8} {m:>7.3f} {c:>7.3f} {a:>8.3f} {avg3:>7.3f} {n:>5}")
