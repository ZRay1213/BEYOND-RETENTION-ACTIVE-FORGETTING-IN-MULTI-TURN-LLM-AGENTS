"""Paired bootstrap for LiC log folders. Aggregates per task_id (mean of reps) then runs paired test."""
import json, sys, random
from pathlib import Path
from collections import defaultdict

PAPER_TASKS = {'math', 'code', 'actions'}


def load_per_task(folder):
    """Returns dict[task_id] -> mean accuracy across reps. Filtered to math/code/actions."""
    folder = Path(folder)
    by_id = defaultdict(list)
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
                    if tid:
                        by_id[tid].append(int(correct))
                except Exception:
                    pass
    return {tid: sum(vs) / len(vs) for tid, vs in by_id.items() if vs}


def bootstrap(a, b, B=10000, seed=42):
    """One-tailed paired bootstrap: P(delta >= obs | H0: delta=0).
    Returns (delta_obs, p_value, ci_low, ci_high) for delta = mean(a) - mean(b)."""
    rng = random.Random(seed)
    ids = sorted(set(a) & set(b))
    if len(ids) < 3:
        return None, None, None, None
    diffs = [a[i] - b[i] for i in ids]
    n = len(diffs)
    obs = sum(diffs) / n

    # H0-centered diffs for p-value
    centered = [d - obs for d in diffs]
    boot_means_h0 = []
    for _ in range(B):
        sample = [centered[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means_h0.append(sum(sample) / n)
    p = sum(1 for x in boot_means_h0 if x >= obs) / B

    # CI from raw diffs (percentile bootstrap)
    boot_means_raw = []
    for _ in range(B):
        sample = [diffs[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means_raw.append(sum(sample) / n)
    boot_means_raw.sort()
    ci_low = boot_means_raw[int(0.025 * B)]
    ci_high = boot_means_raw[int(0.975 * B)]

    return obs, p, ci_low, ci_high


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: bootstrap_lic.py <protocol_folder_a> <protocol_folder_b> [label]")
        sys.exit(1)
    folder_a, folder_b = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else f"{Path(folder_a).name} vs {Path(folder_b).name}"

    a = load_per_task(folder_a)
    b = load_per_task(folder_b)
    common = sorted(set(a) & set(b))
    delta, p, lo, hi = bootstrap(a, b)
    if delta is None:
        print(f"{label}: too few common tasks ({len(common)})")
    else:
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        print(f"{label:<60} Δ={delta:+.3f}  CI=[{lo:+.3f},{hi:+.3f}]  p={p:.3f}  {sig}  (n_common={len(common)})")
