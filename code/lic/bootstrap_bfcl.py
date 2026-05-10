"""Paired bootstrap for BFCL results across multiple datasets."""
import json, sys, random, glob
from pathlib import Path


def bootstrap(a, b, N=10000, seed=42):
    """One-tailed paired bootstrap: P(delta >= obs | H0: delta=0).
    Centers the paired differences under H0 before resampling."""
    rng = random.Random(seed)
    ids = sorted(set(a) & set(b))
    diffs = [int(a[i]) - int(b[i]) for i in ids]
    obs = sum(diffs) / len(diffs)
    # Center under H0
    centered = [d - obs for d in diffs]
    count = 0
    for _ in range(N):
        samp = [rng.choice(centered) for _ in diffs]
        if sum(samp) / len(samp) >= obs:
            count += 1
    p = count / N
    return obs, p, len(ids)


def load(path):
    return {k: bool(v) for k, v in json.load(open(path)).items()}


def report_dataset(name, pid_dir, suffix):
    s = load(f"{pid_dir}/sharded_{suffix}.json")
    fl = load(f"{pid_dir}/fresh-last_{suffix}.json")
    fe = load(f"{pid_dir}/fresh-every_{suffix}.json")
    cg = load(f"{pid_dir}/cacheguard_{suffix}.json")

    ids = sorted(set(s) & set(fl) & set(fe) & set(cg))
    n = len(ids)

    accs = {
        "Sharded":     sum(s[i] for i in ids) / n,
        "Fresh-Last":  sum(fl[i] for i in ids) / n,
        "Fresh-Every": sum(fe[i] for i in ids) / n,
        "CacheGuard":  sum(cg[i] for i in ids) / n,
    }

    print(f"\n{'='*50}")
    print(f"Dataset: {name}  (N={n})")
    print(f"{'='*50}")
    for proto, acc in accs.items():
        print(f"  {proto:<14}: {acc:.3f}  ({int(acc*n)}/{n})")

    print(f"\n  Bootstrap paired tests vs Sharded (N=10000 resamp):")
    for other_name, other in [("CacheGuard", cg), ("Fresh-Last", fl), ("Fresh-Every", fe)]:
        diff, p, _ = bootstrap(other, s)
        sig = "*" if p < 0.05 else ("~" if p < 0.10 else "")
        print(f"    {other_name} vs Sharded:  delta={diff:+.3f}  p={p:.3f} {sig}")

    print(f"\n  CacheGuard vs Fresh-Every:")
    diff, p, _ = bootstrap(cg, fe)
    sig = "*" if p < 0.05 else ("~" if p < 0.10 else "")
    print(f"    delta={diff:+.3f}  p={p:.3f} {sig}")

    return accs, n


if __name__ == "__main__":
    pid_dir = sys.argv[1] if len(sys.argv) > 1 else "per_id"
    mode = sys.argv[2] if len(sys.argv) > 2 else "n30"  # "n30" or "n200"

    if mode == "n200":
        datasets = [("base_N200", "base_n200")]
    elif mode == "partial":
        datasets = [("base_partial", "base_partial")]
    else:
        datasets = [("base", "base_n30"), ("miss_param", "missparam_n30"), ("long_context", "longctx_n30")]

    all_results = {}
    for (name, suffix) in datasets:
        try:
            accs, n = report_dataset(name, pid_dir, suffix)
            all_results[name] = (accs, n)
        except FileNotFoundError as e:
            print(f"\n[SKIP] {name}: {e}")

    print("\n\n=== SUMMARY TABLE ===")
    protos = ["Sharded", "Fresh-Last", "Fresh-Every", "CacheGuard"]
    print(f"{'Dataset':<16}", end="")
    for p in protos:
        print(f"  {p:<14}", end="")
    print()
    print("-" * 76)
    for name, (accs, n) in all_results.items():
        print(f"{name:<16}", end="")
        for p in protos:
            print(f"  {accs[p]:.3f}          ", end="")
        print(f"  (N={n})")
