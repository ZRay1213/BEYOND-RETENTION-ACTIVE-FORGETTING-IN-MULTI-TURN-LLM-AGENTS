"""Deep analysis of BFCL N=200 results.
Usage: python analyze_bfcl.py /path/to/logs_bfcl /path/to/per_id
"""
import json, sys, collections
from pathlib import Path

LOGS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs_bfcl")
PID  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("per_id")
PKG  = Path("/root/miniconda3/lib/python3.12/site-packages/bfcl_eval")
DATA = PKG / "data" / "BFCL_v4_multi_turn_base.json"

instances = {json.loads(l)["id"]: json.loads(l) for l in open(DATA)}

PROTOS = ["sharded", "fresh-last", "fresh-every", "cacheguard"]
LABELS = {"sharded": "Sharded", "fresh-last": "Fresh-Last",
          "fresh-every": "Fresh-Every", "cacheguard": "CacheGuard"}

# ── 1. Per-protocol token stats ──────────────────────────────────────────────
print("=" * 60)
print("1. INPUT TOKEN STATS (mean per instance across all turns)")
print("=" * 60)
tok_stats = {}
for proto in PROTOS:
    f = LOGS / f"{proto}_n200.jsonl"
    if not f.exists():
        f = LOGS / f"{proto}_partial.jsonl"
    records = [json.loads(l) for l in open(f) if not json.loads(l).get("error")]
    def flat_sum(nested):
        if isinstance(nested, list):
            return sum(flat_sum(x) for x in nested)
        return nested
    total_in = [flat_sum(r["input_token_count"]) for r in records if r.get("input_token_count")]
    total_out = [flat_sum(r["output_token_count"]) for r in records if r.get("output_token_count")]
    tok_stats[proto] = total_in
    n = len(total_in)
    mean_in = sum(total_in)/n if n else 0
    mean_out = sum(total_out)/n if n else 0
    print(f"  {LABELS[proto]:<14}: n={n}  mean_in={mean_in:.0f}  mean_out={mean_out:.0f}")

shard_mean = sum(tok_stats["sharded"]) / len(tok_stats["sharded"])
for proto in ["fresh-last", "fresh-every", "cacheguard"]:
    if tok_stats[proto]:
        m = sum(tok_stats[proto]) / len(tok_stats[proto])
        print(f"  {LABELS[proto]:<14} vs Sharded: {(m-shard_mean)/shard_mean*100:+.1f}%")

# ── 2. Per-category accuracy breakdown ──────────────────────────────────────
print()
print("=" * 60)
print("2. ACCURACY BY INVOLVED_CLASSES")
print("=" * 60)

per_id = {}
for proto in PROTOS:
    f = PID / f"{proto}_base_n200.json"
    if f.exists():
        per_id[proto] = {k: bool(v) for k, v in json.load(open(f)).items()}

# collect classes per id
class_acc = collections.defaultdict(lambda: {p: [] for p in PROTOS})
for tid, entry in instances.items():
    cls = tuple(sorted(entry.get("involved_classes", [])))
    for proto in PROTOS:
        if tid in per_id.get(proto, {}):
            class_acc[cls][proto].append(per_id[proto][tid])

# top classes by count
top = sorted(class_acc.items(), key=lambda x: -len(x[1]["sharded"]))[:8]
print(f"  {'Category':<35}", end="")
for p in PROTOS:
    print(f"  {LABELS[p][:9]:<11}", end="")
print()
print("  " + "-" * 82)
for cls, acc in top:
    label = "+".join(cls)[:33]
    n = len(acc["sharded"])
    print(f"  {label:<35}", end="")
    for p in PROTOS:
        a = acc[p]
        if a:
            print(f"  {sum(a)/len(a):.2f}({len(a):3d})", end="")
        else:
            print(f"  {'N/A':>9}", end="")
    print()

# ── 3. Accuracy by number of turns ──────────────────────────────────────────
print()
print("=" * 60)
print("3. ACCURACY BY NUMBER OF TURNS")
print("=" * 60)

turn_acc = collections.defaultdict(lambda: {p: [] for p in PROTOS})
for tid, entry in instances.items():
    n_turns = len(entry.get("question", []))
    for proto in PROTOS:
        if tid in per_id.get(proto, {}):
            turn_acc[n_turns][proto].append(per_id[proto][tid])

print(f"  {'Turns':<8}", end="")
for p in PROTOS:
    print(f"  {LABELS[p][:9]:<11}", end="")
print()
print("  " + "-" * 55)
for nt in sorted(turn_acc.keys()):
    print(f"  {nt:<8}", end="")
    for p in PROTOS:
        a = turn_acc[nt][p]
        if a:
            print(f"  {sum(a)/len(a):.2f}({len(a):3d})", end="")
        else:
            print(f"  {'N/A':>9}", end="")
    print()

# ── 4. CG vs S disagreement analysis ────────────────────────────────────────
print()
print("=" * 60)
print("4. CacheGuard vs Sharded DISAGREEMENT (where they differ)")
print("=" * 60)
if "sharded" in per_id and "cacheguard" in per_id:
    ids = sorted(set(per_id["sharded"]) & set(per_id["cacheguard"]))
    cg_only = [i for i in ids if per_id["cacheguard"][i] and not per_id["sharded"][i]]
    s_only  = [i for i in ids if per_id["sharded"][i] and not per_id["cacheguard"][i]]
    both    = [i for i in ids if per_id["sharded"][i] and per_id["cacheguard"][i]]
    neither = [i for i in ids if not per_id["sharded"][i] and not per_id["cacheguard"][i]]
    print(f"  Both correct:  {len(both)}")
    print(f"  S only:        {len(s_only)}")
    print(f"  CG only:       {len(cg_only)}")
    print(f"  Neither:       {len(neither)}")
    print(f"  Total:         {len(ids)}")
    if s_only:
        print(f"\n  S-only instance classes (first 5):")
        for tid in s_only[:5]:
            cls = instances.get(tid, {}).get("involved_classes", [])
            print(f"    {tid}: {cls}")
    if cg_only:
        print(f"\n  CG-only instance classes (first 5):")
        for tid in cg_only[:5]:
            cls = instances.get(tid, {}).get("involved_classes", [])
            print(f"    {tid}: {cls}")
