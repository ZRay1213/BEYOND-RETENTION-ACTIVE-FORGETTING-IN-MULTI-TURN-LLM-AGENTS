import json, os, sys
from collections import defaultdict
import numpy as np

ROOT = '/root/autodl-tmp/DCC/data/lost_in_conversation/logs_stage3_baseline/math'
schedules = ['ctrl_NONE','ctrl_VERIFY-K4','ctrl_VERIFY-K3','ctrl_RESET-K4','ctrl_SUMMARY-EVERY']
data = defaultdict(lambda: defaultdict(list))  # tid -> sched -> [scores]

for sched in schedules:
    d = os.path.join(ROOT, sched)
    if not os.path.isdir(d): continue
    for fn in os.listdir(d):
        if not fn.endswith('.jsonl'): continue
        with open(os.path.join(d, fn)) as f:
            for ln in f:
                r = json.loads(ln)
                if r.get('score') is None: continue
                data[r['task_id']][sched].append(float(r['score']))

# Per-instance mean acc per schedule
tids = sorted(data.keys())
print(f'instances: {len(tids)}\n')
print(f'{"task_id":24s} {"NONE":>6s} {"VER4":>6s} {"VER3":>6s} {"RST4":>6s} {"SUM":>6s}  pattern')
print('-' * 80)

# count per-pattern flips between NONE and VERIFY-K4
flips_helped = 0  # NONE wrong, VER4 right
flips_hurt   = 0  # NONE right, VER4 wrong
both_right   = 0
both_wrong   = 0

# oracle per-instance: max over schedules
def m(s, t):
    v = data[t].get(s, [])
    return np.mean(v) if v else float('nan')

for t in tids:
    n = m('ctrl_NONE', t); v4 = m('ctrl_VERIFY-K4', t)
    v3 = m('ctrl_VERIFY-K3', t); r4 = m('ctrl_RESET-K4', t); su = m('ctrl_SUMMARY-EVERY', t)
    pat = ''
    if n >= 0.5 and v4 < 0.5: flips_hurt += 1; pat='HURT'
    elif n < 0.5 and v4 >= 0.5: flips_helped += 1; pat='HELPED'
    elif n >= 0.5 and v4 >= 0.5: both_right += 1; pat='both-OK'
    else: both_wrong += 1; pat='both-WRONG'
    print(f'{t:24s} {n:>6.2f} {v4:>6.2f} {v3:>6.2f} {r4:>6.2f} {su:>6.2f}  {pat}')

print()
print('=== NONE vs VERIFY-K4 (instance-level majority) ===')
print(f'helped (N wrong → V right): {flips_helped}')
print(f'hurt   (N right → V wrong): {flips_hurt}')
print(f'both right:                 {both_right}')
print(f'both wrong:                 {both_wrong}')

# Oracle: per instance pick the best schedule
oracle_scores = []
for t in tids:
    best = max((m(s,t) for s in schedules if not np.isnan(m(s,t))), default=0)
    oracle_scores.append(best)
print(f'\nORACLE acc (per-instance best schedule): {np.mean(oracle_scores):.3f}')

# Oracle restricted to {NONE, VERIFY-K4}
oracle2 = []
for t in tids:
    cands = [m('ctrl_NONE',t), m('ctrl_VERIFY-K4',t)]
    cands = [c for c in cands if not np.isnan(c)]
    oracle2.append(max(cands) if cands else 0)
print(f'ORACLE_{{NONE+V4}} acc:                    {np.mean(oracle2):.3f}')
print(f'best fixed schedule acc (max of NONE/V4): {max(np.mean([m(s,t) for t in tids]) for s in ["ctrl_NONE","ctrl_VERIFY-K4"]):.3f}')
print(f'GAP (oracle headroom):                    {np.mean(oracle2) - 0.367:+.3f}')
