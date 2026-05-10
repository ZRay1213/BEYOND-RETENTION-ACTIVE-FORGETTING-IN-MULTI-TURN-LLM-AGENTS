#!/usr/bin/env bash
# Auto-iterate: wait for current serial to finish, then run G2 eval, then auto-G3 proposer.
set -e
LOGDIR_ROOT=/root/autodl-tmp/DCC/data/lost_in_conversation
HARNESS_POP=/root/autodl-tmp/DCC/harness_population
SUBSET=/root/autodl-tmp/DCC/train_subset_10.json

echo "=== AUTO PIPELINE: $(date) ==="

# Step 1: wait for serial to end
while screen -ls | grep -q s_serial; do
  echo "[waiting] s_serial still running... $(date)"
  sleep 600
done
echo "[done] G1_C1 serial finished at $(date)"

# Step 2: copy G2 variants into lost_in_conversation
cp $HARNESS_POP/harness_g2_c0.py $LOGDIR_ROOT/harness_g2_c0.py
cp $HARNESS_POP/harness_g2_c1.py $LOGDIR_ROOT/harness_g2_c1.py

# Step 3: serial G2 eval (G2_C0 + G2_C1 on same 10-task subset)
for cond in "harness_g2_c0.py:g2c0_t10:logs_g2c0_t10" "harness_g2_c1.py:g2c1_t10:logs_g2c1_t10"; do
  IFS=":" read -r hf cv ld <<< "$cond"
  rm -rf $LOGDIR_ROOT/$ld
  echo "=== $cv start: $(date) ==="
  N=2 WORKERS=2 bash /root/autodl-tmp/DCC/run_eval3.sh $hf $cv $ld
  echo "=== $cv done: $(date) ==="
done

# Step 4: pick best G2 (per-task mean), run G3 proposer
echo "=== auto-G3 proposer: $(date) ==="
/root/miniconda3/bin/python << "PY"
import json, glob, statistics
from collections import defaultdict
def per_task_mean(d):
    sims = []
    for f in glob.glob(f'{d}/*.jsonl'):
        for ln in open(f):
            try:
                r = json.loads(ln)
                if r.get('score') is not None: sims.append((r['task_id'], r['score']))
            except: pass
    by = defaultdict(list)
    for t,s in sims: by[t].append(s)
    if not by: return 0, 0
    return statistics.mean([statistics.mean(rs) for rs in by.values()]), len(by)

m_c0, n0 = per_task_mean('/root/autodl-tmp/DCC/data/lost_in_conversation/logs_g2c0_t10/math/g2c0_t10')
m_c1, n1 = per_task_mean('/root/autodl-tmp/DCC/data/lost_in_conversation/logs_g2c1_t10/math/g2c1_t10')
print(f'G2_C0: per-task={m_c0:.3f} n_tasks={n0}')
print(f'G2_C1: per-task={m_c1:.3f} n_tasks={n1}')
best = 'g2_c0' if m_c0 >= m_c1 else 'g2_c1'
best_traces = f'logs_{best}_t10/math/{best}_t10'
with open('/tmp/g2_best.txt','w') as f: f.write(best+'\n'+best_traces)
print(f'BEST: {best}')
PY
BEST=$(head -1 /tmp/g2_best.txt)
TRACES_REL=$(tail -1 /tmp/g2_best.txt)
cd $LOGDIR_ROOT
export OPENAI_API_KEY=sk-local
export OPENAI_BASE_URL_14B=http://127.0.0.1:8003/v1
/root/miniconda3/bin/python proposer.py \
  --parent $HARNESS_POP/harness_${BEST}.py \
  --traces_dir $LOGDIR_ROOT/$TRACES_REL \
  --gen 3 --n_variants 2 \
  --output_dir $HARNESS_POP

echo "=== AUTO PIPELINE done: $(date) ==="
