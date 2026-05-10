#!/usr/bin/env bash
# G3 eval → G4 proposer
set -e
LOGDIR_ROOT=/root/autodl-tmp/DCC/data/lost_in_conversation
HARNESS_POP=/root/autodl-tmp/DCC/harness_population

echo "=== AUTO_G34: $(date) ==="
cp $HARNESS_POP/harness_g3_c0.py $LOGDIR_ROOT/harness_g3_c0.py
cp $HARNESS_POP/harness_g3_c1.py $LOGDIR_ROOT/harness_g3_c1.py

for cond in "harness_g3_c0.py:g3c0_t10:logs_g3c0_t10" "harness_g3_c1.py:g3c1_t10:logs_g3c1_t10"; do
  IFS=":" read -r hf cv ld <<< "$cond"
  rm -rf $LOGDIR_ROOT/$ld
  echo "=== $cv start: $(date) ==="
  N=2 WORKERS=2 bash /root/autodl-tmp/DCC/run_eval3.sh $hf $cv $ld
  echo "=== $cv done: $(date) ==="
done

# pick best G3, run G4 proposer
echo "=== auto-G4 proposer: $(date) ==="
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
m_c0, _ = per_task_mean('/root/autodl-tmp/DCC/data/lost_in_conversation/logs_g3c0_t10/math/g3c0_t10')
m_c1, _ = per_task_mean('/root/autodl-tmp/DCC/data/lost_in_conversation/logs_g3c1_t10/math/g3c1_t10')
print(f'G3_C0: {m_c0:.3f}, G3_C1: {m_c1:.3f}')
best = 'g3_c0' if m_c0 >= m_c1 else 'g3_c1'
with open('/tmp/g3_best.txt','w') as f: f.write(best)
print(f'BEST: {best}')
PY
BEST=$(cat /tmp/g3_best.txt)
TRACES=/root/autodl-tmp/DCC/data/lost_in_conversation/logs_${BEST}_t10/math/${BEST}_t10
cd $LOGDIR_ROOT
export OPENAI_API_KEY=sk-local
export OPENAI_BASE_URL_14B=http://127.0.0.1:8003/v1
/root/miniconda3/bin/python proposer.py \
  --parent $HARNESS_POP/harness_${BEST}.py \
  --traces_dir $TRACES \
  --gen 4 --n_variants 2 \
  --output_dir $HARNESS_POP

echo "=== AUTO_G34 done: $(date) ==="
