#!/usr/bin/env bash
set -e
BASE=/root/autodl-tmp/DCC/data/lost_in_conversation
PY=/root/miniconda3/bin/python3
cd $BASE

export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://localhost:8001/v1
export OPENAI_BASE_URL_7B=http://localhost:8002/v1
export OPENAI_BASE_URL_14B=http://localhost:8001/v1
export ASSISTANT_MODEL=qwen2.5-14b
export GENERATOR_MODEL=qwen2.5-7b

LOG=/tmp/exp_cf_v2.log

echo "=== [Counterfactual v2] A/B/C/D on multi_baseline ===" | tee -a $LOG

$PY harness_counterfactual.py \
    --source_logs_dir logs_multi_baseline \
    --source_conv_type sharded \
    --N_per_task 4 \
    --max_samples 60 \
    --conditions A_original B_correct C_wrong D_drop \
    --workers 4 \
    --log_folder logs_counterfactual_v2 \
    2>&1 | tee -a $LOG

echo "=== Scoring counterfactual v2 ===" | tee -a $LOG
$PY -c "
import json, glob
from collections import defaultdict
d = defaultdict(list)
for f in glob.glob('logs_counterfactual_v2/**/*.jsonl', recursive=True):
    for line in open(f):
        try:
            r = json.loads(line)
            ct = r.get('conv_type','?')
            d[ct].append(r.get('is_correct', None))
        except: pass
print('=== Counterfactual v2 results ===')
for ct in ['cf_A_original','cf_B_correct','cf_C_wrong','cf_D_drop']:
    vals = d[ct]
    ok = [v for v in vals if v is not None]
    acc = sum(1 for v in ok if v)/len(ok) if ok else 0
    print(f'  {ct}: n={len(ok)}, acc={acc:.3f}')
" 2>&1 | tee -a $LOG

echo "=== COUNTERFACTUAL DONE ===" | tee -a $LOG
