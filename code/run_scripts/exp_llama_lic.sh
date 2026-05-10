#!/usr/bin/env bash
set -e
BASE=/root/autodl-tmp/DCC/data/lost_in_conversation
PY=/root/miniconda3/bin/python3
cd $BASE

export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://localhost:8004/v1
export OPENAI_BASE_URL_7B=http://localhost:8002/v1
export OPENAI_BASE_URL_14B=http://localhost:8001/v1
export ASSISTANT_MODEL=llama-3.1-8b

LOG=/tmp/exp_llama_lic.log

echo "=== [LiC Llama-3.1-8B] Sharded N=150 ===" | tee -a $LOG
$PY harness.py --dataset_file data/sharded_stage1_K25.json \
    --N 150 --workers 6 \
    --log_folder logs_sharded_llama --conv_type sharded_llama \
    2>&1 | tee -a $LOG

echo "=== [LiC Llama] Fresh-Last N=150 ===" | tee -a $LOG
$PY harness_fresh_last.py --dataset_file data/sharded_stage1_K25.json \
    --N 150 --workers 6 \
    --log_folder logs_fresh_last_llama --conv_type fresh_last_llama \
    2>&1 | tee -a $LOG

echo "=== [LiC Llama] Fresh-Every N=150 ===" | tee -a $LOG
$PY harness_fresh_every.py --dataset_file data/sharded_stage1_K25.json \
    --N 150 --workers 6 \
    --log_folder logs_fresh_every_llama --conv_type fresh_every_llama \
    2>&1 | tee -a $LOG

echo "=== [LiC Llama] CacheGuard N=150 ===" | tee -a $LOG
$PY harness_cacheguard.py --dataset_file data/sharded_stage1_K25.json \
    --N 150 --workers 6 \
    --log_folder logs_cacheguard_llama --conv_type cacheguard_llama \
    2>&1 | tee -a $LOG

echo "=== [LiC Llama] Scoring ===" | tee -a $LOG
$PY -c "
import json, glob
rows = []
for proto in ['sharded','fresh_last','fresh_every','cacheguard']:
    recs = []
    for f in glob.glob(f'logs_{proto}_llama/**/*.jsonl', recursive=True):
        for line in open(f):
            try: recs.append(json.loads(line))
            except: pass
    ok = [r for r in recs if r.get('is_correct')]
    acc = len(ok)/len(recs) if recs else 0
    print(f'{proto}: {len(ok)}/{len(recs)} = {acc:.3f}')
" 2>&1 | tee -a $LOG
echo "=== ALL LiC Llama DONE ===" | tee -a $LOG
