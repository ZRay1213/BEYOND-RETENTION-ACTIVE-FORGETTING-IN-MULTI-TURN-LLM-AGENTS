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

LOG=/tmp/exp_lic14b.log

# Record PCHR baseline
curl -s http://localhost:8001/metrics | grep -E "^vllm:prefix_cache_(queries|hits)_total" | awk '{print "S_before", $0}' >> /tmp/pchr_raw.txt

echo "=== [LiC 14B] Sharded N=150 ===" | tee -a $LOG
$PY harness.py --dataset_file data/sharded_stage1_K25.json \
    --N 150 --workers 6 \
    --log_folder logs_sharded_stage1 --conv_type sharded_stage1 \
    2>&1 | tee -a $LOG

curl -s http://localhost:8001/metrics | grep -E "^vllm:prefix_cache_(queries|hits)_total" | awk '{print "S_after", $0}' >> /tmp/pchr_raw.txt

echo "=== [LiC 14B] Fresh-Last N=150 ===" | tee -a $LOG
curl -s http://localhost:8001/metrics | grep -E "^vllm:prefix_cache_(queries|hits)_total" | awk '{print "FL_before", $0}' >> /tmp/pchr_raw.txt

$PY harness_fresh_last.py --dataset_file data/sharded_stage1_K25.json \
    --N 150 --workers 6 \
    --log_folder logs_fresh_last_stage1 --conv_type fresh_last_stage1 \
    2>&1 | tee -a $LOG

curl -s http://localhost:8001/metrics | grep -E "^vllm:prefix_cache_(queries|hits)_total" | awk '{print "FL_after", $0}' >> /tmp/pchr_raw.txt

echo "=== [LiC 14B] Fresh-Every N=150 ===" | tee -a $LOG
curl -s http://localhost:8001/metrics | grep -E "^vllm:prefix_cache_(queries|hits)_total" | awk '{print "FE_before", $0}' >> /tmp/pchr_raw.txt

$PY harness_fresh_every.py --dataset_file data/sharded_stage1_K25.json \
    --N 150 --workers 6 \
    --log_folder logs_fresh_every_stage1 --conv_type fresh_every_stage1 \
    2>&1 | tee -a $LOG

curl -s http://localhost:8001/metrics | grep -E "^vllm:prefix_cache_(queries|hits)_total" | awk '{print "FE_after", $0}' >> /tmp/pchr_raw.txt

echo "=== [LiC 14B] CacheGuard N=150 ===" | tee -a $LOG
curl -s http://localhost:8001/metrics | grep -E "^vllm:prefix_cache_(queries|hits)_total" | awk '{print "CG_before", $0}' >> /tmp/pchr_raw.txt

$PY harness_cacheguard.py --dataset_file data/sharded_stage1_K25.json \
    --N 150 --workers 6 \
    --log_folder logs_cacheguard_stage1 --conv_type cacheguard_stage1 \
    2>&1 | tee -a $LOG

curl -s http://localhost:8001/metrics | grep -E "^vllm:prefix_cache_(queries|hits)_total" | awk '{print "CG_after", $0}' >> /tmp/pchr_raw.txt

echo "=== [LiC 14B] Scoring ===" | tee -a $LOG
for proto in sharded fresh_last fresh_every cacheguard; do
    dir=logs_${proto}_stage1
    $PY -c "
import json, glob
from collections import Counter
recs = []
for f in glob.glob('${dir}/**/*.jsonl', recursive=True):
    for line in open(f):
        try: recs.append(json.loads(line))
        except: pass
ok = [r for r in recs if r.get('is_correct')]
print(f'${proto}: {len(ok)}/{len(recs)} = {len(ok)/len(recs):.3f}' if recs else '${proto}: 0 recs')
" 2>&1 | tee -a $LOG
done

echo "=== ALL LiC 14B DONE ===" | tee -a $LOG
