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
export DCC_GRADER_MODEL=qwen2.5-7b   # avoid gpt-4o grader

DS=data/sharded_stage1_K25_local.json  # 75 tasks: BFCL+code+math
LOG=/tmp/exp_lic14b_v2.log

for proto in sharded fresh_last fresh_every cacheguard; do
    case $proto in
        sharded)     harness=harness.py;            ct=sharded_stage1;    ;;
        fresh_last)  harness=harness_fresh_last.py; ct=fresh_last_stage1; ;;
        fresh_every) harness=harness_fresh_every.py;ct=fresh_every_stage1; ;;
        cacheguard)  harness=harness_cacheguard.py; ct=cacheguard_stage1; ;;
    esac
    echo "=== [LiC 14B] $proto ===" | tee -a $LOG
    curl -s http://localhost:8001/metrics | grep -E "^vllm:prefix_cache_(queries|hits)_total" \
        | awk "{print \"${proto}_before\", \$0}" >> /tmp/pchr_raw_v2.txt
    $PY $harness --dataset_file $DS --N 75 --workers 6 \
        --log_folder logs_${proto}_stage1 --conv_type ${ct} \
        2>&1 | tee -a $LOG
    curl -s http://localhost:8001/metrics | grep -E "^vllm:prefix_cache_(queries|hits)_total" \
        | awk "{print \"${proto}_after\", \$0}" >> /tmp/pchr_raw_v2.txt
done

echo "=== Scoring ===" | tee -a $LOG
$PY -c "
import json, glob
for proto in ['sharded','fresh_last','fresh_every','cacheguard']:
    recs=[]
    for f in glob.glob(f'logs_{proto}_stage1/**/*.jsonl', recursive=True):
        for l in open(f):
            try: recs.append(json.loads(l))
            except: pass
    ok=[r for r in recs if r.get('is_correct')]
    print(f'{proto}: {len(ok)}/{len(recs)} = {len(ok)/len(recs):.3f}' if recs else f'{proto}: 0')
" 2>&1 | tee -a $LOG
echo "=== LiC 14B DONE ===" | tee -a $LOG
