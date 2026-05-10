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
export DCC_GRADER_MODEL=qwen2.5-7b   # avoid gpt-4o grader

DS=data/sharded_stage1_K25_local.json  # 75 tasks: BFCL+code+math
LOG=/tmp/exp_llama_lic3.log

for proto in sharded fresh_last fresh_every; do
    case $proto in
        sharded)     harness=harness.py;            ct=sharded_llama;    ;;
        fresh_last)  harness=harness_fresh_last.py; ct=fresh_last_llama; ;;
        fresh_every) harness=harness_fresh_every.py;ct=fresh_every_llama; ;;
    esac
    echo "=== [LiC Llama-3.1-8B] $proto ===" | tee -a $LOG
    $PY $harness --dataset_file $DS --N 75 --workers 4 \
        --log_folder logs_${proto}_llama --conv_type ${ct} \
        2>&1 | tee -a $LOG
done

echo "=== Scoring ===" | tee -a $LOG
$PY -c "
import json, glob
for proto in ['sharded','fresh_last','fresh_every']:
    recs=[]
    for f in glob.glob(f'logs_{proto}_llama/**/*.jsonl', recursive=True):
        for l in open(f):
            try: recs.append(json.loads(l))
            except: pass
    ok=[r for r in recs if r.get('is_correct')]
    print(f'{proto}: {len(ok)}/{len(recs)} = {len(ok)/len(recs):.3f}' if recs else f'{proto}: 0')
" 2>&1 | tee -a $LOG
echo "=== Llama LiC DONE ===" | tee -a $LOG
