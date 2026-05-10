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
export DCC_GRADER_MODEL=qwen2.5-14b

DS=data/multi_mca24.json
LOG=/tmp/exp_llama_mca.log
PCHR=/tmp/pchr_llama_mca.txt

for proto in sharded fresh_last fresh_every; do
    case $proto in
        sharded)     harness=harness.py;             ct=sharded_llama_mca;    lf=logs_sharded_llama_mca;    ;;
        fresh_last)  harness=harness_fresh_last.py;  ct=fresh_last_llama_mca; lf=logs_fresh_last_llama_mca; ;;
        fresh_every) harness=harness_fresh_every.py; ct=fresh_every_llama_mca; lf=logs_fresh_every_llama_mca; ;;
    esac
    echo "=== [Llama-8B mca24] $proto ===" | tee -a $LOG
    curl -s --max-time 3 http://localhost:8004/metrics \
        | grep -E "^vllm:prefix_cache_(queries|hits)_total" \
        | sed "s/^/${proto}_before /" >> $PCHR 2>/dev/null || true
    $PY $harness --dataset_file $DS --N 4 --workers 6 \
        --log_folder $lf --conv_type ${ct} \
        2>&1 | tee -a $LOG
    curl -s --max-time 3 http://localhost:8004/metrics \
        | grep -E "^vllm:prefix_cache_(queries|hits)_total" \
        | sed "s/^/${proto}_after /" >> $PCHR 2>/dev/null || true
done
echo "=== ALL DONE ===" | tee -a $LOG
