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
LOG=/tmp/exp_cg_llama.log

echo "=== [Llama-8B CacheGuard] ===" | tee -a $LOG
$PY harness_cacheguard.py --dataset_file $DS --N 4 --workers 6 \
    --log_folder logs_cacheguard_llama_mca --conv_type cacheguard_llama_mca \
    2>&1 | tee -a $LOG

echo "=== ALL DONE ===" | tee -a $LOG
