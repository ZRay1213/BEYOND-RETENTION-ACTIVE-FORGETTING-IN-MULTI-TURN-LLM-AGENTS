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
LOG=/tmp/exp_llama_mech.log

# 1. Llama K=0.5 Drop-EARLIEST (test cliff cross-arch)
export DROP_FRAC=0.5
echo "=== [Llama-8B Drop-EARLIEST K=0.5] ===" | tee -a $LOG
$PY harness_selective_drop.py --dataset_file $DS --N 4 --workers 6 \
    --log_folder logs_drop_earliest_K05_llama_mca --conv_type drop_earliest_K05_llama \
    2>&1 | tee -a $LOG

# 2. Llama K=0.5 Drop-LATEST
echo "=== [Llama-8B Drop-LATEST K=0.5] ===" | tee -a $LOG
$PY harness_drop_latest.py --dataset_file $DS --N 4 --workers 6 \
    --log_folder logs_drop_latest_K05_llama_mca --conv_type drop_latest_K05_llama \
    2>&1 | tee -a $LOG

# 3. Llama Marked-History (test H1 cross-arch)
echo "=== [Llama-8B Marked-History] ===" | tee -a $LOG
$PY harness_marked_history.py --dataset_file $DS --N 4 --workers 6 \
    --log_folder logs_marked_llama_mca --conv_type marked_llama \
    2>&1 | tee -a $LOG

echo "=== ALL DONE ===" | tee -a $LOG
