#!/usr/bin/env bash
# Run on Mistral-7B and Gemma-2-9B sequentially.
# Each: Sharded + Fresh-Last + Fresh-Every + Concat (skip CG/Marked for time).
set -e
BASE=/root/autodl-tmp/DCC/data/lost_in_conversation
PY=/root/miniconda3/bin/python3
cd $BASE

export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL_7B=http://localhost:8002/v1
export OPENAI_BASE_URL_14B=http://localhost:8102/v1
export DCC_GRADER_MODEL=qwen2.5-14b-tool

DS=data/multi_mca24.json
LOG=/tmp/exp_gemma2.log

run_model () {
    local mname=$1
    local mport=$2
    local tag=$3
    export OPENAI_BASE_URL=http://localhost:${mport}/v1
    export ASSISTANT_MODEL=${mname}

    echo "=== [${tag} Sharded] ===" | tee -a $LOG
    $PY harness.py --dataset_file $DS --N 4 --workers 6 \
        --log_folder logs_sharded_${tag}_mca --conv_type sharded_${tag}_mca \
        2>&1 | tee -a $LOG

    echo "=== [${tag} Fresh-Last] ===" | tee -a $LOG
    $PY harness_fresh_last.py --dataset_file $DS --N 4 --workers 6 \
        --log_folder logs_fresh_last_${tag}_mca --conv_type fresh_last_${tag}_mca \
        2>&1 | tee -a $LOG

    echo "=== [${tag} Fresh-Every] ===" | tee -a $LOG
    $PY harness_fresh_every.py --dataset_file $DS --N 4 --workers 6 \
        --log_folder logs_fresh_every_${tag}_mca --conv_type fresh_every_${tag}_mca \
        2>&1 | tee -a $LOG

    echo "=== [${tag} Concat] ===" | tee -a $LOG
    $PY run_simulations.py --dataset_file $DS \
        --tasks math code actions \
        --N_concat_runs 4 --N_full_runs 0 --N_sharded_runs 0 \
        --models ${mname} \
        --system_model qwen2.5-7b --user_model qwen2.5-7b \
        --N_workers 6 --log_folder logs_concat_${tag}_mca \
        2>&1 | tee -a $LOG
}

# Edit ports/names below after vLLM is up
run_model "gemma-2-9b" 8005 "gemma"

echo "=== ALL DONE ===" | tee -a $LOG
