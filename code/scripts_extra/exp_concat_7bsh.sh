#!/usr/bin/env bash
set -e
BASE=/root/autodl-tmp/DCC/data/lost_in_conversation
PY=/root/miniconda3/bin/python3
cd $BASE

export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL_7B=http://localhost:8002/v1
export OPENAI_BASE_URL_14B=http://localhost:8001/v1

DS=data/multi_mca24.json
LOG=/tmp/exp_concat_7bsh.log

# ---- Run 1: Llama-8B Concat (single-turn ceiling for Llama) ----
export OPENAI_BASE_URL=http://localhost:8004/v1
export ASSISTANT_MODEL=llama-3.1-8b
export DCC_GRADER_MODEL=qwen2.5-14b
echo "=== [Llama-8B Concat] ===" | tee -a $LOG
$PY run_simulations.py --dataset_file $DS \
    --tasks math code actions \
    --N_concat_runs 4 --N_full_runs 0 --N_sharded_runs 0 \
    --models llama-3.1-8b \
    --system_model qwen2.5-7b --user_model qwen2.5-7b \
    --N_workers 6 --log_folder logs_concat_llama_mca \
    2>&1 | tee -a $LOG

# ---- Run 2: Qwen-7B Sharded (baseline for cross-model table) ----
export OPENAI_BASE_URL=http://localhost:8002/v1
export ASSISTANT_MODEL=qwen2.5-7b
export DCC_GRADER_MODEL=qwen2.5-7b
echo "=== [Qwen-7B Sharded] ===" | tee -a $LOG
$PY harness.py --dataset_file $DS --N 4 --workers 6 \
    --log_folder logs_sharded_qwen7b_mca --conv_type sharded_qwen7b_mca \
    2>&1 | tee -a $LOG

# ---- Run 3: Qwen-7B Concat (Concat ceiling for 7B) ----
echo "=== [Qwen-7B Concat] ===" | tee -a $LOG
$PY run_simulations.py --dataset_file $DS \
    --tasks math code actions \
    --N_concat_runs 4 --N_full_runs 0 --N_sharded_runs 0 \
    --models qwen2.5-7b \
    --system_model qwen2.5-7b --user_model qwen2.5-7b \
    --N_workers 6 --log_folder logs_concat_qwen7b_mca \
    2>&1 | tee -a $LOG

echo "=== ALL DONE ===" | tee -a $LOG
