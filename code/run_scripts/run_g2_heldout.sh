#!/usr/bin/env bash
set -e
cd /root/autodl-tmp/DCC/data/lost_in_conversation
export OPENAI_API_KEY=sk-local
export OPENAI_BASE_URL_14B=http://127.0.0.1:8003/v1
export OPENAI_BASE_URL_7B=http://127.0.0.1:8002/v1
export DCC_GRADER_MODEL=qwen2.5-7b
echo "=== G2_C0 held-out start: $(date) ==="
rm -rf logs_g2c0_heldout
/root/miniconda3/bin/python harness_g2_c0.py \
  --dataset_file data/sharded_stage3_math100.json \
  --task_subset /root/autodl-tmp/DCC/heldout_tasks.json \
  --N 3 --workers 2 \
  --log_folder logs_g2c0_heldout --conv_type g2c0_heldout
echo "=== G2_C0 done: $(date) ==="
echo "=== G2_C1 held-out start: $(date) ==="
rm -rf logs_g2c1_heldout
/root/miniconda3/bin/python harness_g2_c1.py \
  --dataset_file data/sharded_stage3_math100.json \
  --task_subset /root/autodl-tmp/DCC/heldout_tasks.json \
  --N 3 --workers 2 \
  --log_folder logs_g2c1_heldout --conv_type g2c1_heldout
echo "=== G2_C1 done: $(date) ==="
