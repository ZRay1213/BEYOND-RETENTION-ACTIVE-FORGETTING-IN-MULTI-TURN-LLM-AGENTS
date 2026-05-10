#!/usr/bin/env bash
set -e
LOGDIR_ROOT=/root/autodl-tmp/DCC/data/lost_in_conversation
HARNESS_POP=/root/autodl-tmp/DCC/harness_population

echo "=== AUTO_G4_AFTER_G1C1: $(date) ==="
while screen -ls | grep -q s_g1c1_heldout; do
  echo "[wait] g1c1_heldout still running... $(date)"
  sleep 600
done
echo "[done] G1_C1 held-out finished at $(date)"

# Copy + run G4_v2c0 on held-out
cp $HARNESS_POP/harness_g4_v2c0.py $LOGDIR_ROOT/harness_g4_v2c0.py
cd $LOGDIR_ROOT
export OPENAI_API_KEY=sk-local
export OPENAI_BASE_URL_14B=http://127.0.0.1:8003/v1
export OPENAI_BASE_URL_7B=http://127.0.0.1:8002/v1
export DCC_GRADER_MODEL=qwen2.5-7b
echo "=== G4_v2c0 held-out start: $(date) ==="
rm -rf $LOGDIR_ROOT/logs_g4v2c0_heldout
/root/miniconda3/bin/python harness_g4_v2c0.py \
  --dataset_file data/sharded_stage3_math100.json \
  --task_subset /root/autodl-tmp/DCC/heldout_tasks.json \
  --N 3 --workers 2 \
  --log_folder logs_g4v2c0_heldout --conv_type g4v2c0_heldout
echo "=== G4_v2c0 held-out done: $(date) ==="
