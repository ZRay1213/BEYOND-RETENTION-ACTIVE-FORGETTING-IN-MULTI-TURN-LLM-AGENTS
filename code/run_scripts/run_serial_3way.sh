#!/usr/bin/env bash
set -e
SUBSET=/root/autodl-tmp/DCC/train_subset_10.json
for cond in "harness.py:g0_t10:logs_g0_t10" "harness_g1_c0.py:g1c0_t10:logs_g1c0_t10" "harness_g1_c1.py:g1c1_t10:logs_g1c1_t10"; do
  IFS=":" read -r hf cv ld <<< "$cond"
  rm -rf /root/autodl-tmp/DCC/data/lost_in_conversation/$ld
  echo "=== Running $cv ==="
  date
  /root/autodl-tmp/DCC/run_eval3.sh $hf $cv $ld
  echo "=== Done $cv ==="
done
