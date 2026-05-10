#!/usr/bin/env bash
set -e
PROTO=$1   # sharded | recap | fresh-last | fresh-every | cacheguard
cd /root/autodl-tmp/DCC/data/lost_in_conversation
export OPENAI_API_KEY=sk-local
export OPENAI_BASE_URL_14B=http://127.0.0.1:8001/v1
export OPENAI_BASE_URL_7B=http://127.0.0.1:8002/v1
export DCC_GRADER_MODEL=qwen2.5-7b
export ASSISTANT_MODEL=qwen2.5-14b
case $PROTO in
  fresh-last)
    /root/miniconda3/bin/python harness_fresh_last.py --dataset_file data/sharded_f1_additive.json --N 2 --workers 2 --log_folder logs_f1_fresh_last --conv_type f1_fresh_last
    ;;
  fresh-every)
    /root/miniconda3/bin/python harness_fresh_every.py --dataset_file data/sharded_f1_additive.json --N 2 --workers 2 --log_folder logs_f1_fresh_every --conv_type f1_fresh_every
    ;;
  cacheguard)
    /root/miniconda3/bin/python harness_cacheguard.py --dataset_file data/sharded_f1_additive.json --N 2 --workers 2 --log_folder logs_f1_cacheguard --conv_type f1_cacheguard
    ;;
  sharded)
    /root/miniconda3/bin/python run_simulations.py --dataset_file data/sharded_f1_additive.json --N_full_runs 0 --N_concat_runs 2 --N_sharded_runs 2 --tasks math --models qwen2.5-14b --system_model qwen2.5-7b --user_model qwen2.5-7b --N_workers 2 --log_folder logs_f1_baseline
    ;;
esac
