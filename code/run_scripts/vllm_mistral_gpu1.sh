#!/usr/bin/env bash
MODEL=/root/autodl-tmp/models/_modelscope_cache/LLM-Research/Mistral-7B-Instruct-v0.2
CUDA_VISIBLE_DEVICES=1 /root/miniconda3/bin/vllm serve "$MODEL"     --tensor-parallel-size 1     --port 8004     --served-model-name mistral-7b     --gpu-memory-utilization 0.18     --max-model-len 16384     --dtype bfloat16     --enable-prefix-caching     --trust-remote-code
