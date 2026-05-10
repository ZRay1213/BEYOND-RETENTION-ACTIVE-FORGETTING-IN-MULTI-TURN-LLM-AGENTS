#!/usr/bin/env bash
MODEL=/root/autodl-tmp/models/_modelscope_cache/LLM-Research/gemma-2-9b-it
TMPL=/root/autodl-tmp/DCC/gemma_chat_template.jinja
CUDA_VISIBLE_DEVICES=1 /root/miniconda3/bin/vllm serve "$MODEL" \
    --tensor-parallel-size 1 \
    --port 8005 \
    --served-model-name gemma-2-9b \
    --gpu-memory-utilization 0.45 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --trust-remote-code \
    --chat-template "$TMPL" 2>&1 | tee /tmp/gemma_vllm.log
