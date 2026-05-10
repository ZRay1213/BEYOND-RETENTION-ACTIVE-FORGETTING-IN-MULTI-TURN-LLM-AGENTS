# Beyond Retention: Active Forgetting in Multi-Turn LLM Agents

Code and core results for the paper **"Beyond Retention: Active Forgetting in Multi-Turn LLM Agents"**.

## Headline finding

Multi-turn LLM degradation on underspecified conversations is largely an artifact of the chat protocol, not a model-capability deficit. On Lost-in-Conversation with Qwen2.5-14B:

- **Fresh-Last** (drop prior assistant tokens at the final turn, 5 lines of code): recovers **87%** of the multi-turn gap
- **Fresh-Every** (drop them at every turn): reaches **103%**, exceeding the single-turn ceiling
- The fix is **binary, not graded**: keeping any 25–75% of prior tokens recovers only 4–23%

The mechanism is attention-mediated, replicates across 5 models in 4 architecture families, and **reverses cleanly on stateful tool use** (BFCL v4): there `Fresh-Every` collapses to 0.090 while typed `CacheGuard` matches the strongest baseline at 0.280.

## Repo layout

```
code/
  lic/                           # Lost-in-Conversation simulators + harnesses
    simulator_*.py               # Base protocol implementations
    harness_*.py                 # Cross-protocol harnesses (fresh_last/every, cacheguard, marked_history, ...)
    harness_cacheguard.py        # CacheGuard typed-eligibility renderer
    harness_fresh_every.py       # Fresh-Every full-history-purge protocol
    harness_marked_history.py    # [OUTDATED] marker baseline (H3 falsifier)
    harness_selective_drop.py    # Drop-EARLIEST K-fraction sweep
    harness_drop_latest.py       # Drop-LATEST K-fraction sweep
    cacheguard_bfcl.py           # CacheGuard adapter for BFCL v4
    bootstrap_bfcl.py            # Bootstrap CIs for BFCL
    attention_probe_n30.py       # Attention probe (n=22 final after OOM filter)
    model_openai.py              # Multi-endpoint vLLM/OpenAI client
    requirements.txt
  scripts/                       # Top-level analysis + experiment scripts
    per_turn_v3.py               # Per-turn accuracy trajectory (Appendix H)
    sentence_ablation_v3.py      # Cross-turn sentence-level anchor identification (Appendix H)
    bootstrap_lic.py             # Paired-bootstrap stats
    score_lic.py / score_bestofN.py # LiC scoring + best-of-N control
    extract_features_v2.py       # Bandit feature extraction (legacy)
    train_bandit_v2.py           # Bandit controller (legacy)
  run_scripts/                   # Bash launchers for individual experiments
    run_fresh_every.sh
    run_fresh_last.sh
    run_cacheguard.sh
    run_full_concat.sh
    run_drop_latest.sh
    run_f1_all.sh / run_f2_all.sh # Synthetic benchmarks
    run_counterfactual.sh

datasets/                        # JSON dataset files used in experiments
  multi_heldout48.json           # 48-instance multi-task held-out (8 per task × 6 tasks)
  multi_mca24.json               # 24-instance cross-model (8 each math/code/actions)
  sharded_stage1_K20.json        # Stage-1 sharded data
  sharded_stage1_K25.json
  sharded_f1_additive.json       # Synthetic F1 additive-constraints
  sharded_f2_corrective.json     # Synthetic F2 corrective-constraints
  sharded_smoke*.json

results/
  summaries/                     # summary.json from each experiment
  per_turn_v3/                   # n=80 per-turn accuracy trajectory
  sentence_ablation_v3/          # n=20 cross-turn anchor sentences
  dsr1_sharded/, dsr1_fresh_every/ # DeepSeek-R1-Distill-7B partial results (incomplete; see Notes)
  cross_model/                   # Aggregate JSONs from cross-model runs
  stats/                         # Bootstrap output
```

## Reproduction quick-start

### 1. Serve the models

```bash
# vLLM endpoints (assistant + grader/user-simulator)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct --port 8001 \
    --served-model-name qwen2.5-14b --enable-prefix-caching

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct --port 8002 \
    --served-model-name qwen2.5-7b --enable-prefix-caching
```

### 2. Set environment

```bash
export OPENAI_API_KEY=sk-local
export OPENAI_BASE_URL_14B=http://127.0.0.1:8001/v1
export OPENAI_BASE_URL_7B=http://127.0.0.1:8002/v1
export DCC_GRADER_MODEL=qwen2.5-7b
export ASSISTANT_MODEL=qwen2.5-14b
```

### 3. Run protocols

```bash
# Sharded floor + Concat ceiling
bash code/run_scripts/run_full_concat.sh

# Fresh-Last (the 5-line forgetting policy)
bash code/run_scripts/run_fresh_last.sh

# Fresh-Every (every-turn purge)
bash code/run_scripts/run_fresh_every.sh

# CacheGuard (typed forgetting)
bash code/run_scripts/run_cacheguard.sh
```

### 4. Mechanism experiments

```bash
# Marked-history baseline (H3 falsifier; predicts no recovery)
python code/lic/harness_marked_history.py --dataset_file datasets/multi_heldout48.json --N 2

# Selective-drop K-sweep (predicts cliff at K=1.0)
bash code/run_scripts/run_drop_latest.sh

# Per-turn accuracy trajectory (Appendix H)
python code/scripts/per_turn_v3.py \
    --log_dir <sharded_logs> --dataset_dir datasets \
    --tasks math data2text --n_conv 80 --max_turns 12 \
    --output_dir results/per_turn_v3

# Cross-turn sentence-level anchor identification (Appendix H)
python code/scripts/sentence_ablation_v3.py \
    --log_dir <sharded_logs> --dataset_dir datasets \
    --tasks math data2text --n_conv 20 \
    --max_asst_turns 8 --max_sentences_per_conv 30
```

## Core protocols (one-line descriptions)

| Protocol | Idea | LiC accuracy (Qwen-14B) |
|---|---|---|
| Sharded | Standard chat protocol; full history retained | 0.208 (floor) |
| Concat | All shards concatenated as one user message | 0.854 (ceiling) |
| Recap | Append summary user turn at the end; keep assistant history | 0.625 |
| Marked-History | Prefix prior assistant turns with `[OUTDATED]` | 0.307 (H3 null) |
| **Fresh-Last** | Drop all prior assistant tokens at final turn | **0.771** |
| **Fresh-Every** | Drop all prior assistant tokens at every turn | **0.875** |
| **CacheGuard** | Type each block; drop provisional claims, keep verified evidence | **0.875** |

## Key result tables

See the paper for full numbers. Summary JSONs in `results/summaries/`:

- `logs_per_turn_v3_summary.json`: per-turn accuracy trajectory (T0 0.080 → T2 0.430 peak → T3 0.321 dip)
- `logs_sentence_ablation_v3_summary.json`: 4/20 conversations have anchor sentences; mean flip rate 0.04
- `logs_fact_ctrl_v3_summary.json`: FACT positive control across math/data2text/actions

## Notes on DeepSeek-R1 (incomplete)

`results/dsr1_sharded/` and `results/dsr1_fresh_every/` contain **partial** DeepSeek-R1-Distill-Qwen-7B runs. The numbers should not be used as-is because:

1. Many sims did not complete due to server-side gateway throttling
2. DeepSeek-R1 produces `<think>...</think>` reasoning chains that the standard grader (Qwen2.5-7B) interprets differently from a direct answer
3. Code/actions tasks rely on strict output-format extraction that the reasoning-chain format doesn't satisfy

Reasoning-model evaluation requires a different harness; this is documented in the paper's Limitations §Reasoning models. The raw partial data is included for transparency; do not derive cross-model claims from it.

## Citation

Paper draft (ICLR 2026): *Beyond Retention: Active Forgetting in Multi-Turn LLM Agents*. Anonymous Authors.

## License

MIT (see LICENSE).
