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
  lic/                              # 73 files — Lost-in-Conversation simulators + harnesses
    simulator_{sharded,full,recap,snowball,...}.py  Base protocol implementations
    harness_{fresh_last,fresh_every,cacheguard,    Protocol harnesses
             marked_history,drop_latest,
             selective_drop,counterfactual}.py
    cacheguard_bfcl.py / score_bfcl.py /            BFCL v4 stateful tool-use suite
        bootstrap_bfcl.py / analyze_bfcl.py
    attention_probe_n30.py                          Attention probe (n=22 after OOM filter)
    model_openai.py                                 Multi-endpoint vLLM/OpenAI client
    state_tracker.py / system_agent.py              Stage-2 state tracking (legacy)
    proposer.py / proposer_v2.py /                  Meta-Harness style mutator
        proposer_gepa_style.py
    t12_hchr_server.py / t14_fact_ctrl_*.py         HCHR + FACT positive control
  scripts/                          # 16 top-level analysis scripts
    per_turn_v3.py                                  Per-turn accuracy trajectory (Appendix H)
    sentence_ablation_v3.py                         Cross-turn anchor sentence ID (Appendix H)
    bootstrap_lic.py                                Paired-bootstrap stats (Appendix B)
    score_lic.py / score_bestofN.py                 LiC scoring + best-of-N control
    train_bandit_v2.py / extract_features_v2.py     Bandit controller (legacy Stage-3)
    mine_few_shot.py / per_instance.py
  scripts_extra/                    # 23 cross-model experiment launchers + task splits
    exp_llama_*.sh / exp_gemma*.sh / exp_newmodels.sh   Cross-model experiment scripts
    exp_lic14b{,_v2}.sh / exp_concat_7bsh.sh             Main-result launchers
    exp_cf_v2.sh / exp_cg_llama.sh                       Counterfactual + CacheGuard cross-arch
    auto_pipeline*.sh / auto_g4_after_g1c1.sh            Multi-generation harness evolution
    deploy_deepseek_r1.sh / run_deepseek_r1_bench.sh     DSR1 vLLM deployment
    code_{eval,sel}.json / eval_tasks.json /             Task-split JSONs (3-way split)
        f1_heldout.json / multi_heldout.json
    few_shot_examples.json                              Few-shot for LLM-classifier controller
  run_scripts/                      # 69 protocol-level launchers
    run_{fresh_every,fresh_last,cacheguard,full_concat,
         drop_latest,selective_drop,marked_history,
         f1_all,f2_all,f3_all,recap_sh,counterfactual}.sh
    vllm_serve_{14b,7b,32b}*.sh / vllm_{llama,mistral,gemma}*.sh   Model serving
  harness_gepa_population/          # GEPA-style evolved harnesses + rationales
  controllers/                      # Trained bandit Q-models (.joblib)

datasets/                           # 8 JSON dataset files used in experiments
  multi_heldout48.json              48-inst multi-task held-out (8 per × 6 tasks)
  multi_mca24.json                  24-inst cross-model (math/code/actions × 8)
  sharded_stage1_K{20,25}.json      Stage-1 sharded data
  sharded_f1_additive.json          F1 synthetic additive-constraints (Appendix A)
  sharded_f2_corrective.json        F2 synthetic corrective-constraints (Appendix A)

results/
  summaries/                        # summary.json from each Appendix-H run
  per_turn_v3/                      n=80 per-turn accuracy trajectory (Fig 7)
  sentence_ablation_v3/             n=20 cross-turn anchor sentences (Appendix H)
  bfcl/                             BFCL v4 N=200 jsonl per protocol
                                    (cacheguard_n200, fresh-every_n200, longctx, missparam, ...)
  cross_model_aggregates/           25 sub-folders, one per (model × protocol):
                                    Llama / Mistral / Gemma / Qwen-7B × Sharded / Concat /
                                    Fresh-Every / Fresh-Last / CacheGuard / Marked-History;
                                    plus drop-K and drop-latest sweeps
  sharded_stage1_traces/            Source traces for per-turn + sentence-ablation re-analysis
                                    (math; data2text; actions skipped due to 250-turn outliers)
  dsr1_sharded/, dsr1_fresh_every/  DeepSeek-R1-Distill-7B PARTIAL results (see Notes)
  stats/                            Bootstrap output (empty placeholder)

paper/                              # Paper source (ICLR 2026 template)
  main.tex / main.pdf               21-page paper
  iclr2026_conference.{sty,bst,bib} ICLR 2026 style files + bib
  math_commands.tex / fancyhdr.sty / natbib.sty
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
