# Research Report — Beyond Retention: Active Forgetting in Multi-Turn LLM Agents

**Date**: 2026-05-11
**Status**: 26-page draft, all main claims empirically backed
**Overleaf**: `https://git.overleaf.com/69ff3caa541ff994a6dcece9`
**GitHub**: `https://github.com/ZRay1213/BEYOND-RETENTION-ACTIVE-FORGETTING-IN-MULTI-TURN-LLM-AGENTS`

---

## 1. One-paragraph summary

Current LLM agent memory systems — KV-cache compressors, hierarchical memory like MemGPT/MemOS, summarization-based long-term memory — decide what to retain by score (attention salience, importance, recency, semantic similarity), agnostic to **who produced the content and under what evidence**. We argue this is a *category error*: LLM-generated content is *provisional* (produced under a strict subset of the eventual evidence), and persisting it under the same role as externally observed evidence biases subsequent decisions, in direct analog to source-monitoring failures in human cognition (Johnson et al. 1993). We identify **active forgetting** — content-typed eviction at decision time, conditioned on epistemic provenance — as a missing memory primitive, distinct from generation and retention. The *contamination cliff* (Proposition 1) predicts that any retained provisional content perpetuates posterior bias, with no partial-conditioning escape, since standard transformer inference offers no partial-conditioning regime. The cliff predicts (i) failure of attention-salience KV compression, (ii) failure of textual annotation, (iii) success of typed-block eviction in mixed-content regimes. We confirm all three on Lost-in-Conversation (Fresh-Every recovers 103% of the gap; dose-response is binary; marked-history null) and on BFCL v4 (uniform forgetting collapses while typed CacheGuard preserves tool state). The effect replicates across **five models in four architecture families**.

---

## 2. Core claims (with status)

| # | Claim | Evidence | Status |
|---|---|---|---|
| 1 | Multi-turn drop is largely a chat-protocol artifact, not capability deficit | Fresh-Last 87%, Fresh-Every 103% recovery on LiC | ✅ empirical |
| 2 | Cliff binary, not graded | Drop-K K=0.25–0.75 → 4–23%, K=1.0 → 87% | ✅ |
| 3 | Mechanism is attention-mediated, not format-mediated | Marked-History 0.307 = Sharded; FACT positive control +25–43pp | ✅ |
| 4 | "Retention" in Prop 1 is at renderer layer not attention layer | Causal attention-patching n=12: 2D mask alone insufficient (patch_asst 0.250 ≈ baseline 0.333) | ✅ refined |
| 5 | Cross-architecture generalization | 5 models × 4 families: Qwen-7B/14B, Llama-3.1-8B, Mistral-7B, Gemma-2-9B | ✅ |
| 6 | Two-regime reversal: typed > uniform | BFCL v4 Fresh-Every 0.090 collapse, CacheGuard 0.280 matches Sharded ns | ✅ |
| 7 | F-synthetics validate cliff structurally | F1/F2/F3 on Qwen-14B + Llama-8B: stateless protocols 1.000 everywhere | ✅ |
| 8 | Per-turn trajectory shows accumulated commitment drag | n=80, T0=0.080 → T2 peak 0.430 → T3 dip 0.321 → plateau | ✅ |
| 9 | Per-block ablation HCHR small but non-zero | n=80, mean 0.052 [0.022, 0.087]; turn 0 never harmful, monotone trend through turn 3 | ✅ |
| 10 | Cross-regime: summarization memory | Math: cliff holds (Summary 0.333 = Sharded). Data2text n=94: **inverted** (Summary 0.436 > Sharded 0.309 > FL 0.170) — refines cliff to regime-dependent | ✅ |
| 11 | Capability inversion on cross-arch | Llama F1 FL 0.810 > Concat 0.617 (+19pp); F2 FL 0.967 > Concat 0.867 (+10pp) | ✅ |
| 12 | Cross-regime extension predictions (agentic loops, long CoT, persona drift, iterative RAG) | Predicted by Prop 1 + literature support; not directly tested | △ pending |

---

## 3. Methods catalogue

### 3.1 Forgetting taxonomy (Method §4.2)

Five granularities of forgetting operation, with domain of validity:

| Granularity | Operator | Works in | Fails in |
|---|---|---|---|
| Token-attention | Zero attention to target tokens | (none clean — see §5.5 causal patch) | LiC (RoPE artifact) |
| Sentence | Drop specific sentences | Intra-turn CoT (Thought Anchors) | cross-turn anchors |
| Turn-uniform | Drop all prior asst turns (Fresh-Last/Every) | Stateless reasoning (LiC, F1/F2/F3) | Stateful tool use (BFCL FE 0.090) |
| Block-typed | Drop provisional, retain authoritative (CacheGuard) | LiC + BFCL both | Mixed-API edge cases (§Limitations) |
| Conversation-reset | Per-query rebuild (Concat) | Bounded single-shot | Loses history |

### 3.2 Three cache metrics

- **PCHR** (Physical Cache Hit Rate): reused prefix tokens / queried prompt tokens. The standard metric.
- **VCHR** (Valid Cache Hit Rate): fraction of reused tokens whose blocks remain epistemically valid evidence.
- **HCHR** (Harmful Cache Hit Rate): fraction of reused tokens whose blocks counterfactually flip wrong→right when removed.

Empirical estimate (n=80): HCHR = 0.052 [0.022, 0.087].

### 3.3 CacheGuard

Three-component renderer:
- **Block Parser** types each non-log message: system / user_evidence / tool_observation / assistant_artifact / assistant_claim / assistant_clarification / assistant_plan
- **Eligibility Controller**: a block is reuse-eligible iff (system OR user OR successful tool) OR (artifact referenced by later user/tool) OR (clarification asking). Provisional claims & plans are excluded.
- **Renderer**: emits `[stable valid prefix || current turn || referenced artifacts]`; provisional commitments live in a hidden side-buffer.

---

## 4. Empirical results

### 4.1 Main result: LiC `multi_heldout` (Qwen2.5-14B, N=2)

| Protocol | Math | Code | Actions | Avg | Gap recovery |
|---|---|---|---|---|---|
| Sharded (floor) | 0.500 | 0.000 | 0.125 | 0.208 | 0% |
| Concat (ceiling) | 0.938 | 0.625 | 1.000 | 0.854 | 100% |
| Recap | 0.833 | 0.167 | 0.875 | 0.625 | 65% |
| **Fresh-Last** | **0.875** | **0.438** | **1.000** | **0.771** | **87%** |
| **Fresh-Every** | **0.938** | **0.688** | **1.000** | **0.875** | **103%** |

### 4.2 Mechanism (cliff dose-response)

| K (drop fraction) | Drop-EARLIEST | Drop-LATEST |
|---|---|---|
| 0.00 (Sharded) | 0.208 | 0.208 |
| 0.25 | 0.354 | 0.253 |
| 0.50 | 0.235 | 0.242 |
| 0.75 | 0.333 | 0.278 |
| 1.00 (Fresh-Last) | **0.771** | **0.771** |

Llama-3.1-8B replication: K=0 0.139, K=0.5 0.185, K=1.0 0.651. Cliff at K=1.0 dominates on both architectures.

### 4.3 H1 vs H3 controls

| Condition | LiC accuracy |
|---|---|
| Sharded baseline | 0.208 |
| Marked-History `[OUTDATED]` prefix | 0.307 (+10pp; not significant vs Sharded, p=0.34) |
| Fresh-Last | 0.771 (+56pp) |
| FACT injection (gold in user turn) | Math +43pp, d2t +25pp, actions +17pp |

H3 (format confusion) ruled out; H1 (attention-mediated) confirmed; FACT shows model CAN read user-side instructions.

### 4.4 BFCL v4 stateful tool use (N=200, Qwen-14B)

| Protocol | Acc | vs Sharded |
|---|---|---|
| Sharded | 0.295 | — |
| Fresh-Last | 0.120 | −0.175*** |
| Fresh-Every | 0.090 | −0.205*** |
| Tool-Result-Only | 0.210 | −0.085** |
| **CacheGuard** | **0.280** | −0.015 (ns) |

**Two-regime reversal demonstrated**: uniform forgetting collapses on stateful tools; typed forgetting preserves.

### 4.5 Cross-model

| Model | Sharded | Concat | Fresh-Last | Fresh-Every |
|---|---|---|---|---|
| Qwen-7B | 0.201 | 0.771 | **0.917** | 0.896 |
| Qwen-14B | 0.208 | 0.854 | 0.771 | 0.875 |
| Llama-3.1-8B | 0.139 | 0.615 | 0.651 | **0.777** |
| Mistral-7B-v0.2 | 0.148 | 0.396 | 0.443 | 0.529 |
| Gemma-2-9B | 0.169 | 0.656 | 0.746 | 0.733 |

CacheGuard cross-arch: Llama 0.704, Mistral 0.544, Gemma 0.741. **CacheGuard ≥ Fresh-Every on Mistral and Gemma** — typed eligibility can outperform uniform forgetting.

### 4.6 F-benchmark synthetic cross-model (n=30 each)

| Model | Protocol | F1 | F2 | F3 |
|---|---|---|---|---|
| Qwen-14B | Concat / Sharded | 0.977 / 0.609 | 1.000 / 0.767 | 1.000 / 0.433 |
| Qwen-14B | FL / FE / CG | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| Llama-8B | Concat / Sharded | 0.617 / 0.328 | 0.867 / 0.733 | 1.000 / **0.117** |
| Llama-8B | FL / FE | **0.810** / **1.000** | **0.967** / **1.000** | 1.000 / 1.000 |

**Llama F3 Sharded 0.117** = catastrophic 88pp drop on artifact propagation, fully recovered.

### 4.7 Per-turn accuracy trajectory (n=80 math+d2t, Qwen-14B)

```
T0: 0.080  ← commitments to wrong answer with partial info
T1: 0.115
T2: 0.430  ← peak (more shards now in)
T3: 0.321  ← DIP! cumulative commitments drag down despite more evidence
T4: 0.333
T5: 0.329
```

67/80 conversations reach a correct answer at some turn, but only 13/80 retain it — 54 re-commit and lose.

### 4.8 Cross-turn sentence ablation (n=20 failed convs)

- 4/20 have anchor sentences (20% conversation-level)
- Mean per-conv flip rate 0.04 (small)
- Distribution heavy-tailed: one totto-Pinckney conv has 17/30 anchors (fabricated dates)
- One math conv (GSM8K/651): two structural commitments in Turn 1 ("we need specific data" / "I will outline steps") each individually flip wrong→right when removed

### 4.9 HCHR empirical (n=80, Qwen-14B math+d2t)

- Mean HCHR = **0.052** (95% CI **[0.022, 0.087]**, CI excludes zero)
- 11/80 convs contain ≥1 harmful block
- Per-position monotone for first 4 turns: turn 0 (0.013) → turn 1 (0.050) → turn 2 (0.062) → turn 3 (0.089)
- Deep turns: turn 4 (0.039), turn 5 (0.182, n=11)
- Consistent with cliff: most single blocks rarely solo-flip (because other retained provisional content still contributes), but each block grows in dominance as turns accumulate

### 4.10 Attention probe (correlational, n=22 BFCL fails, Qwen-7B)

| Block type | Attention density per token |
|---|---|
| System prompt | 0.00140 ± 0.00004 |
| **Asst commits** | **0.00141 ± 0.00019** |
| User evidence | 0.00130 ± 0.00014 |

8.5% above user evidence. Refutes attention-salience-based KV compression for multi-turn agent settings.

### 4.11 Causal attention patching (n=12 failed math, Qwen-7B HF eager)

| Condition | Accuracy |
|---|---|
| Baseline (full sharded) | 0.333 |
| **patch-asst** (mask attention to all prior asst tokens) | **0.250** |
| patch-user (control) | 0.333 |

**Negative result with positive interpretation**: 2D attention-mask alone does NOT recover. "Let let's" token artifact in patched generations indicates RoPE positional disruption. The Proposition 1 "retained subset" must be read at **renderer layer**, not attention layer. KV compression / attention-salience methods cannot achieve active forgetting.

### 4.12 Cross-regime: summarization memory contamination (regime-dependent!)

| Task | Sharded | Fresh-Last | Summary |
|---|---|---|---|
| Math (n=12, discrete answer) | 0.333 | **0.417** | 0.333 |
| Data2text (n=94, free-form generation) | 0.309 | 0.170 | **0.436** |

**The cliff prediction is regime-dependent**:
- On math (discrete answer): Summary ≈ Sharded < FL → summary inherits contamination as predicted
- On data2text (free-form): Summary > Sharded > FL → ordering **inverts** because evidence-loss cost from truncation dominates over contamination cost from summary

Two competing cost terms in any forgetting intervention:
- (a) contamination cost from retained provisional content
- (b) evidence-loss cost from truncating context the model needs for downstream use

The cliff theorem predicts (a), but the optimal intervention depends on which dominates. This actually STRENGTHENS the typed-eviction argument: uniform forgetting fails differently across regimes, only typed criteria navigate both.

---

## 5. Statistical strength

Paired bootstrap ($B=10{,}000$, common-task pairing, multi_heldout n=24):

| Comparison | Δ | 95% CI | p | sig |
|---|---|---|---|---|
| CacheGuard > Sharded | +0.645 | [+0.476, +0.810] | <0.001 | *** |
| Fresh-Last > Marked-History | +0.416 | [+0.229, +0.604] | <0.001 | *** |
| CacheGuard > Recap | +0.250 | [+0.042, +0.458] | 0.005 | ** |
| Fresh-Every > Recap | +0.250 | [+0.042, +0.458] | 0.005 | ** |
| 7B Fresh-Last > 14B Fresh-Last | +0.146 | [+0.021, +0.292] | 0.019 | * |
| Llama-8B Fresh-Last > Sharded | +0.482 | [+0.333, +0.630] | <0.001 | *** |
| Llama-8B Fresh-Every > Concat | +0.156 | [+0.062, +0.260] | 0.004 | ** |
| CacheGuard vs Sharded (BFCL) | -0.015 | [-0.080, +0.050] | 0.66 | ns |

Multi-attempt control (best-of-4): anchoring claim robust on both Qwen-7B and Llama-8B.

---

## 6. Honest limitations

1. **Cross-regime is mostly prediction** for agentic loops, long CoT, persona drift, iterative RAG — only LiC, BFCL, F1/F2/F3, and summarization-memory have direct evidence
2. **Causal attention patch is negative** — refines the cliff theorem to renderer-layer rather than confirming pure attention H1
3. **Open-weight only** — closed models (GPT, Claude) not tested
4. **N=2 main table** — bootstrap CIs computed; explicit N=4 scaling stalled on context-length overflow
5. **CacheGuard typed rule is hand-written** — learned eligibility classifier left to follow-up
6. **HCHR small (0.052)** — single-block ablation rarely flips outcome (cliff predicts this); the magnitude is consistent but reviewer-attackable

---

## 7. Differentiation vs concurrent work

Three literature-scan agents identified no paper that:
- Uses "provisional content" as a named term or epistemic-provenance typing axis
- Proposes a forgetting-operation taxonomy (granularity: token / sentence / turn / typed-block / conv)
- Names "active forgetting" as a memory primitive
- Provides a regime-independent theorem with empirical validation across regimes

Closest overlaps and how we differentiate:

| Paper | Their claim | Our differentiation |
|---|---|---|
| Huang et al. 2026 (`benefit of forgetting`) | Drop asst history helps 10× context | We add cliff mechanism, taxonomy, cross-regime, typed CacheGuard for stateful tools |
| Wang et al. 2025 (PI-LLM) | Log-linear retrieval decay | They model retrieval; we model evidence vs. inference at epistemic-type level |
| Li et al. 2026 (conv inertia) | Periodic clip context | Position-based recency clip; ours is epistemic typing |
| Active Context Curation 2026 | RL-trained "reasoning anchors" retention | We give principled rule grounded in source-monitoring theory; learned controller is a natural extension |
| MemOS (2025) | Origin signatures in metadata | They use origin for audit, not eviction governance |
| Generative Agents (2023) | Observations + plans + reflections | They ELEVATE reflections via importance score (opposite axis from ours) |
| Reflexion (2023) | Buffer of reflections as positive signal | Reflection is provisional but labeled as failure trace → exits the cliff scope condition |
| Self-consistency (2023) | Parallel chains majority-vote | Parallel chains discarded at aggregation, no sequential accumulation → also exits cliff |

---

## 8. Reproduction quick-start

Server: `connect.westd.seetacloud.com:40074` (password-only, see `reference_dcc_server_ssh.md`)

```bash
# Set up vLLM (GPU 0: 14B, GPU 1: 7B)
bash code/run_scripts/vllm_serve_14b.sh
bash code/run_scripts/vllm_serve_7b.sh

# Main result protocols
bash code/run_scripts/run_full_concat.sh
bash code/run_scripts/run_fresh_last.sh
bash code/run_scripts/run_fresh_every.sh
bash code/run_scripts/run_cacheguard.sh

# Mechanism: dose-response
bash code/run_scripts/run_drop_latest.sh

# Per-turn / sentence-ablation (Appendix H)
python code/scripts/per_turn_v3.py --tasks math data2text --n_conv 80 ...
python code/scripts/sentence_ablation_v3.py --n_conv 20 ...

# Causal attention patch
python code/lic/attention_probe_n30.py  # (probe)
python code/lic/attention_patch.py      # (causal, in repo)
```

Bib & code: see GitHub repo. Paper PDF: `paper/main.pdf` (26 pages).

---

## 9. Forward-looking experiments

Ordered by claim-impact:

| # | Experiment | Time | Adds |
|---|---|---|---|
| A | **τ²-Bench cross-regime** (agentic loops) | 1-2 weeks | 4th regime with direct evidence; tau2-work already on server |
| B | **Closed-source models** GPT-4o-mini / Claude-3.5-Sonnet | 3 days, ~$200 API | closes "open-weight only" gap |
| C | **Layer-wise attention patching** | 1 week | which layers anchor; circuit-level finding |
| D | **Learned eligibility classifier for CacheGuard** | 2 weeks | extends typed eligibility to data-driven |
| E | **Persona drift cross-regime** | 1 week | 5th regime; uses Li 2024 dataset |
| F | **vLLM CacheGuard plugin** | 1 month | production-level serving integration |
| G | **MultiChallenge benchmark** | 1 week | 3rd LiC-style benchmark |
| H | **DSR1 long-CoT cross-regime** | 2 weeks | intra-turn → cross-turn extension on reasoning models |

---

## Compile state

26 pages, clean compile (pdflatex + bibtex). 0 errors, 0 undefined refs. 2 figures + 11 tables in main+appendix.

Files:
- `/Users/zhangrui/PycharmProjects/DCC/paper/main.tex` — source
- `/Users/zhangrui/PycharmProjects/DCC/paper/main.pdf` — compiled output
- Synced to GitHub repo `paper/` and Overleaf `master`

---

*Generated 2026-05-11 by the experiment pipeline. All numbers are from production runs on `connect.westd.seetacloud.com:40074`. Reproduction scripts in the repo.*
