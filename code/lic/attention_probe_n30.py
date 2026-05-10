"""
Attention probe scale-up: n=5 → n=30 on failed sharded conversations.

Loads Qwen2.5-7B-Instruct via HuggingFace transformers (eager attention),
runs forward pass on each failed sharded conversation up to the final user
turn, computes mean attention density (per-token) over four block types:
- system prompt
- user evidence (revealed shards)
- prior assistant provisional commitments
- current turn (final user turn before answer)

Aggregates by computing per-conversation mean, then mean ± SE across
n=30 conversations. Reports the "X% higher than user evidence" claim
with a real CI.

Usage:
    cd /root/autodl-tmp/DCC/data/lost_in_conversation
    python3 attention_probe_n30.py --N 30
"""
import os, sys, json, argparse, math
sys.path.insert(0, "/root/autodl-tmp/DCC/data/lost_in_conversation")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_block_token_ranges(tokenizer, messages):
    """Return list of (block_type, token_start, token_end) for each message,
    based on the tokenizer's chat template rendering."""
    def _enc(msgs):
        out = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        # BatchEncoding: dict-like with 'input_ids'
        if hasattr(out, "keys") and "input_ids" in out:
            ids = out["input_ids"]
        elif hasattr(out, "ids"):
            ids = out.ids
        else:
            ids = out
        # ids may be nested list, tensor, etc. — flatten to a 1-D list of ints
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)
    full = _enc(messages)
    ranges = []
    cum = []
    for i in range(1, len(messages) + 1):
        partial = _enc(messages[:i])
        cum.append(len(partial))
    prev = 0
    for i, msg in enumerate(messages):
        end = cum[i]
        role = msg["role"]
        if role == "system":
            btype = "system"
        elif role == "user":
            btype = "user_evidence" if i < len(messages) - 1 else "current_turn"
        elif role == "assistant":
            btype = "asst_commitment"
        else:
            btype = "other"
        ranges.append((btype, prev, end))
        prev = end
    return ranges, full


@torch.no_grad()
def compute_attention_density(model, tokenizer, messages, device):
    """Run forward pass with output_attentions=True, return mean attention
    density (per-token) for each block type over the final-position attention."""
    ranges, token_ids = find_block_token_ranges(tokenizer, messages)
    if len(token_ids) > 8000:  # avoid OOM on long convs
        return None
    inp = torch.tensor([token_ids], device=device)
    out = model(inp, output_attentions=True, return_dict=True)
    # attentions: tuple of [B, H, S, S] per layer
    # Aggregate: average across all layers and heads, take attention from the LAST token to all earlier tokens
    last_token_idx = inp.shape[1] - 1
    # Stack: [L, H, S]
    attn_to_last = torch.stack([a[0, :, last_token_idx, :] for a in out.attentions])
    # Mean over layers and heads -> [S]
    attn_per_token = attn_to_last.mean(dim=(0, 1))  # [S]
    # Compute density (mean attention per token) per block
    per_block = {}
    for btype, s, e in ranges:
        if e <= s:
            continue
        density = attn_per_token[s:e].mean().item()
        per_block.setdefault(btype, []).append(density)
    # Average per block type within this conv
    return {k: sum(v) / len(v) for k, v in per_block.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--model_path", type=str,
        default="/root/autodl-tmp/DCC/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--source_log", type=str,
        default="/root/autodl-tmp/DCC/data/lost_in_conversation/logs_sharded_qwen7b_mca/actions/sharded_qwen7b_mca/sharded_qwen7b_mca_actions_qwen2.5-7b.jsonl")
    ap.add_argument("--out", type=str, default="/tmp/attention_probe_n30.json")
    args = ap.parse_args()

    print(f"Loading {args.model_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="eager", trust_remote_code=True,
    ).to("cuda:0").eval()
    device = next(model.parameters()).device
    print(f"Model on {device}", flush=True)

    # Load failed sharded conversations
    convs = []
    with open(args.source_log) as f:
        for line in f:
            try:
                c = json.loads(line)
                if c.get("is_correct") is False:
                    convs.append(c)
            except Exception:
                continue
    print(f"Found {len(convs)} failed sharded convs", flush=True)
    convs = convs[:args.N]
    print(f"Probing {len(convs)} convs", flush=True)

    per_conv_results = []
    for i, c in enumerate(convs):
        msgs = []
        for m in c["trace"]:
            if m.get("role") in ("system", "user", "assistant"):
                if isinstance(m["content"], str):
                    msgs.append({"role": m["role"], "content": m["content"]})
        if len(msgs) < 3:
            continue
        try:
            res = compute_attention_density(model, tokenizer, msgs, device)
            if res is None:
                print(f"[{i+1}/{len(convs)}] skip (too long)", flush=True)
                continue
            per_conv_results.append(res)
            print(f"[{i+1}/{len(convs)}] {res}", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[{i+1}/{len(convs)}] OOM, skip", flush=True)
            continue
        except Exception as e:
            print(f"[{i+1}/{len(convs)}] err: {e}", flush=True)
            continue

    # Aggregate
    keys = ["system", "user_evidence", "asst_commitment", "current_turn"]
    summary = {}
    for k in keys:
        vals = [r[k] for r in per_conv_results if k in r]
        if not vals:
            summary[k] = None
            continue
        n = len(vals)
        mean = sum(vals) / n
        var = sum((x - mean) ** 2 for x in vals) / max(1, n - 1)
        se = math.sqrt(var / n)
        summary[k] = {"mean": mean, "se": se, "n": n}
    # Compute relative density vs user_evidence
    if summary.get("user_evidence") and summary.get("asst_commitment"):
        ratio = summary["asst_commitment"]["mean"] / summary["user_evidence"]["mean"]
        summary["asst_vs_user_ratio"] = ratio
        summary["asst_vs_user_pct_higher"] = (ratio - 1) * 100

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "per_conv": per_conv_results}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
