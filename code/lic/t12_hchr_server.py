"""T1.2 HCHR empirical pilot — per-block ablation on failed sharded conversations.

For each failed conversation: drop one prior assistant block at a time and re-score.
HCHR = fraction of blocks whose single-block removal recovers accuracy.

Key claim: even individual blocks causally contaminate. If HCHR is high (many blocks
individually flip to correct), it strengthens the Proposition 1 argument beyond the
dose-response cliff.

Sources: failed sharded conversations from logs_multi_baseline (multi_heldout48).
"""
import os, sys, json, glob, copy, random
sys.path.insert(0, '/root/autodl-tmp/DCC/data/lost_in_conversation')
os.chdir('/root/autodl-tmp/DCC/data/lost_in_conversation')

from model_openai import generate
from system_agent import SystemAgent
from tasks import get_task
from utils import date_str

ASSISTANT_MODEL = os.environ.get('ASSISTANT_MODEL', 'qwen2.5-14b-tool')
SYSTEM_MODEL    = os.environ.get('SYSTEM_MODEL', 'qwen2.5-7b')
LOG_BASE        = '/root/autodl-tmp/DCC/data/lost_in_conversation'
OUT_DIR         = os.path.join(LOG_BASE, 'logs_hchr_pilot')
os.makedirs(OUT_DIR, exist_ok=True)

def load_failed_sharded(base_dir, n_max=40):
    recs = []
    for f in glob.glob(base_dir + '/logs_multi_baseline/**/sharded/*.jsonl', recursive=True):
        with open(f) as fp:
            for line in fp:
                rec = json.loads(line)
                score = rec.get('score', None)
                is_ok = rec.get('is_correct')
                if is_ok == False or (is_ok is None and score is not None and float(score) < 0.5):
                    rec['_src'] = f
                    recs.append(rec)
    random.seed(99); random.shuffle(recs)
    return recs[:n_max]

def trace_to_msgs(trace):
    return [m for m in trace if m.get('role') in ('system', 'user', 'assistant')]

def get_gold(task_name, task_id):
    try:
        sample = get_task(task_name).get_sample(task_id)
        for k in ('answer', 'gold', 'ground_truth', 'reference', 'target'):
            v = sample.get(k)
            if v: return str(v), sample
    except:
        pass
    return None, None

def call_and_score(msgs_context, task_name, task_id, gold_str, sample):
    while msgs_context and msgs_context[-1]['role'] == 'assistant':
        msgs_context = msgs_context[:-1]
    try:
        resp = generate(msgs_context, model=ASSISTANT_MODEL, temperature=1.0,
                        return_metadata=True, max_tokens=800)
        response_text = resp['message']
        full_trace = msgs_context + [{'role': 'assistant', 'content': response_text, 'timestamp': date_str()}]
        sa = SystemAgent(task_name, SYSTEM_MODEL, sample)
        sv_resp, _ = sa.verify_system_response(full_trace)
        if sv_resp['response_type'] == 'answer_attempt':
            ea = sa.extract_answer(full_trace)
            ev = get_task(task_name).evaluator_function(ea, sample)
            score = float(ev.get('score', 0))
            is_correct = ev.get('is_correct')
            if score >= 1.0 and not is_correct: is_correct = True
            return float(score), response_text[:200]
    except Exception as e:
        print(f'    score error: {e}')
    return 0.0, ''

def drop_asst_block(msgs, block_idx):
    """Drop the block_idx-th assistant message (0-indexed)."""
    out = []
    asst_count = 0
    for m in msgs:
        if m['role'] == 'assistant':
            if asst_count == block_idx:
                asst_count += 1
                continue
            asst_count += 1
        out.append(m)
    return out

def main():
    recs = load_failed_sharded(LOG_BASE, n_max=40)
    print(f'Loaded {len(recs)} failed conversations')

    all_results = []
    conv_hchr = []

    for ci, rec in enumerate(recs):
        task_name = rec.get('task')
        task_id   = rec.get('task_id')
        gold_str, sample = get_gold(task_name, task_id)
        if gold_str is None:
            print(f'  [{ci}] skip gold')
            continue

        msgs = trace_to_msgs(rec.get('trace', []))
        asst_blocks = [i for i, m in enumerate(msgs) if m['role'] == 'assistant']
        n_asst = len(asst_blocks)
        if n_asst < 2:
            print(f'  [{ci}] skip: only {n_asst} asst blocks')
            continue

        # Don't ablate the LAST assistant block (that's the one we're re-generating)
        # Ablate the prior n_asst-1 blocks
        n_prior = n_asst - 1

        flips = 0
        block_results = []

        for bi in range(n_prior):
            dropped = drop_asst_block(msgs, bi)
            score, pred = call_and_score(dropped, task_name, task_id, gold_str, sample)
            flip = (score >= 0.5)
            if flip: flips += 1
            block_results.append({'block_idx': bi, 'score': score, 'flip': flip})

        hchr = flips / n_prior
        conv_hchr.append(hchr)

        entry = {
            'conv_idx': ci, 'task': task_name, 'task_id': task_id,
            'n_asst': n_asst, 'n_prior': n_prior,
            'flips': flips, 'hchr': hchr,
            'block_results': block_results,
        }
        all_results.append(entry)

        mean_hchr = sum(conv_hchr) / len(conv_hchr)
        print(f'  [{ci+1}/{len(recs)}] task={task_name} n_prior={n_prior} '
              f'flips={flips}/{n_prior} hchr={hchr:.2f} | running_mean={mean_hchr:.3f}')

    n = len(all_results)
    mean_hchr = sum(conv_hchr) / n if n else 0
    med_hchr  = sorted(conv_hchr)[n//2] if n else 0
    print(f'\nFINAL (n={n}):')
    print(f'  mean HCHR = {mean_hchr:.4f}')
    print(f'  median HCHR = {med_hchr:.4f}')
    print(f'  HCHR > 0 (any block flips): {sum(h > 0 for h in conv_hchr)}/{n}')
    print(f'  HCHR >= 0.5 (majority flip): {sum(h >= 0.5 for h in conv_hchr)}/{n}')
    print(f'  HCHR = 1.0 (all blocks flip): {sum(h == 1.0 for h in conv_hchr)}/{n}')

    out_f = os.path.join(OUT_DIR, 'hchr_results.jsonl')
    with open(out_f, 'w') as fp:
        for r in all_results:
            fp.write(json.dumps(r) + '\n')
    
    summary = {
        'n': n, 'mean_hchr': mean_hchr, 'median_hchr': med_hchr,
        'hchr_gt0': sum(h > 0 for h in conv_hchr),
        'hchr_ge05': sum(h >= 0.5 for h in conv_hchr),
        'hchr_eq1': sum(h == 1.0 for h in conv_hchr),
    }
    with open(os.path.join(OUT_DIR, 'hchr_summary.json'), 'w') as fp:
        json.dump(summary, fp, indent=2)
    print(f'\nResults → {out_f}')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
