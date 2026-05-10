"""T1.4 FACT positive control — uses existing DCC harness infrastructure.

Tests: model follows [FACT: gold] in user turn (high acc) vs ignores [OUTDATED] on
asst turns (low acc, ~same as marked-history null). Gap confirms H1 (attention/trust
asymmetry, not general instruction-blindness).

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
OUT_DIR         = os.path.join(LOG_BASE, 'logs_fact_ctrl')
os.makedirs(OUT_DIR, exist_ok=True)

FACT_TMPL    = '\n\n[FACT: The correct answer to this problem is: {gold}. You MUST use this in your final answer.]'
OUTDATED_PFX = '[OUTDATED: This prior response may be incorrect due to incomplete context.]\n\n'

def load_failed_sharded(base_dir, n_max=60):
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
    random.seed(42); random.shuffle(recs)
    return recs[:n_max]

def trace_to_msgs(trace):
    """Extract system/user/assistant messages from trace (filter log entries)."""
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

def inject_fact_last_user(msgs, gold):
    msgs = copy.deepcopy(msgs)
    for i in range(len(msgs)-1, -1, -1):
        if msgs[i]['role'] == 'user':
            msgs[i]['content'] = msgs[i]['content'] + FACT_TMPL.format(gold=gold)
            break
    return msgs

def inject_outdated_all_asst(msgs):
    msgs = copy.deepcopy(msgs)
    for m in msgs:
        if m['role'] == 'assistant':
            m['content'] = OUTDATED_PFX + m['content']
    return msgs

def call_and_score(msgs_context, task_name, task_id, gold, sample):
    """Re-issue the final assistant turn and score."""
    # Remove last assistant turn if present so we're generating fresh
    while msgs_context and msgs_context[-1]['role'] == 'assistant':
        msgs_context = msgs_context[:-1]
    
    resp = generate(msgs_context, model=ASSISTANT_MODEL, temperature=1.0,
                    return_metadata=True, max_tokens=800)
    response_text = resp['message']
    
    full_trace = msgs_context + [{'role': 'assistant', 'content': response_text, 'timestamp': date_str()}]
    
    try:
        sa = SystemAgent(task_name, SYSTEM_MODEL, sample)
        sv_resp, _ = sa.verify_system_response(full_trace)
        if sv_resp['response_type'] == 'answer_attempt':
            ea = sa.extract_answer(full_trace)
            ev = get_task(task_name).evaluator_function(ea, sample)
            is_correct = ev.get('is_correct')
            score = float(ev.get('score', 0))
            if score >= 1.0 and not is_correct: is_correct = True
            return float(score), response_text[:300]
    except Exception as e:
        pass
    return 0.0, response_text[:300]

def main():
    recs = load_failed_sharded(LOG_BASE, n_max=60)
    print(f'Loaded {len(recs)} failed conversations')

    results = []
    fact_scores, outdated_scores, orig_scores = [], [], []

    for ci, rec in enumerate(recs):
        task_name = rec.get('task')
        task_id   = rec.get('task_id')
        gold_str, sample = get_gold(task_name, task_id)
        if gold_str is None:
            print(f'  [{ci}] skip: cannot get gold for task={task_name} id={task_id}')
            continue

        msgs = trace_to_msgs(rec.get('trace', []))
        if len(msgs) < 3:
            print(f'  [{ci}] skip: too few messages ({len(msgs)})')
            continue

        n_asst = sum(1 for m in msgs if m['role'] == 'assistant')

        # Condition A: original sharded (baseline — already failed, should be ~0)
        orig_s, orig_pred = call_and_score(msgs, task_name, task_id, gold_str, sample)
        orig_scores.append(orig_s)

        # Condition B: FACT injected into last user turn (model should use it)
        fact_msgs = inject_fact_last_user(msgs, gold_str)
        fact_s, fact_pred = call_and_score(fact_msgs, task_name, task_id, gold_str, sample)
        fact_scores.append(fact_s)

        # Condition C: OUTDATED marker on all asst turns (should stay ~0 per H1 prediction)
        outdated_msgs = inject_outdated_all_asst(msgs)
        outdated_s, outdated_pred = call_and_score(outdated_msgs, task_name, task_id, gold_str, sample)
        outdated_scores.append(outdated_s)

        entry = {
            'conv_idx': ci, 'task': task_name, 'task_id': task_id,
            'gold': gold_str, 'n_asst': n_asst,
            'orig_score': orig_s, 'fact_score': fact_s, 'outdated_score': outdated_s,
            'fact_pred': fact_pred, 'outdated_pred': outdated_pred,
        }
        results.append(entry)

        if (ci+1) % 5 == 0:
            n = len(fact_scores)
            print(f'  [{ci+1}/{len(recs)}] FACT={sum(fact_scores)/n:.3f} '
                  f'OUTDATED={sum(outdated_scores)/n:.3f} ORIG={sum(orig_scores)/n:.3f}')

    n = len(results)
    fa = sum(fact_scores)/n if n else 0
    oa = sum(outdated_scores)/n if n else 0
    sa_ = sum(orig_scores)/n if n else 0

    print(f'\nFINAL (n={n}):')
    print(f'  FACT (gold injected):   {fa:.4f}')
    print(f'  OUTDATED (marked-hist): {oa:.4f}')
    print(f'  ORIG (baseline shrd):   {sa_:.4f}')
    print(f'  Delta FACT-OUTDATED:    {fa-oa:+.4f}')
    if fa > 0.5 and (fa - oa) > 0.2:
        print('  → H1 CONFIRMED: model follows user [FACT] but ignores asst [OUTDATED]')
    elif fa < 0.3:
        print('  → Neither hint works well — possible format/attention issue')
    else:
        print(f'  → Ambiguous: inspect individual results')

    out_f = os.path.join(OUT_DIR, 'fact_ctrl_results.jsonl')
    with open(out_f, 'w') as fp:
        for r in results:
            fp.write(json.dumps(r) + '\n')
    
    summary = {'n': n, 'fact_avg': fa, 'outdated_avg': oa, 'orig_avg': sa_,
                'delta': fa-oa, 'h1_confirmed': bool(fa > 0.5 and fa-oa > 0.2)}
    with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as fp:
        json.dump(summary, fp, indent=2)
    print(f'\nResults → {out_f}')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
