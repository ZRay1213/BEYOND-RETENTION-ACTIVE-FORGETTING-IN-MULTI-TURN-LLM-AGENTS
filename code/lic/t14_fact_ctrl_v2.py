"""T1.4 FACT positive control v2 — handles all task types."""
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
OUT_DIR         = os.path.join(LOG_BASE, 'logs_fact_ctrl_v2')
os.makedirs(OUT_DIR, exist_ok=True)

FACT_TMPL    = '\n\n[FACT: The correct answer to this problem is: {gold}. You MUST use this in your final answer.]'
OUTDATED_PFX = '[OUTDATED: This prior response may be incorrect due to incomplete context.]\n\n'

def get_gold(task_name, task_id):
    """Return (gold_str, sample) for any supported task type."""
    try:
        sample = get_task(task_name).get_sample(task_id)
    except Exception as e:
        return None, None
    
    # math: answer ends with #### <number>
    if 'answer' in sample:
        ans = sample['answer']
        if '####' in ans:
            return ans.split('####')[1].strip(), sample
        return str(ans)[:200], sample
    
    # data2text: references list
    refs = sample.get('references')
    if refs and isinstance(refs, list) and refs:
        return refs[0][:300], sample
    
    # actions: reference_answer JSON
    ref_ans = sample.get('reference_answer')
    if ref_ans is not None:
        return json.dumps(ref_ans)[:300], sample
    
    # code: skip (no simple gold)
    if task_name == 'code':
        return None, None
    
    # summary: skip (complex eval)
    if task_name == 'summary':
        return None, None
    
    # fallback: any string field
    for k in ('gold', 'ground_truth', 'target'):
        v = sample.get(k)
        if v:
            return str(v)[:200], sample
    
    return None, None

def load_failed_sharded(base_dir, n_max=60):
    recs = []
    for f in glob.glob(base_dir + '/logs_multi_baseline/**/sharded/*.jsonl', recursive=True):
        with open(f) as fp:
            for line in fp:
                rec = json.loads(line)
                score = rec.get('score', None)
                is_ok = rec.get('is_correct')
                if is_ok == False or (is_ok is None and score is not None and float(score) < 0.5):
                    # Skip tasks with no simple gold
                    if rec.get('task') not in ('code', 'summary'):
                        rec['_src'] = f
                        recs.append(rec)
    random.seed(42); random.shuffle(recs)
    return recs[:n_max]

def trace_to_msgs(trace):
    return [m for m in trace if m.get('role') in ('system', 'user', 'assistant')]

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

def call_and_score(msgs_context, task_name, task_id, gold_str, sample):
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
            score = float(ev.get('score', 0))
            if score >= 1.0 and not ev.get('is_correct'): ev['is_correct'] = True
            return score, response_text[:300]
    except Exception as e:
        print(f'    score err: {e}')
    return 0.0, response_text[:300]

def main():
    recs = load_failed_sharded(LOG_BASE, n_max=60)
    print(f'Loaded {len(recs)} failed conversations (math+data2text+actions)')

    results = []
    fact_s, out_s, orig_s = [], [], []

    for ci, rec in enumerate(recs):
        task_name = rec.get('task')
        task_id   = rec.get('task_id')
        gold_str, sample = get_gold(task_name, task_id)
        if gold_str is None:
            print(f'  [{ci}] skip: no gold for task={task_name}')
            continue

        msgs = trace_to_msgs(rec.get('trace', []))
        if len(msgs) < 3:
            continue

        n_asst = sum(1 for m in msgs if m['role'] == 'assistant')

        os_, op = call_and_score(list(msgs), task_name, task_id, gold_str, sample)
        orig_s.append(os_)

        fs, fp = call_and_score(inject_fact_last_user(msgs, gold_str), task_name, task_id, gold_str, sample)
        fact_s.append(fs)

        xs, xp = call_and_score(inject_outdated_all_asst(msgs), task_name, task_id, gold_str, sample)
        out_s.append(xs)

        results.append({'conv_idx': ci, 'task': task_name, 'task_id': task_id,
                        'n_asst': n_asst,
                        'orig_score': os_, 'fact_score': fs, 'outdated_score': xs})

        if (ci+1) % 5 == 0:
            n = len(fact_s)
            print(f'  [{ci+1}/{len(recs)}] FACT={sum(fact_s)/n:.3f} OUT={sum(out_s)/n:.3f} ORIG={sum(orig_s)/n:.3f}')

    n = len(results)
    fa = sum(fact_s)/n if n else 0
    oa = sum(out_s)/n if n else 0
    sa_ = sum(orig_s)/n if n else 0

    print(f'\nFINAL (n={n}):')
    print(f'  FACT (gold injected):   {fa:.4f}')
    print(f'  OUTDATED (marked-hist): {oa:.4f}')
    print(f'  ORIG (sharded base):    {sa_:.4f}')
    print(f'  Delta FACT-OUTDATED:    {fa-oa:+.4f}')

    out_f = os.path.join(OUT_DIR, 'fact_ctrl_results.jsonl')
    with open(out_f, 'w') as fp:
        for r in results:
            fp.write(json.dumps(r) + '\n')
    
    by_task = {}
    for r in results:
        t = r['task']
        if t not in by_task: by_task[t] = []
        by_task[t].append(r)

    summary = {'n': n, 'fact_avg': fa, 'outdated_avg': oa, 'orig_avg': sa_,
                'delta': fa-oa, 'h1_confirmed': bool(fa > 0.5 and fa-oa > 0.2),
                'by_task': {t: {'n': len(v), 
                                'fact': sum(r['fact_score'] for r in v)/len(v),
                                'outdated': sum(r['outdated_score'] for r in v)/len(v)}
                            for t, v in by_task.items()}}
    with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as fp:
        json.dump(summary, fp, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
