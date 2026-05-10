import json, os, re, sys
from openai import OpenAI

# Load dataset (questions + ground-truth answers)
DATA = '/root/autodl-tmp/DCC/data/lost_in_conversation/data/sharded_stage1_K20.json'
with open(DATA) as f:
    raw = json.load(f)
math_meta = {x['task_id']: x for x in raw if x.get('task') == 'math'}

# Load Stage 1 sharded math run
LOG = '/root/autodl-tmp/DCC/data/lost_in_conversation/logs_stage1/math/sharded/sharded_math_qwen2.5-14b.jsonl'
recs = []
with open(LOG) as f:
    for ln in f:
        d = json.loads(ln)
        if d.get('score') is None:
            continue
        recs.append(d)

# Sample 10 correct and 10 wrong (distinct task_ids)
seen = set()
correct, wrong = [], []
for r in recs:
    tid = r['task_id']
    if tid in seen: continue
    if r['score'] >= 0.99 and len(correct) < 10:
        correct.append(r); seen.add(tid)
    elif r['score'] < 0.01 and len(wrong) < 10:
        wrong.append(r); seen.add(tid)
    if len(correct) >= 10 and len(wrong) >= 10:
        break
print(f'samples: {len(correct)} correct, {len(wrong)} wrong', file=sys.stderr)

# Verifier: given question + assistant's final response, score 0-1
client = OpenAI(api_key='sk-local', base_url='http://127.0.0.1:8001/v1')

VERIFY_PROMPT = '''You are a math step verifier. Given a math problem and a candidate solution, judge whether the final numerical answer is CORRECT.

Problem:
{question}

Candidate solution:
{solution}

Think step by step about whether the candidate's reasoning AND final numerical answer are correct. Then on the LAST line output exactly: VERDICT: CORRECT or VERDICT: WRONG'''

def extract_final(trace):
    asst = [m for m in trace if m.get('role') == 'assistant']
    return asst[-1]['content'] if asst else ''

def verify(question, solution):
    msg = [{'role': 'user', 'content': VERIFY_PROMPT.format(question=question, solution=solution)}]
    r = client.chat.completions.create(model='qwen2.5-14b', messages=msg, temperature=0.0, max_tokens=600)
    txt = r.choices[0].message.content
    last = txt.strip().split('\n')[-1]
    if 'CORRECT' in last and 'WRONG' not in last:
        v = 1
    elif 'WRONG' in last:
        v = 0
    else:
        v = -1  # unparseable
    return v, txt[-200:]

import time
def run(label, items):
    out = []
    for r in items:
        meta = math_meta.get(r['task_id'])
        if not meta: continue
        q = meta['question']
        sol = extract_final(r['trace'])
        if not sol: continue
        t0 = time.time()
        v, tail = verify(q, sol[:3000])
        out.append({'tid': r['task_id'], 'gold': int(r['score'] >= 0.99), 'verifier': v, 'dt': time.time()-t0, 'tail': tail})
        print(f'  {label} tid={r["task_id"]:24s} gold={int(r["score"]>=0.99)} verifier={v} dt={time.time()-t0:.1f}s', file=sys.stderr, flush=True)
    return out

print('=== CORRECT ===', file=sys.stderr)
out_c = run('correct', correct)
print('=== WRONG ===', file=sys.stderr)
out_w = run('wrong', wrong)

# Score the verifier
all_out = out_c + out_w
parseable = [x for x in all_out if x['verifier'] in (0, 1)]
acc = sum(1 for x in parseable if x['verifier'] == x['gold']) / max(1, len(parseable))
tp = sum(1 for x in parseable if x['gold'] == 1 and x['verifier'] == 1)
fn = sum(1 for x in parseable if x['gold'] == 1 and x['verifier'] == 0)
tn = sum(1 for x in parseable if x['gold'] == 0 and x['verifier'] == 0)
fp = sum(1 for x in parseable if x['gold'] == 0 and x['verifier'] == 1)
print(f'\n=== VERIFIER STATS ===\n')
print(f'parseable: {len(parseable)}/{len(all_out)}')
print(f'agreement acc: {acc:.3f}')
print(f'TP={tp} FN={fn} TN={tn} FP={fp}')
print(f'recall_correct: {tp/(tp+fn) if tp+fn else 0:.3f}')
print(f'recall_wrong:   {tn/(tn+fp) if tn+fp else 0:.3f}')

with open('/tmp/verifier_smoke_out.json', 'w') as f:
    json.dump(all_out, f, indent=1, ensure_ascii=False)
