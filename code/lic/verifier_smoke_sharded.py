import json, os, sys, time
from openai import OpenAI

DATA = '/root/autodl-tmp/DCC/data/lost_in_conversation/data/sharded_stage1_K20.json'
LOG = '/root/autodl-tmp/DCC/data/lost_in_conversation/logs_stage1/math/sharded/sharded_math_qwen2.5-14b.jsonl'

with open(DATA) as f:
    raw = json.load(f)
math_meta = {x['task_id']: x for x in raw if x.get('task') == 'math'}

recs = []
with open(LOG) as f:
    for ln in f:
        d = json.loads(ln)
        if d.get('score') is None: continue
        recs.append(d)

seen=set(); correct=[]; wrong=[]
for r in recs:
    if r['task_id'] in seen: continue
    if r['score']>=0.99 and len(correct)<10:
        correct.append(r); seen.add(r['task_id'])
    elif r['score']<0.01 and len(wrong)<10:
        wrong.append(r); seen.add(r['task_id'])
    if len(correct)>=10 and len(wrong)>=10: break

client = OpenAI(api_key='sk-local', base_url='http://127.0.0.1:8001/v1')

# Sharded-only verifier: sees user shards (NOT original question) + assistant's current answer
SHARD_VERIFY = '''You are a math reasoning verifier. The user has revealed information piece-by-piece across multiple turns to ask a math question. Based ONLY on the information revealed by the user, check whether the assistant's solution is consistent and arithmetically correct.

User-revealed information so far:
{shards}

Assistant's current solution:
{solution}

Check:
1. Did the assistant use information consistent with what the user said?
2. Is the arithmetic correct?
3. Does the assistant's answer follow from the user's revealed facts?

Output reasoning, then on the LAST line: VERDICT: CORRECT or VERDICT: WRONG'''

def extract_user_shards(trace):
    return [m['content'] for m in trace if m.get('role')=='user']

def extract_final(trace):
    a = [m for m in trace if m.get('role')=='assistant']
    return a[-1]['content'] if a else ''

def verify_sharded(shards, solution):
    shard_text = '\n'.join(f'- Turn {i+1}: {s}' for i, s in enumerate(shards))
    msg = [{'role':'user','content':SHARD_VERIFY.format(shards=shard_text, solution=solution)}]
    r = client.chat.completions.create(model='qwen2.5-14b', messages=msg, temperature=0.0, max_tokens=600)
    txt = r.choices[0].message.content
    last = txt.strip().split('\n')[-1]
    if 'CORRECT' in last and 'WRONG' not in last: return 1, txt[-300:]
    if 'WRONG' in last: return 0, txt[-300:]
    return -1, txt[-300:]

def run(label, items):
    out=[]
    for r in items:
        shards = extract_user_shards(r['trace'])
        sol = extract_final(r['trace'])
        if not shards or not sol: continue
        t0=time.time()
        v, tail = verify_sharded(shards, sol[:3000])
        out.append({'tid':r['task_id'],'gold':int(r['score']>=0.99),'verifier':v,'dt':time.time()-t0,'tail':tail})
        print(f'  {label} tid={r["task_id"]:24s} gold={int(r["score"]>=0.99)} v={v} dt={time.time()-t0:.1f}s', file=sys.stderr, flush=True)
    return out

print('=== CORRECT ===', file=sys.stderr); out_c = run('correct', correct)
print('=== WRONG ===', file=sys.stderr); out_w = run('wrong', wrong)

all_out = out_c + out_w
p = [x for x in all_out if x['verifier'] in (0,1)]
acc = sum(1 for x in p if x['verifier']==x['gold'])/max(1,len(p))
tp=sum(1 for x in p if x['gold']==1 and x['verifier']==1)
fn=sum(1 for x in p if x['gold']==1 and x['verifier']==0)
tn=sum(1 for x in p if x['gold']==0 and x['verifier']==0)
fp=sum(1 for x in p if x['gold']==0 and x['verifier']==1)
print(f'\n=== SHARDED-VIEW VERIFIER STATS ===')
print(f'parseable: {len(p)}/{len(all_out)}')
print(f'agreement acc: {acc:.3f}')
print(f'TP={tp} FN={fn} TN={tn} FP={fp}')
print(f'recall_correct: {tp/(tp+fn) if tp+fn else 0:.3f}')
print(f'recall_wrong:   {tn/(tn+fp) if tn+fp else 0:.3f}')

with open('/tmp/verifier_sharded_out.json','w') as f: json.dump(all_out, f, indent=1, ensure_ascii=False)
