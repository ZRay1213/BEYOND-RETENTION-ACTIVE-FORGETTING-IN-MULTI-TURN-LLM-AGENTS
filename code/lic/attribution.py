import json, sys, time
from openai import OpenAI

DATA = '/root/autodl-tmp/DCC/data/lost_in_conversation/data/sharded_stage1_K20.json'
LOG  = '/root/autodl-tmp/DCC/data/lost_in_conversation/logs_stage1/math/sharded/sharded_math_qwen2.5-14b.jsonl'

with open(DATA) as f:
    raw = json.load(f)
META = {x['task_id']: x for x in raw if x.get('task') == 'math'}

# Pull ALL wrong cases (one per task_id; pick first wrong run per task to get distinct tasks)
recs = []
with open(LOG) as f:
    for ln in f:
        d = json.loads(ln)
        if d.get('score') is None: continue
        recs.append(d)

distinct_wrong = {}
distinct_correct = {}
for r in recs:
    tid = r['task_id']
    if r['score'] < 0.01 and tid not in distinct_wrong:
        distinct_wrong[tid] = r
    elif r['score'] >= 0.99 and tid not in distinct_correct:
        distinct_correct[tid] = r
print(f'distinct: {len(distinct_correct)} correct, {len(distinct_wrong)} wrong', file=sys.stderr)

client = OpenAI(api_key='sk-local', base_url='http://127.0.0.1:8001/v1')

# ======== TEST 1 (re-run): oracle verifier on larger sample ========
ORACLE_VERIFY = '''Math problem:
{question}

Ground-truth solution:
{gold}

Candidate's solution:
{cand}

Did the candidate produce the same final numerical answer as the ground truth? Output reasoning, then on the LAST line: VERDICT: CORRECT or VERDICT: WRONG'''

# ======== TEST 2: attribution on wrong cases ========
ATTR_PROMPT = '''Original math problem (the user-simulator was given THIS to summarize into shards):
{question}

Ground-truth solution to the original:
{gold}

The user-simulator revealed this information piece-by-piece across turns:
{shards}

The assistant's final answer: {cand_final}

Your task: classify the failure mode. Read the user-revealed shards carefully and check:
(A) Did the user simulator introduce facts inconsistent with the original problem (e.g., wrong derived numbers, missing constraints, contradictory shards)? If YES, the failure is partly upstream.
(B) Were the user-revealed shards a faithful (if partial) version of the original problem, but the assistant still got the wrong answer through its own reasoning errors (premature commitment, ignoring later shards, arithmetic mistake)?

Output a brief reasoning, then on the LAST TWO LINES output exactly:
SIMULATOR_OK: YES or NO    (YES = shards faithfully encode original problem)
ASSISTANT_ERROR: YES or NO (YES = even with these shards, a careful solver could have got the right answer)'''

def call_llm(prompt, max_tokens=600):
    msg=[{'role':'user','content':prompt}]
    r=client.chat.completions.create(model='qwen2.5-14b', messages=msg, temperature=0.0, max_tokens=max_tokens)
    return r.choices[0].message.content

def extract_user_shards(trace):
    return [m['content'] for m in trace if m.get('role')=='user']

def extract_final(trace):
    a=[m for m in trace if m.get('role')=='assistant']
    return a[-1]['content'] if a else ''

def parse_verdict(txt):
    last = txt.strip().split('\n')[-1]
    if 'CORRECT' in last and 'WRONG' not in last: return 1
    if 'WRONG' in last: return 0
    return -1

def parse_attribution(txt):
    sim_ok = None; asst_err = None
    for ln in txt.strip().split('\n')[-6:]:
        ln = ln.strip().upper()
        if ln.startswith('SIMULATOR_OK'):
            sim_ok = 'YES' in ln.split(':',1)[1] if ':' in ln else None
        if ln.startswith('ASSISTANT_ERROR'):
            asst_err = 'YES' in ln.split(':',1)[1] if ':' in ln else None
    return sim_ok, asst_err

# ======== TEST 1 ========
print('\n=== TEST 1: oracle verifier on all distinct (correct + wrong) ===', file=sys.stderr)
t1_results = []
for tid, r in list(distinct_correct.items()) + list(distinct_wrong.items()):
    if tid not in META: continue
    cand = extract_final(r['trace'])[:3000]
    if not cand: continue
    p = ORACLE_VERIFY.format(question=META[tid]['question'], gold=META[tid]['answer'], cand=cand)
    t0=time.time()
    txt = call_llm(p)
    v = parse_verdict(txt)
    g = int(r['score']>=0.99)
    t1_results.append({'tid':tid,'gold':g,'verifier':v,'dt':time.time()-t0})
    print(f'  T1 {tid:24s} gold={g} v={v} dt={time.time()-t0:.1f}s', file=sys.stderr, flush=True)

# ======== TEST 2: attribution on wrong cases ========
print('\n=== TEST 2: attribution on all wrong cases ===', file=sys.stderr)
t2_results = []
for tid, r in distinct_wrong.items():
    if tid not in META: continue
    shards = extract_user_shards(r['trace'])
    cand = extract_final(r['trace'])[:1500]
    if not shards or not cand: continue
    shard_text = '\n'.join(f'  Turn {i+1}: {s}' for i,s in enumerate(shards))
    p = ATTR_PROMPT.format(question=META[tid]['question'], gold=META[tid]['answer'], shards=shard_text, cand_final=cand[-800:])
    t0=time.time()
    txt = call_llm(p, max_tokens=800)
    sim_ok, asst_err = parse_attribution(txt)
    t2_results.append({'tid':tid,'sim_ok':sim_ok,'asst_err':asst_err,'tail':txt[-400:],'dt':time.time()-t0})
    print(f'  T2 {tid:24s} sim_ok={sim_ok} asst_err={asst_err} dt={time.time()-t0:.1f}s', file=sys.stderr, flush=True)

# ======== Aggregate ========
p1 = [x for x in t1_results if x['verifier'] in (0,1)]
acc = sum(1 for x in p1 if x['verifier']==x['gold'])/max(1,len(p1))
print(f'\n=== TEST 1 SUMMARY ===')
print(f'oracle-verifier acc: {acc:.3f} on {len(p1)}/{len(t1_results)} parseable')
tp=sum(1 for x in p1 if x['gold']==1 and x['verifier']==1); fn=sum(1 for x in p1 if x['gold']==1 and x['verifier']==0)
tn=sum(1 for x in p1 if x['gold']==0 and x['verifier']==0); fp=sum(1 for x in p1 if x['gold']==0 and x['verifier']==1)
print(f'TP={tp} FN={fn} TN={tn} FP={fp}')

cat = {'A_only':0,'B_only':0,'both':0,'neither':0,'unparsed':0}
for x in t2_results:
    s, a = x['sim_ok'], x['asst_err']
    if s is None or a is None: cat['unparsed']+=1; continue
    if not s and a: cat['both']+=1
    elif not s and not a: cat['A_only']+=1
    elif s and a: cat['B_only']+=1
    else: cat['neither']+=1

print(f'\n=== TEST 2 SUMMARY (attribution on {len(t2_results)} wrong cases) ===')
print(f'A_only (simulator broke, assistant could not have recovered): {cat["A_only"]}')
print(f'B_only (simulator faithful, assistant errored):              {cat["B_only"]}')
print(f'both   (simulator broke AND assistant could have recovered): {cat["both"]}')
print(f'neither: {cat["neither"]}')
print(f'unparsed: {cat["unparsed"]}')

with open('/tmp/attribution_out.json','w') as f:
    json.dump({'t1':t1_results,'t2':t2_results,'cats':cat}, f, indent=1, ensure_ascii=False)
print('\nfull dump → /tmp/attribution_out.json')
