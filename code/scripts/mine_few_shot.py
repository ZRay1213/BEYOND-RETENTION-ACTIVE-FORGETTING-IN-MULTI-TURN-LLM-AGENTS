"""Mine concrete few-shot examples from TRAINING traces.

For each of the 6 actions, find a task where:
  - Schedule featuring that action significantly outperformed alternatives
  - The conversation has a clean state before the action fired

Output: 6 hand-grounded examples, formatted for LLM prompt.
"""
import json, glob, os, statistics
from collections import defaultdict

ROOT = '/root/autodl-tmp/DCC/data/lost_in_conversation/logs_stage3_a1_math100/math'
heldout = set(json.load(open('/root/autodl-tmp/DCC/heldout_tasks.json')))

# task -> schedule -> [scores]
task_sched = defaultdict(lambda: defaultdict(list))
# (task, schedule) -> sample trace (first one)
task_sched_trace = {}
for sd in glob.glob(f'{ROOT}/mediator_*'):
    sch = os.path.basename(sd).replace('mediator_', '')
    for jl in glob.glob(f'{sd}/*.jsonl'):
        for ln in open(jl):
            try:
                rec = json.loads(ln)
            except: continue
            t = rec.get('task_id')
            s = rec.get('score')
            if t in heldout or s is None: continue
            task_sched[t][sch].append(s)
            if (t, sch) not in task_sched_trace and rec.get('trace'):
                task_sched_trace[(t, sch)] = rec['trace']

# For each task, find dominant schedule (best - second-best margin)
SCHED_TO_PRIMARY_ACTION = {
    'NONE': 'CONTINUE',
    'VERIFY-K3': 'VERIFY', 'VERIFY-K4': 'VERIFY',
    'CLARIFY-K3': 'CLARIFY',
    'SUMMARY-EVERY': 'INJECT_SUMMARY',
    'CONCLUDE-K3': 'CONCLUDE',
}

# Find tasks where one schedule (and hence one action) wins by clear margin
margin_winners = defaultdict(list)  # action -> [(task, margin, schedule)]
for t, scheds in task_sched.items():
    means = {sch: statistics.mean(rs) for sch, rs in scheds.items() if rs}
    if len(means) < 4: continue
    sorted_m = sorted(means.items(), key=lambda x: -x[1])
    best_sch, best_acc = sorted_m[0]
    second_sch, second_acc = sorted_m[1]
    margin = best_acc - second_acc
    if margin >= 0.4 and best_acc >= 0.6:  # decisive winner
        margin_winners[SCHED_TO_PRIMARY_ACTION[best_sch]].append((t, margin, best_sch, best_acc))

print('Decisive-winner tasks per action (margin >= 0.4, best acc >= 0.6):')
for act, lst in margin_winners.items():
    lst.sort(key=lambda x: -x[1])
    print(f'  {act}: {len(lst)} tasks; top: {lst[:3]}')

# Build examples: pick 1 task per action (top by margin)
examples = {}
for act, lst in margin_winners.items():
    if not lst: continue
    t, margin, sch, acc = lst[0]
    trace = task_sched_trace.get((t, sch))
    if not trace: continue
    # Extract first few user shards + assistant turns + mediator rewrites
    user_msgs, asst_msgs, mediator_rws = [], [], []
    for m in trace:
        if not isinstance(m, dict): continue
        role, c = m.get('role'), m.get('content')
        if role == 'user' and isinstance(c, str): user_msgs.append(c)
        elif role == 'assistant' and isinstance(c, str): asst_msgs.append(c)
        elif role == 'log' and isinstance(c, dict) and c.get('type') == 'mediator_rewrite':
            mediator_rws.append(c.get('rewritten', ''))
    excerpt = []
    for i in range(min(4, len(user_msgs))):
        excerpt.append(f'[User turn {i+1}]: {user_msgs[i][:140]}')
        if i < len(asst_msgs):
            excerpt.append(f'[Assistant turn {i+1}]: {asst_msgs[i][:200]}')
    examples[act] = {
        'task_id': t, 'schedule': sch, 'margin': margin, 'best_acc': acc,
        'excerpt': '\n'.join(excerpt),
    }
    print(f'\n=== {act} EXAMPLE (task={t}, schedule={sch}, margin={margin:.2f}, acc={acc:.2f}) ===')
    print(examples[act]['excerpt'][:800])

# Save formatted examples for prompt
with open('/root/autodl-tmp/DCC/few_shot_examples.json', 'w') as f:
    json.dump(examples, f, indent=2)
print(f'\nSaved {len(examples)} examples to /root/autodl-tmp/DCC/few_shot_examples.json')
