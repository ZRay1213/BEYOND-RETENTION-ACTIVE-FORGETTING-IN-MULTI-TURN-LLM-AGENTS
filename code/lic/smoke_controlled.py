import json, os, sys, time
os.environ.setdefault('OPENAI_API_KEY','sk-local')
os.environ.setdefault('OPENAI_BASE_URL_14B','http://127.0.0.1:8001/v1')
os.environ.setdefault('OPENAI_BASE_URL_7B','http://127.0.0.1:8002/v1')
os.environ.setdefault('DCC_GRADER_MODEL','qwen2.5-7b')

from simulator_controlled import ControlledSharded, fixed_schedule_policy

# Pick 3 known-wrong math instances
DATA = '/root/autodl-tmp/DCC/data/lost_in_conversation/data/sharded_stage1_K20.json'
samples = json.load(open(DATA))
math_meta = {x['task_id']: x for x in samples if x.get('task') == 'math'}
target_tids = ['sharded-GSM8K/427', 'sharded-GSM8K/189', 'sharded-GSM8K/1066']
target_samples = [math_meta[t] for t in target_tids if t in math_meta]
print(f'target: {len(target_samples)} samples', file=sys.stderr)

schedules = ['NONE', 'RESET-K2', 'VERIFY-K2']
results = []
for sched in schedules:
    print(f'\n=== schedule={sched} ===', file=sys.stderr)
    if sched == 'NONE':
        policy = fixed_schedule_policy('CONTINUE', 1)
    elif sched == 'RESET-K2':
        policy = fixed_schedule_policy('RESET', 2)
    elif sched == 'VERIFY-K2':
        policy = fixed_schedule_policy('VERIFY', 2)
    for s in target_samples:
        t0 = time.time()
        try:
            sim = ControlledSharded(
                sample=s, policy_fn=policy, track_state=False,
                assistant_model='qwen2.5-14b',
                system_model='qwen2.5-7b', user_model='qwen2.5-7b',
                assistant_temperature=0.0, user_temperature=1.0,
                dataset_fn=DATA, log_folder='logs_smoke_ctrl',
            )
            sim.conv_type = f'ctrl_{sched}'
            done, correct, score = sim.run(verbose=False, save_log=False)
            results.append({'tid':s['task_id'],'sched':sched,'score':score,'correct':correct,'dt':time.time()-t0,'actions':[a for _,a in sim.action_history if a!='CONTINUE']})
            print(f'  {s["task_id"]:24s} sched={sched:10s} score={score} correct={correct} dt={time.time()-t0:.1f}s actions={results[-1]["actions"]}', file=sys.stderr, flush=True)
        except Exception as e:
            import traceback; print(f'ERR on {s["task_id"]}/{sched}: {e}', file=sys.stderr); traceback.print_exc()

print('\n=== summary ===')
from collections import defaultdict
agg = defaultdict(list)
for r in results:
    agg[r['sched']].append(r['score'] or 0)
for sched, scores in agg.items():
    print(f'  {sched:12s}: scores={scores} avg={sum(scores)/len(scores):.2f}')

with open('/tmp/smoke_ctrl.json','w') as f: json.dump(results, f, indent=1)
