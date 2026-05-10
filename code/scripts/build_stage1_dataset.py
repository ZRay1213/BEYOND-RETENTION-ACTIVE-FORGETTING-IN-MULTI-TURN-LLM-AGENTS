import json, sys, random
K = int(sys.argv[1]) if len(sys.argv) > 1 else 20
TASKS = sys.argv[2].split(',') if len(sys.argv) > 2 else None
random.seed(42)
src = 'data/lost_in_conversation/data/sharded_instructions_600.json'
d = json.load(open(src))
by_task = {}
for x in d:
    by_task.setdefault(x['task'], []).append(x)
out = []
for task, items in sorted(by_task.items()):
    if TASKS and task not in TASKS: continue
    random.shuffle(items)
    out.extend(items[:K])
fn = f'data/lost_in_conversation/data/sharded_stage1_K{K}.json'
json.dump(out, open(fn, 'w'))
print(f'Wrote {len(out)} samples ({K} per task) to {fn}')
print('Tasks:', sorted({x["task"] for x in out}))
