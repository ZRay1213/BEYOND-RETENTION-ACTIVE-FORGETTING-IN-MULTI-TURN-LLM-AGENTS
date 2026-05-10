"""Runner for MediatorRewrittenSharded (A1 + optional directive controller)."""
import argparse, random, json, multiprocessing
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import tqdm

from simulator_mediator import MediatorRewrittenSharded, noop_policy
from simulator_controlled import fixed_schedule_policy, always_policy
from utils_log import get_run_counts


def make_policy(schedule):
    if schedule in ('NONE', 'A1-ONLY', 'MEDIATOR'):
        return noop_policy
    if schedule.startswith('VERIFY-K'):
        return fixed_schedule_policy('VERIFY', int(schedule.split('K')[1]))
    if schedule.startswith('CLARIFY-K'):
        return fixed_schedule_policy('CLARIFY', int(schedule.split('K')[1]))
    if schedule.startswith('RESET-K'):
        return fixed_schedule_policy('RESET', int(schedule.split('K')[1]))
    if schedule == 'SUMMARY-EVERY':
        return always_policy('INJECT_SUMMARY')
    if schedule.startswith('CONCLUDE-K'):
        k = int(schedule.split('K')[1])
        def policy(state, t, history):
            if t >= k and not any(a == 'CONCLUDE' for _, a in history):
                return 'CONCLUDE'
            return 'CONTINUE'
        return policy
    raise ValueError(schedule)


def run_one(todo):
    try:
        policy_fn = make_policy(todo['schedule'])
        track_state = todo['schedule'] == 'SUMMARY-EVERY'
        sim = MediatorRewrittenSharded(
            sample=todo['sample'],
            mediator_model=todo.get('mediator_model', 'qwen2.5-14b'),
            policy_fn=policy_fn,
            tracker_model=todo.get('tracker_model', 'qwen2.5-7b'),
            track_state=track_state,
            assistant_model=todo['assistant_model'],
            system_model=todo['system_model'],
            user_model=todo['user_model'],
            assistant_temperature=todo.get('assistant_temperature', 1.0),
            user_temperature=todo.get('user_temperature', 1.0),
            dataset_fn=todo['dataset_fn'],
            log_folder=todo['log_folder'],
        )
        sim.conv_type = f'mediator_{todo["schedule"]}'
        sim.run(verbose=todo.get('verbose', False))
    except Exception:
        import traceback
        tqdm.tqdm.write(f'\033[91m[Error on {todo["sample"]["task_id"]}; {todo["schedule"]}]:\n{traceback.format_exc()}\033[0m')


if __name__ == '__main__':
    multiprocessing.freeze_support()
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_file', type=str, required=True)
    p.add_argument('--schedule', type=str, required=True,
                   help='NONE | RESET-K{2,4} | VERIFY-K{2,3,4} | SUMMARY-EVERY | CONCLUDE-K{2,3} | CLARIFY-K{2,3}')
    p.add_argument('--models', nargs='+', default=['qwen2.5-14b'])
    p.add_argument('--mediator_model', type=str, default='qwen2.5-14b')
    p.add_argument('--system_model', type=str, default='qwen2.5-7b')
    p.add_argument('--user_model', type=str, default='qwen2.5-7b')
    p.add_argument('--tracker_model', type=str, default='qwen2.5-7b')
    p.add_argument('--tasks', nargs='+', default=['math'])
    p.add_argument('--N', type=int, default=3)
    p.add_argument('--N_workers', type=int, default=4)
    p.add_argument('--log_folder', type=str, default='logs_mediator')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    samples = json.load(open(args.dataset_file))
    samples = [s for s in samples if s['task'] in args.tasks]
    print(f'Loaded {len(samples)} samples; schedule={args.schedule}')
    random.shuffle(samples)

    todos = []
    for assistant_model in args.models:
        run_counts = Counter()
        for task in args.tasks:
            run_counts.update(get_run_counts(f'mediator_{args.schedule}', task, assistant_model, args.dataset_file, log_folder=args.log_folder))
        for s in samples:
            need = args.N - run_counts.get(s['task_id'], 0)
            for _ in range(max(0, need)):
                todos.append({
                    'sample': s, 'schedule': args.schedule,
                    'mediator_model': args.mediator_model,
                    'tracker_model': args.tracker_model,
                    'assistant_model': assistant_model,
                    'system_model': args.system_model, 'user_model': args.user_model,
                    'dataset_fn': args.dataset_file, 'log_folder': args.log_folder,
                    'verbose': args.verbose,
                })

    random.shuffle(todos)
    print(f'Running {len(todos)} mediator sims (schedule={args.schedule})')
    with ThreadPoolExecutor(max_workers=args.N_workers) as ex:
        list(tqdm.tqdm(ex.map(run_one, todos), total=len(todos)))
