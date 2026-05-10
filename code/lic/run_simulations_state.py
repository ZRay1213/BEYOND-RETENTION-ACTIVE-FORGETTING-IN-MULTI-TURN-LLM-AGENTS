import argparse, random, json, multiprocessing
from concurrent.futures import ThreadPoolExecutor
import tqdm
from collections import Counter

from simulator_state_tracked import StateTrackedSharded
from utils_log import get_run_counts


def run_one(todo):
    try:
        sim = StateTrackedSharded(
            sample=todo['sample'],
            mode=todo['mode'],
            tracker_model=todo['tracker_model'],
            assistant_model=todo['assistant_model'],
            system_model=todo['system_model'],
            user_model=todo['user_model'],
            assistant_temperature=todo.get('assistant_temperature', 1.0),
            user_temperature=todo.get('user_temperature', 1.0),
            dataset_fn=todo['dataset_fn'],
            log_folder=todo['log_folder'],
        )
        sim.run(verbose=todo.get('verbose', False))
    except Exception as e:
        import traceback
        tqdm.tqdm.write(f'\033[91m[Error on {todo["sample"]["task_id"]}; {todo["assistant_model"]}; {todo["mode"]}]:\n{traceback.format_exc()}\033[0m')


if __name__ == '__main__':
    multiprocessing.freeze_support()
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_file', type=str, required=True)
    p.add_argument('--models', nargs='+', default=['qwen2.5-14b'])
    p.add_argument('--system_model', type=str, default='qwen2.5-7b')
    p.add_argument('--user_model', type=str, default='qwen2.5-7b')
    p.add_argument('--tracker_model', type=str, default='qwen2.5-7b')
    p.add_argument('--mode', type=str, default='state_only', choices=['state_only', 'state_aug'])
    p.add_argument('--tasks', nargs='+', default=['math'])
    p.add_argument('--N', type=int, default=3)
    p.add_argument('--N_workers', type=int, default=8)
    p.add_argument('--log_folder', type=str, default='logs_stage2')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    samples = json.load(open(args.dataset_file))
    samples = [s for s in samples if s['task'] in args.tasks]
    print(f'Loaded {len(samples)} samples')
    random.shuffle(samples)

    todos = []
    for assistant_model in args.models:
        run_counts = Counter()
        for task in args.tasks:
            run_counts.update(get_run_counts(args.mode, task, assistant_model, args.dataset_file, log_folder=args.log_folder))
        print(f'Run counts for {args.mode}: {sum(run_counts.values())} prior runs')
        for s in samples:
            need = args.N - run_counts.get(s['task_id'], 0)
            for _ in range(max(0, need)):
                todos.append({
                    'sample': s, 'mode': args.mode, 'tracker_model': args.tracker_model,
                    'assistant_model': assistant_model, 'system_model': args.system_model,
                    'user_model': args.user_model, 'dataset_fn': args.dataset_file,
                    'log_folder': args.log_folder, 'verbose': args.verbose,
                })

    random.shuffle(todos)
    print(f'Running {len(todos)} state-tracked sims (mode={args.mode})')
    with ThreadPoolExecutor(max_workers=args.N_workers) as ex:
        list(tqdm.tqdm(ex.map(run_one, todos), total=len(todos)))
