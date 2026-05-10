import argparse, random, json, multiprocessing
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import tqdm
from simulator_bandit import BanditMediatorSharded
from utils_log import get_run_counts


def run_one(todo):
    try:
        sim = BanditMediatorSharded(
            sample=todo['sample'],
            q_model_path=todo['q_model_path'],
            mediator_model='qwen2.5-14b',
            assistant_model='qwen2.5-14b',
            system_model='qwen2.5-7b',
            user_model='qwen2.5-7b',
            dataset_fn=todo['dataset_fn'],
            log_folder=todo['log_folder'],
        )
        sim.conv_type = 'bandit'
        sim.run(verbose=False)
    except Exception:
        import traceback
        tqdm.tqdm.write(f'\033[91m[Error on {todo["sample"]["task_id"]}]:\n{traceback.format_exc()}\033[0m')


if __name__ == '__main__':
    multiprocessing.freeze_support()
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_file', type=str, default='data/sharded_stage3_math100.json')
    p.add_argument('--heldout_file', type=str, default='/root/autodl-tmp/DCC/heldout_tasks.json')
    p.add_argument('--q_model_path', type=str, default='/root/autodl-tmp/DCC/bandit_q_model.joblib')
    p.add_argument('--N', type=int, default=3)
    p.add_argument('--N_workers', type=int, default=4)
    p.add_argument('--log_folder', type=str, default='logs_bandit_heldout')
    args = p.parse_args()

    samples = json.load(open(args.dataset_file))
    heldout_tasks = set(json.load(open(args.heldout_file)))
    samples = [s for s in samples if s['task_id'] in heldout_tasks]
    print(f'Loaded {len(samples)} held-out samples (target N={args.N})')

    todos = []
    run_counts = Counter(get_run_counts('bandit', 'math', 'qwen2.5-14b', args.dataset_file, log_folder=args.log_folder))
    for s in samples:
        need = args.N - run_counts.get(s['task_id'], 0)
        for _ in range(max(0, need)):
            todos.append({
                'sample': s, 'q_model_path': args.q_model_path,
                'dataset_fn': args.dataset_file, 'log_folder': args.log_folder,
            })
    random.shuffle(todos)
    print(f'Running {len(todos)} bandit sims')
    with ThreadPoolExecutor(max_workers=args.N_workers) as ex:
        list(tqdm.tqdm(ex.map(run_one, todos), total=len(todos)))
