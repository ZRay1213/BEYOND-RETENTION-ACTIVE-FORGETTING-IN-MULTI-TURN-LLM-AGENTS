import argparse, random, json, multiprocessing
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import tqdm
from simulator_llm_classifier import LLMClassifierMediatorSharded
from utils_log import get_run_counts


def run_one(todo):
    try:
        sim = LLMClassifierMediatorSharded(
            sample=todo['sample'],
            controller_model='qwen2.5-7b',
            mediator_model='qwen2.5-14b',
            assistant_model='qwen2.5-14b',
            system_model='qwen2.5-7b',
            user_model='qwen2.5-7b',
            dataset_fn=todo['dataset_fn'],
            log_folder=todo['log_folder'],
        )
        sim.conv_type = 'llm_classifier'
        sim.run(verbose=False)
    except Exception:
        import traceback
        tqdm.tqdm.write(f'\033[91m[Error on {todo["sample"]["task_id"]}]:\n{traceback.format_exc()[:300]}\033[0m')


if __name__ == '__main__':
    multiprocessing.freeze_support()
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_file', type=str, default='data/sharded_stage3_math100.json')
    p.add_argument('--heldout_file', type=str, default='/root/autodl-tmp/DCC/heldout_tasks.json')
    p.add_argument('--N', type=int, default=3)
    p.add_argument('--N_workers', type=int, default=4)
    p.add_argument('--log_folder', type=str, default='logs_llmclf_heldout')
    args = p.parse_args()

    samples = json.load(open(args.dataset_file))
    heldout = set(json.load(open(args.heldout_file)))
    samples = [s for s in samples if s['task_id'] in heldout]
    print(f'Loaded {len(samples)} held-out (target N={args.N})')

    todos = []
    rc = Counter(get_run_counts('llm_classifier', 'math', 'qwen2.5-14b', args.dataset_file, log_folder=args.log_folder))
    for s in samples:
        need = args.N - rc.get(s['task_id'], 0)
        for _ in range(max(0, need)):
            todos.append({'sample': s, 'dataset_fn': args.dataset_file, 'log_folder': args.log_folder})
    random.shuffle(todos)
    print(f'Running {len(todos)} llm-classifier sims')
    with ThreadPoolExecutor(max_workers=args.N_workers) as ex:
        list(tqdm.tqdm(ex.map(run_one, todos), total=len(todos)))
