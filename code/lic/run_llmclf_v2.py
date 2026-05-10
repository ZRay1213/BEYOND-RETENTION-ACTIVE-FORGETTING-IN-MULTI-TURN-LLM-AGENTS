import argparse, random, json, multiprocessing
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import tqdm
from simulator_llm_classifier_v2 import LLMClassifierMediatorShardedV2
from utils_log import get_run_counts


def run_one(todo):
    try:
        sim = LLMClassifierMediatorShardedV2(
            sample=todo['sample'],
            controller_model='qwen2.5-14b',
            mediator_model='qwen2.5-14b',
            assistant_model='qwen2.5-14b',
            system_model='qwen2.5-7b', user_model='qwen2.5-7b',
            dataset_fn=todo['dataset_fn'], log_folder=todo['log_folder'],
        )
        sim.conv_type = 'llm_classifier_v2'
        sim.run(verbose=False)
    except Exception:
        import traceback
        tqdm.tqdm.write(f'\033[91m[Error on {todo["sample"]["task_id"]}]:\n{traceback.format_exc()[:300]}\033[0m')


if __name__ == '__main__':
    multiprocessing.freeze_support()
    p = argparse.ArgumentParser()
    p.add_argument('--N', type=int, default=3)
    p.add_argument('--N_workers', type=int, default=3)
    p.add_argument('--log_folder', type=str, default='logs_llmclf_v2_heldout')
    args = p.parse_args()
    samples = json.load(open('data/sharded_stage3_math100.json'))
    heldout = set(json.load(open('/root/autodl-tmp/DCC/heldout_tasks.json')))
    samples = [s for s in samples if s['task_id'] in heldout]
    print(f'Loaded {len(samples)} held-out')
    todos = []
    rc = Counter(get_run_counts('llm_classifier_v2', 'math', 'qwen2.5-14b',
                                 'data/sharded_stage3_math100.json', log_folder=args.log_folder))
    for s in samples:
        need = args.N - rc.get(s['task_id'], 0)
        for _ in range(max(0, need)):
            todos.append({'sample': s, 'dataset_fn': 'data/sharded_stage3_math100.json',
                          'log_folder': args.log_folder})
    random.shuffle(todos)
    print(f'Running {len(todos)} llmclf-v2 sims')
    with ThreadPoolExecutor(max_workers=args.N_workers) as ex:
        list(tqdm.tqdm(ex.map(run_one, todos), total=len(todos)))
