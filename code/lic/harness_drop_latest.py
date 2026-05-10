"""SELECTIVE-DROP experiment.

At the last turn (when all shards revealed), drop FIRST K% of LLM's prior
assistant messages from the context, keep last (1-K)%. K ∈ {0, 25, 50, 75, 100}
(K=0 == default sharded, K=100 == Fresh-Last).

Use to identify dose-response of history pollution: does early or late
commitment matter more?

Reuses LiC sharded conversation flow until last turn, then customizes the
final assistant call.
"""
import os
from simulator_sharded import ConversationSimulatorSharded
from utils import extract_conversation, date_str
from utils_log import log_conversation
from model_openai import generate

ASSISTANT_MODEL = os.environ.get('ASSISTANT_MODEL', 'qwen2.5-14b-tool')
DROP_FRAC = float(os.environ.get('DROP_FRAC', '0.5'))  # latest variant  # fraction of EARLY asst messages to drop


class SelectiveDropSimulator(ConversationSimulatorSharded):
    def __init__(self, sample, log_folder, dataset_fn=None, conv_type='selective_drop', drop_frac=0.5):
        super().__init__(
            sample,
            assistant_model=ASSISTANT_MODEL,
            system_model='qwen2.5-7b',
            user_model='qwen2.5-7b',
            dataset_fn=dataset_fn,
            log_folder=log_folder,
        )
        self.conv_type = conv_type
        self.drop_frac = drop_frac

    def _drop_late_assistant(self, conversation):
        """Take a list of {role,content} messages, identify ALL assistant messages,
        and replace the LAST drop_frac of them with empty (drop)."""
        asst_idx = [i for i, m in enumerate(conversation) if m.get('role') == 'assistant']
        n_drop = int(len(asst_idx) * self.drop_frac)
        drop_set = set(asst_idx[-n_drop:] if n_drop > 0 else [])
        # also drop the user message immediately preceding each dropped asst (the user shard)
        # NO — keep all user shards, only drop LLM's own outputs
        return [m for i, m in enumerate(conversation) if i not in drop_set]

    def run(self, verbose=False, save_log=True):
        max_assistant_tokens = 1000
        is_completed, is_correct, score = False, False, None
        shards = self.sample['shards']

        while not is_completed:
            revealed_ids = set(
                m['content']['shard_id']
                for m in self.trace
                if m['role'] == 'log' and m['content']['type'] == 'shard_revealed'
            )
            if len(revealed_ids) == len(shards):
                break

            is_last_turn = len(revealed_ids) == len(shards) - 1

            ur, sid, cu = self.user_agent.generate_response(self.trace, self.sample, temperature=self.user_temperature)
            self.trace.append({'role': 'user', 'content': ur, 'timestamp': date_str(), 'cost_usd': cu})
            if sid != -1:
                self.trace.append({'role': 'log', 'content': {'type': 'shard_revealed', 'shard_id': sid}, 'timestamp': date_str()})

            if is_last_turn:
                conversation = extract_conversation(self.trace, to_str=False)
                conversation = self._drop_late_assistant(conversation)
                self.trace.append({'role': 'log', 'content': {'type': 'selective-drop', 'drop_frac': self.drop_frac, 'kept_msgs': len(conversation)}, 'timestamp': date_str()})
                resp_obj = generate(
                    conversation,
                    model=self.assistant_model,
                    temperature=self.assistant_temperature,
                    return_metadata=True,
                    max_tokens=max_assistant_tokens,
                )
            else:
                resp_obj = generate(
                    extract_conversation(self.trace, to_str=False),
                    model=self.assistant_model,
                    temperature=self.assistant_temperature,
                    return_metadata=True,
                    max_tokens=max_assistant_tokens,
                )

            ar = resp_obj['message']
            self.trace.append({'role': 'assistant', 'content': ar, 'timestamp': date_str(), 'cost_usd': resp_obj['total_usd']})

            sv_resp, sv_cost = self.system_agent.verify_system_response(self.trace)
            self.trace.append({'role': 'log', 'content': {'type': 'system-verification', 'response': sv_resp}, 'timestamp': date_str(), 'cost_usd': sv_cost})

            if sv_resp['response_type'] == 'answer_attempt':
                ea = self.system_agent.extract_answer(self.trace)
                if self.task_name == 'summary' and not is_last_turn:
                    ev = {'score': 0.0}; score = 0.0; is_correct = None
                else:
                    ev = self.task.evaluator_function(ea, self.sample)
                    is_correct = ev.get('is_correct'); score = ev.get('score')
                if score == 1.0 and not is_correct:
                    is_correct = True
                self.trace.append({'role': 'log', 'content': {'type': 'answer-evaluation', 'exact_answer': ea, 'is_correct': is_correct, 'score': score, 'evaluation_return': ev}, 'timestamp': date_str()})
                if is_correct:
                    is_completed = True
                    self.trace.append({'role': 'log', 'content': {'type': 'conversation-completed', 'is_correct': is_correct}, 'timestamp': date_str()})

        if save_log:
            log_conversation(
                self.conv_type, self.task.get_task_name(), self.sample['task_id'],
                self.dataset_fn, self.assistant_model, self.system_model, self.user_model,
                self.trace, is_correct, score, log_folder=self.log_folder,
            )
        return is_correct, score


def run_selective_drop(sample, log_folder, dataset_fn=None, conv_type='selective_drop', drop_frac=0.5):
    sim = SelectiveDropSimulator(sample, log_folder, dataset_fn=dataset_fn, conv_type=conv_type, drop_frac=drop_frac)
    sim.run()


if __name__ == '__main__':
    import argparse, json, random
    from concurrent.futures import ThreadPoolExecutor
    from collections import Counter
    import tqdm
    from utils_log import get_run_counts

    p = argparse.ArgumentParser()
    p.add_argument('--dataset_file', default='data/sharded_stage3_math100.json')
    p.add_argument('--task_subset', default=None)
    p.add_argument('--N', type=int, default=2)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--log_folder', default='logs_selective_drop')
    p.add_argument('--conv_type', default='selective_drop')
    p.add_argument('--drop_frac', type=float, default=0.5)
    args = p.parse_args()

    samples = json.load(open(args.dataset_file))
    if args.task_subset:
        keep = set(json.load(open(args.task_subset)))
        samples = [s for s in samples if s['task_id'] in keep]
    print(f'Loaded {len(samples)} samples; conv_type={args.conv_type}; drop_frac={args.drop_frac}')

    todos = []
    rc = Counter()
    for _tk in set(s['task'] for s in samples):
        rc.update(get_run_counts(args.conv_type, _tk, ASSISTANT_MODEL, args.dataset_file, log_folder=args.log_folder))
    for s in samples:
        need = args.N - rc.get(s['task_id'], 0)
        for _ in range(max(0, need)):
            todos.append(s)
    random.shuffle(todos)
    print(f'Running {len(todos)} sims')

    def _run(s):
        try:
            run_selective_drop(s, args.log_folder, dataset_fn=args.dataset_file, conv_type=args.conv_type, drop_frac=args.drop_frac)
        except Exception as e:
            import traceback
            tqdm.tqdm.write(f'\033[91m[Error on {s["task_id"]}]: {traceback.format_exc()[:300]}\033[0m')

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(tqdm.tqdm(ex.map(_run, todos), total=len(todos)))
