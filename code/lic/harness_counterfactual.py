"""COUNTERFACTUAL injection experiment.

For each existing failed-sharded conversation, replay the last turn under 4 conditions:
  A — original (real LLM commitments preserved)
  B — oracle CORRECT prior commitments injected (synthetic, ground-truth derived)
  C — oracle WRONG prior commitments injected (synthetic, deliberately wrong)
  D — no prior assistant commitments (= Fresh-Last)

Asymmetric B-A vs C-A delta directly tests anchoring hypothesis (H1).

To generate B and C synthetic commitments, we use the 7B model with a separate prompt
that asks for "what an assistant would commit at turn k given partial info, where the
commitment should be (correct|incorrect)".
"""
import os, json, glob, sys
sys.path.insert(0, '/root/autodl-tmp/DCC/data/lost_in_conversation')
os.chdir('/root/autodl-tmp/DCC/data/lost_in_conversation')

from utils import extract_conversation, date_str
from utils_log import log_conversation
from model_openai import generate
from system_agent import SystemAgent
from tasks import get_task

ASSISTANT_MODEL = os.environ.get('ASSISTANT_MODEL', 'qwen2.5-14b')
GENERATOR_MODEL = os.environ.get('GENERATOR_MODEL', 'qwen2.5-7b')


def get_ground_truth_string(sample):
    """Best-effort: extract ground-truth answer text from sample."""
    for k in ['answer', 'gold', 'ground_truth', 'reference', 'target']:
        v = sample.get(k)
        if v: return str(v)
    return None


def synthesize_commitment(sample, partial_shard_text, want_correct):
    """Generate a synthetic LLM-like commitment at a partial-info turn.

    want_correct=True  → commitment that, if taken as final, would be correct
    want_correct=False → commitment that is plausibly wrong
    """
    gt = get_ground_truth_string(sample)
    if want_correct:
        instruction = (
            f"Given the partial problem statement so far, write a brief assistant response "
            f"that COMMITS to a final answer. The committed final answer MUST be: {gt}\n\n"
            f"Make the response sound natural, like a confident attempt. Length ≤80 words.\n\n"
            f"Partial problem: {partial_shard_text}"
        )
    else:
        instruction = (
            f"Given the partial problem statement so far, write a brief assistant response "
            f"that COMMITS to a deliberately INCORRECT final answer (NOT {gt}). "
            f"Make the wrong answer plausible-looking, like a real mistake under partial info. "
            f"Length ≤80 words.\n\n"
            f"Partial problem: {partial_shard_text}"
        )
    msgs = [{'role': 'user', 'content': instruction}]
    resp = generate(msgs, model=GENERATOR_MODEL, temperature=0.7, return_metadata=True, max_tokens=200)
    return resp['message']


def run_counterfactual_sample(sharded_log, condition, log_folder):
    """For a given source sharded log, replay only the last assistant call under `condition`.

    condition ∈ {'A_original', 'B_correct', 'C_wrong', 'D_drop'}
    """
    sample = get_task(sharded_log['task']).get_sample(sharded_log['task_id'])
    trace = sharded_log['trace']

    # Find all (user_shard, asst_response) pairs in the original trace
    asst_idx_list = [i for i, m in enumerate(trace) if m.get('role') == 'assistant']
    if not asst_idx_list:
        return  # no asst messages, skip

    # Build the last-turn context based on condition
    # We rebuild conversation from scratch up to (but not including) the FINAL asst message,
    # then replace prior asst messages per condition, then issue final assistant call.

    # Step 1: gather user-shard contents in order
    convo = []
    convo.append({'role': 'system', 'content': trace[0]['content']})

    # Walk through trace, collect (user, asst) pairs except the LAST asst
    pairs = []  # list of (user_msg, asst_msg, partial_shard_text_so_far)
    cur_user = None
    shards_so_far = []
    for m in trace:
        if m.get('role') == 'log' and m.get('content',{}).get('type') == 'shard_revealed':
            sid = m['content']['shard_id']
            shard_obj = next((s for s in sample['shards'] if (s.get('shard_id') if isinstance(s, dict) else None) == sid), None)
            if isinstance(shard_obj, dict):
                shards_so_far.append(shard_obj.get('shard') or '')
        elif m.get('role') == 'user':
            cur_user = m
        elif m.get('role') == 'assistant':
            partial_text = '\n'.join(shards_so_far)
            pairs.append((cur_user, m, partial_text))
            cur_user = None

    if len(pairs) < 2:
        return  # need at least 1 prior asst before last

    # Last asst is the one we re-generate. Prior asst messages are pairs[:-1].
    n_prior = len(pairs) - 1

    # Step 2: build the new conversation
    new_convo = [{'role': 'system', 'content': trace[0]['content']}]
    for i, (user_m, asst_m, partial_text) in enumerate(pairs[:-1]):
        new_convo.append({'role': 'user', 'content': user_m['content']})
        if condition == 'A_original':
            new_convo.append({'role': 'assistant', 'content': asst_m['content']})
        elif condition == 'B_correct':
            synth = synthesize_commitment(sample, partial_text, want_correct=True)
            new_convo.append({'role': 'assistant', 'content': synth})
        elif condition == 'C_wrong':
            synth = synthesize_commitment(sample, partial_text, want_correct=False)
            new_convo.append({'role': 'assistant', 'content': synth})
        elif condition == 'D_drop':
            pass  # skip the assistant message

    # add the final user shard
    last_user = pairs[-1][0]
    if last_user is not None:
        new_convo.append({'role': 'user', 'content': last_user['content']})

    # Step 3: issue the final assistant call
    resp = generate(new_convo, model=ASSISTANT_MODEL, temperature=1.0, return_metadata=True, max_tokens=1000)
    final_asst = resp['message']

    # Step 4: evaluate
    sa = SystemAgent(sharded_log['task'], 'qwen2.5-7b', sample)
    fake_trace = [{'role': 'system', 'content': trace[0]['content']}] + new_convo[1:] + [{'role': 'assistant', 'content': final_asst}]
    sv_resp, _ = sa.verify_system_response(fake_trace)
    is_correct = None; score = None
    if sv_resp['response_type'] == 'answer_attempt':
        ea = sa.extract_answer(fake_trace)
        ev = get_task(sharded_log['task']).evaluator_function(ea, sample)
        is_correct = ev.get('is_correct'); score = ev.get('score')
        if score == 1.0 and not is_correct: is_correct = True

    # Step 5: log
    log_conversation(
        f'cf_{condition}', sharded_log['task'], sharded_log['task_id'],
        sharded_log['dataset_fn'], ASSISTANT_MODEL, 'qwen2.5-7b', 'qwen2.5-7b',
        fake_trace + [{'role': 'log', 'content': {'type': 'cf-condition', 'condition': condition, 'n_prior': n_prior}, 'timestamp': date_str()}],
        is_correct, score, log_folder=log_folder,
        additional_info={'source_conv_id': sharded_log.get('conv_id'), 'condition': condition},
    )
    return is_correct


if __name__ == '__main__':
    import argparse, random
    from concurrent.futures import ThreadPoolExecutor
    import tqdm

    p = argparse.ArgumentParser()
    p.add_argument('--source_logs_dir', default='logs_multi_baseline')
    p.add_argument('--source_conv_type', default='sharded')
    p.add_argument('--N_per_task', type=int, default=2)
    p.add_argument('--max_samples', type=int, default=30, help='task_id count')
    p.add_argument('--conditions', nargs='+', default=['A_original', 'B_correct', 'C_wrong', 'D_drop'])
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--log_folder', default='logs_counterfactual')
    args = p.parse_args()

    files = glob.glob(f'{args.source_logs_dir}/**/*.jsonl', recursive=True)
    source_logs = []
    for f in files:
        for line in open(f):
            try:
                r = json.loads(line)
                if r.get('conv_type') == args.source_conv_type:
                    source_logs.append(r)
            except: pass

    # Group by task_id, pick max_samples task_ids
    by_task_id = {}
    for log in source_logs:
        by_task_id.setdefault(log['task_id'], []).append(log)
    task_ids = list(by_task_id.keys())[:args.max_samples]
    print(f'Found {len(source_logs)} source logs; using {len(task_ids)} task_ids')

    todos = []
    for tid in task_ids:
        log = by_task_id[tid][0]  # use first log
        for cond in args.conditions:
            for _ in range(args.N_per_task):
                todos.append((log, cond))
    random.shuffle(todos)
    print(f'Running {len(todos)} sims')

    def _run(args_):
        log, cond = args_
        try:
            run_counterfactual_sample(log, cond, args.log_folder)
        except Exception as e:
            import traceback
            tqdm.tqdm.write(f'\033[91m[Error on {log["task_id"]} cond={cond}]: {traceback.format_exc()[:300]}\033[0m')

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(tqdm.tqdm(ex.map(_run, todos), total=len(todos)))
