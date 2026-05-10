"""CacheGuard: validity-aware reuse layer.

Block Parser → Eligibility Controller → Renderer.
At each turn, render context as [stable_valid_prefix | volatile_suffix].

Goal: simultaneously high PCHR (stable prefix is monotone-growing across turns,
maximizing prefix-cache reuse) and high VCHR (no provisional commitments are
included, so all reused blocks are valid).
"""
import os, re
from simulator_sharded import ConversationSimulatorSharded
from utils import extract_conversation, date_str
from utils_log import log_conversation
from model_openai import generate

ASSISTANT_MODEL = os.environ.get('ASSISTANT_MODEL', 'qwen2.5-14b')


# ---------------------------------------------------------------------
# Block Parser
# ---------------------------------------------------------------------

def classify_assistant_message(content: str) -> str:
    """Classify an assistant message into a block type.

    Returns one of:
      - 'claim' (answer attempt)
      - 'artifact' (code or structured output)
      - 'clarification' (asking the user)
      - 'plan' (procedural description)
      - 'misc'
    """
    if not isinstance(content, str):
        content = str(content)
    s = content.strip()
    # boxed or [[...]] answer
    if re.search(r'\\boxed\{[^}]+\}', s) or ('[[' in s and ']]' in s):
        return 'claim'
    # code block
    if '```' in s:
        return 'artifact'
    # ends with question
    if s.endswith('?'):
        return 'clarification'
    # explicit final answer wording
    if re.search(r'\b(final answer|the answer is|answer:)\b', s.lower()):
        return 'claim'
    # plan-y wording
    if re.search(r'\b(i will|i\'ll|let me|first,? .* second,? .* third)\b', s.lower()):
        return 'plan'
    return 'misc'


def parse_blocks(trace):
    """Parse a list-of-message trace into typed blocks.

    Returns list of dicts with keys: role, type, content, turn_idx, block_id.
    Excludes log entries.
    """
    blocks = []
    asst_idx = 0
    for m in trace:
        role = m.get('role')
        if role == 'log':
            continue
        if role == 'system':
            blocks.append({
                'role': 'system', 'type': 'system',
                'content': m['content'], 'turn_idx': 0, 'block_id': 'sys',
            })
        elif role == 'user':
            blocks.append({
                'role': 'user', 'type': 'user_evidence',
                'content': m['content'], 'turn_idx': len(blocks), 'block_id': f'u{len(blocks)}',
            })
        elif role == 'assistant':
            asst_idx += 1
            blocks.append({
                'role': 'assistant',
                'type': classify_assistant_message(m['content']),
                'content': m['content'], 'turn_idx': len(blocks),
                'block_id': f'a{asst_idx}',
            })
    return blocks


# ---------------------------------------------------------------------
# Eligibility Controller
# ---------------------------------------------------------------------

def is_eligible(block, all_blocks_so_far):
    """Decide whether a block should be kept in the context for the next LLM call.

    Default policy v1:
      - system: always
      - user_evidence: always
      - assistant_clarification: keep (asking, not committing)
      - assistant_artifact: keep iff later user/tool block references it
      - assistant_claim/plan/misc: drop
    """
    t = block['type']
    if t == 'system': return True
    if t == 'user_evidence': return True
    if t == 'assistant_clarification': return True
    if t == 'assistant_artifact':
        # crude reference detection: was a token from this artifact echoed by later user?
        artifact_str = str(block['content'])
        # take a code-block snippet
        m = re.search(r'```(?:[a-z]*\n)?(.{20,200}?)```', artifact_str, re.DOTALL)
        if not m: return False
        snippet = m.group(1).strip()[:80]
        if not snippet: return False
        for later in all_blocks_so_far:
            if later['turn_idx'] <= block['turn_idx']: continue
            if later['role'] == 'user' and snippet[:40] in str(later['content']):
                return True
        return False
    return False


# ---------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------

def render_context(blocks):
    """Construct the messages list for the next LLM call.

    Stable prefix = system + all user shards so far + eligible artifacts/clarifications
    Volatile suffix is empty (we let the LLM see the latest user shard already in stable).

    For LiC sharded protocol where every user message is a shard, the stable prefix
    naturally grows monotonically as turns progress -- this is exactly what prefix
    caching wants.
    """
    eligible_msgs = []
    sys_msgs = []
    for b in blocks:
        if not is_eligible(b, blocks):
            continue
        if b['role'] == 'system':
            sys_msgs.append({'role': 'system', 'content': b['content']})
        else:
            eligible_msgs.append({'role': b['role'], 'content': b['content']})
    return sys_msgs + eligible_msgs


# ---------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------

class CacheGuardSimulator(ConversationSimulatorSharded):
    def __init__(self, sample, log_folder, dataset_fn=None, conv_type='cacheguard'):
        super().__init__(
            sample,
            assistant_model=ASSISTANT_MODEL,
            system_model='qwen2.5-7b',
            user_model='qwen2.5-7b',
            dataset_fn=dataset_fn,
            log_folder=log_folder,
        )
        self.conv_type = conv_type

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

            # CacheGuard rendering — at last turn, use task concat prompt for proper formatting
            if is_last_turn:
                concat_prompt = self.task.populate_concat_prompt(self.sample)
                rendered_msgs = [
                    {'role': 'system', 'content': self.system_message},
                    {'role': 'user', 'content': concat_prompt},
                ]
            else:
                blocks = parse_blocks(self.trace)
                rendered_msgs = render_context(blocks)
            n_kept = len(rendered_msgs)
            n_total = sum(1 for m in self.trace if m.get('role') in ('system','user','assistant'))
            self.trace.append({'role': 'log', 'content': {'type': 'cacheguard-render', 'kept_msgs': n_kept, 'total_msgs': n_total}, 'timestamp': date_str()})

            resp_obj = generate(
                rendered_msgs,
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


def run_cacheguard(sample, log_folder, dataset_fn=None, conv_type='cacheguard'):
    sim = CacheGuardSimulator(sample, log_folder, dataset_fn=dataset_fn, conv_type=conv_type)
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
    p.add_argument('--log_folder', default='logs_cacheguard')
    p.add_argument('--conv_type', default='cacheguard')
    args = p.parse_args()

    samples = json.load(open(args.dataset_file))
    if args.task_subset:
        keep = set(json.load(open(args.task_subset)))
        samples = [s for s in samples if s['task_id'] in keep]
    print(f'Loaded {len(samples)} samples; conv_type={args.conv_type}')

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
            run_cacheguard(s, args.log_folder, dataset_fn=args.dataset_file, conv_type=args.conv_type)
        except Exception as e:
            import traceback
            tqdm.tqdm.write(f'\033[91m[Error on {s["task_id"]}]: {traceback.format_exc()[:300]}\033[0m')

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(tqdm.tqdm(ex.map(_run, todos), total=len(todos)))
