"""DCC harness — single-file conversation control program for LiC.

This file is the unit of evolution. The outer-loop proposer mutates it based on
traces. Sections marked with [MUTABLE] can be freely rewritten by the proposer.

Entry point: run_harness(sample, log_folder) -> (is_correct, score)
"""
import json, os, sys
from utils import print_colored, extract_conversation, date_str
from utils_log import log_conversation
from model_openai import generate
from simulator_sharded import ConversationSimulatorSharded


# =============================================================================
# [MUTABLE] HARNESS HYPERPARAMETERS
# =============================================================================
MEDIATOR_MODEL = 'qwen2.5-14b'
CONTROLLER_MODEL = 'qwen2.5-14b'
ASSISTANT_MODEL = 'qwen2.5-14b'

# =============================================================================
# [MUTABLE] MEDIATOR — rewrites multi-turn history into one fully-spec instruction
# =============================================================================
MEDIATOR_SYSTEM = (
    "You are a helpful assistant that reconstructs a user's full question from a multi-turn conversation. "
    "The user reveals information progressively across turns. Produce ONE fully-specified, self-contained instruction "
    "that captures every fact the user has revealed up to and including the latest turn.\n\n"
    "Rules:\n"
    "1. Include ALL facts, numbers, constraints, goals across every turn.\n"
    "2. Do NOT reference previous turns. Write as if asked once.\n"
    "3. Do NOT add facts the user has not stated. Do NOT solve the problem.\n"
    "4. If a later turn updates an earlier value, use the updated one.\n"
    "5. Output ONLY the reconstructed instruction. No preamble, no explanation."
)


def mediator_rewrite(history_str):
    """Convert formatted history into fully-specified instruction."""
    msgs = [
        {'role': 'system', 'content': MEDIATOR_SYSTEM},
        {'role': 'user', 'content': f'Conversation history:\n\n{history_str}\n\nReconstructed instruction:'},
    ]
    try:
        resp = generate(msgs, model=MEDIATOR_MODEL, temperature=0.3, return_metadata=True, max_tokens=800)
        rewritten = resp['message'].strip() if isinstance(resp, dict) else str(resp).strip()
        for p in ('Reconstructed instruction:', 'Instruction:', 'Question:'):
            if rewritten.startswith(p): rewritten = rewritten[len(p):].strip()
        return rewritten
    except Exception:
        return None


# =============================================================================
# [MUTABLE] DIRECTIVE TEMPLATES — action vocabulary
# =============================================================================
ACTION_LIST = ['CONTINUE', 'VERIFY', 'CLARIFY', 'REFRAME', 'RESET', 'CONCLUDE', 'INJECT_SUMMARY']

DIRECTIVE_TEMPLATES = {
    'CONTINUE': None,
    'VERIFY': (
        'Important directive: Before responding, verify your previous numerical answer step-by-step '
        'using only the facts the user has explicitly stated. Recompute any arithmetic. '
        'If you find an inconsistency, correct it now.'
    ),
    'CLARIFY': (
        'Important directive: Before answering, ask the user ONE concise clarifying question '
        'about the most recent fact. Ensure the question is relevant and specific to the problem. '
        'Reply with only the question.'
    ),
    'REFRAME': (
        'Important directive: Re-frame the problem in a different way using the information provided so far. '
        'Do not solve the problem yet.'
    ),
    'RESET': (
        'Important directive: Discard the partial answer you may have given earlier. '
        'Re-read all user-provided information from the start, list each fact, and propose a fresh path. '
        'If a previous claim contradicts a new fact, drop it.'
    ),
    'CONCLUDE': (
        'Important directive: Stop asking for more information. Commit to your best final numerical '
        'answer right now in the form \\boxed{...} based on what you know.'
    ),
    'INJECT_SUMMARY': '__INJECT_TASKSTATE__',  # filled at runtime
}

# =============================================================================
# [MUTABLE] STATE EXTRACTION — what features the controller sees
# =============================================================================
def extract_state(history, mediator_rewrites, action_history, turn_idx):
    """Return a structured state dict the controller can read."""
    # Last assistant content
    asst_msgs = [m['content'] for m in history if isinstance(m, dict) and m.get('role') == 'assistant' and isinstance(m.get('content'), str)]
    last_asst = asst_msgs[-1] if asst_msgs else ''
    cur_rewrite = mediator_rewrites[-1] if mediator_rewrites else ''
    prev_rewrite = mediator_rewrites[-2] if len(mediator_rewrites) >= 2 else ''
    return {
        'turn': turn_idx,
        'cur_rewrite': cur_rewrite,
        'prev_rewrite': prev_rewrite,
        'last_asst': last_asst,
        'last_asst_short_q': len(last_asst) < 200 and last_asst.strip().endswith('?'),
        'last_asst_has_boxed': '\\boxed' in last_asst,
        'last_asst_len': len(last_asst.split()),
        'past_actions': [a for _, a in action_history],
        'rewrite_changed': cur_rewrite != prev_rewrite,
    }

# =============================================================================
# [MUTABLE] CONTROLLER — picks one action given state
# =============================================================================
CONTROLLER_SYSTEM = (
    "You are a meta-controller in a multi-turn dialogue. Each turn pick ONE action:\n"
    "  CONTINUE — proceed normally (DEFAULT)\n"
    "  VERIFY   — assistant must verify its previous numeric answer (use when last reply gave a definite numeric/boxed answer that may be wrong)\n"
    "  CLARIFY  — ask one short clarifying question (use only when CRITICAL info missing AND assistant stuck >2 turns)\n"
    "  REFRAME  — reframe the problem (use when assistant has been repeatedly clarifying without progress)\n"
    "  RESET    — discard partial answers, recompute (use when assistant clearly committed wrong)\n"
    "  CONCLUDE — stop asking, commit to final boxed answer now (use when 3+ turns revealed adequate info but assistant keeps clarifying)\n"
    "  SUMMARY  — re-anchor on TaskState (rarely useful)\n\n"
    "Default to CONTINUE. Avoid same action 2 turns in a row.\n"
    "Output EXACTLY one word: CONTINUE | VERIFY | CLARIFY | REFRAME | RESET | CONCLUDE | SUMMARY"
)


def controller_select_action(state, history_str):
    """Given state, return action name. Default policy: query 14B as classifier."""
    past = state.get('past_actions', [])
    msgs = [
        {'role': 'system', 'content': CONTROLLER_SYSTEM},
        {'role': 'user', 'content': (
            f'Conversation history (truncated):\n\n{history_str[-5000:]}\n\n'
            f'Previous directives: {past[-5:]}\n'
            f'State signals: turn={state["turn"]} '
            f'last_asst_short_q={state["last_asst_short_q"]} '
            f'last_asst_has_boxed={state["last_asst_has_boxed"]} '
            f'rewrite_changed={state["rewrite_changed"]}\n\n'
            f'Pick one action. Output one word.'
        )},
    ]
    try:
        resp = generate(msgs, model=CONTROLLER_MODEL, temperature=0.0, return_metadata=True, max_tokens=10)
        raw = resp['message'].strip().upper() if isinstance(resp, dict) else str(resp).strip().upper()
        for a in ['CONTINUE', 'VERIFY', 'CLARIFY', 'REFRAME', 'RESET', 'CONCLUDE', 'SUMMARY']:
            if a in raw:
                return 'INJECT_SUMMARY' if a == 'SUMMARY' else a
        return 'CONTINUE'
    except Exception:
        return 'CONTINUE'

    # Deterministic rule override
    if state['turn'] >= 4 and state['last_asst_short_q'] and 'How' in state['last_asst']:
        return 'REFRAME'
    if state['last_asst_has_boxed'] and state['rewrite_changed']:
        return 'RESET'
    return 'CONTINUE'


# =============================================================================
# UTILITY — format history for LLMs (keep stable)
# =============================================================================
def format_history(trace):
    lines, idx = [], 0
    for m in trace:
        if not isinstance(m, dict): continue
        role = m.get('role')
        c = m.get('content')
        if role == 'user' and isinstance(c, str):
            idx += 1
            lines.append(f'[User turn {idx}]: {c}')
        elif role == 'assistant' and isinstance(c, str):
            short = c if len(c) < 400 else c[:400] + '...'
            lines.append(f'[Assistant turn {idx}]: {short}')
    return '\n\n'.join(lines)

# =============================================================================
# MAIN LOOP — orchestrates a sharded simulation with mediator + controller
# =============================================================================
class HarnessSharded(ConversationSimulatorSharded):
    """Wraps the official sharded simulator to apply mediator + controller per turn."""

    def __init__(self, sample, log_folder, dataset_fn=None, conv_type='harness'):
        super().__init__(sample,
                         assistant_model=ASSISTANT_MODEL,
                         system_model='qwen2.5-7b',
                         user_model='qwen2.5-7b',
                         dataset_fn=dataset_fn,
                         log_folder=log_folder)
        self.conv_type = conv_type
        self.mediator_rewrites = []
        self.action_history = []
        self.controller_log = []

    def run(self, verbose=False, save_log=True):
        max_assistant_tokens = 1000
        is_completed, is_correct, score = False, False, None
        shards = self.sample['shards']
        turn = 0

        while not is_completed:
            revealed_ids = set([m['content']['shard_id'] for m in self.trace
                               if m['role'] == 'log' and m['content']['type'] == 'shard_revealed'])
            if len(revealed_ids) == len(shards): break
            is_last = len(revealed_ids) == len(shards) - 1
            turn += 1

            # 1. user shard
            ur, sid, cu = self.user_agent.generate_response(self.trace, self.sample, temperature=self.user_temperature)
            self.trace.append({'role': 'user', 'content': ur, 'timestamp': date_str(), 'cost_usd': cu})
            if sid != -1:
                self.trace.append({'role': 'log', 'content': {'type': 'shard_revealed', 'shard_id': sid}, 'timestamp': date_str()})

            # 2. Mediator rewrite
            hist = format_history(self.trace)
            rewrite = mediator_rewrite(hist)
            if rewrite is None: rewrite = ur
            self.mediator_rewrites.append(rewrite)
            self.trace.append({'role': 'log', 'content': {'type': 'mediator_rewrite', 'turn': turn, 'rewritten': rewrite}, 'timestamp': date_str()})

            # 3. Controller picks action
            state = extract_state(self.trace, self.mediator_rewrites, self.action_history, turn)
            action = controller_select_action(state, hist)
            self.action_history.append((turn, action))
            self.controller_log.append({'turn': turn, 'action': action, 'state': {k: v for k, v in state.items() if k not in ('cur_rewrite', 'prev_rewrite', 'last_asst')}})
            self.trace.append({'role': 'log', 'content': {'type': 'controller_action', 'turn': turn, 'action': action}, 'timestamp': date_str()})

            # 4. Build directive
            if action == 'CONTINUE': directive = None
            elif action == 'INJECT_SUMMARY':
                directive = None  # state tracker not used in baseline; proposer may add later
            else:
                directive = DIRECTIVE_TEMPLATES.get(action)

            # 5. Assistant call: system + (directive) + rewritten
            msgs = [{'role': 'system', 'content': self.system_message}]
            if directive: msgs.append({'role': 'system', 'content': directive})
            msgs.append({'role': 'user', 'content': rewrite})
            ar = generate(msgs, model=self.assistant_model, temperature=self.assistant_temperature,
                         return_metadata=True, max_tokens=max_assistant_tokens)
            am = ar['message']
            self.trace.append({'role': 'assistant', 'content': am, 'timestamp': date_str(), 'cost_usd': ar['total_usd']})

            # 6. system verification (unchanged)
            sv, vc = self.system_agent.verify_system_response(self.trace)
            self.trace.append({'role': 'log', 'content': {'type': 'system-verification', 'response': sv}, 'timestamp': date_str(), 'cost_usd': vc})

            if sv['response_type'] == 'answer_attempt':
                ea = self.system_agent.extract_answer(self.trace)
                if self.task_name == 'summary' and not is_last:
                    ev = {'score': 0.0}; score = 0.0; is_correct = None
                else:
                    ev = self.task.evaluator_function(ea, self.sample)
                    is_correct = ev.get('is_correct'); score = ev.get('score')
                if score == 1.0 and not is_correct: is_correct = True
                self.trace.append({'role': 'log', 'content': {'type': 'answer-evaluation', 'exact_answer': ea, 'is_correct': is_correct, 'score': score, 'evaluation_return': ev}, 'timestamp': date_str()})
                if is_correct:
                    is_completed = True
                    self.trace.append({'role': 'log', 'content': {'type': 'conversation-completed', 'is_correct': True}, 'timestamp': date_str()})

        if save_log:
            log_conversation(self.conv_type, self.task.get_task_name(), self.sample['task_id'],
                           self.dataset_fn, self.assistant_model, self.system_model, self.user_model,
                           self.trace, is_correct, score, log_folder=self.log_folder)
        return is_correct, score


# =============================================================================
# ENTRY POINT
# =============================================================================
def run_harness(sample, log_folder, dataset_fn=None, conv_type='harness'):
    sim = HarnessSharded(sample, log_folder, dataset_fn=dataset_fn, conv_type=conv_type)
    return sim.run(save_log=True)


if __name__ == '__main__':
    import argparse, random, multiprocessing
    from concurrent.futures import ThreadPoolExecutor
    from collections import Counter
    import tqdm
    from utils_log import get_run_counts

    p = argparse.ArgumentParser()
    p.add_argument('--dataset_file', default='data/sharded_stage3_math100.json')
    p.add_argument('--task_subset', default=None, help='JSON file with list of task_ids; if given, restrict to those')
    p.add_argument('--N', type=int, default=3)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--log_folder', default='logs_harness_g0')
    p.add_argument('--conv_type', default='harness_g0')
    args = p.parse_args()

    samples = json.load(open(args.dataset_file))
    if args.task_subset:
        keep = set(json.load(open(args.task_subset)))
        samples = [s for s in samples if s['task_id'] in keep]
    # samples = [s for s in samples if s['task'] == 'math']  # multi-task patched
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
            run_harness(s, args.log_folder, dataset_fn=args.dataset_file, conv_type=args.conv_type)
        except Exception as e:
            import traceback
            tqdm.tqdm.write(f'\033[91m[Error on {s["task_id"]}]: {traceback.format_exc()[:300]}\033[0m')

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(tqdm.tqdm(ex.map(_run, todos), total=len(todos)))