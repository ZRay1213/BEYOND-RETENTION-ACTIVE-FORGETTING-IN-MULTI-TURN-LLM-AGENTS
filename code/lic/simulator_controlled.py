'''Controller-augmented sharded simulator. Plug in any policy of signature:
    policy(state, t, action_history) -> action_name
where action_name in CONTINUE / RESET / VERIFY / INJECT_SUMMARY / CONCLUDE.
'''
from utils import print_colored, extract_conversation, date_str
from utils_log import log_conversation
from simulator_sharded import ConversationSimulatorSharded
from model_openai import generate
from state_tracker import TaskState, StateTracker

ACTIONS = ['CONTINUE', 'RESET', 'VERIFY', 'INJECT_SUMMARY', 'CONCLUDE', 'CLARIFY']

ACTION_PROMPTS = {
    'CONTINUE': None,
    'RESET': (
        'Important directive: Discard the partial answer/proposal you may have given in earlier turns. '
        'Do not assume any previous numerical claim is correct. Re-read all of the user-provided '
        'information from the start of this conversation, list each fact, and propose a fresh solution path. '
        'If a previous claim contradicts a new fact, drop the previous claim entirely.'
    ),
    'VERIFY': (
        'Important directive: Before you respond this turn, verify your previous numerical answer step by step '
        'using only the facts the user has explicitly stated. Recompute any arithmetic. '
        'If you find any inconsistency or error, correct it now and explain what you fixed.'
    ),
    'INJECT_SUMMARY': '__INJECT_TASKSTATE__',  # filled at runtime with rendered state
    'CONCLUDE': (
        'Important directive: You have been given enough information. Stop asking questions. '
        'Commit to your best final numerical answer right now in the form \\boxed{...} based on what you know.'
    ),
    'CLARIFY': (
        'Important directive: Before answering or solving anything, ask the user ONE concise clarifying question '
        'about the most recent piece of information they provided. Specifically, check whether the most recent fact '
        'is consistent with what was said earlier and whether any necessary detail is missing or ambiguous. '
        'Do not produce a final numerical answer this turn. Reply with only your single clarifying question.'
    ),
}


def fixed_schedule_policy(action_name, every_k):
    '''Returns a policy function that fires action_name every K turns.'''
    def policy(state, t, history):
        if action_name == 'CONTINUE':
            return 'CONTINUE'
        # fire action every K turns starting at turn K
        if t > 0 and t % every_k == 0:
            return action_name
        return 'CONTINUE'
    return policy


def always_policy(action_name):
    def policy(state, t, history):
        return action_name
    return policy


class ControlledSharded(ConversationSimulatorSharded):
    '''Sharded sim with per-turn controller injecting directives.'''

    def __init__(self, sample, policy_fn, tracker_model='qwen2.5-7b', track_state=True, **kw):
        super().__init__(sample, **kw)
        self.policy_fn = policy_fn
        self.track_state = track_state
        self.tracker = StateTracker(model=tracker_model) if track_state else None
        self.task_state = TaskState()
        self.action_history = []  # list of (turn, action)

    def _render_directive(self, action):
        if action == 'CONTINUE': return None
        if action == 'INJECT_SUMMARY':
            if not (self.task_state.goal or self.task_state.facts or self.task_state.constraints):
                return None
            return f'Current TaskState (use it to verify your reasoning):\n{self.task_state.render_for_assistant()}'
        return ACTION_PROMPTS[action]

    def _build_assistant_messages(self, directive):
        msgs = extract_conversation(self.trace, to_str=False)
        if directive:
            msgs.append({'role': 'system', 'content': directive})
        return msgs

    def run(self, verbose=False, save_log=True):
        is_reasoning_model = ('o1' in self.assistant_model or 'o3' in self.assistant_model or 'deepseek-r1' in self.assistant_model)
        max_assistant_tokens = 10000 if is_reasoning_model else 1000
        is_completed, is_correct, score = False, False, None
        shards = self.sample['shards']

        turn_idx = 0
        while not is_completed:
            revealed_shard_ids = set([msg['content']['shard_id'] for msg in self.trace if msg['role'] == 'log' and msg['content'].get('type') == 'shard_revealed'])
            if len(revealed_shard_ids) == len(shards):
                break
            is_last_turn = len(revealed_shard_ids) == len(shards) - 1

            # 1. user
            user_response, shard_revealed_id, cost_usd = self.user_agent.generate_response(self.trace, self.sample, temperature=self.user_temperature)
            self.trace.append({'role': 'user', 'content': user_response, 'timestamp': date_str(), 'cost_usd': cost_usd})
            if verbose: print_colored(f'[user] {user_response}', 'green')
            if shard_revealed_id != -1:
                self.trace.append({'role': 'log', 'content': {'type': 'shard_revealed', 'shard_id': shard_revealed_id}, 'timestamp': date_str()})

            # 1b. update TaskState (skip if track_state=False to save calls)
            if self.tracker is not None:
                last_assistant = next((m['content'] for m in reversed(self.trace) if m['role'] == 'assistant'), '')
                new_state, tracker_cost, tracker_err = self.tracker.update(self.task_state, user_response, last_assistant)
                self.task_state = new_state
                self.trace.append({'role': 'log', 'content': {'type': 'task_state', 'state': self.task_state.to_dict(), 'tracker_err': tracker_err}, 'timestamp': date_str(), 'cost_usd': tracker_cost})

            # 2. controller decides action
            action = self.policy_fn(self.task_state, turn_idx, self.action_history)
            self.action_history.append((turn_idx, action))
            self.trace.append({'role': 'log', 'content': {'type': 'controller_action', 'turn': turn_idx, 'action': action}, 'timestamp': date_str()})
            if verbose and action != 'CONTINUE': print_colored(f'[controller] turn {turn_idx} → {action}', 'yellow' if 'yellow' in dir() else 'blue')

            # 2b. realize directive
            directive = self._render_directive(action)
            messages = self._build_assistant_messages(directive)
            assistant_response_obj = generate(messages, model=self.assistant_model, temperature=self.assistant_temperature, return_metadata=True, max_tokens=max_assistant_tokens)
            assistant_response = assistant_response_obj['message']
            self.trace.append({'role': 'assistant', 'content': assistant_response, 'timestamp': date_str(), 'cost_usd': assistant_response_obj['total_usd']})
            if verbose: print_colored(f'[assistant] {assistant_response}', 'red')

            # 3. system verify
            system_verification_response, verification_cost_usd = self.system_agent.verify_system_response(self.trace)
            self.trace.append({'role': 'log', 'content': {'type': 'system-verification', 'response': system_verification_response}, 'timestamp': date_str(), 'cost_usd': verification_cost_usd})

            if system_verification_response['response_type'] == 'answer_attempt':
                extracted_answer = self.system_agent.extract_answer(self.trace)
                is_correct, score = None, None
                if self.task_name == 'summary' and not is_last_turn:
                    evaluation_return = {'score': 0.0}; score = 0.0
                else:
                    evaluation_return = self.task.evaluator_function(extracted_answer, self.sample)
                    is_correct = evaluation_return.get('is_correct')
                    score = evaluation_return.get('score')
                if score == 1.0 and not is_correct: is_correct = True
                self.trace.append({'role': 'log', 'content': {'type': 'answer-evaluation', 'exact_answer': extracted_answer, 'is_correct': is_correct, 'score': score, 'evaluation_return': evaluation_return}, 'timestamp': date_str()})
                if is_correct or is_last_turn:
                    is_completed = True

            turn_idx += 1
            if turn_idx > 30: break  # safety

        if save_log:
            conv_type = getattr(self, "conv_type", "controlled")
            log_conversation(conv_type, self.task.get_task_name(), self.sample["task_id"], self.dataset_fn, self.assistant_model, self.system_model, self.user_model, self.trace, is_correct, score, log_folder=self.log_folder)
        return is_completed, is_correct, score
