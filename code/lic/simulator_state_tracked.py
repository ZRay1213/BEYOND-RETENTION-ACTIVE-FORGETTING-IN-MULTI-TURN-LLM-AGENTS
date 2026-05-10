import json
from utils import print_colored, extract_conversation, date_str
from utils_log import log_conversation
from simulator_sharded import ConversationSimulatorSharded
from model_openai import generate
from state_tracker import TaskState, StateTracker


class StateTrackedSharded(ConversationSimulatorSharded):
    '''Sharded simulator with TaskState extraction.
    mode='state_only': assistant sees only system + TaskState
    mode='state_aug':  assistant sees full trace + TaskState appended as system msg
    '''

    def __init__(self, sample, mode='state_only', tracker_model='qwen2.5-7b', **kw):
        super().__init__(sample, **kw)
        assert mode in ('state_only', 'state_aug')
        self.mode = mode
        self.tracker = StateTracker(model=tracker_model)
        self.task_state = TaskState()

    def _build_assistant_messages(self):
        rendered = self.task_state.render_for_assistant()
        if self.mode == 'state_only':
            user_block = (
                'You are answering a multi-turn user request. The conversation so far has been '
                'summarized into a structured TaskState below. Use only this state to formulate '
                'your response. If you have enough information, give a final answer; if something '
                'is missing, ask one targeted clarifying question.\n\n'
                f'{rendered}'
            )
            return [
                {'role': 'system', 'content': self.system_message},
                {'role': 'user', 'content': user_block},
            ]
        # state_aug
        msgs = extract_conversation(self.trace, to_str=False)
        msgs.append({'role': 'system', 'content': f'Current TaskState (refreshed each turn):\n{rendered}'})
        return msgs

    def run(self, verbose=False, save_log=True):
        is_reasoning_model = ('o1' in self.assistant_model or 'o3' in self.assistant_model or 'deepseek-r1' in self.assistant_model)
        max_assistant_tokens = 10000 if is_reasoning_model else 1000
        is_completed, is_correct, score = False, False, None
        shards = self.sample['shards']

        while not is_completed:
            revealed_shard_ids = set([msg['content']['shard_id'] for msg in self.trace if msg['role'] == 'log' and msg['content'].get('type') == 'shard_revealed'])
            if len(revealed_shard_ids) == len(shards):
                if verbose:
                    print_colored(f'[log] all shards revealed', 'blue')
                break
            is_last_turn = len(revealed_shard_ids) == len(shards) - 1

            # 1. user
            user_response, shard_revealed_id, cost_usd = self.user_agent.generate_response(self.trace, self.sample, temperature=self.user_temperature)
            self.trace.append({'role': 'user', 'content': user_response, 'timestamp': date_str(), 'cost_usd': cost_usd})
            if verbose:
                print_colored(f'[user] {user_response}', 'green')

            if shard_revealed_id != -1:
                self.trace.append({'role': 'log', 'content': {'type': 'shard_revealed', 'shard_id': shard_revealed_id}, 'timestamp': date_str()})
                if verbose:
                    print_colored(f'[log] shard revealed: {shard_revealed_id}', 'blue')

            # 1b. update TaskState
            last_assistant = next((m['content'] for m in reversed(self.trace) if m['role'] == 'assistant'), '')
            new_state, tracker_cost, tracker_err = self.tracker.update(self.task_state, user_response, last_assistant)
            self.task_state = new_state
            self.trace.append({'role': 'log', 'content': {'type': 'task_state', 'state': self.task_state.to_dict(), 'tracker_err': tracker_err}, 'timestamp': date_str(), 'cost_usd': tracker_cost})
            if verbose:
                short = json.dumps(self.task_state.to_dict(), ensure_ascii=False)[:200]
                print_colored(f'[state] {short}', 'cyan' if 'cyan' in dir() else 'blue')

            # 2. assistant — use TaskState-driven messages
            messages = self._build_assistant_messages()
            assistant_response_obj = generate(messages, model=self.assistant_model, temperature=self.assistant_temperature, return_metadata=True, max_tokens=max_assistant_tokens)
            assistant_response = assistant_response_obj['message']
            self.trace.append({'role': 'assistant', 'content': assistant_response, 'timestamp': date_str(), 'cost_usd': assistant_response_obj['total_usd']})
            if verbose:
                print_colored(f'[assistant] {assistant_response}', 'red')

            # 3. system verify
            system_verification_response, verification_cost_usd = self.system_agent.verify_system_response(self.trace)
            self.trace.append({'role': 'log', 'content': {'type': 'system-verification', 'response': system_verification_response}, 'timestamp': date_str(), 'cost_usd': verification_cost_usd})
            if verbose:
                print_colored(f'[log] system verification: {system_verification_response}', 'blue')

            if system_verification_response['response_type'] == 'answer_attempt':
                extracted_answer = self.system_agent.extract_answer(self.trace)
                is_correct, score = None, None
                if self.task_name == 'summary' and not is_last_turn:
                    evaluation_return = {'score': 0.0}; score = 0.0
                else:
                    evaluation_return = self.task.evaluator_function(extracted_answer, self.sample)
                    is_correct = evaluation_return.get('is_correct')
                    score = evaluation_return.get('score')
                if score == 1.0 and not is_correct:
                    is_correct = True
                self.trace.append({'role': 'log', 'content': {'type': 'answer-evaluation', 'exact_answer': extracted_answer, 'is_correct': is_correct, 'score': score, 'evaluation_return': evaluation_return}, 'timestamp': date_str()})
                if verbose:
                    print_colored(f'[log] eval: correct={is_correct} score={score}', 'blue')
                if is_correct:
                    is_completed = True
                    self.trace.append({'role': 'log', 'content': {'type': 'conversation-completed', 'is_correct': is_correct}, 'timestamp': date_str()})

        if save_log:
            conv_type = self.mode  # 'state_only' or 'state_aug'
            log_conversation(conv_type, self.task.get_task_name(), self.sample['task_id'], self.dataset_fn, self.assistant_model, self.system_model, self.user_model, self.trace, is_correct, score, log_folder=self.log_folder)
        return is_correct, score
