"""Mediator-Assistant simulator (simplified A1 / arXiv 2602.07338) with optional
directive controller for composability tests.

Each turn: (1) user reveals shard, (2) Mediator rewrites full history into a
fully-specified instruction, (3) optional controller picks a directive
(VERIFY/CLARIFY/RESET/INJECT_SUMMARY/CONCLUDE/CLARIFY/CONTINUE) which is
prepended as a system message to the assistant's input, (4) assistant generates.
"""
from utils import print_colored, extract_conversation, date_str
from utils_log import log_conversation
from model_openai import generate
from simulator_sharded import ConversationSimulatorSharded
from simulator_controlled import ACTION_PROMPTS, fixed_schedule_policy
from state_tracker import TaskState, StateTracker


MEDIATOR_SYSTEM = (
    "You are a helpful assistant that reconstructs a user's full question from a multi-turn conversation. "
    "The user reveals information progressively across turns. Your job: produce ONE fully-specified, self-contained instruction "
    "that captures every piece of information the user has revealed up to and including the latest turn.\n\n"
    "Strict rules:\n"
    "1. Include ALL facts, numbers, constraints, and goals the user stated, across every turn.\n"
    "2. Do NOT reference previous turns (no 'as I said before', no 'in addition'). Write as if asked once.\n"
    "3. Do NOT add facts the user has not stated. Do NOT solve the problem. Output only the reconstructed question.\n"
    "4. Preserve the user's exact intent. If a later turn updates an earlier value, use the updated one.\n"
    "5. Output ONLY the reconstructed instruction text. No preamble. No explanation."
)


def _format_history_for_mediator(trace):
    lines, user_idx = [], 0
    for m in trace:
        role = m.get('role')
        if role == 'user':
            user_idx += 1
            lines.append(f'[User turn {user_idx}]: {m.get("content","")}')
        elif role == 'assistant':
            c = (m.get('content') or '')
            if len(c) > 400:
                c = c[:400] + '...'
            lines.append(f'[Assistant turn {user_idx}]: {c}')
    return '\n\n'.join(lines)


def noop_policy(state, t, history):
    return 'CONTINUE'


class MediatorRewrittenSharded(ConversationSimulatorSharded):
    """Sharded simulator with A1-style Mediator + optional directive controller.

    policy_fn(state, t, history) -> action_name controls per-turn directive.
    Default `noop_policy` => no directive (pure A1).
    """

    def __init__(self, sample, mediator_model='qwen2.5-14b', policy_fn=None,
                 tracker_model='qwen2.5-7b', track_state=False, **kw):
        super().__init__(sample, **kw)
        self.mediator_model = mediator_model
        self.policy_fn = policy_fn if policy_fn is not None else noop_policy
        self.track_state = track_state
        self.tracker = StateTracker(model=tracker_model) if track_state else None
        self.task_state = TaskState()
        self.action_history = []
        self.mediation_log = []

    def _mediate(self, trace_so_far):
        history_text = _format_history_for_mediator(trace_so_far)
        msgs = [
            {'role': 'system', 'content': MEDIATOR_SYSTEM},
            {'role': 'user', 'content': f'Conversation history:\n\n{history_text}\n\nReconstructed instruction:'},
        ]
        try:
            resp = generate(msgs, model=self.mediator_model, temperature=0.3, return_metadata=True, max_tokens=800)
            rewritten = resp['message'].strip() if isinstance(resp, dict) else str(resp).strip()
            for p in ('Reconstructed instruction:', 'Instruction:', 'Question:'):
                if rewritten.startswith(p):
                    rewritten = rewritten[len(p):].strip()
            return rewritten, resp.get('total_usd', 0.0) if isinstance(resp, dict) else 0.0
        except Exception:
            return None, 0.0

    def _render_directive(self, action):
        if action == 'CONTINUE':
            return None
        if action == 'INJECT_SUMMARY':
            if not (self.task_state.goal or self.task_state.facts or self.task_state.constraints):
                return None
            return f'Current TaskState (use it to verify your reasoning):\n{self.task_state.render_for_assistant()}'
        return ACTION_PROMPTS.get(action)

    def run(self, verbose=False, save_log=True):
        is_reasoning_model = ("o1" in self.assistant_model or "o3" in self.assistant_model or "deepseek-r1" in self.assistant_model)
        max_assistant_tokens = 10000 if is_reasoning_model else 1000
        is_completed, is_correct, score = False, False, None
        shards = self.sample["shards"]
        turn_idx = 0

        while not is_completed:
            revealed_shard_ids = set([msg["content"]["shard_id"] for msg in self.trace if msg["role"] == "log" and msg["content"]["type"] == "shard_revealed"])
            if len(revealed_shard_ids) == len(shards):
                break
            is_last_turn = len(revealed_shard_ids) == len(shards) - 1
            turn_idx += 1

            # 1. user shard
            user_response, shard_revealed_id, cost_usd = self.user_agent.generate_response(self.trace, self.sample, temperature=self.user_temperature)
            self.trace.append({"role": "user", "content": user_response, "timestamp": date_str(), "cost_usd": cost_usd})
            if verbose:
                print_colored(f"[user] {user_response}", "green")
            if shard_revealed_id != -1:
                self.trace.append({"role": "log", "content": {"type": "shard_revealed", "shard_id": shard_revealed_id}, "timestamp": date_str()})

            # 2. Mediator rewrite
            rewritten, mediator_cost = self._mediate(self.trace)
            if rewritten is None:
                rewritten = user_response  # fallback
            self.mediation_log.append({'turn': turn_idx, 'rewritten': rewritten})
            self.trace.append({"role": "log", "content": {"type": "mediator_rewrite", "turn": turn_idx, "rewritten": rewritten}, "timestamp": date_str(), "cost_usd": mediator_cost})
            if verbose:
                print_colored(f"[mediator] {rewritten[:200]}", "magenta")

            # 3. Optional state tracking
            if self.track_state and self.tracker:
                try:
                    self.tracker.update(self.task_state, user_response)
                except Exception:
                    pass

            # 4. Controller: pick directive
            action = self.policy_fn(self.task_state, turn_idx, self.action_history)
            self.action_history.append((turn_idx, action))
            directive = self._render_directive(action)
            if verbose:
                print_colored(f"[controller] action={action} directive={'<none>' if directive is None else directive[:80]}", "yellow")

            # 5. Build assistant input: system + (optional directive) + rewritten instruction
            assistant_input = [{'role': 'system', 'content': self.system_message}]
            if directive:
                assistant_input.append({'role': 'system', 'content': directive})
            assistant_input.append({'role': 'user', 'content': rewritten})

            assistant_response_obj = generate(assistant_input, model=self.assistant_model, temperature=self.assistant_temperature, return_metadata=True, max_tokens=max_assistant_tokens)
            assistant_response = assistant_response_obj["message"]
            self.trace.append({"role": "assistant", "content": assistant_response, "timestamp": date_str(), "cost_usd": assistant_response_obj["total_usd"]})
            if verbose:
                print_colored(f"[assistant] {assistant_response[:200]}", "red")

            # 6. system verification (unchanged)
            system_verification_response, verification_cost_usd = self.system_agent.verify_system_response(self.trace)
            self.trace.append({"role": "log", "content": {"type": "system-verification", "response": system_verification_response}, "timestamp": date_str(), "cost_usd": verification_cost_usd})

            if system_verification_response["response_type"] == "answer_attempt":
                extracted_answer = self.system_agent.extract_answer(self.trace)
                is_correct, score = None, None
                if self.task_name == "summary" and not is_last_turn:
                    evaluation_return = {"score": 0.0}; score = 0.0
                else:
                    evaluation_return = self.task.evaluator_function(extracted_answer, self.sample)
                    is_correct = evaluation_return.get("is_correct", None)
                    score = evaluation_return.get("score", None)
                if score == 1.0 and not is_correct:
                    is_correct = True
                self.trace.append({"role": "log", "content": {"type": "answer-evaluation", "exact_answer": extracted_answer, "is_correct": is_correct, "score": score, "evaluation_return": evaluation_return}, "timestamp": date_str()})
                if is_correct:
                    is_completed = True
                    self.trace.append({"role": "log", "content": {"type": "conversation-completed", "is_correct": is_correct}, "timestamp": date_str()})
            elif system_verification_response["response_type"] in ["clarification", "discussion"]:
                continue

        if save_log:
            conv_type = getattr(self, 'conv_type', 'mediator')
            log_conversation(conv_type, self.task.get_task_name(), self.sample["task_id"], self.dataset_fn, self.assistant_model, self.system_model, self.user_model, self.trace, is_correct, score, log_folder=self.log_folder)
        return is_correct, score
