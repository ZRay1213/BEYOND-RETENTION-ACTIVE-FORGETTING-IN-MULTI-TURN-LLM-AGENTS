"""LLM-as-classifier controller. Each turn, ask 7B to pick one of 6 directives
given the (sharded) conversation history + brief action descriptions.

This bypasses hand-engineered features entirely: the 7B sees raw conversation
and chooses semantically. No training data, no Q-regression, no feature
engineering. Pure semantic policy via prompt.
"""
import os
from utils import print_colored, extract_conversation, date_str
from utils_log import log_conversation
from model_openai import generate
from simulator_mediator import MediatorRewrittenSharded, _format_history_for_mediator

CONTROLLER_SYSTEM = (
    "You are a meta-controller that chooses, each turn, what behavioral directive an assistant should follow "
    "in a multi-turn conversation. Your job: read the conversation so far and pick exactly ONE of these actions:\n\n"
    "1. CONTINUE — assistant proceeds normally with no extra directive.\n"
    "2. VERIFY — assistant must verify its previous numerical answer step-by-step before this turn.\n"
    "3. CLARIFY — assistant must ask one concise clarifying question instead of answering.\n"
    "4. RESET — assistant must discard any partial answer from earlier turns and recompute from scratch.\n"
    "5. CONCLUDE — assistant must stop asking, commit to a final boxed answer now.\n"
    "6. SUMMARY — assistant should re-anchor on the structured task state (facts/constraints).\n\n"
    "Decision rules (use your judgement):\n"
    "- CLARIFY when the user has just revealed information that conflicts with earlier facts, or when essential info is missing.\n"
    "- VERIFY when the assistant's last numeric output is suspect (long chain, error-prone arithmetic).\n"
    "- CONCLUDE when assistant keeps asking but enough info has been revealed for a final answer.\n"
    "- RESET when the assistant has clearly committed to a wrong direction.\n"
    "- SUMMARY rarely useful unless conversation is very long and confusion is evident.\n"
    "- CONTINUE by default.\n\n"
    "Output EXACTLY one of: CONTINUE, VERIFY, CLARIFY, RESET, CONCLUDE, SUMMARY. No explanation, no extra text."
)
VALID_ACTIONS = {'CONTINUE', 'VERIFY', 'CLARIFY', 'RESET', 'CONCLUDE', 'SUMMARY'}


class LLMClassifierMediatorSharded(MediatorRewrittenSharded):
    def __init__(self, sample, controller_model='qwen2.5-7b', **kw):
        # Default policy_fn closure that calls _llm_classifier
        kw.setdefault('policy_fn', lambda state, t, history: self._llm_classifier(t, history))
        super().__init__(sample, **kw)
        self.controller_model = controller_model
        self.controller_log = []  # decision observability

    def _llm_classifier(self, t, history):
        # Build history excerpt
        hist_text = _format_history_for_mediator(self.trace)
        if len(hist_text) > 6000:
            hist_text = hist_text[-6000:]
        msgs = [
            {'role': 'system', 'content': CONTROLLER_SYSTEM},
            {'role': 'user', 'content': f'Conversation history:\n\n{hist_text}\n\nThe assistant is about to respond at turn {t}. Past directives: {[a for _,a in history]}\n\nWhich action? Output one word.'},
        ]
        try:
            resp = generate(msgs, model=self.controller_model, temperature=0.0, return_metadata=True, max_tokens=10)
            raw = resp['message'].strip().upper() if isinstance(resp, dict) else str(resp).strip().upper()
            # find any valid action token in the response
            for a in VALID_ACTIONS:
                if a in raw:
                    chosen = a
                    break
            else:
                chosen = 'CONTINUE'  # fallback
            self.controller_log.append({'turn': t, 'raw': raw[:30], 'chosen': chosen})
            # Map SUMMARY to INJECT_SUMMARY (the internal action name our parent expects)
            return 'INJECT_SUMMARY' if chosen == 'SUMMARY' else chosen
        except Exception:
            return 'CONTINUE'
