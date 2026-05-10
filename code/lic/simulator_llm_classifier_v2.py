"""V2 LLM-as-classifier: 14B controller + few-shot examples + default-CONTINUE bias."""
import json, os
from utils import print_colored, extract_conversation, date_str
from utils_log import log_conversation
from model_openai import generate
from simulator_mediator import MediatorRewrittenSharded, _format_history_for_mediator

# Load mined examples (data-grounded, not hand-written)
EXAMPLES_PATH = '/root/autodl-tmp/DCC/few_shot_examples.json'
EXAMPLES = json.load(open(EXAMPLES_PATH))

# Format examples for the prompt
def format_examples():
    out = []
    for action, ex in EXAMPLES.items():
        out.append(f'\n--- Example: {action} was the right action for this task ---')
        out.append(f'Conversation excerpt:\n{ex["excerpt"][:600]}\n')
        out.append(f'→ Best action: {action}')
    return '\n'.join(out)

CONTROLLER_SYSTEM = (
    "You are a meta-controller in a multi-turn dialogue. Each turn, you choose ONE action that the assistant "
    "should follow. You read the conversation so far, then pick:\n\n"
    "  CONTINUE — proceed normally, no extra directive (DEFAULT, use unless evidence below)\n"
    "  VERIFY   — assistant must verify its previous numerical answer step-by-step (use when its last response gave a definite numeric/boxed answer that may be wrong)\n"
    "  CLARIFY  — assistant should ask one short clarifying question (use only when CRITICAL info is missing AND assistant has been stuck >2 turns)\n"
    "  RESET    — discard partial answers, recompute from scratch (use when assistant has clearly committed to wrong direction)\n"
    "  CONCLUDE — stop asking, commit to final boxed answer now (use when 3+ turns have revealed adequate info but assistant keeps clarifying)\n"
    "  SUMMARY  — re-anchor on TaskState (use rarely, only when conversation is very long and confused)\n\n"
    "Decision principles:\n"
    "1. CONTINUE is the default. Choose it unless one of the conditions below clearly applies.\n"
    "2. CONCLUDE is for assistants that over-clarify when they should commit.\n"
    "3. VERIFY is for assistants that gave a specific numeric answer that needs double-checking.\n"
    "4. CLARIFY is rare — only when the user has not given enough to compute the answer.\n"
    "5. Avoid the same action 2 turns in a row unless strong reason.\n\n"
    "FEW-SHOT EXAMPLES (mined from data; show what action helped):\n"
    + format_examples() +
    "\n\nNow you will see a real conversation. Output EXACTLY one of: CONTINUE, VERIFY, CLARIFY, RESET, CONCLUDE, SUMMARY. "
    "No explanation, no extra text. ONE WORD."
)
VALID_ACTIONS = {'CONTINUE', 'VERIFY', 'CLARIFY', 'RESET', 'CONCLUDE', 'SUMMARY'}


class LLMClassifierMediatorShardedV2(MediatorRewrittenSharded):
    def __init__(self, sample, controller_model='qwen2.5-14b', **kw):
        kw.setdefault('policy_fn', lambda state, t, history: self._llm_classifier(t, history))
        super().__init__(sample, **kw)
        self.controller_model = controller_model
        self.controller_log = []

    def _llm_classifier(self, t, history):
        hist_text = _format_history_for_mediator(self.trace)
        if len(hist_text) > 5000:
            hist_text = hist_text[-5000:]
        past_actions = [a for _, a in history]
        prev_a = past_actions[-1] if past_actions else 'NONE'
        msgs = [
            {'role': 'system', 'content': CONTROLLER_SYSTEM},
            {'role': 'user', 'content': (
                f'Real conversation history so far (turn {t} about to start):\n\n{hist_text}\n\n'
                f'Previous directives chosen by you: {past_actions[-5:]}\n'
                f'(Previous turn directive: {prev_a})\n\n'
                f'What ONE action should the assistant follow at turn {t}? Output one word.'
            )},
        ]
        try:
            resp = generate(msgs, model=self.controller_model, temperature=0.0, return_metadata=True, max_tokens=10)
            raw = resp['message'].strip().upper() if isinstance(resp, dict) else str(resp).strip().upper()
            chosen = 'CONTINUE'
            for a in VALID_ACTIONS:
                if a in raw:
                    chosen = a; break
            self.controller_log.append({'turn': t, 'raw': raw[:30], 'chosen': chosen})
            return 'INJECT_SUMMARY' if chosen == 'SUMMARY' else chosen
        except Exception:
            return 'CONTINUE'
