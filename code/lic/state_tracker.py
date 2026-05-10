import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from model_openai import generate

EMPTY_STATE = {
    'goal': '',
    'constraints': [],
    'facts': {},
    'current_proposal': None,
    'open_questions': [],
}

PROMPT_TEMPLATE = '''You maintain a structured task state for a multi-turn conversation between a user and an assistant. After each user turn, update the state with new information.

Current TaskState (JSON):
{state_json}

Most recent user message:
"""
{user_msg}
"""

Most recent assistant response (may be empty for the first turn):
"""
{assistant_msg}
"""

Output an UPDATED TaskState as a single JSON object with exactly these fields:
- goal (string): the user's overall objective, refined as more is revealed
- constraints (array of strings): explicit constraints the user has stated
- facts (object): structured key-value facts the user has provided (numbers, parameters, names). Use snake_case keys.
- current_proposal (string or null): the assistant's current best answer or draft, if any
- open_questions (array of strings): things still uncertain that affect solving the task

Rules:
1. Preserve existing fields unless new information clearly supersedes them.
2. Only record what was EXPLICITLY stated in messages — never invent.
3. If the user contradicts earlier info, update the relevant field and remove it from open_questions.
4. Keep entries short (one phrase each).
5. Output ONLY the JSON object, no commentary, no markdown fences.
'''


@dataclass
class TaskState:
    goal: str = ''
    constraints: list = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    current_proposal: Optional[str] = None
    open_questions: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(
            goal=d.get('goal', '') or '',
            constraints=list(d.get('constraints') or []),
            facts=dict(d.get('facts') or {}),
            current_proposal=d.get('current_proposal'),
            open_questions=list(d.get('open_questions') or []),
        )

    def render_for_assistant(self) -> str:
        lines = []
        lines.append(f'GOAL: {self.goal or "(not yet specified)"}')
        lines.append('')
        lines.append('CONSTRAINTS:')
        if self.constraints:
            for c in self.constraints:
                lines.append(f'- {c}')
        else:
            lines.append('(none stated)')
        lines.append('')
        lines.append('KEY FACTS:')
        if self.facts:
            for k, v in self.facts.items():
                lines.append(f'- {k}: {v}')
        else:
            lines.append('(none stated)')
        lines.append('')
        lines.append(f'CURRENT PROPOSAL: {self.current_proposal or "(none yet)"}')
        lines.append('')
        lines.append('OPEN QUESTIONS:')
        if self.open_questions:
            for q in self.open_questions:
                lines.append(f'- {q}')
        else:
            lines.append('(none)')
        return '\n'.join(lines)


class StateTracker:
    def __init__(self, model: str = 'qwen2.5-7b', temperature: float = 0.0, max_tokens: int = 800):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def update(self, state: TaskState, user_msg: str, assistant_msg: str = '') -> tuple:
        prompt = PROMPT_TEMPLATE.format(
            state_json=json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
            user_msg=user_msg.strip(),
            assistant_msg=(assistant_msg or '').strip(),
        )
        messages = [{'role': 'user', 'content': prompt}]
        try:
            obj = generate(messages, model=self.model, temperature=self.temperature, return_metadata=True, max_tokens=self.max_tokens, is_json=True)
            text = obj['message']
            cost = obj.get('total_usd', 0.0)
            data = json.loads(text)
            new_state = TaskState.from_dict(data)
            return new_state, cost, None
        except Exception as e:
            return state, 0.0, f'{type(e).__name__}: {str(e)[:200]}'
