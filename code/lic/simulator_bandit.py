"""Deploy trained Q-regression bandit as DCC controller in MediatorRewrittenSharded."""
import joblib, numpy as np, os
from simulator_mediator import MediatorRewrittenSharded

ACTIONS = ['CONTINUE', 'VERIFY', 'CLARIFY', 'RESET', 'INJECT_SUMMARY', 'CONCLUDE']
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}

class BanditMediatorSharded(MediatorRewrittenSharded):
    def __init__(self, sample, q_model_path='/root/autodl-tmp/DCC/bandit_q_model.joblib', **kw):
        # Build a stateful policy that holds state we need
        bundle = joblib.load(q_model_path)
        self.q_model = bundle['q_model']
        self.q_actions = bundle['actions']
        self._predicted_qs = []  # decision observability log
        self._cum_asst_tokens = 0
        self._prev_asst = ''
        self._prev_prev_asst = ''
        self._prev_med = ''
        self._last_verif = -1.0
        # closure over self
        def policy_fn(state, t, history):
            return self._bandit_decide(t, history)
        super().__init__(sample, policy_fn=policy_fn, **kw)

    def _extract_state_at_turn(self, t, history):
        # State as in extract_bandit_features.py: rebuild from current trace
        med_len = len(self._cur_rewritten.split()) if hasattr(self, '_cur_rewritten') else 0
        med_len_delta = med_len - len(self._prev_med.split()) if self._prev_med else 0
        prev_asst_tokens = len(self._prev_asst.split())
        prev_has_boxed = 1.0 if '\\boxed' in self._prev_asst else 0.0
        prev_asst_len_delta = prev_asst_tokens - len(self._prev_prev_asst.split()) if self._prev_prev_asst else 0.0
        ah_counts = [0.0]*6
        for _, a in history:
            ah_counts[ACTION_IDX.get(a, 0)] += 1
        feats = [
            float(t), float(med_len), float(med_len_delta),
            float(prev_asst_tokens), prev_has_boxed, float(prev_asst_len_delta),
            float(self._cum_asst_tokens), self._last_verif,
        ] + ah_counts
        return feats

    def _bandit_decide(self, t, history):
        feats = self._extract_state_at_turn(t, history)
        # argmax_a Q(s, a)
        best_a, best_q = 0, -1e9
        Qs = []
        for a_idx in range(len(ACTIONS)):
            ah = [0.0]*len(ACTIONS); ah[a_idx] = 1.0
            x = np.array(feats + ah).reshape(1, -1)
            q = float(self.q_model.predict(x)[0])
            Qs.append(q)
            if q > best_q:
                best_q, best_a = q, a_idx
        self._predicted_qs.append({'turn': t, 'Qs': Qs, 'chosen': ACTIONS[best_a]})
        return ACTIONS[best_a]

    # We need to update state-tracking variables before each policy call.
    # Easiest: override the parts of run() that we need. Reuse parent run by
    # injecting state tracking through a wrapper. Concretely override _mediate
    # to capture rewritten, and patch self.policy_fn with closure.
    def _mediate(self, trace_so_far):
        rew, cost = super()._mediate(trace_so_far)
        if rew is not None:
            self._prev_med = getattr(self, '_cur_rewritten', '')
            self._cur_rewritten = rew
        return rew, cost

    def run(self, verbose=False, save_log=True):
        # Wrap parent run by patching state updates inline
        # Hack: re-run base loop but with state tracking
        # Simpler: monkey-patch generate-call timing by tracking last assistant after each turn.
        # We'll manually scan trace after parent.run completes... but policy needs state DURING run.
        # Solution: use parent run; parent invokes policy_fn which calls _bandit_decide which reads
        # state from self.* — those self.* must be updated each turn. Update them via overriding by
        # pre-policy peek: scan self.trace before policy call.
        # We do this by overriding policy_fn assignment to first refresh state.
        original_policy = self.policy_fn
        def wrapped_policy(state, t, history):
            self._refresh_state_from_trace()
            return original_policy(state, t, history)
        self.policy_fn = wrapped_policy
        return super().run(verbose=verbose, save_log=save_log)

    def _refresh_state_from_trace(self):
        # Walk current self.trace and refresh state-tracking vars
        asst_msgs, verifs = [], []
        for m in self.trace:
            if not isinstance(m, dict): continue
            if m.get('role') == 'assistant' and isinstance(m.get('content'), str):
                asst_msgs.append(m['content'])
            elif m.get('role') == 'log' and isinstance(m.get('content'), dict):
                if m['content'].get('type') == 'system-verification':
                    rt = m['content'].get('response', {}).get('response_type', 'none') if isinstance(m['content'].get('response'), dict) else 'none'
                    verifs.append(rt)
        self._prev_asst = asst_msgs[-1] if asst_msgs else ''
        self._prev_prev_asst = asst_msgs[-2] if len(asst_msgs) >= 2 else ''
        self._cum_asst_tokens = sum(len(m.split()) for m in asst_msgs)
        v_map = {'answer_attempt': 1.0, 'clarification': 0.0, 'discussion': 2.0, 'hedge': 0.5}
        self._last_verif = v_map.get(verifs[-1], -1.0) if verifs else -1.0
