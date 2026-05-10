"""Extract per-turn (state_features, action, reward) tuples from A1 mediator traces.

For each saved sim trace, walk turns and extract:
- state_features: per-turn observable signals (turn_idx, lengths, assistant patterns, action history)
- action: directive injected at this turn (CONTINUE if none)
- reward: sim's terminal task accuracy

Output: a pickle of dict with X (n_samples, n_features), y (action), r (reward), task_id, turn_idx.
"""
import json, glob, os, pickle, sys
from collections import Counter, defaultdict
import numpy as np

ROOT_A1 = '/root/autodl-tmp/DCC/data/lost_in_conversation/logs_stage3_a1_math100/math'
OUT_PATH = '/root/autodl-tmp/DCC/bandit_train_data.pkl'

ACTIONS = ['CONTINUE', 'VERIFY', 'CLARIFY', 'RESET', 'INJECT_SUMMARY', 'CONCLUDE']
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}


def schedule_action_at_turn(schedule, t, action_history):
    """Reproduce the policy that was used when generating this trace."""
    if schedule == 'NONE': return 'CONTINUE'
    if schedule == 'SUMMARY-EVERY': return 'INJECT_SUMMARY'
    if schedule.startswith('VERIFY-K'):
        k = int(schedule.split('K')[1]); return 'VERIFY' if (t > 0 and t % k == 0) else 'CONTINUE'
    if schedule.startswith('CLARIFY-K'):
        k = int(schedule.split('K')[1]); return 'CLARIFY' if (t > 0 and t % k == 0) else 'CONTINUE'
    if schedule.startswith('RESET-K'):
        k = int(schedule.split('K')[1]); return 'RESET' if (t > 0 and t % k == 0) else 'CONTINUE'
    if schedule.startswith('CONCLUDE-K'):
        k = int(schedule.split('K')[1])
        if t >= k and not any(a == 'CONCLUDE' for _, a in action_history): return 'CONCLUDE'
        return 'CONTINUE'
    return 'CONTINUE'


def extract_per_turn(trace, schedule, sim_score):
    """Walk a trace and emit one (features, action, reward) per turn."""
    rows = []
    user_msgs, asst_msgs, mediator_rewrites, verifs = [], [], [], []
    for m in trace:
        if not isinstance(m, dict): continue
        role = m.get('role')
        c = m.get('content')
        if role == 'user' and isinstance(c, str):
            user_msgs.append(c)
        elif role == 'assistant' and isinstance(c, str):
            asst_msgs.append(c)
        elif role == 'log' and isinstance(c, dict):
            t = c.get('type')
            if t == 'mediator_rewrite':
                mediator_rewrites.append(c.get('rewritten', ''))
            elif t == 'system-verification':
                rt = c.get('response', {}).get('response_type', 'none') if isinstance(c.get('response'), dict) else 'none'
                verifs.append(rt)
    n_turns = min(len(user_msgs), len(mediator_rewrites))
    if n_turns == 0:
        return rows
    cum_asst_tokens = 0
    action_hist = []
    for t in range(n_turns):
        # state features at turn t+1 (before policy decides)
        med = mediator_rewrites[t] if t < len(mediator_rewrites) else ''
        med_len = len(med.split())
        med_len_delta = med_len - len(mediator_rewrites[t-1].split()) if t > 0 and t-1 < len(mediator_rewrites) else 0
        if t > 0 and t-1 < len(asst_msgs):
            prev_asst = asst_msgs[t-1]
            prev_asst_tokens = len(prev_asst.split())
            prev_has_boxed = 1.0 if '\\boxed' in prev_asst or '\\boxed{' in prev_asst else 0.0
            prev_asst_len_delta = prev_asst_tokens - len(asst_msgs[t-2].split()) if t >= 2 else 0.0
        else:
            prev_asst_tokens, prev_has_boxed, prev_asst_len_delta = 0.0, 0.0, 0.0
        cum_asst_tokens += prev_asst_tokens
        # last verification response type (encoded numerically)
        if t > 0 and t-1 < len(verifs):
            v_map = {'answer_attempt': 1.0, 'clarification': 0.0, 'discussion': 2.0, 'hedge': 0.5}
            last_verif = v_map.get(verifs[t-1], -1.0)
        else:
            last_verif = -1.0
        # action history: count of each action so far
        ah_counts = [0]*6
        for _, a in action_hist:
            ah_counts[ACTION_IDX.get(a, 0)] += 1
        # action taken at this turn (deterministic from schedule)
        action_at_t = schedule_action_at_turn(schedule, t+1, action_hist)
        feats = [
            float(t+1),                        # turn_idx (1-based)
            med_len, float(med_len_delta),     # mediator-rewrite signals
            prev_asst_tokens, prev_has_boxed, float(prev_asst_len_delta),
            cum_asst_tokens, last_verif,
        ] + ah_counts
        rows.append({
            'features': feats,
            'action': ACTION_IDX.get(action_at_t, 0),
            'reward': sim_score,
            'turn': t+1,
            'schedule': schedule,
        })
        action_hist.append((t+1, action_at_t))
    return rows


def main():
    all_rows, by_task = [], defaultdict(list)
    n_sims = 0
    for sch_dir in sorted(glob.glob(f'{ROOT_A1}/mediator_*')):
        schedule = os.path.basename(sch_dir).replace('mediator_', '')
        for jsonl in glob.glob(f'{sch_dir}/*.jsonl'):
            for line in open(jsonl):
                try:
                    rec = json.loads(line)
                except: continue
                sc = rec.get('score')
                if sc is None: continue
                trace = rec.get('trace', [])
                rows = extract_per_turn(trace, schedule, float(sc))
                if not rows: continue
                n_sims += 1
                for r in rows:
                    r['task_id'] = rec.get('task_id', 'unknown')
                    all_rows.append(r)
                    by_task[r['task_id']].append(r)
    print(f'Extracted {len(all_rows)} (state,action,reward) rows from {n_sims} sims')
    print(f'Tasks covered: {len(by_task)}')
    print(f'Action distribution:', Counter([ACTIONS[r["action"]] for r in all_rows]).most_common())
    print(f'Reward mean: {np.mean([r["reward"] for r in all_rows]):.3f}')
    print(f'Mean turns/sim: {len(all_rows)/n_sims:.1f}')
    pickle.dump({
        'rows': all_rows, 'actions': ACTIONS, 'feature_dim': len(all_rows[0]['features']),
        'feature_names': ['turn_idx','med_len','med_len_delta','prev_asst_tokens','prev_has_boxed',
                          'prev_asst_len_delta','cum_asst_tokens','last_verif',
                          'ah_CONT','ah_VER','ah_CLR','ah_RES','ah_SUM','ah_CON'],
    }, open(OUT_PATH, 'wb'))
    print(f'Saved to {OUT_PATH}')


if __name__ == '__main__': main()
