"""Train Q-regression bandit on extracted features.

Q(s, a) = predicted final reward given state s and chosen action a.
At deployment: argmax_a Q(s, a).

Held-out: 28 tasks reserved for online deployment eval.
"""
import pickle, numpy as np, sys, os, json, random
from collections import defaultdict, Counter
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score

random.seed(42)
np.random.seed(42)

data = pickle.load(open('/root/autodl-tmp/DCC/bandit_train_data.pkl', 'rb'))
rows, ACTIONS = data['rows'], data['actions']
print(f'Loaded {len(rows)} rows, action space: {ACTIONS}')

# Train/test split BY TASK (not by row) to avoid leakage
all_tasks = sorted({r['task_id'] for r in rows})
random.shuffle(all_tasks)
held_out = set(all_tasks[:28])
train_tasks = set(all_tasks) - held_out
print(f'Train tasks: {len(train_tasks)}, held-out: {len(held_out)}')

train_rows = [r for r in rows if r['task_id'] in train_tasks]
heldout_rows = [r for r in rows if r['task_id'] in held_out]
print(f'Train rows: {len(train_rows)}, held-out rows: {len(heldout_rows)}')

# Feature: state + action one-hot. Target: reward.
def to_xy(rows):
    X, y = [], []
    for r in rows:
        feat = list(r['features'])
        ah = [0.0]*len(ACTIONS); ah[r['action']] = 1.0
        X.append(feat + ah); y.append(r['reward'])
    return np.array(X), np.array(y)

X_tr, y_tr = to_xy(train_rows)
X_he, y_he = to_xy(heldout_rows)

# Q-regression
print('\nFitting GradientBoostingRegressor...')
q_model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
q_model.fit(X_tr, y_tr)
y_pred_he = q_model.predict(X_he)
print(f'  train R²: {r2_score(y_tr, q_model.predict(X_tr)):.3f}')
print(f'  held-out R²: {r2_score(y_he, y_pred_he):.3f}')

# Per-action Q on held-out states (offline policy eval)
def predict_best_action(model, state_features):
    """For a single state vector (no action), return best action index."""
    Qs = []
    for a in range(len(ACTIONS)):
        ah = [0.0]*len(ACTIONS); ah[a] = 1.0
        x = np.array(state_features + ah).reshape(1, -1)
        Qs.append(model.predict(x)[0])
    return int(np.argmax(Qs)), Qs

# Estimate held-out policy reward via DR-style:
# For each held-out (task, turn), bandit predicts action a*. Look up empirical reward of that
# (task, schedule) where schedule's directive at turn = a*. If no exact match, use action's mean reward across all tasks.
print('\n--- Offline evaluation on held-out tasks ---')

# Build (task_id, turn, schedule) -> reward lookup
lookup = defaultdict(dict)  # (task_id, turn) -> {schedule: reward}
for r in rows:
    lookup[(r['task_id'], r['turn'])].setdefault(r['schedule'], []).append(r['reward'])

# Best/worst per (task, turn)
def empirical_reward_for_action(task_id, turn, action_idx):
    """Average reward across all training rows where (task=task_id, turn=turn, action=action_idx)."""
    rs = [r['reward'] for r in rows if r['task_id']==task_id and r['turn']==turn and r['action']==action_idx]
    return np.mean(rs) if rs else None

# For each held-out task, compute what bandit predicts at each turn under "no-rollout" eval:
# average action across turns => proxy "would-be schedule"
print('\nBandit predicted action distribution on held-out states:')
pred_counter = Counter()
for r in heldout_rows:
    pred_a, Qs = predict_best_action(q_model, list(r['features']))
    pred_counter[ACTIONS[pred_a]] += 1
total = sum(pred_counter.values())
for a, c in pred_counter.most_common():
    print(f'  {a}: {c} ({100*c/total:.1f}%)')

# Save model
import joblib
joblib.dump({'q_model': q_model, 'feature_names': data['feature_names'], 'actions': ACTIONS,
             'train_tasks': sorted(train_tasks), 'held_out_tasks': sorted(held_out)},
            '/root/autodl-tmp/DCC/bandit_q_model.joblib')
print('\nSaved /root/autodl-tmp/DCC/bandit_q_model.joblib')
print(f'Held-out tasks (28): {sorted(held_out)[:6]}... (use these for deployment eval)')

# Also save held-out task list to a file for the runner
with open('/root/autodl-tmp/DCC/heldout_tasks.json', 'w') as f:
    json.dump(sorted(held_out), f)
print('Saved /root/autodl-tmp/DCC/heldout_tasks.json')
