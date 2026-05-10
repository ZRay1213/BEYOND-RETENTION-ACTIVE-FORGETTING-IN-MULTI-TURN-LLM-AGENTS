"""V2 training: state features + TF-IDF + action one-hot → reward.

Also evaluate two policy variants:
- Q-regression: argmax_a Q(s, a)
- Imitation of per-task oracle action at each turn
"""
import pickle, numpy as np, sys, os, json, random
from collections import defaultdict, Counter
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, accuracy_score
import joblib

random.seed(42); np.random.seed(42)
data = pickle.load(open('/root/autodl-tmp/DCC/bandit_train_data_v2.pkl', 'rb'))
rows, ACTIONS, tfidf = data['rows'], data['actions'], data['tfidf_features']
heldout = set(json.load(open('/root/autodl-tmp/DCC/heldout_tasks.json')))

# Build train/test by task
train_idx = [i for i,r in enumerate(rows) if r['task_id'] not in heldout]
he_idx = [i for i,r in enumerate(rows) if r['task_id'] in heldout]
print(f'train rows {len(train_idx)}, held-out rows {len(he_idx)}')

def build_xy(idxs):
    X, y, ah = [], [], []
    for i in idxs:
        r = rows[i]
        feat = list(r['features']) + tfidf[i].tolist()
        ah_oh = [0.0]*len(ACTIONS); ah_oh[r['action']] = 1.0
        X.append(feat + ah_oh); y.append(r['reward']); ah.append(r['action'])
    return np.array(X), np.array(y), np.array(ah)

X_tr, y_tr, a_tr = build_xy(train_idx)
X_he, y_he, a_he = build_xy(he_idx)
print(f'X_tr shape: {X_tr.shape}, X_he shape: {X_he.shape}')

# Q-regression with GBR
print('\n[Q-regression GradientBoostingRegressor]')
q = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
q.fit(X_tr, y_tr)
print(f'  train R²: {r2_score(y_tr, q.predict(X_tr)):.3f}, held-out R²: {r2_score(y_he, q.predict(X_he)):.3f}')

# Q-regression with Ridge (less overfitting on high-d)
print('[Q-regression Ridge]')
qr = Ridge(alpha=1.0)
qr.fit(X_tr, y_tr)
print(f'  train R²: {r2_score(y_tr, qr.predict(X_tr)):.3f}, held-out R²: {r2_score(y_he, qr.predict(X_he)):.3f}')

# Pick which to deploy: GBR vs Ridge based on held-out R²
he_r2_gbr = r2_score(y_he, q.predict(X_he))
he_r2_ridge = r2_score(y_he, qr.predict(X_he))
deploy_q = q if he_r2_gbr > he_r2_ridge else qr
deploy_name = 'GBR' if he_r2_gbr > he_r2_ridge else 'Ridge'
print(f'\nDeploy {deploy_name} (held-out R² = {max(he_r2_gbr, he_r2_ridge):.3f})')

# Bandit prediction distribution on held-out
def predict_action(model, state_with_tfidf):
    Qs = []
    for a in range(len(ACTIONS)):
        ah = [0.0]*len(ACTIONS); ah[a] = 1.0
        x = np.array(list(state_with_tfidf) + ah).reshape(1, -1)
        Qs.append(float(model.predict(x)[0]))
    return int(np.argmax(Qs)), Qs

# Use *each held-out row's state* (excluding action one-hot) as input
state_dim = 14 + 200  # original + tfidf
predicted = []
for i in he_idx:
    r = rows[i]
    s = list(r['features']) + tfidf[i].tolist()
    a_pred, _ = predict_action(deploy_q, s)
    predicted.append(a_pred)
print(f'\nHeld-out predicted action distribution:')
for a, c in Counter(predicted).most_common():
    print(f'  {ACTIONS[a]}: {c} ({100*c/len(predicted):.1f}%)')

# Imitation classifier: features → action (use only "winning" rows: high-reward where possible)
# Per (task, turn) find best (action, reward) and use that as label
print('\n[Imitation: train classifier on best-action-per-(task,turn) labels]')
best_action_label = {}  # (tid, turn) -> action
for r in rows:
    if r['task_id'] in heldout: continue
    key = (r['task_id'], r['turn'])
    if key not in best_action_label or r['reward'] > best_action_label[key][1]:
        best_action_label[key] = (r['action'], r['reward'])
imit_X, imit_y = [], []
for i in train_idx:
    r = rows[i]
    key = (r['task_id'], r['turn'])
    if key in best_action_label:
        feat = list(r['features']) + tfidf[i].tolist()
        imit_X.append(feat); imit_y.append(best_action_label[key][0])
imit_X, imit_y = np.array(imit_X), np.array(imit_y)
print(f'Imitation training set: {imit_X.shape}, label dist: {Counter(imit_y).most_common()}')
clf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
clf.fit(imit_X, imit_y)
print(f'  train acc: {clf.score(imit_X, imit_y):.3f}')

# Save both models
joblib.dump({'q_model': deploy_q, 'q_kind': deploy_name,
             'imit_model': clf, 'state_feature_dim': 14, 'tfidf_dim': 200,
             'tfidf_vectorizer': data['vectorizer'], 'actions': ACTIONS,
             'held_out_tasks': sorted(heldout)},
            '/root/autodl-tmp/DCC/bandit_q_model_v2.joblib')
print('Saved /root/autodl-tmp/DCC/bandit_q_model_v2.joblib')
