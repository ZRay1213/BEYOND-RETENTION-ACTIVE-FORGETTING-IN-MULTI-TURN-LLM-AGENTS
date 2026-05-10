"""V2 feature extraction: original 14 features + TF-IDF on mediator-rewrite text."""
import json, glob, os, pickle, sys
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT_A1 = '/root/autodl-tmp/DCC/data/lost_in_conversation/logs_stage3_a1_math100/math'
OUT_PATH = '/root/autodl-tmp/DCC/bandit_train_data_v2.pkl'

ACTIONS = ['CONTINUE', 'VERIFY', 'CLARIFY', 'RESET', 'INJECT_SUMMARY', 'CONCLUDE']
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}


def schedule_action_at_turn(schedule, t, action_history):
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


def extract_rows(trace, schedule, sim_score, task_id):
    rows = []
    user_msgs, asst_msgs, mediator_rewrites, verifs = [], [], [], []
    for m in trace:
        if not isinstance(m, dict): continue
        role, c = m.get('role'), m.get('content')
        if role == 'user' and isinstance(c, str): user_msgs.append(c)
        elif role == 'assistant' and isinstance(c, str): asst_msgs.append(c)
        elif role == 'log' and isinstance(c, dict):
            t = c.get('type')
            if t == 'mediator_rewrite': mediator_rewrites.append(c.get('rewritten', ''))
            elif t == 'system-verification':
                rt = c.get('response', {}).get('response_type', 'none') if isinstance(c.get('response'), dict) else 'none'
                verifs.append(rt)
    n_turns = min(len(user_msgs), len(mediator_rewrites))
    cum_asst_tokens = 0
    action_hist = []
    for t in range(n_turns):
        med = mediator_rewrites[t] if t < len(mediator_rewrites) else ''
        med_len = len(med.split())
        med_len_delta = med_len - len(mediator_rewrites[t-1].split()) if t > 0 else 0
        if t > 0 and t-1 < len(asst_msgs):
            prev_asst = asst_msgs[t-1]
            prev_asst_tokens = len(prev_asst.split())
            prev_has_boxed = 1.0 if '\\boxed' in prev_asst else 0.0
            prev_asst_len_delta = prev_asst_tokens - len(asst_msgs[t-2].split()) if t >= 2 else 0.0
        else:
            prev_asst_tokens, prev_has_boxed, prev_asst_len_delta = 0.0, 0.0, 0.0
        cum_asst_tokens += prev_asst_tokens
        if t > 0 and t-1 < len(verifs):
            v_map = {'answer_attempt': 1.0, 'clarification': 0.0, 'discussion': 2.0, 'hedge': 0.5}
            last_verif = v_map.get(verifs[t-1], -1.0)
        else:
            last_verif = -1.0
        ah_counts = [0]*6
        for _, a in action_hist:
            ah_counts[ACTION_IDX.get(a, 0)] += 1
        action_at_t = schedule_action_at_turn(schedule, t+1, action_hist)
        feats = [
            float(t+1), med_len, float(med_len_delta),
            prev_asst_tokens, prev_has_boxed, float(prev_asst_len_delta),
            cum_asst_tokens, last_verif,
        ] + ah_counts
        rows.append({
            'features': feats, 'action': ACTION_IDX.get(action_at_t, 0),
            'reward': sim_score, 'turn': t+1, 'schedule': schedule,
            'mediator_text': med, 'task_id': task_id,
        })
        action_hist.append((t+1, action_at_t))
    return rows


def main():
    all_rows, n_sims = [], 0
    for sch_dir in sorted(glob.glob(f'{ROOT_A1}/mediator_*')):
        schedule = os.path.basename(sch_dir).replace('mediator_', '')
        for jsonl in glob.glob(f'{sch_dir}/*.jsonl'):
            for line in open(jsonl):
                try:
                    rec = json.loads(line)
                except: continue
                sc = rec.get('score')
                if sc is None: continue
                rows = extract_rows(rec.get('trace', []), schedule, float(sc), rec.get('task_id', '?'))
                if not rows: continue
                n_sims += 1
                all_rows.extend(rows)
    print(f'Extracted {len(all_rows)} rows from {n_sims} sims')

    # TF-IDF over mediator_text. Fit on training mediator-rewrites only would be cleanest,
    # but task split happens later — fitting on ALL is fine because TF-IDF doesn't see rewards.
    texts = [r['mediator_text'] for r in all_rows]
    vec = TfidfVectorizer(max_features=200, ngram_range=(1,2), stop_words='english', min_df=5)
    tfidf_mat = vec.fit_transform(texts)  # sparse (n, 200)
    print(f'TF-IDF: shape={tfidf_mat.shape}, nnz density={tfidf_mat.nnz / (tfidf_mat.shape[0]*tfidf_mat.shape[1]):.3f}')
    print(f'Top vocab samples: {list(vec.vocabulary_.keys())[:15]}')

    # Save: rows + vectorizer + tfidf matrix (sparse, but dense version is 14k*200 = 2.8M floats fine)
    tfidf_dense = tfidf_mat.toarray()
    pickle.dump({
        'rows': all_rows, 'tfidf_features': tfidf_dense, 'vectorizer': vec,
        'actions': ACTIONS,
        'feature_names': ['turn_idx','med_len','med_len_delta','prev_asst_tokens','prev_has_boxed',
                          'prev_asst_len_delta','cum_asst_tokens','last_verif',
                          'ah_CONT','ah_VER','ah_CLR','ah_RES','ah_SUM','ah_CON'] + [f'tfidf_{i}' for i in range(200)],
    }, open(OUT_PATH, 'wb'))
    print(f'Saved {OUT_PATH} (rows + 14d state + 200d tfidf = 214d total)')


if __name__ == '__main__': main()
