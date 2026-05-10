import json, os, sys
from collections import defaultdict
import numpy as np

def main(log_root):
    rows = []
    for task in os.listdir(log_root):
        td = os.path.join(log_root, task)
        if not os.path.isdir(td): continue
        for ct in os.listdir(td):
            cd = os.path.join(td, ct)
            if not os.path.isdir(cd): continue
            for fn in os.listdir(cd):
                if not fn.endswith('.jsonl'): continue
                with open(os.path.join(cd, fn)) as f:
                    for line in f:
                        d = json.loads(line)
                        rows.append({
                            'task': task,
                            'conv_type': ct,
                            'model': d['assistant_model'],
                            'task_id': d['task_id'],
                            'score': float(d.get("score") or 0.0),
                        })
    if not rows:
        print('No data'); return
    groups = defaultdict(list)
    for r in rows:
        groups[(r['model'], r['conv_type'], r['task'])].append(r)
    print('%-20s %-12s %-10s %-7s %-7s %-7s %-6s %-6s %-7s' % ('model','conv','task','N_inst','N_runs','acc','P10','P90','reliab'))
    summary = {}
    for key, recs in sorted(groups.items()):
        per_inst = defaultdict(list)
        for r in recs:
            per_inst[r['task_id']].append(r['score'])
        means = [np.mean(v) for v in per_inst.values()]
        n_inst = len(per_inst); n_runs = len(recs)
        acc = float(np.mean(means))
        p10 = float(np.percentile(means, 10))
        p90 = float(np.percentile(means, 90))
        rel = p90 - p10
        summary[key] = (acc, p10, p90, rel, n_inst, n_runs)
        print('%-20s %-12s %-10s %-7d %-7d %.3f   %.3f  %.3f  %.3f' % (key[0], key[1], key[2], n_inst, n_runs, acc, p10, p90, rel))
    print()
    print('=== Drop (full vs sharded) ===')
    print('%-20s %-10s %-7s %-8s %-7s' % ('model','task','full','sharded','drop'))
    by_mt = defaultdict(dict)
    for (m,c,t), (acc, *_) in summary.items():
        cb = c.split('-')[0]
        by_mt[(m,t)][cb] = acc
    for (m,t), d in sorted(by_mt.items()):
        if 'full' in d and 'sharded' in d:
            drop = d['full'] - d['sharded']
            print('%-20s %-10s %.3f   %.3f    %+.3f' % (m, t, d['full'], d['sharded'], drop))

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'data/lost_in_conversation/logs')
