"""Population-aware proposer V2 — sees ALL prior variants + their per-task scores.

Key difference from V1:
- V1: proposer sees only parent harness + traces from one variant
- V2: proposer sees ALL variants (G0..G3) source + per-task fitness matrix +
  highlights tasks each variant uniquely solves/fails

This gives the LLM a "synthesis" view: it can spot which mutations matter
for which task patterns and combine them.
"""
import argparse, json, glob, os, re, sys, statistics
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_openai import generate

PROPOSER_MODEL = 'qwen2.5-14b'

# Variants to include (path, log_dir name)
VARIANTS = [
    ('G0',     '/root/autodl-tmp/DCC/data/lost_in_conversation/harness.py',           'g0_t10'),
    ('G1_C0',  '/root/autodl-tmp/DCC/harness_population/harness_g1_c0.py',           'g1c0_t10'),
    ('G1_C1',  '/root/autodl-tmp/DCC/harness_population/harness_g1_c1.py',           'g1c1_t10'),
    ('G2_C0',  '/root/autodl-tmp/DCC/harness_population/harness_g2_c0.py',           'g2c0_t10'),
    ('G2_C1',  '/root/autodl-tmp/DCC/harness_population/harness_g2_c1.py',           'g2c1_t10'),
    ('G3_C0',  '/root/autodl-tmp/DCC/harness_population/harness_g3_c0.py',           'g3c0_t10'),
    ('G3_C1',  '/root/autodl-tmp/DCC/harness_population/harness_g3_c1.py',           'g3c1_t10'),
]
LOG_ROOT = '/root/autodl-tmp/DCC/data/lost_in_conversation'


def load_per_task(d):
    sims = []
    for f in glob.glob(f'{d}/*.jsonl'):
        for ln in open(f):
            try:
                r = json.loads(ln)
                if r.get('score') is not None:
                    sims.append((r['task_id'], r['score']))
            except: pass
    by = defaultdict(list)
    for t, s in sims: by[t].append(s)
    return {t: statistics.mean(rs) for t, rs in by.items()}


def extract_mutable_section(src):
    """Pull just the [MUTABLE] sections from a harness, to keep prompt compact."""
    out = []
    in_mut = False
    for ln in src.split('\n'):
        if '[MUTABLE]' in ln: in_mut = True
        if in_mut:
            out.append(ln)
            if ln.strip() == '' and len(out) > 3 and out[-2].strip() == '':
                # blank-blank ends a section
                pass
        if in_mut and ln.startswith('# =========') and 'MUTABLE' not in ln:
            in_mut = False
    return '\n'.join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gen', type=int, default=4)
    p.add_argument('--n_variants', type=int, default=2)
    p.add_argument('--output_dir', default='/root/autodl-tmp/DCC/harness_population')
    args = p.parse_args()

    # Load all variant sources + per-task scores
    population = []
    all_tasks = set()
    for name, src_path, log_name in VARIANTS:
        if not os.path.exists(src_path):
            print(f'[skip] {name}: source not found at {src_path}')
            continue
        src = open(src_path).read()
        log_dir = f'{LOG_ROOT}/logs_{log_name}/math/{log_name}'
        pt = load_per_task(log_dir) if os.path.isdir(log_dir) else {}
        all_tasks.update(pt)
        population.append({'name': name, 'src': src, 'per_task': pt})
        print(f'{name}: {len(src)} chars, {len(pt)} tasks scored')

    # Build score matrix
    all_tasks = sorted(all_tasks)
    print(f'\nTotal unique tasks across all variants: {len(all_tasks)}')

    # Build matrix string
    matrix_lines = ['## Per-task acc matrix (rows=tasks, cols=variants):']
    header = 'task_id'.ljust(30) + '|' + ''.join(p['name'].rjust(8) for p in population)
    matrix_lines.append(header)
    matrix_lines.append('-' * len(header))
    for t in all_tasks:
        row = t.ljust(30) + '|'
        for p in population:
            sc = p['per_task'].get(t)
            row += (f'{sc:.2f}' if sc is not None else '   -').rjust(8)
        matrix_lines.append(row)
    score_matrix = '\n'.join(matrix_lines)

    # Identify uniquely-solved tasks per variant
    insights = []
    for p in population:
        unique_solved = []
        for t, sc in p['per_task'].items():
            if sc < 0.5: continue
            other_solved = [q for q in population if q['name']!=p['name'] and q['per_task'].get(t,0) >= 0.5]
            if not other_solved:
                unique_solved.append(t)
        if unique_solved:
            insights.append(f'  {p["name"]} uniquely solves: {unique_solved}')
    insights_str = '\n'.join(insights) if insights else '  (no clear unique solvers)'

    # Per-variant per-task summary
    per_variant_summary = []
    for p in population:
        if not p['per_task']: continue
        m = statistics.mean(p['per_task'].values())
        per_variant_summary.append(f'  {p["name"]}: per-task acc = {m:.3f} (n_tasks={len(p["per_task"])})')

    # Build prompt
    # Show source of best variant + diffs/summary of others to keep prompt size manageable
    best = max(population, key=lambda x: statistics.mean(x['per_task'].values()) if x['per_task'] else 0)
    print(f'\nBest variant: {best["name"]}')

    sys_prompt = """You are an expert harness mutator. You see a POPULATION of harness variants and their per-task scores. Your job: synthesize a NEW variant that combines the strengths of multiple variants and addresses weaknesses revealed by the score matrix.

RULES:
1. Output the COMPLETE new harness.py inside ```python``` fences.
2. Maintain the structural API (run_harness entry, HarnessSharded class, imports unchanged).
3. Look for ROBUST mutations across variants — patterns that helped multiple times.
4. Look for UNIQUE solves — what mutation cracked task X that no other variant solved?
5. Combine cleanly. Don't kitchen-sink — pick 2-3 high-leverage mutations to integrate.
6. After code, add a CHANGE_RATIONALE paragraph citing the matrix evidence."""

    user_prompt_parts = [
        f'## POPULATION SUMMARY:\n', '\n'.join(per_variant_summary), '\n\n',
        score_matrix, '\n\n',
        '## UNIQUE-SOLVE INSIGHTS:\n', insights_str, '\n\n',
        '## BEST VARIANT FULL SOURCE (',  best['name'], '):\n```python\n', best['src'], '\n```\n\n',
        '## OTHER VARIANTS — diff from best (showing the mutable bits):\n',
    ]
    for p in population:
        if p['name'] == best['name']: continue
        # Just show their CONTROLLER + DIRECTIVE_TEMPLATES sections
        # Quick: extract lines around DIRECTIVE_TEMPLATES and controller_select_action
        src = p['src']
        # Find DIRECTIVE_TEMPLATES block
        m1 = re.search(r"(DIRECTIVE_TEMPLATES = \{[\s\S]+?\n\})", src)
        m2 = re.search(r"(def controller_select_action[\s\S]+?\n        return 'CONTINUE'\n)", src)
        excerpt = []
        if m1: excerpt.append(m1.group(1)[:1200])
        if m2: excerpt.append(m2.group(1)[:1500])
        user_prompt_parts.append(f'\n### {p["name"]} (per-task={statistics.mean(p["per_task"].values()):.3f}):\n```python\n')
        user_prompt_parts.append('\n# ...\n'.join(excerpt) if excerpt else src[:2000])
        user_prompt_parts.append('\n```\n')

    user_prompt_parts.append('\n## YOUR TASK:\nProduce ONE synthesized harness that combines the best mutations across this population. Output complete file in ```python``` fences, then CHANGE_RATIONALE paragraph citing matrix evidence.')

    user_prompt = ''.join(user_prompt_parts)
    print(f'\nPrompt: ~{len(user_prompt)//4} tokens')
    if len(user_prompt) > 50000:
        print(f'WARNING: prompt very long, may need trimming')

    os.makedirs(args.output_dir, exist_ok=True)
    for vi in range(args.n_variants):
        print(f'\n[gen {args.gen} variant {vi}] calling V2 proposer...')
        msgs = [{'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': user_prompt}]
        try:
            resp = generate(msgs, model=PROPOSER_MODEL, temperature=0.7, return_metadata=True, max_tokens=8000)
            text = resp['message'] if isinstance(resp, dict) else str(resp)
        except Exception as e:
            print(f'  fail: {e}'); continue
        m = re.search(r'```python\s*\n(.*?)\n```', text, re.DOTALL)
        if not m:
            print(f'  no python block. Head: {text[:300]}'); continue
        code = m.group(1)
        rm = re.search(r'CHANGE_RATIONALE[:\s]*(.+)', text, re.DOTALL)
        rationale = rm.group(1).strip()[:800] if rm else '(none)'
        out = os.path.join(args.output_dir, f'harness_g{args.gen}_v2c{vi}.py')
        open(out, 'w').write(code)
        open(out.replace('.py','_rationale.md'),'w').write(rationale)
        import py_compile
        try:
            py_compile.compile(out, doraise=True)
            print(f'  OK {out}')
            print(f'  rationale: {rationale[:200]}')
        except py_compile.PyCompileError as e:
            print(f'  SYNTAX ERROR: {str(e)[:200]}')


if __name__ == '__main__':
    main()
