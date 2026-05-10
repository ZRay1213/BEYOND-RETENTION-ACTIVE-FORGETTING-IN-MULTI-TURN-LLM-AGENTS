"""DCC harness proposer — Meta-Harness style.

Reads current harness.py + execution traces (failed + succeeded), uses an LLM to
propose a mutated harness.py. Saves variants to a filesystem-style population.

Usage:
  python proposer.py --parent harness.py --traces_dir logs/.../math/conv_type \
                     --gen 1 --n_variants 2 --output_dir harness_population
"""
import argparse, json, glob, os, re, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_openai import generate

PROPOSER_MODEL = 'qwen2.5-14b'  # uses port 8003


def load_traces(traces_dir, n_failed=4, n_succeeded=3, max_excerpt_chars=1200):
    """Pick a balanced sample of failed and succeeded sims, format as compact excerpts."""
    sims = []
    for f in glob.glob(f'{traces_dir}/*.jsonl'):
        for ln in open(f):
            try:
                r = json.loads(ln)
                sc = r.get('score')
                if sc is None: continue
                sims.append({'task_id': r['task_id'], 'score': sc, 'trace': r.get('trace', [])})
            except: pass
    failed = sorted([s for s in sims if s['score'] == 0.0], key=lambda x: x['task_id'])[:n_failed]
    succeeded = sorted([s for s in sims if s['score'] == 1.0], key=lambda x: x['task_id'])[:n_succeeded]

    def fmt_excerpt(s):
        actions, qa_pairs, final_answer = [], [], None
        for m in s['trace']:
            if not isinstance(m, dict): continue
            t = m.get('content')
            if m.get('role') == 'log' and isinstance(t, dict):
                if t.get('type') == 'controller_action':
                    actions.append(f't{t["turn"]}={t["action"]}')
                elif t.get('type') == 'answer-evaluation':
                    final_answer = f'extracted={t.get("exact_answer", "")}; correct={t.get("is_correct")}'
            elif m.get('role') == 'user' and isinstance(t, str) and len(qa_pairs) < 4:
                qa_pairs.append(('U', t[:200]))
            elif m.get('role') == 'assistant' and isinstance(t, str) and len(qa_pairs) < 6:
                qa_pairs.append(('A', t[:250]))
        lines = [f'TASK={s["task_id"]} SCORE={s["score"]} ACTIONS=[{",".join(actions)}]']
        for r, c in qa_pairs[:4]:
            lines.append(f'{r}: {c}')
        if final_answer: lines.append(f'FINAL: {final_answer}')
        out = '\n'.join(lines)
        return out[:max_excerpt_chars]

    return [fmt_excerpt(s) for s in failed], [fmt_excerpt(s) for s in succeeded]


PROPOSER_SYSTEM = """You are an expert code mutator for a multi-turn dialogue control harness. The harness is a single Python file that controls how an LLM assistant responds in a sharded-information conversation. Your job: read the CURRENT HARNESS, read EXECUTION TRACES (failed + succeeded), and produce a MUTATED HARNESS that is likely to perform better.

IMPORTANT RULES:
1. Output the COMPLETE mutated harness.py file, surrounded by ```python ... ``` fences.
2. You MAY modify any of: MEDIATOR_SYSTEM, DIRECTIVE_TEMPLATES, CONTROLLER_SYSTEM, controller_select_action(), extract_state(), ACTION_LIST.
3. You MUST keep the structure: imports, run_harness() entry point, the HarnessSharded class, the main loop logic. Do not rename functions or break the API.
4. Make ONE FOCUSED CHANGE per variant (not 5 random tweaks). Examples of focused changes:
   - Rewrite the CLARIFY directive to be more specific about what to ask
   - Add a new action to ACTION_LIST and DIRECTIVE_TEMPLATES (e.g. REFRAME)
   - Modify CONTROLLER_SYSTEM rules to be more discriminating
   - Add a deterministic rule in controller_select_action that overrides the LLM call when state matches a clear pattern
5. After the code, output a one-paragraph CHANGE_RATIONALE explaining what you changed and why, citing the trace evidence.
6. Be conservative: a small targeted change beats a big rewrite."""


def build_prompt(harness_src, failed_excerpts, succeeded_excerpts):
    parts = [
        '## CURRENT HARNESS:\n```python\n', harness_src, '\n```\n\n',
        '## TRACES — what went wrong (failed sims):\n',
    ]
    for i, e in enumerate(failed_excerpts):
        parts.append(f'\n### FAILED #{i+1}\n{e}\n')
    parts.append('\n## TRACES — what worked (succeeded sims):\n')
    for i, e in enumerate(succeeded_excerpts):
        parts.append(f'\n### SUCCESS #{i+1}\n{e}\n')
    parts.append('\n## YOUR TASK:\nProduce ONE mutated harness.py implementing a focused improvement. Output the complete file inside ```python``` fences, then a CHANGE_RATIONALE paragraph.')
    return ''.join(parts)


CODE_BLOCK_RE = re.compile(r'```python\s*\n(.*?)\n```', re.DOTALL)


def extract_python_code(response):
    matches = CODE_BLOCK_RE.findall(response)
    if not matches: return None
    # Take the longest match (most likely the full file)
    return max(matches, key=len)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', required=True, help='Parent harness.py to mutate')
    p.add_argument('--traces_dir', required=True, help='Dir containing .jsonl traces from parent')
    p.add_argument('--gen', type=int, required=True, help='Generation number')
    p.add_argument('--n_variants', type=int, default=2)
    p.add_argument('--output_dir', default='harness_population')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    harness_src = open(args.parent).read()
    print(f'Parent harness: {args.parent} ({len(harness_src)} chars)')

    failed, succ = load_traces(args.traces_dir)
    print(f'Loaded {len(failed)} failed + {len(succ)} succeeded excerpts from {args.traces_dir}')

    prompt = build_prompt(harness_src, failed, succ)
    print(f'Prompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)')

    for vi in range(args.n_variants):
        print(f'\n[gen {args.gen} variant {vi}] calling proposer...')
        msgs = [
            {'role': 'system', 'content': PROPOSER_SYSTEM},
            {'role': 'user', 'content': prompt},
        ]
        try:
            resp = generate(msgs, model=PROPOSER_MODEL, temperature=0.7, return_metadata=True, max_tokens=8000)
            text = resp['message'] if isinstance(resp, dict) else str(resp)
        except Exception as e:
            print(f'  proposer call failed: {e}')
            continue
        code = extract_python_code(text)
        if code is None:
            print(f'  no python block in response (head: {text[:200]})')
            continue
        rationale_match = re.search(r'CHANGE_RATIONALE[:\s]*(.+)', text, re.DOTALL)
        rationale = rationale_match.group(1).strip()[:600] if rationale_match else '(none provided)'
        out_path = os.path.join(args.output_dir, f'harness_g{args.gen}_c{vi}.py')
        with open(out_path, 'w') as f:
            f.write(code)
        rat_path = out_path.replace('.py', '_rationale.md')
        with open(rat_path, 'w') as f:
            f.write(rationale)
        # Try py_compile
        import py_compile
        try:
            py_compile.compile(out_path, doraise=True)
            print(f'  OK saved {out_path} (rationale: {rationale[:120]})')
        except py_compile.PyCompileError as e:
            print(f'  SYNTAX ERROR in {out_path}: {str(e)[:200]}')


if __name__ == '__main__':
    main()
