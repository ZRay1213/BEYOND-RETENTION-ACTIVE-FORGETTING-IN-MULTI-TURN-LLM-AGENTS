"""GEPA-style ablation proposer — only mutates DIRECTIVE_TEMPLATES dict.

Controller logic, state extraction, action vocabulary, main loop are FROZEN.
Only the 6 prompt strings inside DIRECTIVE_TEMPLATES are evolved.

This is the closest in-spirit replication of GEPA on our LiC harness.
"""
import argparse, json, glob, os, re, sys, statistics
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_openai import generate

PROPOSER_MODEL = 'qwen2.5-14b'


def load_traces(traces_dir, n_failed=4, n_succeeded=3):
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
    def fmt(s):
        actions, qa = [], []
        for m in s['trace']:
            if not isinstance(m, dict): continue
            t = m.get('content')
            if m.get('role') == 'log' and isinstance(t, dict):
                if t.get('type') == 'controller_action':
                    actions.append(f't{t["turn"]}={t["action"]}')
            elif m.get('role') == 'user' and isinstance(t, str) and len(qa) < 4:
                qa.append(('U', t[:200]))
            elif m.get('role') == 'assistant' and isinstance(t, str) and len(qa) < 6:
                qa.append(('A', t[:200]))
        lines = [f'TASK={s["task_id"]} SCORE={s["score"]} ACTIONS=[{",".join(actions)}]']
        for r, c in qa[:4]:
            lines.append(f'{r}: {c}')
        return '\n'.join(lines)[:1100]
    return [fmt(s) for s in failed], [fmt(s) for s in succeeded]


def extract_directive_block(src):
    """Extract DIRECTIVE_TEMPLATES = {...} block."""
    m = re.search(r'(DIRECTIVE_TEMPLATES = \{[\s\S]+?\n\})', src)
    return m.group(1) if m else None


PROPOSER_SYSTEM = """You are a prompt evolver in the GEPA style. You evolve ONLY the 6 directive prompt strings in DIRECTIVE_TEMPLATES — the texts that get injected as system messages to the assistant.

CRITICAL CONSTRAINTS:
1. Output ONLY a new DIRECTIVE_TEMPLATES = {...} dict, inside ```python``` fences. NO other code.
2. Keep all 6 keys: 'CONTINUE', 'VERIFY', 'CLARIFY', 'RESET', 'CONCLUDE', 'INJECT_SUMMARY'.
3. 'CONTINUE' must remain None. 'INJECT_SUMMARY' must remain '__INJECT_TASKSTATE__'.
4. Mutate the OTHER 4 directive strings (VERIFY, CLARIFY, RESET, CONCLUDE) based on trace evidence.
5. Make changes minimal — refine wording for clarity/effect, don't add new fields.

After the python block, write CHANGE_RATIONALE as a paragraph citing trace evidence."""


def build_prompt(parent_directives, failed, succ):
    parts = ['## CURRENT DIRECTIVE_TEMPLATES:\n```python\n', parent_directives, '\n```\n\n',
             '## FAILED TRACES (where current directives didn\'t prevent failure):\n']
    for i,e in enumerate(failed): parts.append(f'\n### FAILED #{i+1}\n{e}\n')
    parts.append('\n## SUCCEEDED TRACES (current directives worked):\n')
    for i,e in enumerate(succ): parts.append(f'\n### SUCCESS #{i+1}\n{e}\n')
    parts.append('\n## TASK:\nProduce ONE evolved DIRECTIVE_TEMPLATES dict refining the 4 mutable directive strings. Output dict only in ```python``` fences, then CHANGE_RATIONALE.')
    return ''.join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', required=True)  # parent harness.py
    p.add_argument('--traces_dir', required=True)
    p.add_argument('--gen', type=int, required=True)
    p.add_argument('--n_variants', type=int, default=2)
    p.add_argument('--output_dir', default='/root/autodl-tmp/DCC/harness_gepa_population')
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    parent_src = open(args.parent).read()
    parent_dir = extract_directive_block(parent_src)
    if not parent_dir:
        print('Could not extract DIRECTIVE_TEMPLATES from parent'); return
    print(f'Parent directives: {len(parent_dir)} chars')

    failed, succ = load_traces(args.traces_dir)
    prompt = build_prompt(parent_dir, failed, succ)
    print(f'Prompt: ~{len(prompt)//4} tokens')

    for vi in range(args.n_variants):
        print(f'\n[gen {args.gen} v{vi}] calling GEPA-style proposer...')
        msgs = [{'role':'system','content':PROPOSER_SYSTEM},
                {'role':'user','content':prompt}]
        try:
            resp = generate(msgs, model=PROPOSER_MODEL, temperature=0.7, return_metadata=True, max_tokens=3000)
            text = resp['message'] if isinstance(resp, dict) else str(resp)
        except Exception as e:
            print(f'  fail: {e}'); continue
        m = re.search(r'```python\s*\n(.*?)\n```', text, re.DOTALL)
        if not m:
            print(f'  no python block (head: {text[:200]})'); continue
        new_dir = m.group(1).strip()
        # Splice into parent harness
        new_src = parent_src.replace(parent_dir, new_dir)
        rm = re.search(r'CHANGE_RATIONALE[:\s]*(.+)', text, re.DOTALL)
        rat = rm.group(1).strip()[:500] if rm else '(none)'
        out = os.path.join(args.output_dir, f'harness_gepa_g{args.gen}_c{vi}.py')
        open(out, 'w').write(new_src)
        open(out.replace('.py','_rationale.md'),'w').write(rat)
        import py_compile
        try:
            py_compile.compile(out, doraise=True)
            print(f'  OK {out}')
            print(f'  rationale: {rat[:150]}')
        except py_compile.PyCompileError as e:
            print(f'  SYNTAX ERR: {str(e)[:200]}')


if __name__ == '__main__': main()
