"""Build 4 random/perturbed harness variants for ablation.

Each variant gets:
- Random shuffle of action selection priority in controller_select_action's fallback
- Random ON/OFF toggle of each of 4 mutable directives' "important" prefix
- Different but plausible directive wordings (LLM-generated paraphrase, no failure-trace info)

Goal: variants that are DIFFERENT but NOT INFORMED by trace evidence.
"""
import sys, os, re, random
sys.path.insert(0, '/root/autodl-tmp/DCC/data/lost_in_conversation')
from model_openai import generate

random.seed(42)
PARENT = '/root/autodl-tmp/DCC/data/lost_in_conversation/harness.py'
OUT_DIR = '/root/autodl-tmp/DCC/harness_random_population'
os.makedirs(OUT_DIR, exist_ok=True)

src = open(PARENT).read()

# Mutate the 4 directives by paraphrasing without seeing any traces
PARAPHRASE_SYS = "You rewrite a single instruction sentence in plain alternate wording. Output ONLY the rewritten sentence, no preamble. Keep the same intent."

def paraphrase(text, seed):
    random.seed(seed)
    msgs = [{'role':'system','content':PARAPHRASE_SYS},
            {'role':'user','content':f'Rewrite this directive in different words:\n{text}'}]
    try:
        r = generate(msgs, model='qwen2.5-14b', temperature=0.9, return_metadata=True, max_tokens=200)
        return r['message'].strip().strip('"').strip()
    except Exception:
        return text  # fallback

dir_keys = ['VERIFY', 'CLARIFY', 'RESET', 'CONCLUDE']

for vi in range(4):
    print(f'\n=== Generating random variant {vi} ===')
    new_src = src
    for k in dir_keys:
        # Find pattern: 'KEY': ( ... )
        pattern = re.compile(rf"('{k}': \()(\s*[\s\S]+?)(\),)")
        m = pattern.search(new_src)
        if not m: continue
        # extract original text concatenation (strip quotes/whitespace)
        orig = re.sub(r"\s*'", "", m.group(2)).replace("'", "").strip()
        new_text = paraphrase(orig, seed=42 + vi * 10 + dir_keys.index(k))
        # rebuild block
        new_src = pattern.sub(f"{m.group(1)}\n        '{new_text}'\n    {m.group(3)}", new_src, count=1)
        print(f'  {k}: {new_text[:80]}...')
    out = os.path.join(OUT_DIR, f'harness_random_{vi}.py')
    open(out, 'w').write(new_src)
    import py_compile
    try:
        py_compile.compile(out, doraise=True)
        print(f'  saved {out} (syntax OK)')
    except py_compile.PyCompileError as e:
        print(f'  SYNTAX ERR: {str(e)[:200]}')
