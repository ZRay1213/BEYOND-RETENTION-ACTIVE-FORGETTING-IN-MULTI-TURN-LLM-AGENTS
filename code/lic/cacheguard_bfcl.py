"""BFCL multi-turn adapter for our 4 protocols.

Wraps QwenFCHandler.inference_multi_turn_prompting to render the message list
according to a protocol (sharded / fresh-last / fresh-every / cacheguard).

The 4 protocols differ ONLY in what is included in inference_data["message"]
at _query_prompting time:

  Sharded     : full history (default)
  Fresh-Last  : at last turn only, render = [sys, U1+U2+...+Un]
  Fresh-Every : every turn,        render = [sys, U1+U2+...+Ucurr]
  CacheGuard  : every turn,        render = [sys, U1, asst_tool_calls_only_1,
                                              tool_results_1, U2, ..., Ucurr]
                  -- drops only natural-language asst rationale; keeps verified
                     state via tool_results and intent log via tool_calls.
  TRO         : every turn,        render = [sys, U1, tool_results_1, U2, ...]
                  -- drops ALL assistant messages (both rationale and tool_calls);
                     keeps only verified tool results. Tests whether tool results
                     alone (without asst intent log) are the irreducible state.
  TRO         : every turn,        render = [sys, U1, tool_results_1, U2, ...]
                  -- drops ALL assistant messages (both rationale and tool_calls);
                     keeps only verified tool results. Tests whether tool results
                     alone (without asst intent log) are the irreducible state.
"""
import json, os, sys, time, copy, traceback, argparse
from pathlib import Path

# Late import to allow LOCAL_SERVER_* env vars to take effect
os.environ.setdefault("LOCAL_SERVER_ENDPOINT", "localhost")
os.environ.setdefault("REMOTE_OPENAI_BASE_URL", "http://localhost:8001/v1")
os.environ.setdefault("REMOTE_OPENAI_API_KEY", "EMPTY")

from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from bfcl_eval.model_handler.utils import system_prompt_pre_processing_chat_model
from overrides import override


def tag(msg, kind, turn):
    msg = dict(msg)
    msg["_kind"] = kind
    msg["_turn"] = turn
    return msg


def strip_meta(m):
    return {k: v for k, v in m.items() if k not in ("_kind", "_turn")}


def render(messages, protocol, turn_idx, last_turn_idx):
    """Render the LLM-visible message list under a given protocol.

    Pruning is applied only to PRIOR turns (msg._turn < turn_idx). The current
    turn's assistant + tool exchanges are always kept intact so the inner
    step-loop can converge.
    """
    sys_msgs = [m for m in messages if m["role"] == "system"]
    other = [m for m in messages if m["role"] != "system"]

    prior = [m for m in other if m.get("_turn", 0) < turn_idx]
    current = [m for m in other if m.get("_turn", 0) >= turn_idx]

    def keep_all(items):
        return [strip_meta(m) for m in items]

    if protocol == "sharded":
        return keep_all(messages)

    if protocol == "fresh-last":
        if turn_idx < last_turn_idx:
            return keep_all(messages)
        # Last turn: drop ALL prior history; concat user msgs from all turns
        # into one user message (current turn user msg is in `current`).
        users_prior = [m for m in prior if m.get("_kind") == "user"]
        users_curr = [m for m in current if m.get("_kind") == "user"]
        contents = [u["content"] for u in users_prior + users_curr if u.get("content")]
        concat = "\n\n".join(contents)
        # Replace user msgs with concat; keep current turn's asst+tool steps.
        non_user_curr = [m for m in current if m.get("_kind") != "user"]
        out = keep_all(sys_msgs) + [{"role": "user", "content": concat}] + keep_all(non_user_curr)
        return out

    if protocol == "fresh-every":
        # Every turn: drop ALL prior asst+tool, keep prior user msgs concatenated
        # with current user msg. Keep current turn asst+tool exchanges.
        users_prior = [m for m in prior if m.get("_kind") == "user"]
        users_curr = [m for m in current if m.get("_kind") == "user"]
        contents = [u["content"] for u in users_prior + users_curr if u.get("content")]
        concat = "\n\n".join(contents)
        non_user_curr = [m for m in current if m.get("_kind") != "user"]
        out = keep_all(sys_msgs) + [{"role": "user", "content": concat}] + keep_all(non_user_curr)
        return out

    if protocol == "cacheguard":
        # Prior turns: keep user + tool results; drop asst rationale; keep
        # asst tool_calls (intent record) but blank content.
        # Current turn: keep all (so inner loop converges).
        out = keep_all(sys_msgs)
        for m in prior:
            k = m.get("_kind")
            if k == "user" or k == "tool":
                out.append(strip_meta(m))
            elif k == "asst_tool_call":
                m2 = strip_meta(m)
                m2["content"] = ""
                out.append(m2)
            elif k == "asst_text":
                pass  # drop rationale
            else:
                out.append(strip_meta(m))
        out.extend(keep_all(current))
        return out

    if protocol == "tool-result-only":
        # Prior turns: keep ONLY user messages + verified tool results.
        # Drops all assistant messages (both asst_text rationale and asst_tool_call records).
        # This isolates whether tool results alone are the irreducible state, or whether
        # the asst_tool_call intent records also contribute.
        out = keep_all(sys_msgs)
        for m in prior:
            k = m.get("_kind")
            if k == "user" or k == "tool":
                out.append(strip_meta(m))
            # drop asst_tool_call and asst_text
        out.extend(keep_all(current))
        return out

    if protocol == "tool-result-only":
        # Prior turns: keep ONLY user messages + verified tool results.
        # Drops all assistant messages (both asst_text rationale and asst_tool_call records).
        # This isolates whether tool results alone are the irreducible state, or whether
        # the asst_tool_call intent records also contribute.
        out = keep_all(sys_msgs)
        for m in prior:
            k = m.get("_kind")
            if k == "user" or k == "tool":
                out.append(strip_meta(m))
            # drop asst_tool_call and asst_text
        out.extend(keep_all(current))
        return out

    raise ValueError(f"Unknown protocol: {protocol}")


class ProtocolQwenHandler(QwenFCHandler):
    def __init__(self, *args, protocol="sharded", **kwargs):
        super().__init__(*args, **kwargs)
        self.protocol = protocol
        self._turn_idx = 0
        self._last_turn_idx = 0

    def set_total_turns(self, n):
        self._last_turn_idx = n - 1
        self._turn_idx = 0

    @override
    def add_first_turn_message_prompting(self, inference_data: dict, first_turn_message: list[dict]) -> dict:
        self._turn_idx = 0
        for m in first_turn_message:
            kind = m["role"] if m["role"] in ("system", "user") else "user"
            inference_data["message"].append(tag(m, kind, self._turn_idx))
        return inference_data

    @override
    def _add_next_turn_user_message_prompting(self, inference_data: dict, user_message: list[dict]) -> dict:
        self._turn_idx += 1
        for m in user_message:
            inference_data["message"].append(tag(m, "user", self._turn_idx))
        return inference_data

    @override
    def _add_assistant_message_prompting(self, inference_data: dict, model_response_data: dict) -> dict:
        msg = model_response_data["model_responses_message_for_chat_history"]
        kind = "asst_tool_call" if msg.get("tool_calls") else "asst_text"
        inference_data["message"].append(tag(msg, kind, self._turn_idx))
        return inference_data

    @override
    def _add_execution_results_prompting(self, inference_data: dict, execution_results: list[str], model_response_data: dict) -> dict:
        for r, decoded in zip(execution_results, model_response_data["model_responses_decoded"]):
            inference_data["message"].append(
                tag({"role": "tool", "name": decoded, "content": r}, "tool", self._turn_idx)
            )
        return inference_data

    @override
    def _query_prompting(self, inference_data):
        # Snapshot tagged list, render by protocol, swap in for the LLM call.
        full = inference_data["message"]
        rendered = render(full, self.protocol, self._turn_idx, self._last_turn_idx)
        inference_data["message"] = rendered
        try:
            resp = super()._query_prompting(inference_data)
        finally:
            # Restore so subsequent _add_* hooks operate on the full tagged log.
            inference_data["message"] = full
        return resp


def make_handler(protocol, model_id, port, tokenizer_path):
    os.environ["LOCAL_SERVER_PORT"] = str(port)
    os.environ["REMOTE_OPENAI_BASE_URL"] = f"http://localhost:{port}/v1"
    h = ProtocolQwenHandler(
        model_name=model_id,
        temperature=0.001,
        registry_name=model_id,
        is_fc_model=True,
        protocol=protocol,
    )
    h.model_path_or_id = model_id  # used by completions API
    from transformers import AutoTokenizer, AutoConfig
    h.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    cfg = AutoConfig.from_pretrained(tokenizer_path)
    h.max_context_length = getattr(cfg, "max_position_embeddings", 32768)
    return h


def run_one(handler, entry):
    handler.set_total_turns(len(entry["question"]))
    return handler.inference_multi_turn_prompting(
        copy.deepcopy(entry), include_input_log=False, exclude_state_log=True
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True,
                    choices=["sharded", "fresh-last", "fresh-every", "cacheguard", "tool-result-only"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen2.5-14b")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--data", default=None)
    ap.add_argument("--tokenizer", default="/root/autodl-tmp/models/Qwen2.5-14B-Instruct")
    args = ap.parse_args()

    # Load functions/category info
    PKG = Path("/root/miniconda3/lib/python3.12/site-packages/bfcl_eval")
    data_path = args.data or str(PKG / "data" / "BFCL_v4_multi_turn_base.json")
    instances = [json.loads(l) for l in open(data_path)]
    func_doc_path = PKG / "data" / "multi_turn_func_doc"

    from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING
    def populate_funcs(entry):
        funcs = []
        for cls in entry["involved_classes"]:
            fname = MULTI_TURN_FUNC_DOC_FILE_MAPPING.get(cls)
            if not fname: continue
            fd = func_doc_path / fname
            if fd.exists():
                for line in fd.read_text().splitlines():
                    line = line.strip()
                    if line:
                        funcs.append(json.loads(line))
        excl = set(entry.get("excluded_function", []))
        funcs = [f for f in funcs if f.get("name") not in excl]
        entry["function"] = funcs
        return entry

    handler = make_handler(args.protocol, args.model, args.port, args.tokenizer)

    out_f = open(args.out, "w")
    for i, entry in enumerate(instances[args.start:args.start + args.n]):
        entry = populate_funcs(entry)
        idx = args.start + i
        t0 = time.time()
        try:
            responses, metadata = run_one(handler, entry)
            rec = {
                "id": entry["id"],
                "result": responses,
                "input_token_count": metadata["input_token_count"],
                "output_token_count": metadata["output_token_count"],
                "latency": metadata["latency"],
            }
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            dt = time.time() - t0
            print(f"[{args.protocol}] {idx}: {entry['id']} ✓ ({dt:.1f}s)")
        except Exception as e:
            print(f"[{args.protocol}] {idx}: {entry['id']} ✗ {type(e).__name__}: {e}")
            traceback.print_exc()
            out_f.write(json.dumps({"id": entry["id"], "result": None, "error": str(e)}) + "\n")
            out_f.flush()
    out_f.close()


if __name__ == "__main__":
    main()
