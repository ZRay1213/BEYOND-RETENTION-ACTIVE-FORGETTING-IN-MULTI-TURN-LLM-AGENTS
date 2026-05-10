from openai import OpenAI, AzureOpenAI
import os, time, json, re

def format_messages(messages, variables={}):
    last_user_msg = [msg for msg in messages if msg["role"] == "user"][-1]

    for k, v in variables.items():
        key_string = f"[[{k}]]"
        if key_string not in last_user_msg["content"]:
            print(f"[prompt] Key {k} not found in prompt; effectively ignored")
        assert type(v) == str, f"[prompt] Variable {k} is not a string"
        last_user_msg["content"] = last_user_msg["content"].replace(key_string, v)

    # find all the keys that are still in the prompt using regex [[STR]] where STR is alnum witout space
    keys_still_in_prompt = re.findall(r"\[\[([^\]]+)\]\]", last_user_msg["content"])
    if len(keys_still_in_prompt) > 0:
        print(f"[prompt] The following keys were not replaced: {keys_still_in_prompt}")

    return messages

class OpenAI_Model:
    def __init__(self):
        self.azure = ("AZURE_OPENAI_API_KEY" in os.environ and "AZURE_OPENAI_ENDPOINT" in os.environ)
        if self.azure:
            self.client = AzureOpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"], azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], api_version="2024-10-01-preview")
        else:
            assert "OPENAI_API_KEY" in os.environ
            self.api_key = os.environ["OPENAI_API_KEY"]
            self.default_base_url = os.environ.get("OPENAI_BASE_URL")
            self.base_urls = {
                "32b": os.environ.get("OPENAI_BASE_URL_32B"),
                "7b": os.environ.get("OPENAI_BASE_URL_7B"),
                "14b": os.environ.get("OPENAI_BASE_URL_14B"),
            }
            self._clients = {}

    def _get_client(self, model: str):
        if self.azure:
            return self.client
        m = (model or "").lower()
        for tag, url in self.base_urls.items():
            if url and tag in m:
                if tag not in self._clients:
                    self._clients[tag] = OpenAI(api_key=self.api_key, base_url=url)
                return self._clients[tag]
        if self.default_base_url:
            if "_default" not in self._clients:
                self._clients["_default"] = OpenAI(api_key=self.api_key, base_url=self.default_base_url)
            return self._clients["_default"]
        if "_plain" not in self._clients:
            self._clients["_plain"] = OpenAI(api_key=self.api_key)
        return self._clients["_plain"]

    def cost_calculator(self, model, usage, is_batch_model=False):
        # local / open-source model: no cost
        if any(x in model.lower() for x in ["qwen", "llama", "mistral", "gemma", "local"]):
            return 0.0
        is_finetuned, base_model = False, model
        if model.startswith("ft:gpt"):
            is_finetuned = True
            base_model = model.split(":")[1]

        prompt_tokens = usage['prompt_tokens']
        if 'prompt_tokens_details' in usage:
            ptd = usage.get('prompt_tokens_details') or {}
            prompt_tokens_cached = ptd.get('cached_tokens', 0) if isinstance(ptd, dict) else 0
        else:
            prompt_tokens_cached = 0
        prompt_tokens_non_cached = prompt_tokens - prompt_tokens_cached

        completion_tokens = usage['completion_tokens']
        if base_model.startswith("gpt-4o-mini"):
            if is_finetuned:
                inp_token_cost, out_token_cost = 0.0003, 0.00015
            else:
                inp_token_cost, out_token_cost = 0.00015, 0.0006
        elif base_model.startswith("gpt-4o"):
            if is_finetuned:
                inp_token_cost, out_token_cost = 0.00375, 0.015
            else:
                inp_token_cost, out_token_cost = 0.0025, 0.01
        elif base_model.startswith("gpt-3.5-turbo"):
            inp_token_cost, out_token_cost = 0.0005, 0.0015
        elif base_model.startswith("o1-mini"):
            inp_token_cost, out_token_cost = 0.003, 0.012
        elif base_model.startswith("gpt-4.5-preview"):
            inp_token_cost, out_token_cost = 0.075, 0.150
        elif base_model.startswith("o1-preview") or base_model == "o1":
            inp_token_cost, out_token_cost = 0.015, 0.06
        else:
            # Local vLLM models: zero cost
            if any(k in base_model.lower() for k in ["dsr1", "deepseek", "qwen", "llama", "mistral", "gemma"]):
                inp_token_cost, out_token_cost = 0.0, 0.0
            else:
                raise Exception(f"Model {model} pricing unknown, please add")

        cache_discount = 0.5 # cached tokens are half the price
        batch_discount = 0.5 # batch API is half the price
        total_usd = ((prompt_tokens_non_cached + prompt_tokens_cached * cache_discount) / 1000) * inp_token_cost + (completion_tokens / 1000) * out_token_cost
        if is_batch_model:
            total_usd *= batch_discount

        return total_usd

    def generate(self, messages, model="gpt-4o-mini", timeout=180, max_retries=3, temperature=1.0, is_json=False, return_metadata=False, max_tokens=None, variables={}):
        kwargs = {}
        if is_json:
            kwargs["response_format"] = { "type": "json_object" }
        N = 0

        messages = format_messages(messages, variables)

        # o1- models do not support system message. If the first message is a system message, and the second message is a user message, then prepend the user message with the system message.
        if model.startswith("o1") and len(messages) > 1 and messages[0]["role"] == "system" and messages[1]["role"] == "user":
            system_message = messages[0]["content"]
            messages[1]["content"] = f"System Message: {system_message}\n{messages[1]['content']}"
            messages = messages[1:]

        while True:
            try:
                response = self._get_client(model).chat.completions.create(model=model, messages=messages, timeout=timeout, max_completion_tokens=max_tokens, temperature=temperature, **kwargs)
                break
            except Exception as _e:
                N += 1
                import sys
                print(f'[generate retry {N}/{max_retries}] model={model} err={type(_e).__name__}: {str(_e)[:300]}', file=sys.stderr, flush=True)
                if N >= max_retries:
                    raise Exception(f'Failed to get response from OpenAI: {type(_e).__name__}: {str(_e)[:200]}')
                else:
                    time.sleep(4)

        response = response.to_dict()
        usage = response['usage']
        response_text = response["choices"][0]["message"]["content"]
        total_usd = self.cost_calculator(model, usage)
        prompt_tokens_cached = 0
        ptd2 = usage.get('prompt_tokens_details') or {}
        if isinstance(ptd2, dict):
            prompt_tokens_cached = ptd2.get('cached_tokens', 0)

        if not return_metadata:
            return response_text
        return {"message": response_text, "total_tokens": usage['total_tokens'], "prompt_tokens": usage['prompt_tokens'], "prompt_tokens_cached": prompt_tokens_cached, "completion_tokens": usage['completion_tokens'], "total_usd": total_usd}

    def generate_json(self, messages, model="gpt-4o-mini", **kwargs):
        response = self.generate(messages, model, is_json=True, **kwargs)
        response["message"] = json.loads(response["message"])
        return response


model = OpenAI_Model()
generate = model.generate
generate_json = model.generate_json





if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "Humor is a way to make people laugh. Tell me a joke about UC Berkeley."},
        {"role": "assistant", "content": '{"joke": '}
    ]

    model = "gpt-4o"
    response = generate(messages, model=model, return_metadata=True)
    
    print(response)

    # batch_id = client.generate_w_delay(messages, model="gpt-4o-mini")
    # print(batch_id)
