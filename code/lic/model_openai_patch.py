"""Patch script that rewrites model_openai.py to support multi-endpoint routing.

Routing rule: model name -> base_url
  - any model containing "32" -> $OPENAI_BASE_URL_32B (or $OPENAI_BASE_URL fallback)
  - any model containing "7b" -> $OPENAI_BASE_URL_7B
  - else $OPENAI_BASE_URL fallback
"""
import re
from pathlib import Path

p = Path("model_openai.py")
s = p.read_text()

# Replace the __init__ to be model-routing aware
old_init = """class OpenAI_Model:
    def __init__(self):
        if "AZURE_OPENAI_API_KEY" in os.environ and "AZURE_OPENAI_ENDPOINT" in os.environ:
            self.client = AzureOpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"], azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], api_version="2024-10-01-preview")
        else:
            assert "OPENAI_API_KEY" in os.environ
            base_url = os.environ.get("OPENAI_BASE_URL")
            if base_url:
                self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base_url)
            else:
                self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])"""

new_init = """class OpenAI_Model:
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
        return self._clients["_plain"]"""

if old_init in s:
    s = s.replace(old_init, new_init)
    print("patched __init__")
else:
    print("ERR: __init__ block not found")

# Replace `response = self.client.chat.completions.create(...)` to use `_get_client(model)`
old_call = "response = self.client.chat.completions.create("
new_call = "response = self._get_client(model).chat.completions.create("
if old_call in s:
    s = s.replace(old_call, new_call)
    print("patched call")
else:
    print("ERR: call block not found")

p.write_text(s)
print("done")
