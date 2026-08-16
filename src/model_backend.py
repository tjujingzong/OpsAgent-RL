"""Model backends for rollout: vLLM HTTP server, local HF model, or rule-based stub."""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


class HTTPBackend:
    """OpenAI-compatible chat backend (vLLM serve / TGI / API)."""

    def __init__(self, base_url: str, api_key: str = "EMPTY", model: str = "default", temperature: float = 0.7, max_tokens: int = 2048):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, messages: list[dict]) -> str:
        payload = json.dumps(
            {"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": self.max_tokens}
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"]


class HFBackend:
    """Local HuggingFace transformers backend (lazy import; GPU needed)."""

    def __init__(self, model_path: str, temperature: float = 0.7, device: str = "cuda"):
        import torch  # noqa
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
        )
        self.temperature = temperature

    def generate(self, messages: list[dict]) -> str:
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs, max_new_tokens=512, do_sample=self.temperature > 0, temperature=self.temperature, top_p=0.95
        )
        new = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new, skip_special_tokens=True)


class RuleBasedBackend:
    """Deterministic no-model backend that replays a scripted expert trajectory.

    Used to sanity-test the full benchmark harness in Docker without a GPU.
    It walks the rule-based trajectory's assistant turns in order, ignoring the
    conversation context, so it is NOT a real model.
    """

    def __init__(self, task):
        from data.sft_generator import rule_based_trajectory

        self._msgs = rule_based_trajectory(task)
        self._idx = 0

    def generate(self, messages: list[dict]) -> str:
        # advance to the next assistant message in the scripted trajectory
        while self._idx < len(self._msgs):
            m = self._msgs[self._idx]
            self._idx += 1
            if m["role"] == "assistant":
                return m["content"]
        return "TASK_COMPLETE"


def build_backend(args, task=None):
    if getattr(args, "rule_based", False):
        return RuleBasedBackend(task)
    if getattr(args, "server_url", None):
        return HTTPBackend(args.server_url, os.getenv("MODEL_API_KEY", "EMPTY"), args.model_name or "default", args.temperature)
    if getattr(args, "model_path", None):
        return HFBackend(args.model_path, temperature=args.temperature)
    raise ValueError("specify --server-url, --model-path, or --rule-based")
