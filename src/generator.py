from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.compressor import ContextCompressor

MAX_CONTEXT_TOKENS = 4096
COMPRESSED_CONTEXT_TOKENS = 1000
MAX_NEW_TOKENS = 96


class Generator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_context_tokens: int = MAX_CONTEXT_TOKENS,
    ):
        # float32: the CPU has no bfloat16 support, which would be emulated
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.compressor = ContextCompressor(
            self.tokenizer, budget=COMPRESSED_CONTEXT_TOKENS
        )
        self.max_context_tokens = max_context_tokens

    def generate(self, query: str, source_text: List[str]) -> str:
        source_text = self.compressor.compress(query, source_text)
        source_text = self._fit_to_budget(source_text)
        prompt = self._build_prompt(query, source_text)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(  # type: ignore[misc]
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(  # type: ignore[return-value]
            new_tokens, skip_special_tokens=True
        )

    def _fit_to_budget(self, sources_text: List[str]) -> List[str]:
        """Keep sources in rank order until the token budget is spent.

        The last source that does not fit entirely is cut at the budget
        boundary instead of being dropped.
        """
        kept: List[str] = []
        remaining = self.max_context_tokens
        for text in sources_text:
            if remaining <= 0:
                break
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) > remaining:
                text = self.tokenizer.decode(ids[:remaining])
            kept.append(text)
            remaining -= len(ids)
        return kept

    def _build_prompt(self, query: str, sources_text: List[str]) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a technical assistant answering questions "
                    "about the vLLM codebase. Answer in one or two "
                    "sentences, using only facts stated in the context "
                    "below, and quote names, paths and values exactly. "
                    "The answer may be worded differently from the "
                    "question or split across several sources. If the "
                    "specific fact asked is not stated anywhere in the "
                    "context, reply: \"I don't know based on the provided "
                    "context.\" Never guess."
                ),
            },
            {
                "role": "user",
                "content": (
                    self._format_context(sources_text)
                    + f"\n\nQuestion: {query}"
                ),
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return prompt  # type: ignore[return-value]

    def _format_context(self, sources_text: List[str]) -> str:
        blocks = [
            f"[Source {i + 1}]\n{text}" for i, text in enumerate(sources_text)
        ]
        return "Context:\n" + "\n\n".join(blocks)
