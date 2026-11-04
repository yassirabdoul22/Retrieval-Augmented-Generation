from typing import Any, List, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.config import (
    MAX_CONTEXT_TOKENS,
    MAX_NEW_TOKENS,
    USED_THREADS_NUMBER,
    BUDGET_MARGIN
)


class Generator:
    """Generate grounded answers with a local causal language model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_context_tokens: int = MAX_CONTEXT_TOKENS,
    ) -> None:
        """Load the local model and tokenizer."""
        torch.set_num_threads(USED_THREADS_NUMBER)
        if max_context_tokens <= MAX_NEW_TOKENS:
            raise ValueError("Token budget must exceed max_new_tokens")
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_context_tokens = max_context_tokens

    def generate(self, query: str, source_text: List[str]) -> str:
        """Use sources in retrieval order within the model token budget."""
        source_text = self._fit_to_budget(source_text, query)
        prompt = self._build_prompt(query, source_text)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            use_cache=True
        )
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        return cast(str, self.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ))

    def _fit_to_budget(
        self, sources_text: List[str], query: str = "",
    ) -> List[str]:
        """Keep sources in retrieval order until the token budget is spent.
        """
        budget = (
            self.max_context_tokens - MAX_NEW_TOKENS - BUDGET_MARGIN
            - self._count_tokens(self._build_prompt(query, []))
        )
        if budget <= 0:
            raise ValueError("Question is too long for the model token budget")
        kept: List[str] = []
        for i, text in enumerate(sources_text):
            budget -= self._count_tokens(f"\n\n[Source {i + 1}]\n")
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) > budget:
                if budget > 0:
                    kept.append(cast(str, self.tokenizer.decode(ids[:budget])))
                break
            kept.append(text)
            budget -= len(ids)
        return kept

    def _count_tokens(self, text: str) -> int:
        """Return how many tokens the tokenizer makes from text."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _build_prompt(self, query: str, sources_text: List[str]) -> str:
        """Render source context and the question using the chat template."""
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
        return cast(str, prompt)

    def _format_context(self, sources_text: List[str]) -> str:
        """Label each retained source for inclusion in the model prompt."""
        blocks = [
            f"[Source {i + 1}]\n{text}" for i, text in enumerate(sources_text)
        ]
        return "Context:\n" + "\n\n".join(blocks)
