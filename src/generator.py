from typing import List, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.compressor import ContextCompressor
from src.config import (
    MAX_CONTEXT_TOKENS,
    COMPRESSED_CONTEXT_TOKENS,
    MAX_NEW_TOKENS,
    USED_THREADS_NUMBER
)
class Generator:
    """Generate grounded answers with a local causal language model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_context_tokens: int = MAX_CONTEXT_TOKENS,
    ) -> None:
        """Load the local model, tokenizer and context compressor."""
        torch.set_num_threads(USED_THREADS_NUMBER)
        if max_context_tokens <= MAX_NEW_TOKENS:
            raise ValueError("Token budget must exceed max_new_tokens")
        # float32: the CPU has no bfloat16 support, which would be emulated
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.compressor = ContextCompressor(
            self.tokenizer, budget=COMPRESSED_CONTEXT_TOKENS
        )
        self.max_context_tokens = max_context_tokens

    def generate(self, query: str, source_text: List[str]) -> str:
        """Compress sources and generate an answer within the token budget."""
        source_text = self.compressor.compress(query, source_text)
        source_text = self._fit_to_budget(source_text, query)
        prompt = self._build_prompt(query, source_text)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            use_cache= True
        )
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        return cast(str, self.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ))

    def _fit_to_budget(
        self, sources_text: List[str], query: str = "",
    ) -> List[str]:
        """Fit the complete prompt while reserving generated tokens.

        Reject questions that cannot fit even without source context.
        Shorten the final retained source until the rendered prompt fits.
        """
        limit = self.max_context_tokens - MAX_NEW_TOKENS

        def prompt_size(sources: List[str]) -> int:
            """Count tokens exactly as generation tokenizes the prompt."""
            return len(self.tokenizer.encode(
                self._build_prompt(query, sources)))

        if prompt_size([]) > limit:
            raise ValueError("Question is too long for the model token budget")
        kept: List[str] = []
        for text in sources_text:
            if prompt_size(kept + [text]) <= limit:
                kept.append(text)
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            while ids:
                candidate = cast(str, self.tokenizer.decode(ids))
                overflow = prompt_size(kept + [candidate]) - limit
                if overflow <= 0:
                    kept.append(candidate)
                    break
                ids = ids[:-max(1, overflow)]
            break
        return kept

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
