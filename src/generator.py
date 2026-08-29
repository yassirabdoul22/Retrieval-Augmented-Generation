from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer


class Generator:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, query: str, source_text: List[str]) -> str:
        prompt = self._build_prompt(query, source_text)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(  # type: ignore[misc]
            **inputs, max_new_tokens=256
        )
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(  # type: ignore[return-value]
            new_tokens, skip_special_tokens=True
        )

    def _build_prompt(self, query: str, sources_text: List[str]) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a technical assistant answering questions "
                    "about the vLLM codebase. Answer ONLY using the "
                    "provided context below. If the context does not "
                    "contain the answer, say you don't know. Be concise "
                    "and factual. Do not invent information."
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
