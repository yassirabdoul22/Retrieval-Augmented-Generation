"""Keep only the passages of the sources closest to the question."""

import re
from typing import List, Set, Tuple

from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerBase

MAX_PASSAGE_LINES = 12


class ContextCompressor:
    """Select the paragraphs most similar to the query, within a budget."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        budget: int,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        """Load the embedding model; budget is in generator tokens."""
        self.tokenizer = tokenizer
        self.budget = budget
        self.model = SentenceTransformer(model_name, device="cpu")

    def compress(self, query: str, sources: List[str]) -> List[str]:
        """Keep the top-ranked source intact, compress the others.

        The first source is the best retrieval match: cutting it often
        loses the answer, so it is kept whole and the remaining budget
        goes to the most relevant paragraphs of the other sources.
        """
        if not sources:
            return []
        first, others = sources[0], sources[1:]
        remaining = self.budget - len(self.tokenizer.encode(first))
        if remaining <= 0:
            return [first]
        return [first] + self._select(query, others, remaining)

    def _select(
        self, query: str, sources: List[str], budget: int
    ) -> List[str]:
        """Return the sources reduced to their most relevant paragraphs."""
        passages = self._split(sources)
        if not passages:
            return []
        texts = [text for _, text in passages]
        query_vec = self.model.encode([query], normalize_embeddings=True)
        vecs = self.model.encode(texts, normalize_embeddings=True)
        scores = (vecs @ query_vec.T).ravel()

        kept: Set[int] = set()
        used = 0
        for i in scores.argsort()[::-1]:
            cost = len(self.tokenizer.encode(texts[i]))
            if used + cost <= budget:
                kept.add(int(i))
                used += cost
        return self._rebuild(passages, kept, len(sources))

    def _split(self, sources: List[str]) -> List[Tuple[int, str]]:
        """Cut each source into paragraphs, tagged with their source index.

        Paragraphs are separated by blank lines; long ones (code) are cut
        every MAX_PASSAGE_LINES lines.
        """
        passages: List[Tuple[int, str]] = []
        for index, text in enumerate(sources):
            for block in re.split(r"\n\s*\n", text):
                lines = block.split("\n")
                for start in range(0, len(lines), MAX_PASSAGE_LINES):
                    piece = "\n".join(
                        lines[start:start + MAX_PASSAGE_LINES]
                    ).strip()
                    if piece:
                        passages.append((index, piece))
        return passages

    def _rebuild(
        self,
        passages: List[Tuple[int, str]],
        kept: Set[int],
        n_sources: int,
    ) -> List[str]:
        """Join the kept paragraphs of each source in their original order."""
        result: List[str] = []
        for index in range(n_sources):
            pieces = [
                text for i, (source, text) in enumerate(passages)
                if i in kept and source == index
            ]
            if pieces:
                result.append("\n...\n".join(pieces))
        return result
