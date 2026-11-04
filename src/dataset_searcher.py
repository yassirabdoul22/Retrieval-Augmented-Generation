import sys
from typing import List, cast

from tqdm import tqdm

from src.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)
from src.retrieval.base import Retriever
from src.utils.exceptions import InvalidPromptError


class DatasetSearcher:

    """Run retrieval over a validated question dataset."""

    def __init__(self, retriever: Retriever, k: int):
        """Store the retriever and the requested number of results."""
        self.retriever = retriever
        self.k = k

    def search(self, dataset: RagDataset) -> StudentSearchResults:
        """Retrieve sources for each question while preserving question IDs."""
        results: List[MinimalSearchResults] = []
        for question in tqdm(dataset.rag_questions):
            try:
                sources = self.retriever.retrieve(question.question, self.k)
            except InvalidPromptError as e:
                print(f"{e} (question_id={question.question_id})",
                      file=sys.stderr)
                sources = []
            results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=cast(List[MinimalSource], sources),
                )
            )
        return StudentSearchResults(search_results=results, k=self.k)
