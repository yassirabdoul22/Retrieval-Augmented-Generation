from typing import List, TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from src.generator import Generator
from src.models import (
    MinimalAnswer,
    MinimalSearchResults,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.utils.io_utils import read_source_text


class DatasetAnswerer:

    """Generate grounded answers for a batch of retrieved questions."""

    def __init__(self, generator: "Generator") -> None:
        """Store the generator used to answer the retrieved questions."""
        self.generator = generator

    def answer_question(self, result: MinimalSearchResults) -> MinimalAnswer:
        """Generate one answer while preserving the question and sources."""
        sources_text = [
            read_source_text(source) for source in result.retrieved_sources
        ]
        answer_text = self.generator.generate(result.question, sources_text)
        return MinimalAnswer(
            question_id=result.question_id,
            question=result.question,
            retrieved_sources=result.retrieved_sources,
            answer=answer_text,
        )

    def answer(
        self, search_results: StudentSearchResults
    ) -> StudentSearchResultsAndAnswer:
        """Generate an answer for each question using its saved sources."""
        results: List[MinimalAnswer] = []
        for result in tqdm(search_results.search_results):
            results.append(self.answer_question(result))
        return StudentSearchResultsAndAnswer(
            search_results=results, k=search_results.k
        )
