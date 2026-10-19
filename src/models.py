import uuid
from typing import List
from pydantic import BaseModel, Field, computed_field


class MinimalSource(BaseModel):
    """Locate a source using its path and exclusive character range."""
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Identify a question whose answer is not yet available."""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Pair a reference answer with its supporting source locations."""
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Collect answered or unanswered questions for batch processing."""
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Associate one question with its ranked retrieved sources."""
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def question_str(self) -> str:
        """Alias expected by the grading moulinette, mirrors ``question``."""
        return self.question


class MinimalAnswer(MinimalSearchResults):
    """Attach a generated answer to one question and its sources."""
    answer: str


class StudentSearchResults(BaseModel):
    """Store batch retrieval results and the requested result count."""
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Store batch answers with sources and the requested result count."""
    search_results: List[MinimalAnswer]
    k: int


class Chunk(MinimalSource):
    """A source location together with its raw text content."""
    text: str

    @property
    def indexable_text(self) -> str:
        """Combine the source path and content for lexical indexing."""
        return f"{self.file_path.replace('/', ' ')}\n{self.text}"
