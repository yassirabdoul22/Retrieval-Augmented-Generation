from src.generator import Generator
from src.retrieval.bm25_retriever import BM25Retriever
from src.utils.validator import Validator
from src.utils.io_utils import read_source_text
class RagService:
    def __init__(self)->None:
        self._retriever = BM25Retriever()
        self._generator = Generator()
        self._validator = Validator() 

    def answer(self,query:str,k:object):
        self._retriever.load("data/processed")
        sources = self._retriever.retrieve(query=query,k=k,validator=self._validator)
        source_text = [read_source_text(source) for source in sources]
        return self._generator.generate(query=query,source_text=source_text)
    