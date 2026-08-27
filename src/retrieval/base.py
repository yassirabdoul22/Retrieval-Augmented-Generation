from abc import ABC , abstractmethod
from typing import List 
from src.models import Chunk, MinimalSource

class Retriever(ABC):
    
    @abstractmethod
    def index(self, chunks:List[Chunk])->None:
        pass 

    @abstractmethod
    def retrieve(self, query:str,k:int)->List[MinimalSource]:
        pass 

    @abstractmethod
    def save(self, path:str)->None:
        pass

    @abstractmethod
    def load(self,path:str)->None:
        pass 

