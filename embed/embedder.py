from abc import ABC, abstractmethod
from typing import List

class Embedder(ABC):
    @abstractmethod
    def load_documents(self, path: str) -> List[dict]:
        """Loads documents from a given path"""
        pass

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Returns a list of embeddings for the input texts"""
        pass