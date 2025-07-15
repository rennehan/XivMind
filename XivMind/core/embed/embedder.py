from abc import ABC, abstractmethod
from typing import List

class Embedder(ABC):
    @abstractmethod
    def embed_text(self, texts: List[str] | str) -> List[List[float]]:
        """Returns a list of embeddings for the input texts"""
        pass
    