from abc import ABC, abstractmethod
from typing import List, Tuple

class Embedder(ABC):
    @abstractmethod
    def validate_text(self, text: str) -> Tuple[bool, str]:
        """Validates the input text for embedding."""
        pass

    @abstractmethod
    def embed_text(self, texts: List[str] | str) -> List[List[float]]:
        """Returns a list of embeddings for the input texts"""
        pass
    