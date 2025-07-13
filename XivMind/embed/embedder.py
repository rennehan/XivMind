from abc import ABC, abstractmethod
from typing import List

class Embedder(ABC):
    @abstractmethod
    def load_documents(self, path: str) -> List[dict]:
        """Loads documents from a given path"""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Returns a list of embeddings for the input texts"""
        pass

    @abstractmethod
    def embed_and_save(self, input_path: str, output_path: str):
        """Embeds documents and saves the results to a specified output path"""
        pass

    