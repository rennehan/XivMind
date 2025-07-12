from abc import ABC, abstractmethod
from typing import List, Tuple
import faiss
import numpy as np

class QA(ABC):
    @abstractmethod
    def load_index(index_path: str) -> faiss.IndexFlatL2:
        """
        Loads a FAISS index from the specified path.
        :param index_path: The path to the FAISS index file.
        :return: A FAISS index object.
        """
        pass
    
    @abstractmethod
    def load_metadata(json_path: str) -> List[dict]:
        """
        Loads metadata from a JSON file.
        :param json_path: The path to the JSON file containing metadata.
        :return: A list of dictionaries containing metadata.
        """
        pass
    
    @abstractmethod
    def embed_question(question: str, model="text-embedding-3-small") -> np.ndarray:
        """
        Embeds a question using a specified model.
        :param question: The question to embed.
        :param model: The embedding model to use.
        :return: A numpy array representing the embedded question.
        """
        pass
    
    @abstractmethod
    def retrieve_top_k(index, metadata: List[dict], query_vec: np.ndarray, k: int = 5) -> List[dict]:
        """
        Retrieves the top k contexts from the index based on the query vector.
        :param index: The FAISS index to search.
        :param metadata: The metadata associated with the indexed documents.
        :param query_vec: The embedded question vector.
        :param k: The number of top contexts to retrieve.
        :return: A list of dictionaries containing the top k contexts.
        """
        pass
    
    @abstractmethod
    def generate_answer(question: str, contexts: List[dict], model="gpt-3.5-turbo") -> str:
        """
        Generates an answer to the question based on the provided contexts.
        :param question: The question to answer.
        :param contexts: The contexts to use for generating the answer.
        :param model: The model to use for generating the answer.
        :return: A string containing the generated answer.
        """
        pass
    
    @abstractmethod
    def rag_query(question: str, k: int = 5) -> Tuple[str, List[dict]]:
        """
        Performs a RAG query to answer the question using the indexed documents.
        :param question: The question to answer.
        :param k: The number of top contexts to retrieve.
        :return: A tuple containing the generated answer and the top contexts.
        """
        pass
