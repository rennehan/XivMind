import openai
import faiss
import numpy as np
import json
from typing import List, Tuple
from openai import OpenAI
from qa import QA

class OpenAIQA(QA):
    def __init__(self, client: OpenAI = None):
        self.client = client or OpenAI()

    def load_index(self, index_path: str) -> faiss.IndexFlatL2:
        return faiss.read_index(index_path)

    def load_metadata(self, json_path: str) -> List[dict]:
        with open(json_path, "r") as f:
            return json.load(f)

    def embed_question(self, question: str, 
                       model="text-embedding-3-small") -> np.ndarray:
        response = self.client.embeddings.create(model=model, input=[question])
        return np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)

    def retrieve_top_k(self, index, metadata: List[dict], 
                       query_vec: np.ndarray, k: int = 5) -> List[dict]:
        distances, indices = index.search(query_vec, k)
        return [metadata[i] for i in indices[0]]

    def generate_answer(self, question: str, contexts: List[dict], 
                        model="gpt-3.5-turbo") -> str:
        context_text = "\n\n".join(f"- {doc['summary']}" for doc in contexts)
        prompt = (
            "You are an AI assistant answering questions based on "
            "scientific paper abstracts. "
            "Use only the provided context to answer the question.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\nAnswer:"
        )

        response = self.client.responses.create(
            model=model,
            input=[
                {"role": "system", 
                 "content": "You only answer using provided scientific context."},
                {"role": "user", 
                 "content": prompt}
            ],
            temperature=0.3
        )
        return response.output_text.strip()

    def rag_query(self, question: str, k: int = 5) -> Tuple[str, List[dict]]:
        index = self.load_index("data/faiss_index.index")
        metadata = self.load_metadata("data/arxiv_astro_ph_summarized.json")
        query_vec = self.embed_question(question)
        top_contexts = self.retrieve_top_k(index, metadata, query_vec, k)
        answer = self.generate_answer(question, top_contexts)
        return answer, top_contexts

