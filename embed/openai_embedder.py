import numpy as np
import json
from typing import List, Dict
from tqdm import tqdm
import os
import time
from openai import OpenAI
from embed.embedder import Embedder

class OpenAIEmbedder(Embedder):
    def __init__(self, client: OpenAI = None,
                 model: str = "text-embedding-3-small", batch_size: int = 20):
        self.model = model
        self.batch_size = batch_size
        self.client = client or OpenAI()

    def load_documents(self, path: str) -> List[Dict]:
        with open(path, "r") as f:
            return json.load(f)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding batches"):
            batch = texts[i:i+self.batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )

                batch_embeddings = [d.embedding for d in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error embedding batch {i}: {e}")
                time.sleep(5)

        return embeddings

    def embed_and_save(self, input_path: str, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        docs = self.load_documents(input_path)
        texts = [doc["summary"] for doc in docs]

        print(f"Embedding {len(texts)} documents...")

        embeddings = self.embed_documents(texts)
        for doc, emb in zip(docs, embeddings):
            doc["embedding"] = emb

        with open(output_path, "w") as f:
            json.dump(docs, f, indent=2)

        print(f"Saved embeddings to {output_path}")
