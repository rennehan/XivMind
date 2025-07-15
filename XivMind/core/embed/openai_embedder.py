from typing import List
from tqdm import tqdm
from openai import OpenAI, AsyncOpenAI
from .embedder import Embedder
import asyncio

class OpenAIEmbedder(Embedder):
    def __init__(self, client: OpenAI | AsyncOpenAI = None,
                 model: str = "text-embedding-3-small", batch_size: int = 20):
        self.model = model
        self.batch_size = batch_size
        self.client = client or AsyncOpenAI()

    def _embed_text_sync(self, texts: List[str]) -> List[List[float]]:
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

        return embeddings
    
    async def _embed_text_async(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding batches"):
            batch = texts[i:i+self.batch_size]
            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )

                batch_embeddings = [d.embedding for d in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error embedding batch {i}: {e}")

        return embeddings
    
    def embed_text(self, texts: List[str] | str) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        
        if isinstance(self.client, AsyncOpenAI):
            return asyncio.run(self.embed_text_async(texts))
        else:
            return self._embed_text_sync(texts)
