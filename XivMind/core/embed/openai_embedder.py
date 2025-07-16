from typing import List, Tuple
from tqdm import tqdm
from openai import OpenAI, AsyncOpenAI
from .embedder import Embedder
import asyncio
from yaml import safe_load as load_config

class OpenAIEmbedder(Embedder):
    def __init__(self, client: OpenAI | AsyncOpenAI = None,
                 model_spec: str = "text-embedding-3-small", batch_size: int = 20):
        
        self.config = load_config(open("./XivMind/configs/config.yaml", "r"))
        self.models = self.config["supported_models"]
        self.embedding_models = self.config["supported_embedding_models"]

        self.model_name = "openai"
        self.model_spec = model_spec
        self.batch_size = batch_size

        # TODO: Remove, this is redundant.
        if self.model_name not in self.embedding_models:
            raise ValueError(self._unsupported_embedding_model_error())
    
        self.client = client or AsyncOpenAI()

    def _unsupported_embedding_model_error(self) -> str:
        return (f"Unsupported embedding model: {self.model_name}. "
                f"Supported models are: {self.embedding_models}")
    
    def _embed_text_sync(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model_spec,
                    input=batch
                )

                batch_embeddings = [d.embedding for d in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error embedding batch {i}: {e}")

        return embeddings
    
    async def _embed_text_async(self, texts: List[str]) -> List[List[float]]:
        async def _embed_batch(batch_idx: int, batch: List[str]) -> Tuple[int, List[List[float]]]:
            try:
                response = await self.client.embeddings.create(
                    model=self.model_spec,
                    input=batch
                )
                return batch_idx, [d.embedding for d in response.data]
            except Exception as e:
                print(f"Error embedding batch: {e}")
                return batch_idx, [None] * len(batch)

        tasks = []
        for i in range(0, len(texts), self.batch_size):
            batch_idx = i // self.batch_size
            print(f"Scheduling batch {batch_idx + 1} for embedding.")
            batch = texts[i:i+self.batch_size]
            tasks.append(_embed_batch(batch_idx, batch))

        results = await asyncio.gather(*tasks)

        # Reorder the batched results to maintain input order
        results.sort(key=lambda x: x[0])
        embeddings = []
        for _, batch_embeddings in results:
            embeddings.extend(batch_embeddings)

        return embeddings

    def validate_text(self, text: str) -> Tuple[bool, str]:
        if not isinstance(text, str):
            return False, "Input text must be a string."
        if len(text.strip()) == 0:
            return False, "Input text cannot be empty or whitespace."
        # TODO: Check actual token length
        if len(text) > 8192:
            return False, "Input text exceeds maximum length of 8192 characters."
        return True, "Valid text."

    async def embed_text(self, texts: List[str] | str) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        
        for i, text in enumerate(texts):
            validated, message = self.validate_text(text)
            if not validated:
                print(f"Error with the input text at index {i}:")
                print(message)
                exit(1)

        if isinstance(self.client, AsyncOpenAI):
            return await self._embed_text_async(texts)
        else:
            return self._embed_text_sync(texts)
