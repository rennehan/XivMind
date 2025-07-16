from typing import List, Dict, Tuple
from .embedder import Embedder
from ..cache.cache import Cache
import h5py
import sqlite3
import os

class CachingEmbedder(Embedder):
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.cache = {}

    def _cache_group_mismatch_error(self, group_path: str, embeddings_file: str) -> str:
        return(f"Fatal cache group mismatch: Group {group_path} not found in "
               f"the cache file {embeddings_file}. This indicates a serious "
               "issue with the cache integrity.")

    def _cache_mismatch_error(self, key: str, group_path: str, embeddings_file: str) -> str:
        return(f"Fatal cache mismatch:Key {key} not found in group "
               f"{group_path} in the cache file {embeddings_file}, but the key "
               "exists in the cache index.")

    def get_keys_in_cache_index(self, keys: List[str] | str) -> List:
        cache_index_path = self.embedder.config["paths"]["cache"]["embeddings_index"]
        cache_index_file = cache_index_path + "/" + self.embedder.config["files"]["cache"]["embeddings_index"]

        if not os.path.isfile(cache_index_file):
            return []
        
        if isinstance(keys, str):
            keys = [keys]

        conn = sqlite3.connect(cache_index_file)
        cursor = conn.cursor()

        found = []
        with conn:
            cursor.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY)")

            cursor.execute("DROP TABLE IF EXISTS temp_cache")
            cursor.execute("CREATE TABLE temp_cache (key TEXT PRIMARY KEY)")

            for i in range(0, len(keys), self.embedder.config["cache"]["batch_size"]):
                batch = keys[i:i+self.embedder.config["cache"]["batch_size"]]
                cursor.executemany("INSERT INTO temp_cache (key) VALUES (?)", [(key,) for key in batch])

            cursor.execute("""
                SELECT c.key FROM cache c
                INNER JOIN temp_cache t ON c.key = t.key
            """)
            rows = cursor.fetchall()

            cursor.execute("DROP TABLE IF EXISTS temp_cache")

            found = [row[0] for row in rows]

        return found
  
    def insert_keys_in_cache_index(self, keys: List[str] | str) -> None:
        cache_index_path = self.embedder.config["paths"]["cache"]["embeddings_index"]
        cache_index_file = cache_index_path + "/" + self.embedder.config["files"]["cache"]["embeddings_index"]

        if isinstance(keys, str):
            keys = [keys]

        conn = sqlite3.connect(cache_index_file)
        cursor = conn.cursor()

        with conn:
            cursor.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY)")

            for i in range(0, len(keys), self.embedder.config["cache"]["batch_size"]):
                batch = keys[i:i+self.embedder.config["cache"]["batch_size"]]
                cursor.executemany("INSERT OR IGNORE INTO cache (key) VALUES (?)", [(key,) for key in batch])

    def get_all_cached_embeddings_by_agent(self, agent_name: str) -> Dict:
        embeddings_path = self.config["paths"]["cache"]["embeddings"]
        embeddings_file = embeddings_path + "/" + self.config["paths"]["cache"]["embeddings_file"]
        
        with h5py.File(embeddings_file, "r") as f:
            group_path = f"/{agent_name}/{self.model_name}/{self.model_spec}"
            if group_path not in f:
                raise NotImplementedError(f"No embeddings found for agent '{agent_name}' with model '{self.model_name}:{self.model_spec}' in cache.")
            
            key_fragment = f"{agent_name}:{self.model_name}:{self.model_spec}:"
            papers = []
            embeddings = []
            for key in f[group_path].keys():
                if not key.startswith(key_fragment):
                    continue
                
                # The key format is agent_name:model_name:model_spec:arxiv_id:label
                _, _, _, arxiv_id, label = Cache.parse_key(key)

                # TODO: IMPORTANT: Currently only abstracts are cached.
                if label != "abstract":
                    continue

                papers.append(arxiv_id)
                embeddings.append(f[group_path][key][...])

        # NOTE: Order matters here, as the FAISS index aligns with the
        # cache order.
        return papers, embeddings

    def get_cached_embeddings(self, keys: List[str]) -> Dict:
        # Checking the existence of the keys in the cache index overwrites
        # the user input keys with only those that exist in the cache.
        keys = self.get_keys_in_cache_index(keys)

        if len(keys) == 0:
            return {}
    
        embeddings_path = self.embedder.config["paths"]["cache"]["embeddings"]
        embeddings_file = embeddings_path + "/" + self.embedder.config["files"]["cache"]["embeddings"]

        if not os.path.isfile(embeddings_file):
            return {}
        
        # Stores key->embedding mapping
        embeddings = {}

        # Get the unique agent_name and full_model_name combinations
        # from the key, since the HDF5 file is organized that way.
        # Use unique_group_paths as a hash table that stores the pointer
        # to the group in the HDF5 file.
        unique_group_paths = {}
        for key in keys:
            agent_name, model_name, model_spec, _, __ = Cache.parse_key(key)
            group_path = f"/{agent_name}/{model_name}/{model_spec}"
            if group_path not in unique_group_paths:
                unique_group_paths.update({group_path: None})
        
        with h5py.File(embeddings_file, "r") as f:
            for group_path in unique_group_paths.keys():
                if group_path in f:
                    unique_group_paths[group_path] = f[group_path]
                else:
                    raise ValueError(self._cache_group_mismatch_error(group_path, embeddings_file))

            for key in keys:
                agent_name, model_name, model_spec, _, __ = Cache.parse_key(key)
                group_path = f"/{agent_name}/{model_name}/{model_spec}"
                group = unique_group_paths[group_path]
                try:
                    # TODO: Read the dataset in chunks
                    embeddings.update({key: group[key][...]})
                except KeyError:
                    # This is a FATAL error. The key should exist in the HDF5
                    # file because it was found in the cache index. 
                    raise ValueError(self._cache_mismatch_error(key, group_path, embeddings_file))
                
        return embeddings

    def cache_embeddings(self, keys: List[str], embeddings: List[float]) -> None:     
        if len(keys) != len(embeddings):
            raise ValueError("Keys and embeddings length mismatch.")
        
        # Make sure the keys are in the cache index!
        # TODO: Remove keys if failure below
        self.insert_keys_in_cache_index(keys)

        embeddings_path = self.embedder.config["paths"]["cache"]["embeddings"]
        embeddings_file = embeddings_path + "/" + self.embedder.config["files"]["cache"]["embeddings"]

        if not os.path.isfile(embeddings_file):
            file_mode = "w"
        else:
            file_mode = "a"

        with h5py.File(embeddings_file, file_mode) as f:
            for i, key in enumerate(keys):
                agent_name, model_name, model_spec, _, __ = Cache.parse_key(key)
                group_path = f"/{agent_name}/{model_name}/{model_spec}"
                if group_path not in f:
                    group = f.create_group(group_path)
                else:
                    group = f[group_path]
                
                if key in group:
                    print(f"WARNING: Cache overwrite: {key} already exists in cache, overwriting.")
                    # TODO: check lengths
                    f[group_path + "/" + key] = embeddings[i]
                else:
                    group.create_dataset(key, data=embeddings[i], dtype=float)

    def validate_text(self, text: str) -> Tuple[bool, str]:
        return self.embedder.validate_text(text)
    
    # The unique key for cached embedding lookups is 
    async def embed_text(self, texts: Dict) -> Dict:
        texts_to_embed_count = len(texts)
        if texts_to_embed_count == 0:
            print("No texts provided to embed.")
            return {}

        embeddings = self.get_cached_embeddings(list(texts.keys()))
        if len(embeddings) > 0:
            print(f"Found {len(embeddings)} cached embeddings.")

            if len(embeddings) == len(texts):
                print("All embeddings found in cache.")
                return embeddings
    
        texts_to_embed_count -= len(embeddings)

        # Order matters here to keep embeddings aligned with input texts
        keys_to_embed = []
        texts_to_embed = []
        for key in texts.keys():
            if key not in embeddings:
                keys_to_embed.append(key)
                texts_to_embed.append(texts[key])

        print(f"Embedding {texts_to_embed_count} new texts.")
        new_embeddings = await self.embedder.embed_text(texts_to_embed)
        for i, key in enumerate(keys_to_embed):
            embeddings.update({key: new_embeddings[i]})

        self.cache_embeddings(keys_to_embed, new_embeddings)

        return embeddings