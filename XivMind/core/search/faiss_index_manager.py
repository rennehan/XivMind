import numpy as np
import faiss
from typing import List, Tuple
import pickle
from yaml import safe_load as load_config

class FAISSIndexManager:
    def __init__(self, index_type: str = "IndexFlatL2"):
        self.config = load_config(open("./XivMind/configs/config.yaml", "r"))
        self.index_type = index_type
        self.faiss_index_file_path = self._construct_faiss_index_file_path()

        # TODO: Support more index types. For now use IndexFlatL2 for a 
        # brute force search over the embeddings, which is fine for a small
        # number of embeddings. 
        if self.index_type != "IndexFlatL2":
            raise NotImplementedError(f"Index type {self.index_type} not supported yet.")
        
    def _construct_faiss_index_file_path(self) -> str:
        faiss_path = self.config["paths"]["cache"]["faiss"]
        faiss_file = faiss_path + "/" + self.config["files"]["cache"]["faiss_index"]
        return faiss_file
    
    def _construct_faiss_keys_file_path(self) -> str:
        faiss_path = self.config["paths"]["cache"]["faiss"]
        keys_file = faiss_path + "/" + self.config["files"]["cache"]["faiss_keys"]
        return keys_file
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:    
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index

    def save_faiss_index(self, index: faiss.Index, key_list: List[str]) -> None:
        faiss.write_index(index, self.faiss_index_file_path)
        # Save the key list alongside the FAISS index for reference
        keys_path = self.config["paths"]["cache"]["faiss"]
        keys_file = keys_path + "/" + self.config["files"]["cache"]["faiss_keys"]
        with open(keys_file, "wb") as f:
            pickle.dump(key_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_faiss_index(self) -> Tuple[List[str], faiss.Index]:
        keys = pickle.load(open(self._construct_faiss_keys_file_path(), "rb"))

        return keys, faiss.read_index(self.faiss_index_file_path)