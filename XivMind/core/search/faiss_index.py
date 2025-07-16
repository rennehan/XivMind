import faiss
import numpy as np
import json
import os

def load_embeddings(filepath: str):
    """
    Load papers and their embeddings from a JSON file.
    Returns a tuple: (list of papers, 2D numpy array of embeddings).
    """
    with open(filepath, "r") as f:
        papers = json.load(f)
    embeddings = np.array([paper["embedding"] for paper in papers]).astype("float32")
    return papers, embeddings

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Build a FAISS index from a 2D numpy array of float32 embeddings.
    Uses IndexFlatL2 (good for cosine similarity when vectors are normalized).
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_faiss_index(index: faiss.IndexFlatL2, filepath: str):
    """
    Save FAISS index to disk.
    """
    faiss.write_index(index, filepath)

def load_faiss_index(filepath: str) -> faiss.IndexFlatL2:
    """
    Load a FAISS index from disk.
    """
    return faiss.read_index(filepath)

if __name__ == "__main__":
    print("Loading embeddings...")
    papers, embeddings = load_embeddings("data/arxiv_astro_ph_embedded.json")
    print(f"Loaded {len(embeddings)} embeddings.")

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("Saving index...")
    os.makedirs("data", exist_ok=True)
    save_faiss_index(index, "data/faiss_index.index")

    print("Done. Index saved to data/faiss_index.index")
