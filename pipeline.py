from openai import OpenAI
from embed import openai_embedder
from summarize import openai_summarizer
from search import faiss_index
from RAG import openai_qa

class Pipeline:
    models = ["openai"]
    embedder = None
    summarizer = None
    client = None
    rag_system = None

    def __init__(self, model_name: str):
        self.model_name = model_name.lower()
        self.client = self.load_client()
        self.embedder = self.get_embedder()
        self.summarizer = self.get_summarizer()
        self.rag_system = self.get_rag_system()

    def load_client(self, model: str = None):
        if self.model_name == "openai":
            return OpenAI()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}.")

    ######
    # Embedding methods
    def get_embedder(self):        
        if self.model_name == "openai":
            return openai_embedder.OpenAIEmbedder(client=self.client)
        else:
            raise ValueError(f"Unsupported model for embedding: {self.model_name}.")

    def embed(self):
        if self.embedder is None:
            self.get_embedder()
  
        self.embedder.embed_and_save("data/arxiv_astro_ph.json",
                                     "data/arxiv_astro_ph_embedded.json")
    ######

    ######
    # Summarization methods
    def get_summarizer(self):
        if self.model_name == "openai":
            return openai_summarizer.OpenAISummarizer(client=self.client)
        else:
            raise ValueError(f"Unsupported model for summarization: {self.model_name}.")
    
    def summarize(self):
        if self.summarizer is None:
            self.get_summarizer()

        self.summarizer.summarize_all("data/arxiv_astro_ph_embedded.json", 
                                      "data/arxiv_astro_ph_summarized.json")
    ######
    # RAG Q&A methods
    def get_rag_system(self):
        if self.model_name == "openai":
            return openai_qa.OpenAIQA(client=self.client)
        else:
            raise ValueError(f"Unsupported model for RAG: {self.model_name}.")
        
    def rag_qa(self):
        if self.rag_system is None:
            self.get_rag_system()

        user_question = input("Ask a question about arXiv papers: ")
        answer, sources = self.rag_system.rag_query(user_question)

        return answer, sources
    
    ######

    ######
    # FAISS index methods
    # Build FAISS index
    def build_faiss_index(self):
        print("Loading embeddings...")
        papers, embeddings = faiss_index.load_embeddings("data/arxiv_astro_ph_embedded.json")
        print(f"Loaded {len(embeddings)} embeddings.")

        print("Building FAISS index...")
        index = faiss_index.build_faiss_index(embeddings)

        print("Saving index...")
        faiss_index.save_faiss_index(index, "data/faiss_index.index")
        print("Done. Index saved to data/faiss_index.index")
    ######

        




