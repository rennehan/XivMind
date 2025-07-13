from XivMind.pipeline import Pipeline

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    pipeline = Pipeline("OpenAI:gpt-3.5-turbo")

    # pipeline.request_model_from_user()

    pipeline.load_agents()

    # Embed documents
    #pipeline.embed()

    # Summarize documents
    #pipeline.summarize()

    # Build FAISS index
    #pipeline.build_faiss_index()

    #while True:
    #    # Run RAG query
    #    answer, sources = pipeline.rag_qa()
        
    #    print("\n📘 Answer:\n", answer)
    #    print("\n🔍 Sources:\n")
    #    for i, paper in enumerate(sources, 1):
    #        print(f"{i}. {paper['title']} ({paper["pdf_url"]})")
    #    print("\n")

