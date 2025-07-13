from XivMind.pipeline import Pipeline

# Only have OpenAI implemented right now
valid_models = ["OpenAI"]
client = None

def request_model_from_user():
    print("Enter an available model name (e.g., OpenAI)")
    model_name = input("Model name: ").strip()

    if model_name.lower() not in [m.lower() for m in valid_models]:
        print(f"Unsupported model: {model_name}.")
        print(f"Available models: {', '.join(valid_models)}")
        print("Please check your input and try again.")
        exit(1)
    return model_name

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    model_name = request_model_from_user()

    # Create pipeline configuration
    # TODO: Load from file or env
    config = {
        "model_name": model_name,
        "model_spec": "gpt-3.5-turbo",
        "agents_path": "./XivMind/configs/agents",
        "dataset_path": "./data"
    }

    pipeline = Pipeline(config)
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

