from XivMind.backend import DataPipeline
from XivMind.pipeline import Pipeline
from XivMind.arxiv.metadatamanager import MetaDataManager
from XivMind.core.embed.openai_embedder import OpenAIEmbedder
from XivMind.core.embed.caching_embedder import CachingEmbedder
from XivMind.core.cache.cache import Cache
from XivMind.core.search.faiss_index_manager import FAISSIndexManager
import datetime as dt
import os
import asyncio
import numpy as np

if __name__ == "__main__":
    data_pipeline = DataPipeline(MetaDataManager("arxiv-metadata-oai-snapshot.json"))

    rebuild = False
    metadata_path = data_pipeline.config["paths"]["arxiv"]["metadata"]
    index_file = metadata_path + "/" + data_pipeline.config["files"]["arxiv"]["index"]
    if os.path.isfile(index_file) and not rebuild:
        index = data_pipeline.load_metadata_index(cache=True)
    else:
        index = data_pipeline.build_metadata_index(cache=True)

    # Filter only entries from a narrow date range
    start_date = dt.datetime(2025, 5, 13)
    end_date = dt.datetime(2025, 5, 14)
    date_range = [start_date, end_date]

    # Only use astro-ph
    categories = ["astro-ph.GA"]

    results = data_pipeline.data_manager.filter(categories=categories,
                                                date_range=date_range)

    if len(results) == 0:
        print("No results found for the given categories and date range.")
        exit(0)

    # Abstracts are needed to summarize and embed. Flatten the results 
    # dictionary into a list of IDs to grab from the JSON file.
    ids = {}
    for category in results.keys():
        for date in results[category].keys():
            for id in results[category][date]:
                ids.update({id: None})

    papers = data_pipeline.load_metadata(fields=["title", "abstract"],
                                         ids=ids)
    
    if len(papers) == 0:
        print("No papers found for the given IDs.")
        exit(0)

    pipeline = Pipeline("OpenAI:gpt-3.5-turbo", "OpenAI:text-embedding-3-small")
    data_pipeline.load_pipeline(pipeline, load_agents=True)

    # TODO: Choose a different summarizer based on the config
    summarizer_name = "DomainExpertSummarizer"

    # async wrapper to run in __main__
    async def get_summaries(papers, agent_name):
        return await data_pipeline.summarize_abstracts(papers, agent_name=agent_name)
    
    # Data format is dict[summarizer name] = tuple(paper ID, summary text)
    summaries = asyncio.run(get_summaries(papers, agent_name=summarizer_name))

    # TODO: Check if already in the cache
    inserted_count = data_pipeline.cache_summaries(summaries)
    if inserted_count > 0:
        print(f"Inserted {inserted_count} new summaries into the cache.")

    # TODO: Choose a different embedder based on the config
    caching_embedder = CachingEmbedder(
        OpenAIEmbedder(model_spec="text-embedding-3-small", batch_size=20)
    )
    
    print("Embedding summaries.")
    
    # The CachingEmbedder will check the cache first,
    # and return anything that is already cached. 
    #
    # However, it requires a dictionary with keys as the unique identifiers and 
    # values as the texts to embed. 
    embedder_data = {}
    for summarizer_name in summaries:
        for paper_id, paper_summary in summaries[summarizer_name]:
            key = Cache.get_key(
                agent_name=summarizer_name,
                model_name=caching_embedder.embedder.model_name,
                model_spec=caching_embedder.embedder.model_spec,
                arxiv_id=paper_id,
                label="abstract"
            )
            embedder_data.update({key: paper_summary})

    # async wrapper to run in __main__
    async def embed_texts(embedder, texts):
        return await embedder.embed_text(texts)
    
    embeddings = asyncio.run(embed_texts(caching_embedder, embedder_data))

    # Build the FAISS index from the embeddings
    if len(embeddings) == 0:
        print("No embeddings to index.")
        exit(0)

    print("Building FAISS index.")

    # NOTE: Order matters here because the FAISS index must align with the 
    # cache order.
    key_list = []
    embeddings_list = []
    for key in embeddings.keys():
        key_list.append(key)
        embeddings_list.append(embeddings[key])

    faiss_manager = FAISSIndexManager()

    embeddings_array = np.array([embedding for embedding in embeddings_list], 
                                dtype=float)

    if embeddings_array.ndim != 2:
        print(f"Error: Embeddings array has invalid shape {embeddings_array.shape}.")
        exit(0)

    faiss_index = faiss_manager.build_faiss_index(embeddings_array)
    faiss_manager.save_faiss_index(faiss_index, key_list)

    print("Done!")

    
    





