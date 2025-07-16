from XivMind.backend import DataPipeline
from XivMind.pipeline import Pipeline
from XivMind.arxiv.metadatamanager import MetaDataManager
from XivMind.core.embed.openai_embedder import OpenAIEmbedder
import datetime as dt
import os
import asyncio

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

    async def get_summaries(papers, agent_name):
        return await data_pipeline.summarize_abstracts(papers, agent_name=agent_name)

    papers = data_pipeline.load_metadata(fields=["title", "abstract"],
                                         ids=ids)
    
    if len(papers) == 0:
        print("No papers found for the given IDs.")
        exit(0)

    pipeline = Pipeline("OpenAI:gpt-3.5-turbo")
    data_pipeline.load_pipeline(pipeline, load_agents=True)

    # TODO: Choose a different summarizer based on the config
    summarizer_name = "DomainExpertSummarizer"

    summaries = asyncio.run(get_summaries(papers, agent_name=summarizer_name))

    # TODO: Check if already in the cache
    data_pipeline.cache_summaries(summaries)

    # TODO: Choose a different embedder based on the config
    embedder = OpenAIEmbedder(model="text-embedding-3-small", batch_size=20)

    async def embed_texts(embedder, texts):
        return await embedder.embed_text(texts)
    
    print("Embedding summaries.")
    
    # Now embed the summaries
    embeddings = asyncio.run(embed_texts(embedder, [summary for _, summary in summaries[summarizer_name]]))

    #data_pipeline.cache_embeddings(embeddings, papers, summaries)

    print("Done!")

    
    





