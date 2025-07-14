from XivMind.backend import DataPipeline
from XivMind.pipeline import Pipeline
from XivMind.arxiv.metadatamanager import MetaDataManager
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
    # dictionary into a listç of IDs to grab from the JSON file.
    ids = {}
    for category in results.keys():
        for date in results[category].keys():
            for id in results[category][date]:
                ids.update({id: None})

    async def get_summaries(papers, agent_name):
        return await data_pipeline.summarize_abstracts(papers, agent_name=agent_name
                                                       )
    papers = data_pipeline.load_metadata(fields=["title", "abstract"],
                                         ids=ids)
    
    pipeline = Pipeline("OpenAI:gpt-3.5-turbo")
    data_pipeline.load_pipeline(pipeline, load_agents=True)

    summaries = asyncio.run(get_summaries(papers, agent_name="DomainExpertSummarizer"))

    data_pipeline.cache_summaries(summaries, papers)

    
    





