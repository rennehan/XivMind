from XivMind.backend import DataPipeline
from XivMind.arxiv.metadatamanager import MetaDataManager
import datetime as dt
import os

if __name__ == "__main__":
    pipeline = DataPipeline(MetaDataManager("arxiv-metadata-oai-snapshot.json"))

    metadata_path = pipeline.config["paths"]["arxiv"]["metadata"]
    index_file = metadata_path + "/" + pipeline.config["files"]["arxiv"]["index"]
    if os.path.isfile(index_file):
        index = pipeline.load_metadata_index(cache=True)
    else:
        index = pipeline.build_metadata_index(cache=True)

    # Filter only entries from 2020 to 2025
    start_date = dt.datetime(2020, 1, 1)
    end_date = dt.datetime(2025, 12, 31)
    date_range = [start_date, end_date]

    # Only use astro-ph
    categories = ["astro-ph.GA"]

    results = pipeline.data_manager.filter(categories=categories, date_range=date_range)


