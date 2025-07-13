from XivMind.backend import DataPipeline
from XivMind.arxiv.metadatamanager import MetaDataManager

if __name__ == "__main__":
    pipeline = DataPipeline(MetaDataManager("arxiv-metadata-oai-snapshot.json"))

    #metadata = pipeline.load_metadata(limit=10)
    index = pipeline.build_metadata_index(cache=True, limit=10)

    print(index)