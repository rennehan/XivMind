from collections import defaultdict
from pyexpat import model
from .arxiv.metadatamanager import MetaDataManager
from .core.sql import SQLQueryContainer
from .pipeline import Pipeline
from typing import Dict
from openai import OpenAI
import pickle 
import json 
import os
import sqlite3

class DataPipeline:
    pipeline = None
    data_manager = None

    def __init__(self, metadata_manager: MetaDataManager, pipeline: Pipeline = None):
        self.data_manager = metadata_manager
        self.config = metadata_manager.get_config()
        if pipeline is not None:
            self.pipeline = pipeline
            self.model = pipeline.model
            self.model_name = pipeline.model_name
            self.model_spec = pipeline.model_spec

    def _pipeline_not_loaded_error(self):
        return ("Pipeline not loaded. Use load_pipeline() to load a Pipeline instance.")

    def load_metadata(self, fields: list[str] = [], limit: int = None, ids: Dict = None) -> defaultdict:
        return self.data_manager.load_metadata(fields=fields, limit=limit)

    def build_metadata_index(self, cache: bool = True, limit: int = None) -> defaultdict:
        return self.data_manager.build_metadata_index(cache=cache, limit=limit)

    def load_metadata_index(self, cache: bool = True, limit: int = None) -> defaultdict:
        return self.data_manager.load_metadata_index(cache=cache, limit=limit)

    def load_pipeline(self, pipeline: Pipeline, load_agents: bool = True):
        self.pipeline = pipeline
        self.model = pipeline.model
        self.model_name = pipeline.model_name
        self.model_spec = pipeline.model_spec
        if load_agents:
            self.pipeline.load_agents()

    def load_summaries_index(self) -> Dict:
        summary_path = self.config["paths"]["arxiv"]["summaries"] + "/" + \
                       self.config["files"]["arxiv"]["summaries_index"]
        
        with open(summary_path, "rb") as f:
            summary_index = pickle.load(f)
    
        return summary_index
    
    def cache_summaries(self, summaries: str) -> int:
        """
        Cache summaries to disk. The input summaries should be a dictionary that
        maps the summarizer name to the dictionary of key->value pairs, where
        the key is the paper ID and the value is the summary text. That will
        allow a quick lookup of summaries by paper ID and summarizer.

        Returns the number of inserted summaries.
        """
        summary_path = self.config["paths"]["arxiv"]["summaries"]
        summary_file = summary_path + "/" + self.config["files"]["arxiv"]["summaries"]

        # Lazy loading the SQL queries from disk as needed
        queries = SQLQueryContainer(self.config)

        conn = sqlite3.connect(summary_file)
        c = conn.cursor()

        c.execute(queries.retrieve("create-summaries-table"))

        full_model = f"{self.model_name}:{self.model_spec}"
        data_container = []
        for summarizer in summaries.keys():
            for arxiv_id, summary in summaries[summarizer]:
                data_container.append((summarizer, full_model, arxiv_id, summary))

        inserted_count = 0
        try:
            with conn:
                c.executemany(queries.retrieve("insert-summary"), data_container)
            inserted_count = len(data_container)
        except sqlite3.Error as e:
            print(f"Summaries were not inserted due to an error: {e}")
        finally:
            conn.close()

        return inserted_count
    
    async def summarize_abstracts(self, papers: Dict, agent_class: str = "summarizer", 
                                  agent_name: str = None) -> str:
        # TODO: Should this really be interactive? 
        if self.pipeline is None:
            model_name = input("Pipeline not loaded. Please enter the name of the model to load: ")
            self.load_pipeline(Pipeline(model_name))

        paper_ids = []
        abstracts = []
        for paper_id in papers.keys():
            paper_ids.append(paper_id)
            abstracts.append(papers[paper_id]["abstract"])

        summaries = await self.pipeline.summarize_text(abstracts, 
                                                       agent_class=agent_class, 
                                                       agent_name=agent_name)
        
        # Reformat the summaries into a structure that stores the 
        # summarizer name -> (paper ID, full model name, summary text)
        # for caching
        reformatted_summaries = {}
        
        for summarizer in summaries.keys():
            reformatted_summaries[summarizer] = []
            for idx, summary in enumerate(summaries[summarizer]):
                paper_id = paper_ids[idx]
                reformatted_summaries[summarizer].append((paper_id, summary))

        # TODO: Make this a Pydantic model to ensure type safety
        return reformatted_summaries
