from collections import defaultdict
from pyexpat import model
from .arxiv.metadatamanager import MetaDataManager
from .core.sql import SQLQueryContainer
from .pipeline import Pipeline
from typing import Dict, List, Tuple
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

    def load_summaries(self, keys: Tuple = None, 
                       keys_values: List[Tuple[str]] | Tuple[str] = None) -> Dict:
        """
        Load cached summaries from disk. If a summarizer name is provided,
        only load summaries for that summarizer. Otherwise, load all summaries.

        Returns a dictionary that maps the summarizer name to the dictionary of
        key->value pairs, where the key is the paper ID and the value is the summary text.
        """
        summary_path = self.config["paths"]["arxiv"]["summaries"]
        summary_file = summary_path + "/" + self.config["files"]["arxiv"]["summaries"]

        # Lazy loading the SQL queries from disk as needed
        queries = SQLQueryContainer(self.config)

        conn = sqlite3.connect(summary_file)
        c = conn.cursor()

        rows = []
        if keys_values is not None:
            if not isinstance(keys_values, List):
                keys_values = [keys_values]


            if keys is None:
                c.executemany(
                    queries.retrieve("select-summaries-by-primary-key"),
                    keys_values
                )
            else:
                summarizer_name_in_keys = "summarizer_name" in keys
                full_model_name_in_keys = "full_model_name" in keys
                arxiv_id_in_keys = "arxiv_id" in keys

                if len(keys) > 1:
                    # TODO: This is messy, probably because I have a rigid
                    # query structure. All of these combinations should be
                    # handled dynamically.
                    if summarizer_name_in_keys and full_model_name_in_keys and arxiv_id_in_keys:
                        if keys != ("summarizer_name", "full_model_name", "arxiv_id"):
                            raise ValueError("If all three keys are provided, they must be in the order: (summarizer_name, full_model_name, arxiv_id)")
                    elif summarizer_name_in_keys and full_model_name_in_keys:
                        if keys != ("summarizer_name", "full_model_name"):
                            raise ValueError("If both summarizer_name and full_model_name are provided, they must be in that order.")
                    elif summarizer_name_in_keys and arxiv_id_in_keys:
                        if keys != ("summarizer_name", "arxiv_id"):
                            raise ValueError("If both summarizer_name and arxiv_id are provided, they must be in that order.")
                    elif full_model_name_in_keys and arxiv_id_in_keys:
                        if keys != ("full_model_name", "arxiv_id"):
                            raise ValueError("If both full_model_name and arxiv_id are provided, they must be in that order.")

                # TODO: This needs to be rethought. Right now the query 
                # structure is very rigid and the queries might just have
                # to be done dynamically.
                query_name = "select-summaries-by-"
                query_parts = []

                if summarizer_name_in_keys:
                    query_parts.append("summarizer")
                if full_model_name_in_keys:
                    query_parts.append("full-model")
                if arxiv_id_in_keys:
                    query_parts.append("arxiv-id")

                if len(query_parts) == 0:
                    raise ValueError("No valid keys provided for summary lookup.")

                query_name += "-and-".join(query_parts)

                for key_value in keys_values:
                    c.execute(
                        queries.retrieve(query_name),
                        key_value
                    )
                    rows.extend(c.fetchall())
        else:
            c.execute(queries.retrieve("select-all-summaries"))
            rows = c.fetchall()

        conn.close()

        summaries = {}
        for row in rows:
            summarizer_name = row[0]
            # TODO: Right now the standard summaries format does not
            # carry the full model name. I might want to change that.
            #full_model_name = row[1]
            paper_id = row[2]
            summary_text = row[3]

            if summarizer_name not in summaries:
                summaries[summarizer_name] = []

            summaries[summarizer_name].append((paper_id, summary_text))

        return summaries

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

        full_model_name = f"{self.model_name}:{self.model_spec}"
        data_container = []
        for summarizer in summaries.keys():
            for arxiv_id, summary in summaries[summarizer]:
                data_container.append((summarizer, full_model_name, arxiv_id, summary))

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
                                  agent_name: str = None,
                                  cache: bool = True) -> str:
        # TODO: Should this really be interactive? 
        if self.pipeline is None:
            model_name = input("Pipeline not loaded. Please enter the name of the model to load: ")
            self.load_pipeline(Pipeline(model_name))

        if cache:
            full_model_name = f"{self.model_name}:{self.model_spec}"
            keys = ("summarizer_name", "full_model_name", "arxiv_id") if agent_name is not None else ("full_model_name", "arxiv_id")
            print(f"Checking for cached summaries for model {full_model_name}.")
            cached_summaries = {}
            cached_summaries_count = 0

        paper_ids = []
        abstracts = []
        for paper_id in papers.keys():
            if cache:
                # TODO: keys order matters here and really shouldn't
                keys_values = (agent_name, full_model_name, paper_id) if agent_name is not None else (full_model_name, paper_id)
                results = self.load_summaries(
                    keys=keys,
                    keys_values=keys_values
                )

                if len(results) > 0:
                    cached_summaries_count += len(results)
                    # load_summaries() returns a dictionary that maps
                    # the summarizer name to a list of (paper ID, summary text)
                    # tuples, which is already the desired format for summarizing.
                    for summarizer in results.keys():
                        if summarizer not in cached_summaries:
                            cached_summaries[summarizer] = []
                        cached_summaries[summarizer].extend(results[summarizer])
                    continue

            paper_ids.append(paper_id)
            abstracts.append(papers[paper_id]["abstract"])

        if cache:
            print(f"Loaded {cached_summaries_count} cached summaries.")

            if len(cached_summaries) > 0 and len(abstracts) == 0:
                # All summaries were cached, so return them directly
                return cached_summaries
            
            print(f"Summarizing the remaining {len(abstracts)} abstracts.")

        summaries = await self.pipeline.summarize_text(abstracts, 
                                                       agent_class=agent_class, 
                                                       agent_name=agent_name)
        
        # Reformat the summaries into a structure that stores the 
        # summarizer name -> (paper ID, summary text)
        # for caching
        reformatted_summaries = {}
        
        for summarizer in summaries.keys():
            reformatted_summaries[summarizer] = []
            for idx, summary in enumerate(summaries[summarizer]):
                paper_id = paper_ids[idx]
                reformatted_summaries[summarizer].append((paper_id, summary))

        if cache:
            # Merge the cached summaries
            for summarizer in cached_summaries.keys():
                if summarizer not in reformatted_summaries:
                    reformatted_summaries[summarizer] = []
                reformatted_summaries[summarizer].extend(cached_summaries[summarizer])

        # TODO: Make this a Pydantic model to ensure type safety
        return reformatted_summaries
