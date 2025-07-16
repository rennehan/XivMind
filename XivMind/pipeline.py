from openai import AsyncOpenAI
from yaml import safe_load as load_config
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
import os
from typing import List
import glob
import asyncio
from .core.embed.embedder import Embedder
from .core.embed.caching_embedder import CachingEmbedder
from .core.cache.cache import Cache
from .core.search.faiss_index_manager import FAISSIndexManager
import numpy as np

class Pipeline:
    model_name = None
    model_spec = None
    models = None
    config = None
    agents = None
    agent_links = None
    summarizer = None
    embedder = None

    def __init__(self, model: str, embedding_model: str):
        # Load main paths for the pipeline
        self.config = load_config(open("./XivMind/configs/config.yaml", "r"))
        self.models = self.config["supported_models"]
        self.embedding_models = self.config["supported_embedding_models"]
        
        self.prepare_model(model)
        self.prepare_embedding_model(embedding_model)

    ######
    # Error handling methods
    def _unimplemented_model_error(model_name) -> str:
        return (f"Model '{model_name}' is not yet implemented but the configuration"
                " exists. The model should be removed from the configuration.")

    def _model_format_error(model_input: str) -> str:
        return (f"Model specification '{model_input}' is invalid. Use the format "
                "'ModelName:ModelSpec', e.g., 'OpenAI:gpt-3.5-turbo'")

    def _unsupported_model_error(model_name: str) -> str:
        return f"Unsupported model: {model_name}. Supported models: {', '.join(self.models)}"
    ######

    ######
    # Model preparation methods
    def _decompose_model(self, text: str) -> List[str]:
        try:
            text_decomposition = text.split(":")
        except Exception as e:
            raise ValueError(self._model_format_error(text)) from e

        if len(text_decomposition) != 2:
            raise ValueError(self._model_format_error(text))

        return text_decomposition
    
    def prepare_model(self, model: str):
        model_decomposition = self._decompose_model(model)

        # These are useful in several cases
        self.model_name = model_decomposition[0].lower()
        self.model_spec = model_decomposition[1]

        if self.model_name in self.models:
            if self.model_name == "openai":
                self.model = OpenAIResponsesModel(self.model_spec)
            else:
                raise NotImplementedError(self._unimplemented_model_error(self.model_name))
        else:
            raise ValueError(self._unsupported_model_error(self.model_name))
        
    def prepare_embedding_model(self, embedding_model: str, cache: bool = False):
        embedding_model_decomposition = self._decompose_model(embedding_model)

        self.embedding_model_name = embedding_model_decomposition[0].lower()
        self.embedding_model_spec = embedding_model_decomposition[1]

        if self.embedding_model_name in self.embedding_models:
            embedder = None
            if self.embedding_model_name == "openai":
                from openai import OpenAI
                from .core.embed.openai_embedder import OpenAIEmbedder
                embedder = OpenAIEmbedder(OpenAI(), self.embedding_model_spec)
            else:
                raise ValueError(self._unsupported_model_error(self.model))

            self.embedder = CachingEmbedder(embedder) if cache else embedder
        else:
            raise NotImplementedError(self._unsupported_model_error(self.embedding_model_name))
    ######

    def request_model_from_user(self):
        if self.models is None:
            raise NotImplementedError("No models available. Please check the configuration file.")
        
        print("Enter an available model name.")
        model_name = input("Model name: ").strip()

        if model_name.lower() not in [m.lower() for m in self.models]:
            print(f"Unsupported model: {model_name}.")
            print(f"Available models: {', '.join(self.models)}")
            print("Please check your input and try again.")

        # Incorporate the model into the pipeline
        self.prepare_model(model_name)

        return model_name

    def load_agents(self, agent_class: List | str = "all", cache: bool = True):
        agent_path = self.config["paths"]["agents"]
        print(f"Loading agents from: {agent_path}")

        if not isinstance(agent_class, List):
            if agent_class == "all":
                # Walk the top-level folders in the agents_path. The inner
                # loop is to get the list of folder names only, and the 
                # outer loop is to get a flattened list of those names.
                agent_classes = [b for b in [a[1] for a in os.walk(agent_path)][0]]
            else:
                agent_classes = [agent_class]

        # Standard format for XivMind is to have the agents have all
        # lowercase names.
        for i, agent_class in enumerate(agent_classes):
            agent_classes[i] = agent_class.lower()

        if self.agents is None:
            self.agents = {}
        if self.agent_links is None:
            self.agent_links = {}

        # Search the agents/summarizers directory for all yml files and
        # load them as agents.
        agent_paths = [f"{agent_path}/{a}/*.yaml" for a in agent_classes]
        for i, glob_path in enumerate(agent_paths):
            # top-level folders in agents_path need keys for caching
            if agent_classes[i] not in self.agents:
                self.agents[agent_classes[i]] = {}
            if agent_classes[i] not in self.agent_links:
                self.agent_links[agent_classes[i]] = {}

            for file_path in glob.glob(glob_path):
                # Get the agent model name to store in the internal
                # dictionary for lookup
                # Get the file name with extension
                file_name = os.path.basename(file_path)
                agent_name, ext = os.path.splitext(file_name)
  
                if cache and agent_name in self.agents[agent_classes[i]]:
                    print("Loading agent from cache:", agent_name)
                    continue

                if ext != ".yaml":
                    print(f"Skipping non-yaml file: {file_path}")
                    continue

                print(f"Loading {agent_classes[i]} agent: {agent_name}")

                agent_config = load_config(open(file_path, "r"))
                
                # TODO: Configure the agent parameters
                self.agents[agent_classes[i]][agent_name] = Agent(
                    self.model_name + ":" + self.model_spec,
                    # NOTE: Instructions are specific to each agent here and
                    # it is important that they are not shared across agents.
                    instructions=agent_config["instructions"]
                )

                self.agent_links[agent_classes[i]][agent_name] = agent_config["agent_link"]

    # TODO: Move to the summarizer class 
    async def summarize_text(self, text: List[str] | str, agent_class: str = "summarizer", 
                             agent_name: str = None,
                             max_concurrent: int = 5) -> str:
        if agent_class not in self.agents.keys():
            raise ValueError(f"Agent class '{agent_class}' not found. Available classes: {list(self.agents.keys())}")
        
        if agent_name is not None:
            if agent_name not in self.agents[agent_class].keys():
                raise ValueError(f"Agent name '{agent_name}' not found in class '{agent_class}'. Available agents: {list(self.agents[agent_class].keys())}")

            agent_names = [agent_name]
        else:
            agent_names = list(self.agents[agent_class].keys())

        # Always process input text as a list
        if isinstance(text, str):
            text = [text]

        # If no API limit is set, use the default from the config file. By
        # default, try to limit the number of max concurrent requests to 1/100
        # of the per-minute limit.
        if max_concurrent is None:
            model_name = self.model_name.lower()
            max_concurrent = int(self.config["API"][model_name]["max_concurrent_requests"])
            if max_concurrent == 0:
                max_concurrent = 1

        semaphore = asyncio.Semaphore(max_concurrent)
        summaries = {name: [""] * len(text) for name in agent_names}

        async def run_agent(agent_name: str, model_input: str, idx: int):
            async with semaphore:
                try:
                    result = await self.agents[agent_class][agent_name].run(model_input)
                    result = result.output
                except Exception as e:
                    result = f"XIVMIND_ERROR: '{agent_name}': {e}"
                
                return agent_name, result, idx
            
        tasks = [run_agent(agent_name, model_input, idx) 
                 for agent_name in agent_names 
                 for idx, model_input in enumerate(text)]

        model_outputs = await asyncio.gather(*tasks)

        for agent_name, model_output, idx in model_outputs:
            summaries[agent_name][idx] = model_output

        return summaries

    ######

    def retrieve_top_k_keys(self, query_embedding, top_k: int = 5):
        faiss_index = FAISSIndexManager()
        keys, index = faiss_index.load_faiss_index()

        # Perform the search
        _, I = index.search(query_embedding, top_k)

        return [keys[i] for i in I[0]]
    
    def retrieve_relevant_papers(self, query_embedding: List[float], agent_name, top_k: int = 5):
        query_embedding = np.array(query_embedding).astype(float).reshape(1, -1)

        top_keys = self.retrieve_top_k_keys(query_embedding, top_k)

        arxiv_ids = []
        # Extract arXiv IDs from the keys
        for key in top_keys:
            _, _, _, arxiv_id, _ = Cache.parse_key(key)
            arxiv_ids.append(arxiv_id)

        # TODO: This is a temporary solution to load the metadata. In the future,
        # need a central database of the papers to make this quicker.
        from .arxiv.metadatamanager import MetaDataManager
        from .backend import DataPipeline
        data_pipeline = DataPipeline(MetaDataManager("arxiv-metadata-oai-snapshot.json"))

        return data_pipeline.load_metadata(fields=["title", "abstract"],
                                           ids=arxiv_ids)

    async def query_and_respond(self):
        query = input("Enter a query:")
        
        # This is a sync call, since it is only a single text input. It validates
        # the input text automatically.
        query_embedding = await self.embedder.embed_text(query)

        # TODO: Make this optional
        agent_name = "PostSecondaryAssistant"
        papers = self.retrieve_relevant_papers(query_embedding, agent_name)

        # Augment the query with the relevant papers
        model_input = f"Question: {query}\n\n"
        model_input += "Context:\n"
        for arxiv_id in papers:
            model_input += f"arXiv:{arxiv_id}\nAbstract: {papers[arxiv_id]['abstract']}\n\n"

        response = await self.agents["researchassistant"][agent_name].run(model_input)

        print("\nAnswer:\n", response.output)
        print("\nSources:\n")
        for arxiv_id in papers:
            print(f"- arXiv:{arxiv_id} Title: {papers[arxiv_id]['title']}")
        print()


        




