from openai import OpenAI
import yaml
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
import os
from typing import List, Dict
import glob
from .embed.embedder import Embedder
from .embed.openai_embedder import OpenAIEmbedder
import asyncio

class Pipeline:
    model_name = None
    model_spec = None
    models = None
    config = None
    agents = None
    summarizer = None

    def __init__(self, model: str = None):
        # Load main paths for the pipeline
        self.config = yaml.load(open("./XivMind/configs/config.yaml", "r"), 
                                Loader=yaml.FullLoader)
        self.models = self.config["supported_models"]

        # Allow Pipeline tests with a direct input.
        if model is None:
            return
        else:
            self.prepare_model(model)

    def prepare_model(self, model: str):
        try:
            model_decomposition = model.split(":")
        except Exception as e:
            raise ValueError(self._model_format_error(model)) from e

        if len(model_decomposition) != 2:
            raise ValueError(self._model_format_error(model))

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

        # Search the agents/summarizers directory for all yml files and
        # load them as agents.
        agent_paths = [f"{agent_path}/{a}/*.yaml" for a in agent_classes]
        for i, glob_path in enumerate(agent_paths):
            # top-level folders in agents_path need keys for caching
            if agent_classes[i] not in self.agents:
                self.agents[agent_classes[i]] = {}

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

                agent_config = yaml.load(open(file_path, "r"), 
                                        Loader=yaml.FullLoader)
                
                # TODO: Configure the agent parameters
                self.agents[agent_classes[i]][agent_name] = Agent(
                    self.model_name + ":" + self.model_spec,
                    # NOTE: Instructions are specific to each agent here and
                    # it is important that they are not shared across agents.
                    instructions=agent_config["instructions"]
                )
    
    def _unimplemented_model_error(model_name) -> str:
        return (f"Model '{model_name}' is not yet implemented but the configuration"
                " exists. The model should be removed from the configuration.")

    def _model_format_error(self, model_input: str) -> str:
        return (f"Model specification '{model_input}' is invalid. Use the format "
                "'ModelName:ModelSpec', e.g., 'OpenAI:gpt-3.5-turbo'")
    
    def _unsupported_model_error(self, model_name: str) -> str:
        return f"Unsupported model: {model_name}. Supported models: {', '.join(self.models)}"
    
    ######
    # Embedding methods
    def get_embedder(self) -> Embedder:        
        if self.embedder is not None:
            return self.embedder

        if self.model_name == "openai":
            from .embed.openai_embedder import OpenAIEmbedder
            self.embedder = OpenAIEmbedder(OpenAI(), self.model_spec)
        else:
            raise ValueError(self._unsupported_model_error(self.model))

        return self.embedder

    def embed(self, text: str):
        if self.embedder is None:
            self.get_embedder()
        
        self.embedder.embed_and_save(text)
    ######

    ######    
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

    ######
    # FAISS index methods
    # Build FAISS index
    def build_faiss_index(self):
        print("Loading embeddings...")
        papers, embeddings = faiss_index.load_embeddings("../data/arxiv_astro_ph_embedded.json")
        print(f"Loaded {len(embeddings)} embeddings.")

        print("Building FAISS index...")
        index = faiss_index.build_faiss_index(embeddings)

        print("Saving index...")
        faiss_index.save_faiss_index(index, "../data/faiss_index.index")
        print("Done. Index saved to ../data/faiss_index.index")
    ######

        




