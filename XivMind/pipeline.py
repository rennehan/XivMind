from openai import OpenAI
import yaml
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
import os
from typing import List, Dict
import glob
from .core.openai_summarizer import OpenAISummarizer

class Pipeline:
    model_name = None
    model_spec = None
    config = None
    agents = None
    summarizer = None

    def __init__(self, model: str):
        try:
            model_decomposition = model.split(":")
        except Exception as e:
            raise ValueError(self._model_format_error(model)) from e

        if len(model_decomposition) != 2:
            raise ValueError(self._model_format_error(model))

        # These are useful in several cases
        self.model_name = model_decomposition[0].lower()
        self.model_spec = model_decomposition[1]

        # Load main paths for the pipeline
        self.config = yaml.load(open("./XivMind/configs/config.yaml", "r"), 
                                Loader=yaml.FullLoader)
        self.models = self.config["supported_models"]

        if self.model_name in self.models:
            if self.model_name == "openai":
                self.model = OpenAIResponsesModel(self.model_spec)
            else:
                raise NotImplementedError(self._unimplemented_model_error(self.model_name))
        else:
            raise ValueError(self._unsupported_model_error(self.model_name))

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
                    system_prompt=agent_config["instructions"]
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
    def get_embedder(self):        
        pass

    def embed(self):
        pass
    ######

    ######
    # Summarization methods
    def get_summarizer(self):
        if self.summarizer is not None:
            return self.summarizer
        
        if self.model_name == "openai":
            self.summarizer = OpenAISummarizer(OpenAI(), self.model_spec)
        else:
            raise ValueError(self._unsupported_model_error(self.model))

        return self.summarizer
    
    def summarize(self):
        self.summarizer.load_papers
        pass
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

        




