from openai import OpenAI
import yaml
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
import os
from typing import List, Dict
import glob

class Pipeline:
    models = ["openai"]
    agents = None

    def __init__(self, config: Dict):
        self.config = config

        # These are useful in several cases
        self.model_name = config["model_name"].lower()
        self.model_spec = config["model_spec"]

        if self.model_name == "openai":
            self.model = OpenAIResponsesModel(config["model_spec"])
        else:
            model_name = config["model_name"]
            raise ValueError(f"Unsupported model: {model_name}. Supported models: {', '.join(self.models)}")

    def load_agents(self, agent_class: List | str = "all", cache: bool = True):
        agent_path = self.config["agents_path"]
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
        pass
    
    def summarize(self):
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

        




