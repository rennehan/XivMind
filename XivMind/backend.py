import json
import yaml
from collections import defaultdict
import pickle
from .arxiv.metadatamanager import MetaDataManager

class DataPipeline:
    def __init__(self, metadata_manager: MetaDataManager):
        self.metadata_manager = metadata_manager
        self.config = metadata_manager.get_config()
    
    def load_metadata(self, fields: list[str] = [], limit: int = None):
        return self.metadata_manager.load_metadata(fields=fields, limit=limit)

    def build_metadata_index(self, cache: bool = True, limit: int = None) -> defaultdict:
        return self.metadata_manager.build_metadata_index(cache=cache, limit=limit)
    
    def load_metadata_index(self, cache: bool = True, limit: int = None) -> defaultdict:
        return self.metadata_manager.load_metadata_index(cache=cache, limit=limit)

  
