import json
import yaml
from collections import defaultdict
import pickle
from .arxiv.metadatamanager import MetaDataManager
from typing import Dict

class DataPipeline:
    def __init__(self, metadata_manager: MetaDataManager):
        self.data_manager = metadata_manager
        self.config = metadata_manager.get_config()
    
    def load_metadata(self, fields: list[str] = [], limit: int = None, ids: Dict = None) -> defaultdict:
        return self.data_manager.load_metadata(fields=fields, limit=limit)

    def build_metadata_index(self, cache: bool = True, limit: int = None) -> defaultdict:
        return self.data_manager.build_metadata_index(cache=cache, limit=limit)
    
    def load_metadata_index(self, cache: bool = True, limit: int = None) -> defaultdict:
        return self.data_manager.load_metadata_index(cache=cache, limit=limit)

  
