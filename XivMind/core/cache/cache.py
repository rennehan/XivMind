from typing import Dict

class Cache:
    def __init__(self, config: Dict):
        self.config = config

    @staticmethod
    def get_key(agent_name: str, model_name: str, model_spec: str,
                arxiv_id: str, label: str) -> str:
        return f"{agent_name}:{model_name}:{model_spec}:{arxiv_id}:{label}"
    
    @staticmethod
    def parse_key(key: str):
        parts = key.split(":")
        if len(parts) < 5:
            raise ValueError("Invalid cache key format.")
        
        agent_name = parts[0]
        model_name = parts[1]
        model_spec = parts[2]
        arxiv_id = parts[3]
        label = ":".join(parts[4:])

        return agent_name, model_name, model_spec, arxiv_id, label