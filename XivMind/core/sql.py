from typing import Dict

class SQLQueryContainer:
    def __init__(self, config: Dict):
        self.sql_path = config["sql"]["main-path"]
        self.files = {}
        self.query_names = {}
        for key, value in config["sql"].items():
            if key != "main-path":
                self.query_names.update({key: None})
                self.files.update({key: config["sql"]["main-path"] + "/" + value})

        self.queries = {}

    def retrieve(self, query_name: str) -> str:
        if query_name not in self.query_names:
            raise ValueError(f"Query '{query_name}' not found in {self.sql_path}.")
        
        if query_name in self.queries:
            return self.queries[query_name]
        
        with open(self.files[query_name], "r") as f:
            query = f.read()
            self.queries.update({query_name: query})
            return query


        
