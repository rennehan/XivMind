from typing import List, Dict
import datetime as dt
from collections import defaultdict
import json
import pickle
import yaml

class MetaDataManager:
    config = None
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.date_format = "%Y-%m-%d"
        self.index = {}
        self.config = yaml.load(open("./XivMind/configs/config.yaml", "r"), 
                                Loader=yaml.FullLoader)
    
    def _get_metadata_file_path(self, file_name: str) -> str:
        return f"{self.config["paths"]["arxiv"]["metadata"]}/{file_name}"
    
    def _get_metadata_index_path(self, file_name: str) -> str:
        return f"{self.config["paths"]["arxiv"]["metadata"]}/{file_name}"

    
    def _unsupported_field_error(self, field: str) -> str:      
        return f"Unsupported field: {field}. Supported fields are: {', '.join(self.config["fields"]["metadata"])}"
    
    def get_config(self) -> Dict:
        if self.config is None:
            raise AttributeError("Configuration not loaded.")
        
        return self.config
    
    def load_metadata(self, fields: List[str] = [], limit: int = None):
        if len(fields) == 0:
            fields = self.config["fields"]["metadata"]
            print("Assuming all fields are requested:")
            print(", ".join(fields))
        else:
            for field in fields:
                if field not in self.config["fields"]["metadata"]:
                    raise ValueError(self._unsupported_field_error(field))
        
        count = 0
        papers = defaultdict(lambda: defaultdict)

        with open(self._get_metadata_file_path(self.file_name), "r") as f:
            for line in f:
                try:
                    paper = json.loads(line)
                    paper_data = {field: paper[field] for field in fields if field != "id"}
                    papers[paper["id"]] = paper_data

                    count += 1
                    if limit and count >= limit:
                        break
                except json.JSONDecodeError:
                    continue

        print(f"Read {count} entries and the following fields:")
        print(", ".join(fields))

        return papers
    
    def build_metadata_index(self, cache: bool = True, limit: int = None) -> defaultdict:
        print("Building metadata index.")
        index = {}
        count = 0
        with open(self._get_metadata_file_path(self.file_name), "r") as f:
            for line in f:
                try:
                  paper = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Only keeping track of the primary category for now
                # to reduce complexity of the system. 
                # TODO: Consider multi-category indexing.
                id = paper["id"]
                category = paper["categories"].split()[0]
                date = paper["update_date"]
          
                if id and category and date:
                    if category not in index:
                        index[category] = {}
                    if date not in index[category]:
                        index[category][date] = []

                    index[category][date].append(id)
                
                count += 1
                if limit and count >= limit:
                    break

        if cache:
            cache_file_name = self.config["files"]["arxiv"]["index"]
            cache_file = self._get_metadata_index_path(cache_file_name)
            with open(cache_file, "wb") as f:
                pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Metadata index cached at {cache_file}")

        return index

    def load_metadata_index(self, cache: bool = True, limit: int = None) -> defaultdict:
        if cache:
            cache_file_name = self.config["files"]["arxiv"]["index"]
            cache_file = self._get_metadata_index_path(cache_file_name)
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except FileNotFoundError:
                yn = input("No cached index found at {cache_file}. Do you want to build a new one? (y/n)")
                if yn.lower() != "y":
                    print("Exiting without building a new index.")
                    exit(1)
                
                return self.build_metadata_index(cache=True, limit=limit)
        else:
            yn = input("Building a new index is expensive. Do you want to proceed? (y/n)")
            if yn.lower() != "y":
                print("Exiting without building a new index.")
                exit(1)

            return self.build_metadata_index(cache=True, limit=limit)
        
    def filter(self, categories: List[str] | str = None, date_range: List[dt.date] | dt.date = None) -> list:
        if isinstance(categories, str):
            categories = [categories]

        for category in categories:
            if category not in self.index.keys():
                raise ValueError(f"Category {category} not found in index.")
            
        if isinstance(date_range, dt.date):
            # Start and end dates are the same
            date_range = [date_range, date_range]

        if len(date_range) > 2:
            raise ValueError("Dates should be a single date or a list of two dates (start and end).")
        
        for date in date_range:
            if not isinstance(date, dt.dates):
                raise ValueError(f"Date {date} is not a valid date object.")
            
        results = {}
        total_results = 0
        for category in categories:
            category_dates = list(self.index[category].keys())
            date_objects = [dt.strptime(date_str, self.date_format).date() for date_str in category_dates]

            for i, date_object in enumerate(date_objects):
                if date_range[0] <= date_object <= date_range[1]:
                    if category not in results:
                        results[category] = {}
                    results[category][category_dates[i]] = self.index[category][category_dates[i]]
                    total_results += 1

        print(f"Total results found: {total_results}")

        return results
    



        
