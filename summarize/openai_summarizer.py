import openai
import json
import time
from tqdm import tqdm
import os
from summarize.summarizer import Summarizer
from openai import OpenAI

class OpenAISummarizer(Summarizer):
    def __init__(self, client: OpenAI = None,
                 model: str = "gpt-3.5-turbo"):
        self.model = model
        self.client = client or OpenAI()

    def summarize(self, abstract: str) -> str:
        return self.summarize_abstract(abstract, model=self.model)
    
    def load_papers(self, path):
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        else:
            return []

    def save_papers(self, papers, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(papers, f, indent=2)

    def summarize_abstract(self, abstract: str) -> str:
        system_prompt = (
            "You are a science communicator. Rephrase the following "
            "scientific abstract in plain English for a general, "
            "curious audience."
        )
        
        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", 
                     "content": system_prompt},
                    {"role": "user", 
                     "content": abstract}
                ],
                temperature=0.5
            )
            return response.output_text.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return ""

    def summarize_all(self, input_path: str, output_path: str):
        input_papers = self.load_papers(input_path)
        output_papers = self.load_papers(output_path)

        # Build cache map from output file
        cache = {paper["entry_id"]: paper for paper in output_papers if "summary_gpt" in paper}

        updated_papers = []
        for paper in tqdm(input_papers, desc="Summarizing"):
            if paper["entry_id"] in cache:
                # Reuse cached summary
                paper["summary_gpt"] = cache[paper["entry_id"]]["summary_gpt"]
            else:
                paper["summary_gpt"] = self.summarize_abstract(paper["summary"])
                time.sleep(1)  # Rate limit safety
            updated_papers.append(paper)

        self.save_papers(updated_papers, output_path)
        print(f"Saved summarized papers to {output_path}")
