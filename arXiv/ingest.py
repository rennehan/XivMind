# ingestion.py

import arxiv
import json
from typing import List, Dict

def fetch_arxiv_papers(query: str = "astro-ph", max_results: int = 100) -> List[Dict]:
    """
    Fetches papers from the arXiv API based on a category or keyword query.
    Returns a list of dictionaries containing metadata.
    """
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers = []
    for result in search.results():
        paper = {
            "title": result.title.strip(),
            "authors": [author.name for author in result.authors],
            "summary": result.summary.strip().replace("\n", " "),
            "published": result.published.isoformat(),
            "updated": result.updated.isoformat(),
            "pdf_url": result.pdf_url,
            "entry_id": result.entry_id
        }
        papers.append(paper)
    return papers

def save_papers_to_json(papers: List[Dict], filepath: str = "data/arxiv_astro_ph.json"):
    import os
    os.makedirs("data", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(papers, f, indent=2)

if __name__ == "__main__":
    print("Fetching papers from arXiv...")
    papers = fetch_arxiv_papers("au:rennehan", 100)
    print(f"Fetched {len(papers)} papers.")
    save_papers_to_json(papers)
    print("Saved to data/arxiv_astro_ph.json")