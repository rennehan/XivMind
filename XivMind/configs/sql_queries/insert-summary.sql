INSERT OR REPLACE INTO summaries(
    summarizer_name, 
    full_model_name, 
    arxiv_id, 
    summary_text, 
    timestamp)
   VALUES (?, ?, ?, ?, datetime('now'));