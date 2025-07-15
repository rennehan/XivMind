SELECT summarizer_name, full_model_name, arxiv_id, summary_text from summaries
WHERE summarizer_name = ?;