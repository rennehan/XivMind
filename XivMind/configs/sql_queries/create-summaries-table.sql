CREATE TABLE IF NOT EXISTS summaries(
  summarizer_name TEXT,
  full_model_name TEXT,
  arxiv_id TEXT,
  summary_text TEXT,
  timestamp DATETIME,
  PRIMARY KEY (summarizer_name, full_model_name, arxiv_id)
);