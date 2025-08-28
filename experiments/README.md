# Experiments

This folder hosts scripts & outputs for evaluating retrieval / generation settings.

## Files
* `run_rag_matrix.py` – matrix runner for a small set (<=7) of config variants.
* `legacy_retriever_test.py` – legacy exploratory retriever quality script (kept for reference).

## Running the Matrix
Set any desired env overrides (or edit the variants list) then:
```
uv run python experiments/run_rag_matrix.py
```
Skip LLM calls (retrieval-only) via:
```
NO_LLM=1 uv run python experiments/run_rag_matrix.py
```

Outputs appear under a timestamped directory: `experiments/rag_eval_YYYYMMDD-HHMMSS/` containing:
* `vectorstores/` per-variant persisted stores
* `results/` detailed JSON per variant
* `summaries/` concise Markdown summaries
* `overview.json` aggregate overview

## TODO
Future: integrate automated metrics & comparative charts.
