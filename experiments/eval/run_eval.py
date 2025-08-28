#!/usr/bin/env python
"""Evaluation harness (Phase 1 -> Phase 2 incremental).

Metrics:
- retrieval_recall_at_k: fraction of gold source papers (by filename prefix match) retrieved within top_k
- answer_keyword_coverage: fraction of expected_keywords present (case-insensitive substring) in answer
- mrr@k: reciprocal rank of first gold source among retrieved chunks (averaged)
- map@k: mean average precision over gold source matches within top_k

Optional (--compare-rerank): runs twice (baseline then rerank) and reports MRR delta.

Usage:
    uv run python experiments/eval/run_eval.py --gold experiments/eval/gold_examples.jsonl \
        --k 8 --mode vector

Environment overrides (same as RAGConfig) respected.
Outputs metrics JSON to experiments/eval/results/{timestamp}.json
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from app.core.rag_pipeline import RAGPipeline
from app.core.config import RAGConfig

@dataclass
class Example:
    id: str
    question: str
    expected_keywords: List[str]
    source_papers: List[str]

def load_examples(path: Path) -> List[Example]:
    examples: List[Example] = []
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            examples.append(Example(**obj))
    return examples

def retrieval_recall(retrieved: List[str], gold: List[str]) -> float:
    if not gold:
        return 0.0
    hits = 0
    for g in gold:
        # match by starting substring of source chunk (normalized)
        g_norm = g.lower().split('.pdf')[0]
        if any(r.lower().startswith(g_norm[:40]) for r in retrieved):
            hits += 1
    return hits / len(gold)

def keyword_coverage(answer: str, expected: List[str]) -> float:
    if not expected:
        return 0.0
    text = answer.lower()
    hits = sum(1 for k in expected if k.lower() in text)
    return hits / len(expected)

def mrr_at_k(retrieved: List[str], gold: List[str]) -> float:
    """Return reciprocal rank of first gold source among retrieved sources (chunk-level order)."""
    if not gold:
        return 0.0
    # normalize gold prefixes
    gold_norm = [g.lower().split('.pdf')[0] for g in gold]
    for idx, r in enumerate(retrieved, start=1):
        r_norm = r.lower()
        if any(r_norm.startswith(g[:40]) for g in gold_norm):
            return 1.0 / idx
    return 0.0

def average_precision_at_k(retrieved: List[str], gold: List[str]) -> float:
    if not gold:
        return 0.0
    gold_norm = [g.lower().split('.pdf')[0] for g in gold]
    hits = 0
    precisions: List[float] = []
    for idx, r in enumerate(retrieved, start=1):
        r_norm = r.lower()
        if any(r_norm.startswith(g[:40]) for g in gold_norm):
            hits += 1
            precisions.append(hits / idx)
    if not precisions:
        return 0.0
    return sum(precisions) / len(gold)

def run_single_eval(pipe: RAGPipeline, examples: List[Example]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows = []
    agg = {"retrieval_recall@k": [], "answer_keyword_coverage": [], "mrr@k": [], "map@k": []}
    for ex in examples:
        res = pipe.answer(ex.question)
        retrieved_sources = [m.metadata.source_name for m in res.raw_chunks]
        recall = retrieval_recall(retrieved_sources, ex.source_papers)
        cov = keyword_coverage(res.answer, ex.expected_keywords)
        mrr = mrr_at_k(retrieved_sources, ex.source_papers)
        ap = average_precision_at_k(retrieved_sources, ex.source_papers)
        rows.append({
            "id": ex.id,
            "question": ex.question,
            "recall_at_k": recall,
            "keyword_coverage": cov,
            "mrr": mrr,
            "ap": ap,
            "retrieved_sources": retrieved_sources,
            "answer": res.answer[:400],
        })
        agg["retrieval_recall@k"].append(recall)
        agg["answer_keyword_coverage"].append(cov)
        agg["mrr@k"].append(mrr)
        agg["map@k"].append(ap)
    summary = {k: (sum(v)/len(v) if v else 0.0) for k, v in agg.items()}
    return summary, rows

def run_eval(gold_path: Path, k: int, mode: str, compare_rerank: bool) -> Dict[str, Any]:
    cfg = RAGConfig.from_env()
    cfg.top_k = k
    cfg.retrieval_mode = mode  # type: ignore
    examples = load_examples(gold_path)
    pipe = RAGPipeline(cfg)
    pipe.ensure_index()
    summary_base, rows_base = run_single_eval(pipe, examples)
    result: Dict[str, Any] = {
        "summary": summary_base,
        "details": rows_base,
        "k": k,
        "mode": mode,
        "rerank_compared": False,
    }
    if compare_rerank:
        # Enable rerank in-place (retriever rebuild only)
        pipe.update_settings(rerank_enable=True)
        summary_rerank, rows_rerank = run_single_eval(pipe, examples)
        # Compute deltas
        result["rerank_compared"] = True
        result["summary_rerank"] = summary_rerank
        result["summary"]["mrr@k_rerank_delta"] = summary_rerank["mrr@k"] - summary_base["mrr@k"]
        result["summary"]["map@k_rerank_delta"] = summary_rerank.get("map@k", 0.0) - summary_base.get("map@k", 0.0)
        # Attach per-example mrr ranks
        for base_row, rr_row in zip(rows_base, rows_rerank):
            base_row["mrr_rerank"] = rr_row["mrr"]
            base_row["mrr_delta"] = rr_row["mrr"] - base_row["mrr"]
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--mode", choices=["vector", "keyword", "hybrid"], default="vector")
    ap.add_argument("--compare-rerank", action="store_true", help="Also evaluate rerank-enabled retriever and report MRR delta")
    args = ap.parse_args()
    start = time.time()
    results = run_eval(args.gold, args.k, args.mode, args.compare_rerank)
    results["elapsed_sec"] = round(time.time() - start, 2)
    out_dir = Path("experiments/eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_file = out_dir / f"eval_{args.mode}_k{args.k}_{ts}.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_file} -> summary: {results['summary']}")

if __name__ == "__main__":
    main()
