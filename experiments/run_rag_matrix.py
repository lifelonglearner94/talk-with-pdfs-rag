#!/usr/bin/env python
"""Matrix runner for a limited set (<=7) of RAG configuration variants.

Moved from former `run_rag_experiments.py.backup` and lightly renamed.
"""
from __future__ import annotations

import json, os, time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from app.core.config import RAGConfig
from app.core.rag_pipeline import RAGPipeline

EXPERIMENT_VARIANTS: List[Dict[str, Any]] = [
    {"label": "sim_small_chunks_k5", "retrieval_strategy": "similarity", "chunk_size": 800, "chunk_overlap": 120, "top_k": 5},
    {"label": "sim_default_baseline", "retrieval_strategy": "similarity", "chunk_size": 1500, "chunk_overlap": 200, "top_k": 10},
    {"label": "mmr_small_lambda0.3", "retrieval_strategy": "mmr", "chunk_size": 800, "chunk_overlap": 120, "top_k": 8, "mmr_lambda_mult": 0.3},
    {"label": "mmr_medium_lambda0.7", "retrieval_strategy": "mmr", "chunk_size": 1500, "chunk_overlap": 200, "top_k": 10, "mmr_lambda_mult": 0.7},
    {"label": "mmr_large_chunks", "retrieval_strategy": "mmr", "chunk_size": 2000, "chunk_overlap": 250, "top_k": 6, "mmr_lambda_mult": 0.5},
]

EVAL_QUESTION = (
    "Welche Vorteile bietet ein Kubernetes Cluster-Management gegenüber einer manuellen Orchestrierung von Containern, "
    "insbesondere im Hinblick auf Skalierung, Ausfallsicherheit, Scheduling und betrieblichen Aufwand?"
)


def build_config(base: RAGConfig, variant: Dict[str, Any], persist_root: Path, idx: int) -> RAGConfig:
    kwargs = base.model_dump()
    kwargs.update(variant)
    kwargs["persist_dir"] = persist_root / f"exp_{idx:02d}_{variant['label']}"
    return RAGConfig(**kwargs)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def serialize_answer_result(answer_result) -> Dict[str, Any]:
    if answer_result is None:
        return {}
    return {
        "answer": answer_result.answer,
        "sources": [asdict(s) for s in answer_result.sources],
        "raw_chunks": [
            {"order": i, "text": r.text, "metadata": asdict(r.metadata), "char_len": len(r.text)}
            for i, r in enumerate(answer_result.raw_chunks)
        ],
    }


def main():
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    root = Path("experiments") / f"rag_eval_{timestamp}"
    persist_root = root / "vectorstores"
    results_dir = root / "results"
    summaries_dir = root / "summaries"
    for d in (persist_root, results_dir, summaries_dir):
        ensure_dir(d)

    base_config = RAGConfig.from_env()
    no_llm = os.getenv("NO_LLM") == "1"
    aggregate_summary = []

    for idx, variant in enumerate(EXPERIMENT_VARIANTS, start=1):
        label = variant["label"]
        print(f"\n=== Experiment {idx}: {label} ===")
        cfg = build_config(base_config, variant, persist_root, idx)
        pipeline = RAGPipeline(cfg)
        t0 = time.time()
        pipeline.ensure_index()
        t_index = time.time() - t0
        t1 = time.time()
        if no_llm:
            if not pipeline._retriever:
                pass
            docs = pipeline._retriever.invoke(EVAL_QUESTION)
            from app.core.models import AnswerResult, RetrievalResult, ChunkMetadata
            retrieval_results = []
            sources_seen = {}
            sources_list = []
            for i, d in enumerate(docs):
                md = ChunkMetadata(
                    source_name=d.metadata.get("source_name", "Unknown"),
                    chunk_id=d.metadata.get("chunk_id", ""),
                    page=d.metadata.get("page"),
                )
                retrieval_results.append(RetrievalResult(text=d.page_content, metadata=md))
                if md.source_name not in sources_seen:
                    sources_seen[md.source_name] = True
                    sources_list.append(md)
            answer_result = AnswerResult(answer="", sources=sources_list, raw_chunks=retrieval_results)
        else:
            answer_result = pipeline.answer(EVAL_QUESTION)
        t_qa = time.time() - t1
        total_time = time.time() - t0

        answer_data = serialize_answer_result(answer_result)
        config_json = json.loads(cfg.to_json())
        result_record = {
            "experiment_index": idx,
            "label": label,
            "question": EVAL_QUESTION,
            "config": config_json,
            "timings_sec": {"index_build": round(t_index, 3), "qa": round(t_qa, 3), "total": round(total_time, 3)},
            "no_llm": no_llm,
            "retrieval_strategy": cfg.retrieval_strategy,
            "top_k": cfg.top_k,
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "mmr_lambda_mult": cfg.mmr_lambda_mult,
            "mmr_fetch_k_factor": cfg.mmr_fetch_k_factor,
            "mmr_min_fetch_k": cfg.mmr_min_fetch_k,
            "result": answer_data,
            "metrics": {
                "n_sources": len(answer_data.get("sources", [])),
                "n_retrieved_chunks": len(answer_data.get("raw_chunks", [])),
                "avg_chunk_chars": round(
                    sum(rc["char_len"] for rc in answer_data.get("raw_chunks", [])) / max(1, len(answer_data.get("raw_chunks", []))), 1
                ),
                "answer_chars": len(answer_data.get("answer", "")),
            },
        }
        (results_dir / f"exp_{idx:02d}_{label}.json").write_text(json.dumps(result_record, ensure_ascii=False, indent=2))
        md_lines = [
            f"# Experiment {idx}: {label}",
            "",
            "## Settings",
            f"- retrieval_strategy: {cfg.retrieval_strategy}",
            f"- chunk_size / overlap: {cfg.chunk_size} / {cfg.chunk_overlap}",
            f"- top_k: {cfg.top_k}",
            f"- mmr_lambda_mult: {cfg.mmr_lambda_mult}",
            f"- mmr_fetch_k_factor: {cfg.mmr_fetch_k_factor}",
            f"- mmr_min_fetch_k: {cfg.mmr_min_fetch_k}",
            "",
            "## Timings (s)",
            f"- index_build: {result_record['timings_sec']['index_build']}",
            f"- qa: {result_record['timings_sec']['qa']}",
            f"- total: {result_record['timings_sec']['total']}",
            "",
            "## Retrieval Metrics",
            f"- sources: {result_record['metrics']['n_sources']}",
            f"- retrieved_chunks: {result_record['metrics']['n_retrieved_chunks']}",
            f"- avg_chunk_chars: {result_record['metrics']['avg_chunk_chars']}",
            f"- answer_chars: {result_record['metrics']['answer_chars']}",
        ]
        if not no_llm and answer_data.get("answer"):
            md_lines += ["", "## Answer (truncated)", answer_data["answer"][:1200] + ("…" if len(answer_data["answer"]) > 1200 else "")]
        md_lines += ["", "## First 2 retrieved chunk previews"]
        for rc in answer_data.get("raw_chunks", [])[:2]:
            preview = rc["text"][:500].replace("\n", " ")
            md_lines.append(f"- {rc['metadata']['source_name']} | len={rc['char_len']} | preview: {preview}…")
        (summaries_dir / f"exp_{idx:02d}_{label}.md").write_text("\n".join(md_lines))
        aggregate_summary.append({
            "idx": idx,
            "label": label,
            "strategy": cfg.retrieval_strategy,
            "chunk_size": cfg.chunk_size,
            "top_k": cfg.top_k,
            "n_chunks": result_record['metrics']['n_retrieved_chunks'],
            "avg_chunk_chars": result_record['metrics']['avg_chunk_chars'],
            "answer_chars": result_record['metrics']['answer_chars'],
            "index_build_s": result_record['timings_sec']['index_build'],
            "qa_s": result_record['timings_sec']['qa'],
        })

    overview_path = root / "overview.json"
    overview_path.write_text(json.dumps({"question": EVAL_QUESTION, "no_llm": no_llm, "experiments": aggregate_summary}, ensure_ascii=False, indent=2))
    print(f"\nAll experiments completed. Results directory: {root}")
    print("Overview written to", overview_path)


if __name__ == "__main__":  # pragma: no cover
    main()
