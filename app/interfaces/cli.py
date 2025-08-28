from __future__ import annotations
import argparse, sys, json
from app.core import RAGPipeline, RAGConfig

def build_index(pipeline: RAGPipeline, force: bool = False):
    pipeline.ensure_index(force=force)
    print("Index ready (rebuild forced)." if force else "Index ready.")

def ask(pipeline: RAGPipeline, question: str):
    result = pipeline.answer(question)
    print(result.answer)
    if result.sources:
        print("\nSources:")
        for s in result.sources:
            print(f"- {s.source_name}")

def main():
    parser = argparse.ArgumentParser(description="RAG PDF Chat CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build index if needed (hash-based)")
    p_build.add_argument("--force", action="store_true", help="Force rebuild regardless of hash")

    p_ask = sub.add_parser("ask", help="Ask a question against the indexed PDFs")
    p_ask.add_argument("question", help="Question text")

    sub.add_parser("list-sources", help="List currently indexed source document names")
    sub.add_parser("rebuild", help="Alias for build --force (always rebuild)")
    sub.add_parser("reset", help="Delete vectorstore directory (clears index)")

    args = parser.parse_args()

    cfg = RAGConfig.from_env()
    pipeline = RAGPipeline(cfg)

    if args.command == "build":
        build_index(pipeline, force=getattr(args, 'force', False))
    elif args.command == "rebuild":
        build_index(pipeline, force=True)
    elif args.command == "ask":
        ask(pipeline, args.question)
    elif args.command == "list-sources":
        sources = pipeline.vs_manager.list_sources()
        print("\n".join(sources))
    elif args.command == "reset":
        pipeline.vs_manager.reset()
        print("Vectorstore reset:", pipeline.config.persist_dir)

if __name__ == "__main__":  # pragma: no cover
    main()
