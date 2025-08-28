from __future__ import annotations
import argparse, sys
from app.core import RAGPipeline, RAGConfig

def build_index(pipeline: RAGPipeline):
    pipeline.ensure_index()
    print("Index ready.")

def ask(pipeline: RAGPipeline, question: str):
    result = pipeline.answer(question)
    print(result.answer)
    if result.sources:
        print("\nSources:")
        for s in result.sources:
            print(f"- {s.source_name}")

def main():
    parser = argparse.ArgumentParser(description="RAG PDF Chat CLI")
    parser.add_argument("command", choices=["build", "ask"], help="Command to run")
    parser.add_argument("question", nargs="?", help="Question for 'ask'")
    args = parser.parse_args()

    cfg = RAGConfig.from_env()
    pipeline = RAGPipeline(cfg)

    if args.command == "build":
        build_index(pipeline)
    elif args.command == "ask":
        if not args.question:
            print("Please provide a question.")
            sys.exit(1)
        ask(pipeline, args.question)

if __name__ == "__main__":  # pragma: no cover
    main()
