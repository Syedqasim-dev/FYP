from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_module.rag import RAGPipeline


def build_index(args: argparse.Namespace) -> None:
    pipeline = RAGPipeline(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    total_chunks = pipeline.ingest_paths(args.inputs)
    pipeline.save_index(args.output)
    print(f"Indexed {total_chunks} chunks into {args.output}")


def ask_question(args: argparse.Namespace) -> None:
    pipeline = RAGPipeline()
    pipeline.load_index(args.index)
    response = pipeline.answer(args.query, top_k=args.top_k)

    if args.pretty:
        print(json.dumps(response, indent=2))
    else:
        print(response["answer"])


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple from-scratch RAG pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Create an index from files/directories")
    index_parser.add_argument("--inputs", nargs="+", required=True, help="Input files/directories to ingest")
    index_parser.add_argument("--output", default="rag_module/index.json", help="Output index file")
    index_parser.add_argument("--chunk-size", type=int, default=120, help="Chunk size in words")
    index_parser.add_argument("--chunk-overlap", type=int, default=30, help="Chunk overlap in words")
    index_parser.set_defaults(func=build_index)

    ask_parser = subparsers.add_parser("ask", help="Ask a question against an index")
    ask_parser.add_argument("--index", required=True, help="Path to saved index file")
    ask_parser.add_argument("--query", required=True, help="Question text")
    ask_parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    ask_parser.add_argument("--pretty", action="store_true", help="Print full JSON response")
    ask_parser.set_defaults(func=ask_question)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
