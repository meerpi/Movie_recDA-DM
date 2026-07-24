#!/usr/bin/env python3
"""
interface/query.py — Step 7: CineVault CLI Search Interface

Command-line query interface for CineVault recommendations.
Fuses BM25, Genome HNSW, and Voyage-4-Large Dense HNSW using RRF.

Usage:
    .venv/bin/python interface/query.py "atmospheric slow burn Korean thriller"
    .venv/bin/python interface/query.py "existential sci-fi" --top-k 5
    .venv/bin/python interface/query.py --interactive
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nlp.hydrator import ResultHydrator
from nlp.reranker import CineVaultReranker
from nlp.retriever import CineVaultRetriever


def format_card_output(rank: int, item: dict) -> str:
    """Format a single hydrated movie result into a rich CLI card."""
    title   = item["title"]
    year    = f"({item['year']})" if item.get("year") else ""
    tier    = f"[{item['tier']}]"
    score   = item.get("rerank_score", item.get("rrf_score", 0.0))
    rating  = f"★ {item['avg_rating']:.2f}" if item.get("avg_rating") else "★ N/A"
    count   = f"({item['num_ratings']:,} ratings)" if item.get("num_ratings") else ""
    genres  = ", ".join(item.get("genres", []))

    lanes_hit = item.get("lanes", [])
    bm25_mark   = "BM25 ✓" if "bm25" in lanes_hit else "BM25 ✗"
    genome_mark = "Genome ✓" if "genome" in lanes_hit else "Genome ✗"
    dense_mark  = "Dense ✓" if "dense" in lanes_hit else "Dense ✗"
    lane_summary = f"{bm25_mark}  │  {genome_mark}  │  {dense_mark}"

    rerank_str = f" [CrossEncoder Score: {item['rerank_score']:.4f} | RRF Rank #{item['rrf_rank']}]" if "rerank_score" in item else ""

    lines = [
        f"#{rank:<2d} {title} {year:<6s} {tier:>10s}  score: {score:.5f}{rerank_str}",
        f"    Genres: {genres if genres else 'N/A'}",
        f"    {rating} {count:<16s} │ {lane_summary}",
    ]

    if "themes" in item and item["themes"]:
        lines.append(f"    Themes: {', '.join(item['themes'])}")
    if "tone" in item and item["tone"]:
        lines.append(f"    Tone  : {', '.join(item['tone'])}")
    if "pacing" in item and item["pacing"]:
        lines.append(f"    Pacing: {item['pacing']}")
    if "top_tags" in item and item["top_tags"]:
        lines.append(f"    Tags  : {', '.join(item['top_tags'][:6])}")
    elif "user_tags" in item and item["user_tags"]:
        lines.append(f"    Tags  : {', '.join(item['user_tags'][:6])}")
    if "comparable_films" in item and item["comparable_films"]:
        lines.append(f"    Similar to: {', '.join(item['comparable_films'][:4])}")

    return "\n".join(lines)


def run_query(
    retriever: CineVaultRetriever,
    hydrator: ResultHydrator,
    query_text: str,
    top_k: int = 10,
    json_output: bool = False,
    reranker: Optional[CineVaultReranker] = None,
    candidate_k: int = 50
):
    print(f"\nSearching CineVault for: '{query_text}' ...")
    print("━" * 65)
    t0 = time.time()

    # Stage 1: RRF Retrieval (pull candidate pool)
    fetch_k = candidate_k if reranker else top_k
    hits = retriever.search(query_text, top_k=fetch_k)
    hydrated = hydrator.hydrate(hits)

    # Stage 2: Cross-Encoder Reranking
    if reranker and hydrated:
        print(f"Reranking top {len(hydrated)} candidates with Cross-Encoder...")
        results = reranker.rerank(query_text, hydrated, top_k=top_k)
    else:
        results = hydrated[:top_k]

    elapsed = (time.time() - t0) * 1000

    if json_output:
        print(json.dumps(results, indent=2))
        return

    if not results:
        print("No movies found matching your query.")
        return

    for rank, item in enumerate(results, 1):
        print(format_card_output(rank, item))
        print("─" * 65)

    mode_str = "RRF + CrossEncoder Reranked" if reranker else "RRF Fused (BM25 + Genome + Voyage-4-Large)"
    print(f"\nRetrieved {len(results)} movies in {elapsed:.1f}ms ({mode_str})")



def main():
    parser = argparse.ArgumentParser(description="CineVault CLI Recommendation & Search Engine")
    parser.add_argument("query", nargs="*", help="Search query string (e.g. 'atmospheric slow burn Korean thriller')")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return (default: 10)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive search mode")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--qwen", action="store_true", help="Use 4-bit Qwen3-Reranker-4B model instead of BAAI")

    args = parser.parse_args()

    # Ensure API key is available or prompt
    if not os.environ.get("VOYAGE_API_KEY"):
        os.environ["VOYAGE_API_KEY"] = "pa-RKRgJB-gEjfRSG32QZ4MsGMtxaJN7P9g6SYUVOOnUev"

    retriever = CineVaultRetriever()
    hydrator  = ResultHydrator()

    model_label = "Qwen3-Reranker-4B (4-bit GGUF)" if args.qwen else "BAAI/bge-reranker-v2-m3"
    print(f"Initializing Stage 2 Cross-Encoder Reranker ({model_label})...")
    reranker = CineVaultReranker(use_qwen_4bit=args.qwen)


    if args.interactive:
        print("\n" + "=" * 65)
        print("  CINEVAULT INTERACTIVE MOVIE SEARCH")
        print("  Type your search query (or 'exit' / 'q' to quit)")
        print("=" * 65 + "\n")
        while True:
            try:
                user_q = input("\ncinevault> ").strip()
                if user_q and user_q.lower() not in ("exit", "q", "quit"):
                    run_query(retriever, hydrator, user_q, top_k=args.top_k, json_output=args.json, reranker=reranker)
                elif user_q.lower() in ("exit", "q", "quit"):
                    print("Goodbye!")
                    break
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
    elif args.query:
        query_text = " ".join(args.query)
        run_query(retriever, hydrator, query_text, top_k=args.top_k, json_output=args.json, reranker=reranker)
    else:
        parser.print_help()



if __name__ == "__main__":
    main()
