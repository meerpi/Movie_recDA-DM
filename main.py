#!/usr/bin/env python3
"""
main.py — Single Unified Entry Point & Application Launcher for CineVault

Provides:
  1. Terminal Log Isolation: Redirects subsystem logs cleanly to `.tmp/cinevault.log`.
  2. Pre-flight Health & Hardware Diagnostics (`--check-health`).
  3. Interactive Cold-Start Onboarding Wizard (`--onboard`).
  4. Personalized CLI Query Search (`--query "..."`).
  5. Default TUI Application Launcher.
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

# Suppress Hugging Face Hub unauthenticated & parallelism warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── 1. Terminal Log Isolation Setup ──────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / ".tmp"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "cinevault.log"

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("cinevault.main")

from interface.controller import CineVaultController


def run_health_check():
    """
    Executes pre-flight hardware, database, and index diagnostics.
    """
    print("\n" + "=" * 70)
    print("  🎬 CINEVAULT PRE-FLIGHT SYSTEM & HARDWARE DIAGNOSTICS")
    print("=" * 70)

    # 1. Environment & Python
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"  ✓ Python Environment : Python {py_ver} ({sys.executable})")

    # 2. Textual Framework
    try:
        import textual
        print(f"  ✓ Textual Framework  : Textual v{textual.__version__} loaded")
    except ImportError:
        print("  ✗ Textual Framework  : NOT INSTALLED (Run: pip install textual)")

    # 3. PyTorch & GPU Hardware
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Only (CUDA unavailable)"
        print(f"  ✓ Deep Learning Engine: PyTorch v{torch.__version__}")
        print(f"  ✓ GPU Acceleration   : {'NVIDIA GPU (' + gpu_name + ')' if cuda_avail else gpu_name}")
    except ImportError:
        print("  ✗ Deep Learning Engine: PyTorch not found")

    # 4. Database Integrity
    db_path = PROJECT_ROOT / "db" / "cinevault.db"
    if db_path.exists():
        import sqlite3
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        movie_count = c.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
        stats_count = c.execute("SELECT COUNT(*) FROM movie_stats").fetchone()[0]
        rules_count = c.execute("SELECT COUNT(*) FROM association_rules").fetchone()[0]
        conn.close()
        print(f"  ✓ Database Integrity : SQLite ({movie_count:,} movies, {stats_count:,} stats, {rules_count:,} association rules)")
    else:
        print(f"  ✗ Database Integrity : Missing db file at {db_path}")

    # 5. Search Index Artifacts
    bm25_pth = PROJECT_ROOT / "nlp" / "bm25_index.pkl"
    genome_pth = PROJECT_ROOT / "nlp" / "genome.hnsw"
    dense_pth = PROJECT_ROOT / "nlp" / "dense.hnsw"

    print(f"  ✓ BM25 Keyword Index : {'Loaded (' + str(bm25_pth.name) + ')' if bm25_pth.exists() else 'Missing'}")
    print(f"  ✓ Genome Tag HNSW    : {'Loaded (' + str(genome_pth.name) + ')' if genome_pth.exists() else 'Missing'}")
    print(f"  ✓ Dense Text HNSW    : {'Loaded (' + str(dense_pth.name) + ')' if dense_pth.exists() else 'Disabled / Optional'}")

    # 6. API Key Readiness
    gemini_key = "SET ✓" if os.environ.get("GEMINI_API_KEY") else "UNSET (Using local RapidFuzz fallback)"
    voyage_key = "SET ✓" if (os.environ.get("VOYAGE_API_KEY") or (PROJECT_ROOT / ".env").exists()) else "UNSET"
    print(f"  ✓ Gemini API Key     : {gemini_key}")
    print(f"  ✓ Voyage API Key     : {voyage_key}")

    print("=" * 70 + "\n")


def run_cli_search(controller: CineVaultController, query: str, top_k: int = 10, json_output: bool = False):
    """
    Executes instant CLI personalized search.
    """
    print(f"\nSearching CineVault for user '{controller.user_id}': '{query}' ...")
    print("━" * 70)
    t0 = time.time()

    results = controller.search(query, top_k=top_k)
    elapsed = (time.time() - t0) * 1000

    if json_output:
        print(json.dumps(results, indent=2))
        return

    if not results:
        print("No recommendations found.")
        return

    for idx, item in enumerate(results, 1):
        final_rank = item.get("final_rank", idx)
        rrf_rank = item.get("rrf_rank")
        rrf_str = f" (RRF Pool Rank: #{rrf_rank})" if rrf_rank else ""
        title = item.get("title", "Unknown")
        year = f"({item['year']})" if item.get("year") else ""
        rating = f"★ {item['avg_rating']:.2f}" if item.get("avg_rating") else "★ Unrated"
        score = item.get("final_score", item.get("rerank_score", 0.0))
        genres = ", ".join(item.get("genres", []))

        print(f"#{final_rank:<2} {title} {year:<6s} | {rating} | Score: {score:.4f}{rrf_str}")
        print(f"    Genres: {genres}")
        if item.get("themes"):
            print(f"    Themes: {', '.join(item['themes'])}")
        if item.get("pacing"):
            print(f"    Pacing: {item['pacing']}")
        print("─" * 70)

    print(f"\nCompleted search in {elapsed:.1f}ms for user '{controller.user_id}'.")


def run_cold_start_cli(controller: CineVaultController):
    """
    Interactive CLI wizard for Cold-Start Onboarding.
    """
    print("\n" + "=" * 70)
    print("  🌟 CINEVAULT COLD-START ONBOARDING WIZARD")
    print("=" * 70)

    print("\n1. Enter 3 to 5 of your favorite genres (comma-separated):")
    print("   Available: Action, Adventure, Animation, Comedy, Crime, Drama, Fantasy, Horror, Mystery, Romance, Sci-Fi, Thriller, Western")
    raw_genres = input("   Genres > ").strip()
    fav_genres = [g.strip() for g in raw_genres.split(",") if g.strip()]

    print("\n2. Enter up to 3 anchor movie IDs or search titles (e.g. 58559, 79132, 1):")
    print("   (Toy Story = 1, The Dark Knight = 58559, Inception = 79132, The Matrix = 603)")
    raw_anchors = input("   Movie IDs > ").strip()
    anchor_ids = []
    for val in raw_anchors.split(","):
        if val.strip().isdigit():
            anchor_ids.append(int(val.strip()))

    print("\n3. Enter any dealbreaker rules (e.g. 'No Slapstick', 'No Gore'):")
    raw_dealbreakers = input("   Dealbreakers > ").strip()
    dealbreakers = [d.strip() for d in raw_dealbreakers.split(",") if d.strip()]

    controller.seed_cold_start(
        favorite_genres=fav_genres,
        anchor_movie_ids=anchor_ids,
        dealbreakers=dealbreakers
    )

    print(f"\n✓ Cold-start profile successfully created and saved for user '{controller.user_id}'!")
    print(f"  Genre Affinities: {controller.profile.genre_affinity}")
    print(f"  Tag Affinities  : {controller.profile.tag_affinity}\n")


def main():
    parser = argparse.ArgumentParser(description="CineVault Personalized Recommendation Engine")
    parser.add_argument("--query", "-q", type=str, help="Search query string")
    parser.add_argument("--user", "-u", type=str, default="default_user", help="Active user ID (default: default_user)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations to return (default: 10)")
    parser.add_argument("--onboard", action="store_true", help="Launch Cold-Start Onboarding Wizard")
    parser.add_argument("--check-health", action="store_true", help="Run system & hardware diagnostics")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    if args.check_health:
        run_health_check()
    elif args.onboard:
        ctrl = CineVaultController(user_id=args.user)
        run_cold_start_cli(ctrl)
    elif args.query:
        ctrl = CineVaultController(user_id=args.user)
        run_cli_search(ctrl, args.query, top_k=args.top_k, json_output=args.json)
    else:
        # Default: Launch Full Interactive Textual TUI Application
        from interface.tui.app import CineVaultApp
        ctrl = CineVaultController(user_id=args.user)
        app = CineVaultApp(user_id=args.user, controller=ctrl)
        app.run()


if __name__ == "__main__":
    main()
