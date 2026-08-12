#!/usr/bin/env python3
"""main.py — CineVault entry point: TUI launcher, CLI search, health check, onboarding."""

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time
import warnings
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")


def _fix_cell_size_for_fractional_scaling():
    """Compensate for Wayland fractional scaling (e.g. Hyprland 1.5×).

    textual-image uses int() truncation for cell-size detection from TIOCGWINSZ,
    which produces wrong values at non-integer scale factors. We pre-seed the
    TEXTUAL_CELL_WIDTH/HEIGHT env vars with round() values so textual-image's
    fallback path picks them up before caching a bad result.
    """
    import struct, fcntl, termios
    try:
        if not sys.__stdout__ or not sys.__stdout__.isatty():
            return
        buf = fcntl.ioctl(sys.__stdout__, termios.TIOCGWINSZ, b'\x00' * 8)
        rows, cols, xpix, ypix = struct.unpack('HHHH', buf)
        if rows > 0 and cols > 0 and xpix > 0 and ypix > 0:
            cell_w = round(xpix / cols)
            cell_h = round(ypix / rows)
            os.environ.setdefault("TEXTUAL_CELL_WIDTH", str(cell_w))
            os.environ.setdefault("TEXTUAL_CELL_HEIGHT", str(cell_h))
    except Exception:
        pass  # Not a TTY or ioctl failed — skip

    # Clear any cached cell size so textual-image re-reads from env vars
    try:
        from textual_image._terminal import get_cell_size
        if hasattr(get_cell_size, "_result"):
            delattr(get_cell_size, "_result")
    except Exception:
        pass

_fix_cell_size_for_fractional_scaling()

# Pre-import textual-image to trigger terminal protocol detection (TGP/Sixel)
# BEFORE Textual starts.  The cell size must already be seeded (above) or
# available via TEXTUAL_CELL_WIDTH/HEIGHT env vars (set by serve_web.sh for SSH).
# If this import happens after Textual owns stdin, the escape-sequence probe
# times out and falls back to blocky halfcell rendering.
try:
    import textual_image.widget  # noqa: F401
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

LOG_DIR  = PROJECT_ROOT / ".tmp"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "cinevault.log"

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("cinevault.main")

from interface.controller import CineVaultController
from user_profile.identity import validate_user_id, lookup_anchor_by_title, resolve_anchor_tokens


def run_health_check():
    print("\n" + "=" * 70)
    print("  🎬 CINEVAULT PRE-FLIGHT SYSTEM & HARDWARE DIAGNOSTICS")
    print("=" * 70)

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"  ✓ Python Environment : Python {py_ver} ({sys.executable})")

    try:
        import textual
        print(f"  ✓ Textual Framework  : Textual v{textual.__version__} loaded")
    except ImportError:
        print("  ✗ Textual Framework  : NOT INSTALLED (Run: pip install textual)")

    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Only (CUDA unavailable)"
        print(f"  ✓ Deep Learning Engine: PyTorch v{torch.__version__}")
        print(f"  ✓ GPU Acceleration   : {'NVIDIA GPU (' + gpu_name + ')' if cuda_avail else gpu_name}")
    except ImportError:
        print("  ✗ Deep Learning Engine: PyTorch not found")

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

    bm25_pth = PROJECT_ROOT / "dirtywork" / "bm25_index.pkl"
    genome_pth = PROJECT_ROOT / "dirtywork" / "genome.hnsw"
    dense_pth = PROJECT_ROOT / "dirtywork" / "dense.hnsw"

    print(f"  ✓ BM25 Keyword Index : {'Loaded (' + str(bm25_pth.name) + ')' if bm25_pth.exists() else 'Missing'}")
    print(f"  ✓ Genome Tag HNSW    : {'Loaded (' + str(genome_pth.name) + ')' if genome_pth.exists() else 'Missing'}")
    print(f"  ✓ Dense Text HNSW    : {'Loaded (' + str(dense_pth.name) + ')' if dense_pth.exists() else 'Disabled / Optional'}")

    gemini_key = "SET ✓" if os.environ.get("GEMINI_API_KEY") else "UNSET (Using local rule-based fallback)"
    voyage_key = "SET ✓" if (os.environ.get("VOYAGE_API_KEY") or (PROJECT_ROOT / ".env").exists()) else "UNSET"
    print(f"  ✓ Gemini API Key     : {gemini_key}")
    print(f"  ✓ Voyage API Key     : {voyage_key}")

    print("=" * 70 + "\n")


def run_cli_search(controller, query, top_k=10, json_output=False):
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


def run_cold_start_cli(controller):
    print("\n" + "=" * 70)
    print("  🌟 CINEVAULT COLD-START ONBOARDING WIZARD")
    print("=" * 70)

    print("\n1. Enter 3 to 5 of your favorite genres (comma-separated):")
    print("   Available: Action, Adventure, Animation, Comedy, Crime, Drama, Fantasy, Horror, Mystery, Romance, Sci-Fi, Thriller, Western")
    raw_genres = input("   Genres > ").strip()
    fav_genres = [g.strip() for g in raw_genres.split(",") if g.strip()]

    print("\n2. Enter up to 3 anchor movie IDs or search titles (e.g. 58559, Inception, 1214):")
    print("   (Toy Story = 1, The Dark Knight = 58559, Inception = 79132, The Matrix = 603)")
    raw_anchors = input("   Movie IDs/Titles > ").strip()
    db_path_ref = PROJECT_ROOT / "db" / "cinevault.db"
    anchor_ids, anchor_warnings = resolve_anchor_tokens(raw_anchors, db_path_ref)
    for msg in anchor_warnings:
        print(f"   ⚠  {msg}")

    print("\n3. Enter any dealbreaker rules (e.g. 'No Slapstick', 'No Gore'):")
    raw_dealbreakers = input("   Dealbreakers > ").strip()
    dealbreakers = [d.strip() for d in raw_dealbreakers.split(",") if d.strip()]

    controller.seed_cold_start(
        favorite_genres=fav_genres,
        anchor_movie_ids=anchor_ids,
        dealbreakers=dealbreakers
    )

    print(f"\n✓ Cold-start profile created for user '{controller.user_id}'!")
    print(f"  Genre Affinities: {controller.profile.genre_affinity}")
    print(f"  Tag Affinities  : {controller.profile.tag_affinity}\n")


def print_user_profile(controller, json_format=False):
    p = controller.profile
    if json_format:
        print(json.dumps(p.to_dict(), indent=2))
        return

    print("\n" + "=" * 70)
    print(f"  👤 CINEVAULT USER PROFILE: '{p.user_id}'")
    print("=" * 70)
    print(f"  Personalization λ : {p.personalization_lambda:.2f}")
    print(f"  Total Rated Movies: {len(p.rating_log)}")
    print(f"  Dealbreakers      : {p.dealbreakers or 'None'}")
    print(f"  Disabled Signals  : {p.disabled_signals or 'None'}")
    print("\n  Top Genre Affinities:")
    for g, aff in sorted(p.genre_affinity.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    - {g:<15s}: {aff:+.2f}")
    print("\n  Top Director Affinities:")
    for d, aff in sorted(p.director_affinity.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    - {d:<15s}: {aff:+.2f}")
    print("\n  Sensitivity Weights:")
    print(f"    - Genre Weight   : {p.genre_weight:.2f}")
    print(f"    - Director Weight: {p.director_weight:.2f}")
    print(f"    - Actor Weight   : {p.actor_weight:.2f}")
    print(f"    - Tag Weight     : {p.tag_weight:.2f}")
    print(f"    - Pacing Weight  : {p.pacing_weight:.2f}")
    print("=" * 70 + "\n")


def run_edit_profile_cli(controller):
    p = controller.profile
    print("\n" + "=" * 70)
    print(f"  ✏️ EDIT PREFERENCES FOR USER: '{p.user_id}'")
    print("=" * 70)
    print(f"Current Dealbreakers: {p.dealbreakers}")
    new_db = input("Enter new dealbreaker to add (or leave blank to skip): ").strip()
    if new_db:
        clean_db = re.sub(r"^no\s+", "", new_db.lower())
        if clean_db not in p.dealbreakers:
            p.dealbreakers.append(clean_db)
            print(f"✓ Added dealbreaker: '{clean_db}'")

    print("\nCurrent Sensitivity Weights:")
    print(f"1. Genre Weight   [{p.genre_weight:.2f}]")
    print(f"2. Director Weight[{p.director_weight:.2f}]")
    print(f"3. Actor Weight   [{p.actor_weight:.2f}]")
    gw_in = input("Enter new Genre Weight (0.0 - 1.0, or blank to keep): ").strip()
    if gw_in:
        try:
            p.genre_weight = max(0.0, min(1.0, float(gw_in)))
            print(f"✓ Genre weight updated to {p.genre_weight:.2f}")
        except ValueError:
            pass

    controller._save_profile_safe()
    print("\n✓ Profile changes saved successfully!\n")


def main():
    parser = argparse.ArgumentParser(description="CineVault Personalized Recommendation Engine")
    parser.add_argument("--query", "-q", type=str, help="Search query string")
    parser.add_argument("--user", "-u", type=str, default="default_user", help="Active user ID (default: default_user)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations to return (default: 10)")
    parser.add_argument("--onboard", action="store_true", help="Launch Cold-Start Onboarding Wizard")
    parser.add_argument("--show-profile", action="store_true", help="Display active user profile preferences")
    parser.add_argument("--edit-profile", action="store_true", help="Interactively edit user profile preferences")
    parser.add_argument("--clear-history", action="store_true", help="Clear search query history for active user")
    parser.add_argument("--check-health", action="store_true", help="Run system & hardware diagnostics")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    db_path = PROJECT_ROOT / "db" / "cinevault.db"
    user_id = validate_user_id(args.user, db_path=db_path)

    if args.check_health:
        run_health_check()
    elif args.show_profile:
        ctrl = CineVaultController(user_id=user_id)
        print_user_profile(ctrl, json_format=args.json)
    elif args.edit_profile:
        ctrl = CineVaultController(user_id=user_id)
        run_edit_profile_cli(ctrl)
    elif args.clear_history:
        ctrl = CineVaultController(user_id=user_id)
        ctrl.clear_query_history()
        print(f"✓ Cleared search query history for user '{user_id}'.")
    elif args.onboard:
        ctrl = CineVaultController(user_id=user_id)
        run_cold_start_cli(ctrl)
    elif args.query:
        ctrl = CineVaultController(user_id=user_id)
        run_cli_search(ctrl, args.query, top_k=args.top_k, json_output=args.json)
    else:
        from interface.tui.app import CineVaultApp
        ctrl = CineVaultController(user_id=user_id)
        app = CineVaultApp(user_id=user_id, controller=ctrl)
        app.run()


if __name__ == "__main__":
    main()
