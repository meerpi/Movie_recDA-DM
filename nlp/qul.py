#!/usr/bin/env python3
"""
nlp/qul.py — Step 9: High-Performance Local Query Understanding Layer (QUL)

100% Offline, Zero-API-Key Local QUL Engine powered by:
  1. spaCy EntityRuler for structural genre/pacing/tag token extraction.
  2. RapidFuzz for typo-tolerant title & genome tag resolution (< 20ms).
  3. Local Profile Card DNA expansion & clause/negation modifier parsing.

Usage:
    from nlp.qul import QueryUnderstandingLayer

    qul = QueryUnderstandingLayer()
    parsed = qul.parse_query("Movies like Dark night but less serious but doesnt loose the phylosopjical depth.")
    print(parsed["expanded_query"])
"""

import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import spacy
from rapidfuzz import fuzz, process

logger = logging.getLogger("cinevault.qul")

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "db" / "cinevault.db"
TIER_A_PATH = PROJECT_ROOT / "tier_a_profile_cards_v3.jsonl"
TIER_B_PATH = PROJECT_ROOT / "tier_b_profile_cards.jsonl"
TIER_C_PATH = PROJECT_ROOT / "tier_c_profile_cards.jsonl"

NEGATION_PATTERNS = re.compile(r"\b(?:not|no|less|without|non|except)\s+([a-zA-Z0-9\s-]+)", re.IGNORECASE)

CONCEPT_EXPANSIONS = {
    "dark magic": ["demons", "supernatural", "occult", "dark fantasy", "magic", "exorcism", "hell"],
    "magic": ["supernatural", "fantasy", "wizards", "sorcery", "magic"],
    "comic book": ["comic", "graphic novel", "superhero", "dc comics", "vertigo"],
    "comic": ["comic", "graphic novel", "superhero"],
    "dc": ["dc comics", "superhero", "constantine", "batman"],
    "marvel": ["marvel comics", "superhero"],
}


class QueryUnderstandingLayer:
    """
    Local, deterministic Query Understanding Layer using spaCy, RapidFuzz, and local Card DNA.
    """

    def __init__(self, db_path: Path = DB_PATH):
        t0 = time.time()
        logger.info("Initializing Local CineVault QUL Engine (spaCy + RapidFuzz)...")
        self.db_path = db_path

        # 1. Load spaCy blank English pipeline + EntityRuler
        self.nlp = spacy.blank("en")
        self.ruler = self.nlp.add_pipe("entity_ruler")

        # 2. In-Memory Card & Title Lookup Indices
        self.cards: Dict[int, dict] = {}
        self.titles_dict: Dict[str, int] = {}
        self.actor_vocab: Set[str] = set()
        self.director_vocab: Set[str] = set()
        self.tag_vocab: Set[str] = set()
        self.genres_vocab: Set[str] = {
            "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
        }

        self._load_memory_assets()
        self._build_spacy_rules()

        t1 = time.time()
        logger.info(f"Local QUL Engine initialized in {t1 - t0:.2f}s with {len(self.titles_dict):,} titles & {len(self.tag_vocab):,} tags.")

    def _load_memory_assets(self):
        """Loads profile cards, titles, actors, and tags into RAM for sub-millisecond lookup."""
        # Load SQLite titles
        if self.db_path.exists():
            try:
                conn = sqlite3.connect(self.db_path)
                c = conn.cursor()
                rows = c.execute("""
                    SELECT m.movie_id, m.title 
                    FROM movies m
                    LEFT JOIN movie_stats ms ON m.movie_id = ms.movie_id
                    ORDER BY ms.num_ratings DESC
                """).fetchall()
                for mid, title in rows:
                    clean_t = title.lower().strip()
                    if clean_t not in self.titles_dict:
                        self.titles_dict[clean_t] = mid
                conn.close()
            except Exception as e:
                logger.error(f"Error loading DB titles: {e}")

        # Load Cards metadata
        for pth in (TIER_A_PATH, TIER_B_PATH, TIER_C_PATH):
            if pth.exists():
                with open(pth, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            card = json.loads(line)
                            mid = card.get("movie_id")
                            if mid:
                                mid = int(mid)
                                self.cards[mid] = card

                                # Index title if missing
                                t = card.get("title")
                                if t and t.lower() not in self.titles_dict:
                                    self.titles_dict[t.lower()] = mid

                                # Index actors & directors & tags
                                for a in card.get("actors", []):
                                    self.actor_vocab.add(a)
                                for d in card.get("directors", []):
                                    self.director_vocab.add(d)
                                for tg in card.get("top_tags", []):
                                    tag_str = tg["tag"] if isinstance(tg, dict) else str(tg)
                                    self.tag_vocab.add(tag_str.lower())
                        except Exception:
                            continue

    def _build_spacy_rules(self):
        """Builds spaCy EntityRuler patterns for genres, pacing, and genome tags."""
        patterns = []
        for g in self.genres_vocab:
            patterns.append({"label": "GENRE", "pattern": g.lower()})

        for p in ["fast", "fast paced", "slow", "slow burn", "moderate", "relentless", "intense"]:
            patterns.append({"label": "PACING", "pattern": p})

        for t in list(self.tag_vocab)[:2000]:
            patterns.append({"label": "TAG", "pattern": t.lower()})

        self.ruler.add_patterns(patterns)

    def parse_query(self, raw_query: str) -> Dict[str, Any]:
        """
        Parses raw user query using spaCy + RapidFuzz + Local Card DNA.
        """
        t0 = time.time()
        q_lower = raw_query.lower().strip()

        # ── 1. spaCy Entity Extractions ─────────────────────────────────────
        doc = self.nlp(q_lower)
        spacy_ents = [(ent.text, ent.label_) for ent in doc.ents]
        matched_genres = list({ent.text.capitalize() for ent in doc.ents if ent.label_ == "GENRE"})
        matched_pacing = next((ent.text for ent in doc.ents if ent.label_ == "PACING"), None)
        matched_tags = list({ent.text for ent in doc.ents if ent.label_ == "TAG"})

        # ── 2. Negation & Clause Modifiers ──────────────────────────────────
        negations = [m.strip() for m in NEGATION_PATTERNS.findall(q_lower)]

        # ── 3. Fuzzy Entity / Title Resolution (RapidFuzz) ───────────────────
        detected_title = None
        matched_movie_id = None
        intent_type = "general_search"

        if "like" in q_lower or "similar to" in q_lower or "if i liked" in q_lower:
            intent_type = "title_reference"
            raw_title_target = re.sub(r".*?(?:like|similar to|if i liked)\s+", "", q_lower, flags=re.I)
            raw_title_target = re.split(r"\b(?:but|with|and|not)\b", raw_title_target)[0].strip()

            if raw_title_target:
                match = process.extractOne(
                    raw_title_target, list(self.titles_dict.keys()), scorer=fuzz.token_sort_ratio
                )
                if match and match[1] >= 75:
                    detected_title = match[0].title()
                    matched_movie_id = self.titles_dict[match[0]]

        # ── 4. Fuzzy Typo Correction for Unmatched Query Words ──────────────
        for word in q_lower.split():
            if len(word) > 7 and word not in matched_tags:
                tag_match = process.extractOne(word, list(self.tag_vocab), scorer=fuzz.token_sort_ratio)
                if tag_match and tag_match[1] >= 80:
                    matched_tags.append(tag_match[0])

        # ── 5. DNA & Concept Query Expansion ────────────────────────────────
        expanded_terms = []
        if matched_movie_id and matched_movie_id in self.cards:
            card = self.cards[matched_movie_id]
            expanded_terms.extend(card.get("genres", []))
            expanded_terms.extend(card.get("themes", []))
            expanded_terms.extend(card.get("tone", []))
            expanded_terms.extend([t["tag"] if isinstance(t, dict) else str(t) for t in card.get("top_tags", [])[:8]])
            for d in card.get("directors", []):
                expanded_terms.append(d)

        for concept, terms in CONCEPT_EXPANSIONS.items():
            if concept in q_lower:
                expanded_terms.extend(terms)

        # ── 6. Assemble Expanded Query String ───────────────────────────────
        if expanded_terms:
            expanded_query_str = f"{raw_query} " + " ".join(expanded_terms[:12])
        else:
            expanded_query_str = raw_query

        # Remove negative terms from expanded query if present
        for neg in negations:
            expanded_query_str = re.sub(r"\b" + re.escape(neg) + r"\b", "", expanded_query_str, flags=re.I)

        expanded_query_str = " ".join(expanded_query_str.split())

        latency_ms = round((time.time() - t0) * 1000, 2)

        return {
            "raw_query": raw_query,
            "expanded_query": expanded_query_str,
            "detected_title": detected_title,
            "matched_movie_id": matched_movie_id,
            "intent_type": intent_type,
            "matched_genres": matched_genres,
            "matched_pacing": matched_pacing,
            "matched_tags": matched_tags,
            "negated_constraints": negations,
            "spacy_entities": spacy_ents,
            "latency_ms": latency_ms,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    qul = QueryUnderstandingLayer()

    test_queries = [
        "Movies like Dark night but less serious but doesnt loose the phylosopjical depth.",
        "movies like Inception but less serious and more action",
        "psychological sci-fi thriller starring Keanu Reeves but not horror",
        "atmospheric slow burn murder mystery like Zodiac",
    ]

    for q in test_queries:
        print(f"\n--- Raw Query: '{q}' ---")
        res = qul.parse_query(q)
        print(json.dumps(res, indent=2))
