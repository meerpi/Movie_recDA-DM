"""nlp/retriever.py — Multi-lane RRF search (BM25 + Genome HNSW + Dense Voyage)."""

import csv
import json
import logging
import math
import os
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

import hnswlib
import numpy as np

logger = logging.getLogger("cinevault.retriever")

PROJECT_ROOT     = Path(__file__).parent.parent
BM25_INDEX_PATH  = PROJECT_ROOT / "dirtywork" / "bm25_index.pkl"
BM25_MAP_PATH    = PROJECT_ROOT / "dirtywork" / "bm25_id_map.json"
GENOME_HNSW_PATH = PROJECT_ROOT / "dirtywork" / "genome.hnsw"
GENOME_MAP_PATH  = PROJECT_ROOT / "dirtywork" / "genome_id_map.json"
GENOME_TAGS_PATH = PROJECT_ROOT / "dirtywork" / "data" / "ml-25m" / "genome-tags.csv"
DENSE_HNSW_PATH  = PROJECT_ROOT / "dirtywork" / "dense_v2.hnsw"
DENSE_MAP_PATH   = PROJECT_ROOT / "dirtywork" / "dense_v2_id_map.json"

VOYAGE_MODEL  = "voyage-4-large"
RRF_K         = 60
LANE_K        = 100

_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def tokenize(text):
    return [t for t in _SPLIT_RE.split(text.lower()) if t]


class CineVaultRetriever:

    def __init__(self, voyage_api_key=None, load_dense=True):
        t0 = time.time()
        logger.info("Initializing CineVault Retriever ...")

        if not BM25_INDEX_PATH.exists():
            raise FileNotFoundError(f"BM25 index missing: {BM25_INDEX_PATH}. Run ETL first.")
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25        = bm25_data["bm25"]
        self.bm25_corpus = bm25_data["corpus"]
        self.bm25_id_map = bm25_data["id_map"]
        logger.info(f"  ✓ Lane 1 (BM25): {len(self.bm25_id_map):,} docs")

        if not GENOME_HNSW_PATH.exists():
            raise FileNotFoundError(f"Genome HNSW missing: {GENOME_HNSW_PATH}. Run ETL first.")
        with open(GENOME_MAP_PATH, encoding="utf-8") as f:
            self.genome_id_map = json.load(f)
        self.genome_dim   = 1128
        self.genome_index = hnswlib.Index(space="cosine", dim=self.genome_dim)
        self.genome_index.load_index(str(GENOME_HNSW_PATH), max_elements=len(self.genome_id_map))
        self.genome_index.set_ef(100)
        logger.info(f"  ✓ Lane 2 (Genome HNSW): {len(self.genome_id_map):,} vectors")

        self.tag_to_idx: dict[str, int] = {}
        self.tag_patterns: dict[str, re.Pattern] = {}
        if GENOME_TAGS_PATH.exists():
            with open(GENOME_TAGS_PATH, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    tag_clean = row["tag"].strip().lower()
                    idx = int(row["tagId"]) - 1
                    self.tag_to_idx[tag_clean] = idx
                    self.tag_patterns[tag_clean] = re.compile(r"\b" + re.escape(tag_clean) + r"\b")
        logger.info(f"  ✓ Genome vocabulary: {len(self.tag_to_idx):,} tags")

        self.dense_index  = None
        self.dense_id_map = []
        self.voyage_client = None

        if load_dense and DENSE_HNSW_PATH.exists():
            with open(DENSE_MAP_PATH, encoding="utf-8") as f:
                self.dense_id_map = json.load(f)
            self.dense_dim   = 1024
            self.dense_index = hnswlib.Index(space="cosine", dim=self.dense_dim)
            self.dense_index.load_index(str(DENSE_HNSW_PATH), max_elements=len(self.dense_id_map))
            self.dense_index.set_ef(100)

            api_key = voyage_api_key or os.environ.get("VOYAGE_API_KEY", "").strip()
            if api_key:
                try:
                    import voyageai
                    self.voyage_client = voyageai.Client(api_key=api_key)
                    logger.info(f"  ✓ Lane 3 (Dense + Voyage): {len(self.dense_id_map):,} vectors")
                except Exception as e:
                    logger.warning(f"  Voyage client failed ({e}). Lane 3 disabled.")
            else:
                logger.info("  Lane 3 disabled — no VOYAGE_API_KEY.")

        logger.info(f"Retriever ready in {time.time() - t0:.2f}s.")

    def build_genome_query_vector(self, query):
        vec = np.zeros(self.genome_dim, dtype=np.float32)
        q_lower  = query.lower().strip()
        q_tokens = set(tokenize(query))

        if q_lower in self.tag_to_idx:
            vec[self.tag_to_idx[q_lower]] += 3.0

        for tag, idx in self.tag_to_idx.items():
            pattern = self.tag_patterns.get(tag) or re.compile(r"\b" + re.escape(tag) + r"\b")
            if pattern.search(q_lower):
                vec[idx] += 2.0
            else:
                overlap = len(q_tokens & set(tokenize(tag)))
                if overlap > 0:
                    vec[idx] += 0.5 * overlap

        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec /= norm
        return vec

    def search_lane1_bm25(self, query, top_k=LANE_K):
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        if not isinstance(scores, np.ndarray):
            scores = np.asarray(scores)

        if len(scores) <= top_k:
            top_idx = np.argsort(-scores)
        else:
            partition = np.argpartition(-scores, top_k)[:top_k]
            top_idx = partition[np.argsort(-scores[partition])]

        return [(self.bm25_id_map[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    def search_lane2_genome(self, query, top_k=LANE_K):
        q_vec = self.build_genome_query_vector(query)
        if np.linalg.norm(q_vec) < 1e-10:
            return []
        labels, dists = self.genome_index.knn_query([q_vec], k=top_k)
        return [
            (self.genome_id_map[int(lbl)], max(0.0, 1.0 - float(dist)))
            for lbl, dist in zip(labels[0], dists[0])
        ]

    def search_lane3_dense(self, query, top_k=LANE_K):
        if not self.dense_index or not self.voyage_client:
            return []
        res   = self.voyage_client.embed([query], model=VOYAGE_MODEL, input_type="query")
        q_vec = np.array(res.embeddings[0], dtype=np.float32)
        labels, dists = self.dense_index.knn_query([q_vec], k=top_k)
        return [
            (self.dense_id_map[int(lbl)], max(0.0, 1.0 - float(dist)))
            for lbl, dist in zip(labels[0], dists[0])
        ]

    def search(self, query, top_k=20, lane_k=LANE_K, k_rrf=RRF_K,
               use_voyage=True, bm25_query=None):
        lanes = {
            "bm25":   self.search_lane1_bm25(bm25_query or query, top_k=lane_k),
            "genome": self.search_lane2_genome(query, top_k=lane_k),
            "dense":  self.search_lane3_dense(query, top_k=lane_k) if (use_voyage and self.voyage_client) else [],
        }

        rrf_scores = defaultdict(float)
        lane_hits  = defaultdict(dict)

        for lane_name, hits in lanes.items():
            for rank, (mid, _) in enumerate(hits):
                rrf_scores[mid] += 1.0 / (k_rrf + rank + 1)
                lane_hits[mid][lane_name] = rank + 1

        sorted_mids = sorted(rrf_scores, key=lambda m: -rrf_scores[m])[:top_k]

        return [
            {
                "movie_id":  mid,
                "rrf_rank":  idx + 1,
                "rrf_score": round(rrf_scores[mid], 5),
                "lanes":     list(lane_hits[mid]),
                "lane_ranks": lane_hits[mid],
            }
            for idx, mid in enumerate(sorted_mids)
        ]
