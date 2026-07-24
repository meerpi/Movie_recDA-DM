#!/usr/bin/env python3
"""
nlp/reranker.py — Step 8: Cross-Encoder Second-Stage Reranker

Supports:
  1. "BAAI/bge-reranker-v2-m3" (Default: Fast, ~0.7GB VRAM, ~250ms)
  2. "QuantFactory/Qwen3-Reranker-4B-GGUF" (Pre-quantized 4-bit GGUF: ~0.67 NDCG, ~2.1GB VRAM)

Usage:
    from nlp.reranker import CineVaultReranker

    reranker = CineVaultReranker(use_qwen_4bit=True)
    reranked = reranker.rerank(query, candidates, top_k=10)
"""

import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*sending unauthenticated requests to the HF Hub.*")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

from sentence_transformers import CrossEncoder

logger = logging.getLogger("cinevault.reranker")

BAAI_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
GGUF_MODEL_PATH = Path.home() / ".cache" / "huggingface" / "hub" / "models--QuantFactory--Qwen3-Reranker-4B-GGUF" / "snapshots" / "2a42c7aa9c702165da87b09dec164a54d973123b" / "Qwen3-Reranker-4B.Q4_K_M.gguf"


class CineVaultReranker:

    def __init__(
        self,
        model_name: str = BAAI_MODEL_NAME,
        device: Optional[str] = None,
        use_qwen_4bit: bool = False
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.use_qwen_4bit = use_qwen_4bit or ("qwen" in model_name.lower())

        logger.info(f"Initializing CineVaultReranker (Qwen 4-bit={self.use_qwen_4bit}) on {self.device}...")
        t0 = time.time()

        if self.use_qwen_4bit and GGUF_MODEL_PATH.exists():
            from llama_cpp import Llama
            self.model_type = "gguf"
            self.gguf_llm = Llama(
                model_path=str(GGUF_MODEL_PATH),
                n_gpu_layers=-1 if self.device == "cuda" else 0,
                embedding=True,
                verbose=False
            )
            logger.info("Loaded pre-quantized Qwen3-Reranker-4B.Q4_K_M.gguf on GPU")
        else:
            self.model_type = "bge"
            self.model = CrossEncoder(BAAI_MODEL_NAME, device=self.device, max_length=256)
            logger.info(f"Loaded {BAAI_MODEL_NAME} cross-encoder on {self.device}")

        t1 = time.time()
        logger.info(f"Reranker model loaded in {t1 - t0:.2f}s")

    def build_passage_text(self, item: Dict[str, Any]) -> str:
        """Formats candidate result item into a compact passage for high-speed GPU cross-encoding."""
        parts = []

        title = item.get("title", "")
        year = item.get("year", "")
        if title:
            parts.append(f"Title: {title} ({year})" if year else f"Title: {title}")

        genres = item.get("genres", [])
        if genres:
            parts.append(f"Genres: {', '.join(genres[:3])}" if isinstance(genres, list) else f"Genres: {genres}")

        themes = item.get("themes", [])
        if themes:
            parts.append(f"Themes: {', '.join(themes[:3])}" if isinstance(themes, list) else f"Themes: {themes}")

        tags = item.get("top_tags", [])
        if tags:
            if isinstance(tags, list):
                top_few = tags[:8]
                tag_strs = [t if isinstance(t, str) else t.get("tag", "") for t in top_few]
                parts.append(f"Tags: {', '.join(filter(None, tag_strs))}")

        overview = item.get("overview") or item.get("wiki_intro")
        if overview:
            clean_ov = overview.replace("\n", " ").strip()
            parts.append(f"Plot: {clean_ov[:120]}")

        return " | ".join(parts) if parts else str(item)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """Reranks candidates using cross-attention with fp16 GPU autocast acceleration."""
        if not candidates:
            return []

        passages = [self.build_passage_text(c) for c in candidates]
        t0 = time.time()

        if self.model_type == "gguf":
            import numpy as np

            q_raw = np.array(self.gguf_llm.create_embedding(query)["data"][0]["embedding"])
            q_vec = np.mean(q_raw, axis=0) if q_raw.ndim > 1 else q_raw
            q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)

            scores = []
            for p in passages:
                p_raw = np.array(self.gguf_llm.create_embedding(p)["data"][0]["embedding"])
                p_vec = np.mean(p_raw, axis=0) if p_raw.ndim > 1 else p_raw
                p_norm = p_vec / (np.linalg.norm(p_vec) + 1e-9)
                scores.append(float(np.dot(q_norm, p_norm)))

        else:
            pairs = [(query, p) for p in passages]
            with torch.inference_mode():
                if self.device == "cuda":
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
                else:
                    scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        t1 = time.time()

        scored_candidates = []
        for orig_idx, (cand, score) in enumerate(zip(candidates, scores)):
            cand_copy = dict(cand)
            cand_copy["rerank_score"] = float(score)
            cand_copy["rrf_rank"] = orig_idx + 1
            scored_candidates.append(cand_copy)

        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        logger.debug(
            f"Reranked {len(candidates)} candidates in {t1 - t0:.3f}s. "
            f"Top score: {scored_candidates[0]['rerank_score']:.4f}"
        )

        return scored_candidates[:top_k]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing CineVaultReranker (Qwen 4-bit)...")
    reranker = CineVaultReranker(use_qwen_4bit=True)

    test_query = "movies like Annabelle"
    test_candidates = [
        {
            "movie_id": 1,
            "title": "Grandma's Boy",
            "year": 2006,
            "genres": ["Comedy"],
            "themes": ["slacker", "video games"],
            "tone": ["silly"],
        },
        {
            "movie_id": 2,
            "title": "The Boy",
            "year": 2016,
            "genres": ["Horror", "Thriller"],
            "themes": ["possessed doll", "demonic entity", "supernatural dread"],
            "tone": ["creepy", "menacing"],
            "comparable_films": ["Annabelle", "The Conjuring"],
        },
    ]

    results = reranker.rerank(test_query, test_candidates)
    for i, res in enumerate(results, 1):
        print(f"#{i} {res['title']} ({res['year']}) — Score: {res['rerank_score']:.4f} (Original RRF Rank: #{res['rrf_rank']})")
