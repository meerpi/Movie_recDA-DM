"""nlp/reranker.py — Cross-encoder reranker (BGE default, optional Qwen3 GGUF)."""

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

BAAI_MODEL   = "BAAI/bge-reranker-v2-m3"
GGUF_PATH    = (
    Path.home() / ".cache" / "huggingface" / "hub"
    / "models--QuantFactory--Qwen3-Reranker-4B-GGUF"
    / "snapshots" / "2a42c7aa9c702165da87b09dec164a54d973123b"
    / "Qwen3-Reranker-4B.Q4_K_M.gguf"
)


class CineVaultReranker:

    def __init__(self, model_name=BAAI_MODEL, device=None, use_qwen_4bit=False):
        self.device       = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_qwen_4bit = use_qwen_4bit or ("qwen" in model_name.lower())

        logger.info(f"Loading reranker (Qwen4bit={self.use_qwen_4bit}) on {self.device} ...")
        t0 = time.time()

        if self.use_qwen_4bit and GGUF_PATH.exists():
            from llama_cpp import Llama
            self.model_type = "gguf"
            self.gguf_llm   = Llama(
                model_path=str(GGUF_PATH),
                n_gpu_layers=-1 if self.device == "cuda" else 0,
                logits_all=True,
                verbose=False,
            )
            self.yes_id       = self.gguf_llm.tokenize(b"Yes")[-1]
            self.no_id        = self.gguf_llm.tokenize(b"No")[-1]
            self.yes_lower_id = self.gguf_llm.tokenize(b"yes")[-1]
            self.no_lower_id  = self.gguf_llm.tokenize(b"no")[-1]
        else:
            self.model_type = "bge"
            self.model      = CrossEncoder(BAAI_MODEL, device=self.device, max_length=256)

        logger.info(f"Reranker ready in {time.time() - t0:.2f}s.")

    def build_passage_text(self, item):
        parts = []
        title = item.get("title", "")
        year  = item.get("year", "")
        if title:
            parts.append(f"Title: {title} ({year})" if year else f"Title: {title}")
        if item.get("directors"):
            dirs = item["directors"]
            parts.append(f"Director: {', '.join(dirs[:2]) if isinstance(dirs, list) else str(dirs)}")
        if item.get("actors"):
            actors = item["actors"]
            parts.append(f"Starring: {', '.join(actors[:4]) if isinstance(actors, list) else str(actors)}")
        if item.get("genres"):
            parts.append(f"Genres: {', '.join(item['genres'][:3])}")
        if item.get("themes"):
            parts.append(f"Themes: {', '.join(item['themes'][:3])}")
        if item.get("top_tags"):
            tags = [t if isinstance(t, str) else t.get("tag", "") for t in item["top_tags"][:8] if t is not None]
            parts.append(f"Tags: {', '.join(filter(None, tags))}")
        overview = item.get("overview") or item.get("wiki_intro")
        if overview:
            parts.append(f"Plot: {overview.replace(chr(10), ' ').strip()[:120]}")
        return " | ".join(parts) if parts else str(item)

    def rerank(self, query, candidates, top_k=10, batch_size=32):
        if not candidates:
            return []

        passages = [self.build_passage_text(c) for c in candidates]
        t0 = time.time()

        if self.model_type == "gguf":
            scores = []
            for p in passages:
                prompt = (
                    f'Given a query "{query}", is the following document relevant?\n'
                    f'Document: {p}\nAnswer:'
                )
                tokens = self.gguf_llm.tokenize(prompt.encode("utf-8"))
                self.gguf_llm.reset()
                self.gguf_llm.eval(tokens)
                logits    = self.gguf_llm._scores[-1]
                score_yes = max(logits[self.yes_id], logits[self.yes_lower_id])
                score_no  = max(logits[self.no_id], logits[self.no_lower_id])
                scores.append(float(score_yes - score_no))
        else:
            pairs = [(query, p) for p in passages]
            with torch.inference_mode():
                if self.device == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
                else:
                    scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        scored = []
        for orig_idx, (cand, score) in enumerate(zip(candidates, scores)):
            item = dict(cand)
            item["rerank_score"] = float(score)
            item["rrf_rank"]     = orig_idx + 1
            scored.append(item)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        logger.debug(f"Reranked {len(candidates)} in {time.time() - t0:.3f}s. Top: {scored[0]['rerank_score']:.4f}")
        return scored[:top_k]
