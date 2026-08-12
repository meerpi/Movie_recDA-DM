"""nlp/mmr.py — MMR diversity filter with franchise capping."""

import re


_FRANCHISE_STOPWORDS = {
    "the", "a", "an", "of", "and", "in", "on", "at", "to",
    "movie", "part", "chapter", "creation", "origins", "collection",
}


class MaximalMarginalRelevance:

    def __init__(self, diversity_lambda=0.75, max_per_franchise=2):
        self.diversity_lambda  = diversity_lambda  # 1.0 = pure relevance, 0.0 = max diversity
        self.max_per_franchise = max_per_franchise

    def _franchise_key(self, item):
        """TMDb collection wins over title stem — catches cross-title universes
        like Conjuring/Annabelle that share no title tokens."""
        collection = item.get("collection")
        if collection:
            normalized = re.sub(r"\bcollection\b", "", collection.strip().lower()).strip()
            tokens     = [t for t in normalized.split() if t not in _FRANCHISE_STOPWORDS]
            key        = " ".join(tokens) if tokens else normalized
            return f"coll:{key}"
        return f"stem:{self._title_stem(item.get('title', ''))}"

    def _title_stem(self, title):
        full_clean = title.strip().lower()
        parts  = re.split(r"[:\-]|\b\d+$", title)
        base   = parts[0].strip().lower() if parts and parts[0].strip() else full_clean
        tokens = [t for t in base.split() if t not in _FRANCHISE_STOPWORDS]
        return " ".join(tokens) if tokens else base

    def _is_stem_match(self, a, b):
        if not a or not b:
            return False
        if a == b:
            return True

        na = " ".join(t.rstrip("s") for t in a.split())
        nb = " ".join(t.rstrip("s") for t in b.split())
        if na == nb:
            return True

        pa = r"\b" + r"\s+".join(re.escape(t.rstrip("s")) + r"s?" for t in a.split()) + r"\b"
        pb = r"\b" + r"\s+".join(re.escape(t.rstrip("s")) + r"s?" for t in b.split()) + r"\b"

        return bool(re.search(pa, b) or re.search(pb, a))

    def _similarity(self, a, b):
        key_a, key_b = self._franchise_key(a), self._franchise_key(b)

        if key_a.startswith("coll:") and key_a == key_b:
            return 0.97

        stem_a = re.sub(r"^(coll:|stem:)", "", key_a)
        stem_b = re.sub(r"^(coll:|stem:)", "", key_b)
        if stem_a and stem_b and self._is_stem_match(stem_a, stem_b):
            return 0.90

        genres_a = set(a.get("genres", []))
        genres_b = set(b.get("genres", []))
        if not genres_a or not genres_b:
            return 0.0
        return len(genres_a & genres_b) / len(genres_a | genres_b) * 0.3

    def filter_diverse(self, candidates, top_k=10):
        if not candidates:
            return []

        def _score(c):
            for key in ("final_score", "rerank_score", "rrf_score"):
                if key in c and c[key] is not None:
                    return float(c[key])
            return 0.0

        scores  = [_score(c) for c in candidates]
        min_s   = min(scores)
        range_s = max(scores) - min_s

        def norm(c):
            if range_s < 1e-9:
                return 1.0
            return (_score(c) - min_s) / range_s

        selected  = []
        remaining = list(candidates)
        fran_counts: dict[str, int] = {}

        while remaining and len(selected) < top_k:
            if not selected:
                best = remaining.pop(0)
                selected.append(best)
                fran_counts[self._franchise_key(best)] = 1
                continue

            best_val, best_idx = -999.0, 0
            for idx, cand in enumerate(remaining):
                fkey = self._franchise_key(cand)
                if fran_counts.get(fkey, 0) >= self.max_per_franchise:
                    continue
                rel     = norm(cand)
                max_sim = max(self._similarity(cand, s) for s in selected)
                mmr     = self.diversity_lambda * rel - (1.0 - self.diversity_lambda) * max_sim
                if mmr > best_val:
                    best_val, best_idx = mmr, idx

            best = remaining.pop(best_idx if best_val > -999.0 else 0)
            selected.append(best)
            fkey = self._franchise_key(best)
            fran_counts[fkey] = fran_counts.get(fkey, 0) + 1

        return selected
