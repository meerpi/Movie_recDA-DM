# 🎬 CineVault

**A GPU-accelerated, fully offline-capable, neural movie recommendation engine with a retro Terminal User Interface.**

Built on a 7-stage hybrid pipeline: BM25 keyword search × Tag-Genome HNSW × Dense Voyage-4-Large embeddings → Reciprocal Rank Fusion → Cross-Encoder Reranking → Personalized Score Fusion → MMR Diversity Filter.

---

## ✨ Features

- **Natural Language Search** — "atmospheric slow-burn Korean thriller with philosophical depth"
- **3-Lane Hybrid Retrieval** — BM25 + Genome HNSW (1,128-dim tag vectors) + Dense Voyage ANN
- **7-Stage Pipeline** — Router → QUL Expansion → Retrieval → Hydration → Cross-Encoder → Personalization → MMR
- **Fully Offline Mode** — Works without any API keys (BM25 + Genome only)
- **GPU Reranking** — BAAI/bge-reranker-v2-m3 (fp16 autocast) or Qwen3-Reranker-4B Q4 GGUF
- **3-Tier Movie Coverage** — 9,526 (Tier A rich) + 4,290 (Tier B genome) + 27,278 (Tier C catalog)
- **Personalization** — Multi-dimensional user profile (genre/tone/tag/pacing/actor/director affinities)
- **Surgical Review System** — Checkbox-based aspect tagging + optional free-text review (LLM or local RapidFuzz)
- **Franchise Deduplication** — MMR with TMDb collection-aware diversity filter

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
git clone <repo>
cd movie_rec
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Required API Keys (Optional but recommended)

Create a `.env` file in the project root:

```env
VOYAGE_API_KEY=your_voyage_ai_key_here   # For Dense HNSW lane (optional)
GEMINI_API_KEY=your_gemini_key_here      # For LLM review extraction (optional)
```

Without API keys, the engine runs fully offline using BM25 + Genome HNSW (2 of 3 retrieval lanes).

### 3. Launch the TUI

```bash
python main.py                          # Interactive TUI (default)
python main.py --user alice             # Launch as specific user
python main.py --check-health           # Run system diagnostics
python main.py --onboard --user alice   # Cold-start onboarding wizard
python main.py --query "dark sci-fi"    # CLI search (no TUI)
python main.py --query "dark sci-fi" --json  # JSON output
```

---

## 🏗️ Architecture

```
main.py
  └── CineVaultController (interface/controller.py)
        ├── CineVaultPipeline (nlp/pipeline.py)
        │     ├── QueryRouter          — fast-path association rules / top-N
        │     ├── QueryUnderstandingLayer (QUL) — spaCy + RapidFuzz expansion
        │     ├── CineVaultRetriever  — BM25 + Genome HNSW + Dense HNSW (RRF)
        │     ├── ResultHydrator      — SQLite + profile card enrichment
        │     ├── CineVaultReranker   — BAAI bge-reranker or Qwen3 Q4
        │     ├── PersonalizationFusion — λ-dial blend (query ↔ profile)
        │     └── MaximalMarginalRelevance — diversity filter + franchise cap
        ├── UserProfileStore (user_profile/store.py)
        └── LLMReviewProcessor (user_profile/review_processor.py)
              ├── Path A: Surgical checkbox credit assignment
              ├── Gemini Flash LLM extraction (if API key set)
              └── Path B: Local RapidFuzz fuzzy tag fallback
```

---

## ⌨️ TUI Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+S` | Focus search input |
| `Ctrl+O` | Profile switcher (switch/create users) |
| `Ctrl+P` | Profile settings (λ, presets, memory) |
| `Enter` | Inspect selected movie |
| `R` | Review selected movie |
| `Escape` | Close modal / go back |
| `Ctrl+Q` | Quit |

---

## 📁 Data Pipeline Build Order

> Run these scripts once to build all indexes from the raw MovieLens / TMDB data:

```bash
python etl/load_movielens.py          # Load MovieLens 25M ratings
python etl/load_metadata.py           # Load TMDB metadata
python etl/enrich_database.py         # Compute movie stats, genres
python etl/compute_stats_and_rules.py # Association rules + popularity stats
python build_tier_a.py               # Build Tier A profile cards (LLM)
python build_tier_b.py               # Build Tier B genome cards
python build_tier_c.py               # Build Tier C catalog cards
python backfill_tier_a_genome.py     # Backfill genome vectors into Tier A
python nlp/build_bm25.py             # Build BM25 index
python nlp/build_genome_hnsw.py      # Build Genome HNSW index
python nlp/embed_tier_a.py           # Embed Tier A cards with Voyage AI
python nlp/build_dense_hnsw.py       # Build Dense HNSW index
```

---

## 🔧 Configuration

- **Personalization λ dial** — Exposed in TUI controls bar (0.0 = pure profile, 1.0 = pure query)
- **Concept Expansions** — `model/concept_expansions.json` — extend without code changes
- **Profile Card Config** — `model/profile_card_config.json`

---

## 🔮 Future Scope

- **Profile ↔ Scoring Pipeline Wiring** — The profile system (presets, signal toggles, learned taste display) is built as UI + DB but not yet wired into the actual recommendation scoring. The λ slider works (it's passed as `personalization_lambda` to the pipeline), but the per-signal toggles (Watch History / Ratings / Reviews on/off) don't yet gate their respective scoring components. This is the next step.
- **Preset-driven re-search** — When activating a preset, auto-re-run the current query so results update immediately.
- **Profile export/import** — JSON export of profiles + presets for backup or sharing.

---

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `textual` | Terminal User Interface framework |
| `hnswlib` | HNSW approximate nearest-neighbour index |
| `rank_bm25` | BM25 keyword search |
| `sentence-transformers` | BAAI cross-encoder reranker |
| `spacy` | EntityRuler for query parsing |
| `rapidfuzz` | Fuzzy string matching for QUL & review processing |
| `voyageai` | Dense text embeddings (optional) |
| `google-genai` | Gemini LLM for review extraction (optional) |
| `torch` | PyTorch for GPU reranker inference |
