https://github.com/user-attachments/assets/8701d557-e051-4bcf-920d-c59a24a5d7ac

# CineVault

A local-first neural movie recommendation engine and search system with a terminal user interface (TUI).

The engine implements a hybrid retrieval and reranking architecture: multi-lane candidate retrieval (BM25 lexical search, 1,128-dimensional Tag Genome HNSW, and 1,024-dimensional dense text embeddings) combined through Reciprocal Rank Fusion (RRF), cross-encoder neural reranking, personalized Bayesian score blending, and Maximal Marginal Relevance (MMR) diversification.

---

## Overview

CineVault indexes 62,423 movies from the MovieLens 25M and TMDb datasets. It supports both free-form natural language querying and deterministic fast-path routing, running either fully offline on local CPU/GPU hardware or augmented with cloud embedding and extraction APIs.

- Multi-Lane Hybrid Retrieval: Combines Okapi BM25 keyword search, cosine similarity over 1,128-dimensional MovieLens tag genome vectors, and optional 1,024-dimensional dense semantic vectors (Voyage AI).
- Cross-Encoder Neural Reranking: Reranks candidate pools using local PyTorch cross-encoders (`BAAI/bge-reranker-v2-m3` in fp16), local 4-bit quantized GGUF models (`Qwen3-Reranker-4B`), or remote API rerankers.
- Multi-Signal User Profiling: Tracks user preferences across genres, tags, directors, actors, pacing, tone, and release eras with exponential interaction decay, custom dealbreakers, and confidence weighting.
- Dual-Path Review Analysis: Ingests user reviews via structured aspect checkboxes and text extraction (Gemini Flash structured JSON or local fuzzy tag matching with sarcasm heuristics).
- Franchise-Aware Diversification: Applies Maximal Marginal Relevance (MMR) with TMDb collection grouping to prevent sequel clumping in recommendation lists.
- Full Offline Capability: Operates without API keys by falling back to BM25, Genome HNSW, local regex-based query understanding, and local cross-encoder models.

---

## System Architecture

```
User Query / CLI / TUI
  │
  ├── 1. Query Router (nlp/router.py)
  │     ├── Fast-Path: Association rules (Apriori co-rating patterns)
  │     └── Fast-Path: Aggregated popularity & rating top-N queries
  │
  ├── 2. Query Understanding Layer (nlp/qul.py)
  │     ├── Intent & demonym detection (e.g. language/country extraction)
  │     ├── Concept expansion via model/concept_expansions.json
  │     └── Gemini structured output / Local regex fallback
  │
  ├── 3. Candidate Retrieval (nlp/retriever.py)
  │     ├── Lane 1: BM25 (Okapi index over titles, overviews, crew, tags)
  │     ├── Lane 2: Tag Genome HNSW (1,128-dim vectors, cosine distance)
  │     ├── Lane 3: Dense HNSW (1,024-dim Voyage embeddings, optional)
  │     └── Fusion: Reciprocal Rank Fusion (RRF, k=60)
  │
  ├── 4. Result Hydration (nlp/hydrator.py)
  │     └── Merges SQLite metadata with Tier A/B/C profile cards
  │
  ├── 5. Neural Reranker (nlp/reranker.py)
  │     ├── Primary: BAAI/bge-reranker-v2-m3 (PyTorch fp16)
  │     ├── Alternative: Qwen3-Reranker-4B Q4 GGUF (llama-cpp-python)
  │     └── Fallback / API: Voyage rerank-2.5
  │
  ├── 6. Personalization & Scoring Fusion (nlp/pipeline.py)
  │     ├── Bayesian rating dampening: (N*R + 100*3.2)/(N + 100)
  │     ├── Profile boost calculation across 8 affinity vectors
  │     ├── Lambda dial: lambda * Relevance + (1 - lambda) * ProfileBoost
  │     └── Genre, tone contradiction, and recency penalty filters
  │
  └── 7. Maximal Marginal Relevance (nlp/mmr.py)
        ├── Diversity lambda scoring
        └── Hard cap per franchise / TMDb collection
```

---

## Data Coverage Tiers

The dataset is divided into three functional tiers based on metadata completeness:

| Tier | Count | Description | Available Metadata |
|------|-------|-------------|-------------------|
| Tier A | 9,526 | High-interest / popular catalog | Rich profile cards: themes, tone, pacing, directorial style notes, standout performances, notable criticisms, comparable films, dense embeddings. |
| Tier B | 4,290 | Mid-tail MovieLens catalog | 1,128-dimensional Tag Genome vectors, scraped Wikipedia summaries, IMDb user reviews, and TMDb metadata. |
| Tier C | 48,607 | Long-tail global catalog | Core database records: title, release year, runtime, genres, aggregated rating stats, TMDb overview, and keyword tokens. |

---

## Quick Start

### 1. Environment Setup

```bash
git clone <repository_url>
cd movie_rec
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables (Optional)

Create a `.env` file in the project root to enable remote APIs:

```env
# Optional: Enables Dense HNSW retrieval lane and cloud reranking
VOYAGE_API_KEY=your_voyage_key_here

# Optional: Enables LLM-driven query understanding and review parsing
GEMINI_API_KEY=your_gemini_key_here

# Optional: Fallback for LLM extraction
OPENAI_API_KEY=your_openai_key_here
```

If no keys are provided, the application runs offline using BM25, Tag-Genome HNSW, local QUL rules, and local cross-encoder models.

### 3. Diagnostics and Pre-Flight Check

Run the system health check to verify database integrity, indices, and GPU availability:

```bash
python main.py --check-health
```

### 4. Running the Application

```bash
# Launch interactive Terminal User Interface (TUI)
python main.py

# Launch TUI under a specific profile
python main.py --user alice

# Run a cold-start onboarding wizard for a new user
python main.py --onboard --user alice

# Run a CLI search query without opening the TUI
python main.py --query "atmospheric slow-burn psychological thriller"

# Output CLI search results as JSON
python main.py --query "neo-noir detective" --top-k 5 --json

# View or interactively edit stored user profile preferences
python main.py --show-profile --user alice
python main.py --edit-profile --user alice
```

---

## TUI Keybindings

The Terminal User Interface is implemented with Textual.

| Shortcut | Context | Action |
|----------|---------|--------|
| `Ctrl+S` | Global | Focus the search input bar |
| `Ctrl+P` | Global | Open User Profile and Settings modal (manage presets, sensitivity weights, dealbreakers) |
| `Ctrl+O` | Global | Open Profile Switcher (switch active user or create new profile) |
| `Enter` | Search Results | Inspect detailed profile card for the highlighted film |
| `R` | Search Results / Inspector | Open interactive Review modal for the highlighted film |
| `Escape` | Modals / Screens | Close active modal, drawer, or screen |
| `Ctrl+Q` | Global | Exit application |

---

## Scoring & Personalization

Final candidate ranking combines neural relevance, overall ratings, and user profile affinities:

- Relevance & Quality: Combines normalized cross-encoder rerank scores with Bayesian-damped ratings.
- Personalization Dial (λ): Blends query relevance and user taste profile (0.0 = pure profile, 1.0 = pure query relevance).
- Contextual Rules: Applies scoring modifiers for national origin (demonym matching), genre constraints, tone consistency, and release era decay.

---

## Data Pipeline Build Steps

To rebuild the database, feature stores, and index files from raw MovieLens 25M and TMDb data:

```bash
# 1. Load raw datasets into SQLite
python etl/load_movielens.py
python etl/load_metadata.py
python etl/load_genome_vectors.py

# 2. Compute aggregate statistics and association rules
python etl/enrich_database.py
python etl/compute_stats_and_rules.py

# 3. Build tiered profile cards
python build_tier_a.py
python build_tier_b.py
python build_tier_c.py
python dirtywork/backfill_tier_a_genome.py

# 4. Compile search indices
python dirtywork/build_bm25.py
python dirtywork/build_genome_hnsw.py
python dirtywork/embed_tier_a.py
python dirtywork/build_dense_hnsw.py
```

---

## Repository Structure

```
movie_rec/
├── main.py                     # CLI entry point, argument parser, and health checks
├── schema.sql                  # Canonical SQLite schema definition
├── requirements.txt            # Python dependencies
├── interface/
│   ├── controller.py           # State management, caching, and pipeline orchestration
│   └── tui/                    # Textual user interface implementation
│       ├── app.py              # Main Textual App class and theme bindings
│       ├── cinevault.tcss      # Phosphor dark terminal styling sheet
│       ├── screens/            # Search, Profile, and Profile Switcher screens
│       └── modals/             # Movie Inspector, Review, and Onboarding dialogs
├── nlp/
│   ├── pipeline.py             # End-to-end recommendation pipeline execution
│   ├── router.py               # Deterministic query router and association rule lookup
│   ├── qul.py                  # Query Understanding Layer (LLM + local fallback)
│   ├── retriever.py            # 3-lane RRF candidate retriever (BM25 + Genome + Dense)
│   ├── hydrator.py             # Result metadata hydration from SQLite and JSONL
│   ├── reranker.py             # Cross-encoder reranker (BGE-M3 / Qwen3 GGUF)
│   ├── mmr.py                  # Maximal Marginal Relevance diversification
│   └── ltr_scorer.py           # Optional Learning-to-Rank (XGBoost) scoring path
├── user_profile/
│   ├── schema.py               # UserProfile, UserPreset, and affinity vector data classes
│   ├── store.py                # Optimistic-concurrency SQLite storage for profiles
│   ├── identity.py             # Profile validation and anchor movie resolution
│   └── review_processor.py     # Aspect credit assignment & sarcasm detection
├── etl/                        # Raw data loading, enrichment, and statistics scripts
├── model/                      # Configuration files, concept expansions, and LTR models
└── dirtywork/                  # HNSW indices, BM25 pickles, and dataset jsonl files
```

---

## Dependencies

- Textual: Terminal User Interface runtime
- PyTorch: Tensor operations and GPU acceleration for cross-encoders
- Sentence-Transformers: Cross-encoder reranking (`BAAI/bge-reranker-v2-m3`)
- hnswlib: Approximate Nearest Neighbors index for Genome and Dense embeddings
- rank-bm25: Lexical search index
- llama-cpp-python: Local GGUF quantized model execution (optional)
- RapidFuzz & spaCy: Fuzzy string matching and text processing
- Google-GenAI / OpenAI: Optional remote LLM query understanding and review parsing
- VoyageAI: Optional dense embeddings and cloud reranking
