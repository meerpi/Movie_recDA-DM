#!/bin/bash
set -euo pipefail

cd /home/meerpi/curr_project/movie_rec

# Load API keys (VOYAGE_API_KEY, GEMINI_API_KEY, etc.)
set -a; source .env; set +a

# Activate venv
source .venv/bin/activate

# Ensure terminal is set (SSH PTY should provide this, but be safe)
export TERM=${TERM:-xterm-256color}

# Point at existing HF cache (avoids re-downloading if fallback triggers)
export HF_HOME=/home/meerpi/.cache/huggingface

# Suppress noisy HF warnings
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export TOKENIZERS_PARALLELISM=false

exec python main.py
