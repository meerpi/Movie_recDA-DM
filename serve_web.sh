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

# Pre-seed terminal cell size for textual-image.
# Over SSH, TIOCGWINSZ returns 0 for pixel dimensions, so the script
# falls back to querying via escape sequence (\e[14t) while bash still
# has raw terminal access (before Textual takes over stdin).
eval "$(python interface/tui/query_cell_size.py 2>/dev/null)" || true

exec python main.py
