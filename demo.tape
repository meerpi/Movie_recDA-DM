# CineVault — Full Showcase Recording
# Record: vhs demo.tape
# Outputs: demo.gif + demo.mp4

Output demo.gif
Output demo.mp4

Set Shell "bash"
Set FontSize 14
Set Width 1400
Set Height 800
Set Theme "Catppuccin Mocha"
Set TypingSpeed 45ms
Set Padding 15

# ═══════════════════════════════════════════════════════════════
# PART 1: Launch the app
# ═══════════════════════════════════════════════════════════════
Type "cd ~/curr_project/movie_rec && source .venv/bin/activate"
Enter
Sleep 1s

Type "python main.py"
Enter
Sleep 4s

# ═══════════════════════════════════════════════════════════════
# PART 2: Preferences & Onboarding (Ctrl+O)
# Shows Genre Selectors, Anchor Movies & Dealbreaker Filters
# ═══════════════════════════════════════════════════════════════
Ctrl+O
Sleep 4s
Escape
Sleep 2s

# ═══════════════════════════════════════════════════════════════
# PART 3: Actor Direct Search — "Robert Downey Jr"
# ═══════════════════════════════════════════════════════════════
Ctrl+S
Sleep 800ms
Type "Robert Downey Jr"
Sleep 500ms
Enter
Sleep 7s

# Navigate down to a film
Down
Sleep 1s
Down
Sleep 1s
Down
Sleep 1s

# ── FEATURE SHOWCASE: Movie Inspector Modal (Enter) ──
# Opens the full movie inspector showing TMDB poster, cast, synopsis & scores
Enter
Sleep 6s
Escape
Sleep 2.5s

# ── FEATURE SHOWCASE: Review Submission Modal (r key) ──
# Press 'r' on a movie to open the interactive aspect-based review system
Type "r"
Sleep 5s
Escape
Sleep 2.5s

# ═══════════════════════════════════════════════════════════════
# PART 4: Complex NLP Queries (Neural Recommendation Engine)
# Prompts that break standard keyword/genre recommendation systems
# ═══════════════════════════════════════════════════════════════

# ── Query 1: Cross-medium emotional comparison ──
Ctrl+S
Sleep 800ms
Type "movies that feel like reading a Haruki Murakami novel"
Sleep 500ms
Enter
Sleep 8s

Down
Sleep 1s
Down
Sleep 1s
# Inspect result
Enter
Sleep 5s
Escape
Sleep 2.5s

# ── Query 2: Negation + emotional context ──
Ctrl+S
Sleep 800ms
Type "something to watch after a breakup that isnt a romantic comedy"
Sleep 500ms
Enter
Sleep 8s

Down
Sleep 1s
Down
Sleep 1s
Down
Sleep 1s
Sleep 2.5s

# ── Query 3: Abstract moral concept ──
Ctrl+S
Sleep 800ms
Type "films where the villain actually has a valid point"
Sleep 500ms
Enter
Sleep 8s

Down
Sleep 1s
Down
Sleep 1s
# Inspect result
Enter
Sleep 5s
Escape
Sleep 2.5s

# ── Query 4: Director style modification ──
Ctrl+S
Sleep 800ms
Type "Kubrick but emotionally warmer"
Sleep 500ms
Enter
Sleep 8s

Down
Sleep 1s
Down
Sleep 1s
Sleep 2.5s

# ── Query 5: Contradictory mood ──
Ctrl+S
Sleep 800ms
Type "a feel-good movie that is actually quite dark underneath"
Sleep 500ms
Enter
Sleep 8s

Down
Sleep 1s
Down
Sleep 1s
Down
Sleep 1s
Sleep 2.5s

# ── Query 6: Niche cinematic element ──
Ctrl+S
Sleep 800ms
Type "movies where architecture or buildings are central to the story"
Sleep 500ms
Enter
Sleep 8s

Down
Sleep 1s
# Inspect result
Enter
Sleep 5s
Escape
Sleep 2.5s

# ═══════════════════════════════════════════════════════════════
# PART 5: Taste Profile Dashboard (Ctrl+P)
# Displays user's evolving preference weights & rating history
# ═══════════════════════════════════════════════════════════════
Ctrl+P
Sleep 4s
Down
Sleep 1s
Down
Sleep 1s
Down
Sleep 1s
Sleep 3s
Escape
Sleep 2s

# Exit app
Sleep 1s
Ctrl+Q
Sleep 1s
