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
# PART 1: Launch CineVault
# ═══════════════════════════════════════════════════════════════
Type "cd ~/curr_project/movie_rec && source .venv/bin/activate"
Enter
Sleep 1s

Type "python main.py"
Enter
Sleep 4s

# ═══════════════════════════════════════════════════════════════
# PART 2: Preferences & Onboarding (Ctrl+O)
# ═══════════════════════════════════════════════════════════════
Ctrl+O
Sleep 3.5s
Escape
Sleep 2s

# ═══════════════════════════════════════════════════════════════
# PART 3: Actor Direct Search — "Robert Downey Jr"
# ═══════════════════════════════════════════════════════════════
Ctrl+S
Sleep 600ms
Type "Robert Downey Jr"
Sleep 500ms
Enter
Sleep 7s

# Browse filmography
Down
Sleep 800ms
Down
Sleep 800ms
Down
Sleep 800ms

# ── SHOWCASE: Movie Inspector (Poster, Cast, Synopsis & Scores) ──
Enter
Sleep 5s
Escape
Sleep 2s

# ── SHOWCASE: Interactive Aspect-Based Review System (r key) ──
Type "r"
Sleep 2s

# Phase 1: Toggle aspect chips (Atmosphere, Plot, Acting)
Enter
Sleep 600ms
Tab
Enter
Sleep 600ms
Tab
Enter
Sleep 1s

# Navigate to NEXT button
Tab
Tab
Tab
Tab
Tab
Tab
Enter
Sleep 1.5s

# Phase 2: Aspect Scoring (Rate 5 stars)
Up
Up
Sleep 600ms
Tab
Enter
Sleep 1s

Up
Up
Sleep 600ms
Tab
Enter
Sleep 1s

Up
Up
Sleep 600ms
Tab
Enter
Sleep 1.5s

# Phase 3: Free-form review submission
Type "Incredible performances and immersive atmosphere. A standout film."
Sleep 1.5s
Tab
Sleep 800ms
Enter
Sleep 3s

# ═══════════════════════════════════════════════════════════════
# PART 4: Complex NLP Queries (Neural Recommendation Engine)
# ═══════════════════════════════════════════════════════════════

# ── Query 1: Cross-medium emotional comparison ──
Ctrl+S
Sleep 600ms
Type "movies that feel like reading a Haruki Murakami novel"
Sleep 500ms
Enter
Sleep 8s

Down
Sleep 800ms
Down
Sleep 800ms
# Inspect top match
Enter
Sleep 5s
Escape
Sleep 2s

# ── Query 2: Negation + emotional context ──
Ctrl+S
Sleep 600ms
Type "something to watch after a breakup that isnt a romantic comedy"
Sleep 500ms
Enter
Sleep 8s

Down
Sleep 800ms
Down
Sleep 800ms
# Inspect match
Enter
Sleep 5s
Escape
Sleep 2s

# ── Query 3: Abstract moral concept ──
Ctrl+S
Sleep 600ms
Type "films where the villain actually has a valid point"
Sleep 500ms
Enter
Sleep 8s

Down
Sleep 800ms
Down
Sleep 800ms
# Inspect match
Enter
Sleep 5s
Escape
Sleep 2s

# ── Query 4: Director style modification ──
Ctrl+S
Sleep 600ms
Type "Kubrick but emotionally warmer"
Sleep 500ms
Enter
Sleep 8s

Down
Sleep 800ms
Down
Sleep 800ms
# Inspect match
Enter
Sleep 5s
Escape
Sleep 2s

# ── Query 5: Contradictory mood ──
Ctrl+S
Sleep 600ms
Type "a feel-good movie that is actually quite dark underneath"
Sleep 500ms
Enter
Sleep 8s

Down
Sleep 800ms
Down
Sleep 800ms
# Inspect match
Enter
Sleep 5s
Escape
Sleep 2s

# ── Query 6: Niche cinematic element ──
Ctrl+S
Sleep 600ms
Type "movies where architecture or buildings are central to the story"
Sleep 500ms
Enter
Sleep 8s

Down
Sleep 800ms
Down
Sleep 800ms
# Inspect match
Enter
Sleep 5s
Escape
Sleep 2s

# ═══════════════════════════════════════════════════════════════
# PART 5: Evolving Taste Profile Dashboard (Ctrl+P)
# Displays updated weights and submitted review in taste profile
# ═══════════════════════════════════════════════════════════════
Ctrl+P
Sleep 3.5s
Down
Sleep 800ms
Down
Sleep 800ms
Down
Sleep 800ms
Sleep 3s
Escape
Sleep 2s

# Exit app
Sleep 1s
Ctrl+Q
Sleep 1s
