"""
interface/tui/modals/inspector.py — Movie Inspector Modal with TMDB Poster
"""

import asyncio
import hashlib
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Markdown, Static

logger = logging.getLogger("cinevault.inspector")

# Poster cache directory
_CACHE_DIR = Path(__file__).parent.parent.parent.parent / ".tmp" / "poster_cache"

TMDB_BASE = "https://image.tmdb.org/t/p"
POSTER_SIZE = "w342"


def _poster_cache_path(poster_path: str) -> Path:
    """Returns local cache file path for a poster_path."""
    safe = hashlib.md5(poster_path.encode()).hexdigest()
    return _CACHE_DIR / f"{safe}.jpg"


async def _fetch_poster(poster_path: str) -> Optional[Path]:
    """Downloads a poster to local cache if not already present. Returns the local path."""
    if not poster_path:
        return None

    cache_path = _poster_cache_path(poster_path)
    if cache_path.exists():
        return cache_path

    url = f"{TMDB_BASE}/{POSTER_SIZE}{poster_path}"
    try:
        import urllib.request
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        def _download():
            req = urllib.request.Request(url, headers={"User-Agent": "CineVault/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
            cache_path.write_bytes(data)
            return cache_path

        return await asyncio.to_thread(_download)
    except Exception as e:
        logger.warning(f"Failed to fetch poster {poster_path}: {e}")
        return None


class MovieInspectorModal(ModalScreen[None]):
    """
    Movie Inspector Modal.
    Displays poster (left) + metadata markdown (right).
    """

    BINDINGS = [
        ("escape", "dismiss", "Close Modal"),
        ("r", "open_review", "Write Review"),
    ]

    def __init__(self, item: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.item = item

    def compose(self) -> ComposeResult:
        title = self.item.get("title", "Unknown Movie")
        year = self.item.get("year", "")
        year_str = f" ({year})" if year else ""

        container = Vertical(
            # Title as text inside the modal, not in the border
            Static(f"[ INSPECTOR ] {title.upper()}{year_str}", id="inspector-title"),

            # Content area: poster left, text right
            Horizontal(
                Vertical(
                    Static("Loading poster...", id="poster-placeholder"),
                    id="inspector-poster-box",
                ),
                VerticalScroll(
                    Markdown(self._build_markdown(), id="inspector-markdown"),
                    id="inspector-body",
                ),
                id="inspector-content",
            ),

            # Action buttons
            Horizontal(
                Button("[ WRITE REVIEW ]", id="btn-inspector-review", classes="-filled"),
                Button("[ CLOSE ]", id="btn-inspector-close", classes="-ghost"),
                id="inspector-buttons",
            ),
            id="inspector-container",
        )

        yield Container(container, id="modal-wrapper")

    def on_mount(self) -> None:
        """Load poster asynchronously after mount."""
        poster_path = self.item.get("poster_path")
        if poster_path:
            self.run_worker(self._load_poster(poster_path))
        else:
            placeholder = self.query_one("#poster-placeholder", Static)
            placeholder.update("No poster available")

    async def _load_poster(self, poster_path: str) -> None:
        """Fetch and display the poster image."""
        local_path = await _fetch_poster(poster_path)
        placeholder = self.query_one("#poster-placeholder", Static)

        if local_path and local_path.exists():
            try:
                from textual_image.widget import Image
                poster_box = self.query_one("#inspector-poster-box", Vertical)
                await placeholder.remove()
                poster_widget = Image(str(local_path))
                await poster_box.mount(poster_widget)
            except Exception as e:
                logger.warning(f"Could not render poster widget: {e}")
                placeholder.update("Poster unavailable")
        else:
            placeholder.update("Poster unavailable")

    def _build_markdown(self) -> str:
        title = self.item.get("title", "Unknown Movie")
        year = self.item.get("year", "")
        year_str = f" ({year})" if year else ""
        rating = f"★ {self.item['avg_rating']:.2f}" if self.item.get("avg_rating") else "★ Unrated"
        tier = self.item.get("tier", "Tier A")

        genres = ", ".join(self.item.get("genres", []))
        directors = ", ".join(self.item.get("directors", [])) or "N/A"
        actors = ", ".join(self.item.get("actors", [])) or "N/A"

        themes = ", ".join(self.item.get("themes", [])) if self.item.get("themes") else "N/A"
        tone = ", ".join(self.item.get("tone", [])) if self.item.get("tone") else "N/A"
        pacing = self.item.get("pacing") or "N/A"
        comp = ", ".join(self.item.get("comparable_films", [])) if self.item.get("comparable_films") else "N/A"
        overview = self.item.get("overview") or self.item.get("wiki_intro") or "No plot summary available."

        content_rating = self.item.get('content_rating', 'NR')
        return f"""# {title}{year_str}

{rating}  |  {tier}  |  {content_rating}

---

**Director(s):** {directors}
**Cast:** {actors}
**Genres:** {genres}

---

### Synopsis

{overview}

---

### Cinema Signals

- **Themes:** {themes}
- **Tone:** {tone}
- **Pacing:** {pacing}
- **Similar Films:** {comp}
"""

    def action_open_review(self) -> None:
        self.dismiss()
        from interface.tui.modals.review import ReviewSubmissionModal
        self.app.push_screen(ReviewSubmissionModal(self.item))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-inspector-close":
            self.dismiss()
        elif event.button.id == "btn-inspector-review":
            self.action_open_review()
