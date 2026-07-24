"""
interface/tui/modals/inspector.py — IMDb-Style Movie Inspector Modal Screen
"""

from typing import Any, Dict, Optional
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static, Markdown


class MovieInspectorModal(ModalScreen[None]):
    """
    IMDb-Style Movie Inspector Modal Screen.
    Displays hydrated metadata, TMDB poster URL, themes, tone, pacing, cast, and director.
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
        rating = f"★ {self.item['avg_rating']:.2f}" if self.item.get("avg_rating") else "★ Unrated"
        tier = self.item.get("tier", "Tier A")

        genres = ", ".join(self.item.get("genres", []))
        directors = ", ".join(self.item.get("directors", [])) or "N/A"
        actors = ", ".join(self.item.get("actors", [])) or "N/A"
        poster_path = self.item.get("poster_path")
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "No Poster Path Available"

        themes = ", ".join(self.item.get("themes", [])) if self.item.get("themes") else "N/A"
        tone = ", ".join(self.item.get("tone", [])) if self.item.get("tone") else "N/A"
        pacing = self.item.get("pacing") or "N/A"
        comp = ", ".join(self.item.get("comparable_films", [])) if self.item.get("comparable_films") else "N/A"
        overview = self.item.get("overview") or self.item.get("wiki_intro") or "No plot summary available."

        markdown_body = f"""# {title}{year_str}
**Rating:** `{rating}` | **Tier:** `{tier}` | **Content Rating:** `{self.item.get('content_rating', 'NR')}`

---

### TMDB Movie Poster
**Poster URL:** [{poster_url}]({poster_url})

```
┌──────────────────────────────────────┐
│            CINEVAULT POSTER          │
│                                      │
│         {title[:20]:^20}         │
│                                      │
│           ★ {rating:^12}          │
│                                      │
└──────────────────────────────────────┘
```

---

### Overview & Synopsis
{overview}

---

### Cast & Crew
- **Director(s):** {directors}
- **Cast:** {actors}
- **Genres:** {genres}

---

### Qualitative Attributes & Cinema Signals
- **Themes:** {themes}
- **Tone:** {tone}
- **Pacing:** {pacing}
- **Similar Films:** {comp}
"""

        container = Vertical(
            VerticalScroll(
                Markdown(markdown_body, id="inspector-markdown"),
                id="inspector-body"
            ),
            Horizontal(
                Button("WRITE REVIEW [R]", id="btn-inspector-review", variant="success"),
                Button("CLOSE [ESC]", id="btn-inspector-close", variant="primary"),
                id="inspector-buttons"
            ),
            id="inspector-container"
        )
        container.border_title = f"▸ INSPECTOR · {title.upper()}"
        container.border_subtitle = "movie metadata"

        yield Container(
            container,
            id="modal-wrapper"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-inspector-close":
            self.dismiss()
        elif event.button.id == "btn-inspector-review":
            self.dismiss()
            # Post message to app to open review modal for this item
            from interface.tui.modals.review import ReviewSubmissionModal
            self.app.push_screen(ReviewSubmissionModal(self.item))
