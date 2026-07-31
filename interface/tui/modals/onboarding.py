"""
interface/tui/modals/onboarding.py — Cold-Start Onboarding Wizard Modal Screen
"""

from pathlib import Path
from typing import List
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, Rule, Static
from user_profile.identity import resolve_anchor_tokens

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

class ColdStartOnboardingModal(ModalScreen[None]):
    """
    30-Second Cold-Start Onboarding Wizard Modal.
    """

    BINDINGS = [
        ("escape", "dismiss", "Cancel"),
    ]

    GENRES = ["Action", "Comedy", "Drama", "Horror", "Mystery", "Sci-Fi", "Thriller", "Western"]

    def compose(self) -> ComposeResult:
        genre_checkboxes = [Checkbox(g, value=False, id=f"genre-{g.lower()}") for g in self.GENRES]

        container = Vertical(
            Static("[ ONBOARDING ]", id="onboard-title"),
            VerticalScroll(
                Label("[ STEP 1 ]  Select favorite genres:"),
                Horizontal(*genre_checkboxes[:4], id="onboard-genres-1"),
                Horizontal(*genre_checkboxes[4:], id="onboard-genres-2"),
                Rule(),
                Label("[ STEP 2 ]  Enter 2-3 anchor movie IDs or titles:"),
                Input(value="58559, 79132", id="input-anchors"),
                Rule(),
                Label("[ STEP 3 ]  Dealbreaker rules (e.g. 'No Slapstick', 'No Gore'):"),
                Input(value="No Slapstick", id="input-dealbreakers"),
                id="onboard-scroll-body"
            ),
            Horizontal(
                Button("[ SAVE PROFILE ]", id="btn-save-onboard", classes="-filled"),
                Button("[ CANCEL ]", id="btn-cancel-onboard", classes="-ghost"),
                id="onboard-buttons"
            ),
            id="onboard-container"
        )

        yield Container(
            container,
            id="modal-wrapper"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel-onboard":
            self.dismiss()
        elif event.button.id == "btn-save-onboard":
            self._save_onboarding()

    def _save_onboarding(self) -> None:
        fav_genres: List[str] = []
        for g in self.GENRES:
            chk = self.query_one(f"#genre-{g.lower()}", Checkbox)
            if chk.value:
                fav_genres.append(g)

        raw_anchors = self.query_one("#input-anchors", Input).value
        db_path = _PROJECT_ROOT / "db" / "cinevault.db"

        anchor_ids, anchor_warnings = resolve_anchor_tokens(raw_anchors, db_path)
        for msg in anchor_warnings:
            self.notify(msg, title="Anchor Lookup", severity="warning")

        raw_dealbreakers = self.query_one("#input-dealbreakers", Input).value
        dealbreakers = [d.strip() for d in raw_dealbreakers.split(",") if d.strip()]

        controller = getattr(self.app, "controller", None)
        if controller:
            controller.seed_cold_start(
                favorite_genres=fav_genres,
                anchor_movie_ids=anchor_ids,
                dealbreakers=dealbreakers
            )
            self.notify("Profile preferences saved.", title="Profile Updated")

        self.dismiss()
