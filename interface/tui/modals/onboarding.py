"""
interface/tui/modals/onboarding.py — Cold-Start Onboarding Wizard Modal Screen
"""

from typing import List
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, Static


class ColdStartOnboardingModal(ModalScreen[None]):
    """
    30-Second Cold-Start Onboarding Wizard Modal.
    """

    BINDINGS = [
        ("escape", "dismiss", "Cancel"),
    ]

    GENRES = ["Action", "Comedy", "Drama", "Horror", "Mystery", "Sci-Fi", "Thriller", "Western"]

    def compose(self) -> ComposeResult:
        genre_checkboxes = [Checkbox(g, value=(g in ["Sci-Fi", "Thriller"]), id=f"genre-{g.lower()}") for g in self.GENRES]

        container = Vertical(
            VerticalScroll(
                Label("1. Select Your Favorite Genres:"),
                Horizontal(*genre_checkboxes[:4], id="onboard-genres-1"),
                Horizontal(*genre_checkboxes[4:], id="onboard-genres-2"),
                Static("─" * 40),
                Label("2. Enter 2-3 Anchor Movie IDs (e.g., 58559, 79132, 1214):"),
                Input(value="58559, 79132", id="input-anchors"),
                Static("─" * 40),
                Label("3. Enter Dealbreaker Rules (e.g. 'No Slapstick', 'No Gore'):"),
                Input(value="No Slapstick", id="input-dealbreakers"),
                id="onboard-scroll-body"
            ),
            Horizontal(
                Button("SAVE PROFILE", id="btn-save-onboard", variant="success"),
                Button("CANCEL [ESC]", id="btn-cancel-onboard", variant="error"),
                id="onboard-buttons"
            ),
            id="onboard-container"
        )
        container.border_title = "▸ ONBOARDING WIZARD"
        container.border_subtitle = "cold-start profile"

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
        anchor_ids: List[int] = []
        for val in raw_anchors.split(","):
            if val.strip().isdigit():
                anchor_ids.append(int(val.strip()))

        raw_dealbreakers = self.query_one("#input-dealbreakers", Input).value
        dealbreakers = [d.strip() for d in raw_dealbreakers.split(",") if d.strip()]

        controller = getattr(self.app, "controller", None)
        if controller:
            controller.seed_cold_start(
                favorite_genres=fav_genres,
                anchor_movie_ids=anchor_ids,
                dealbreakers=dealbreakers
            )
            self.notify("✓ Cold-start profile preferences saved!", title="Profile Updated")

        self.dismiss()
