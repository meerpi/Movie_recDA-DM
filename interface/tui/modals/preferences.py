"""
interface/tui/modals/preferences.py — User Profile Preferences & Data Controls Modal Screen
"""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, Rule, Static


class ProfilePreferencesModal(ModalScreen[None]):

    BINDINGS = [
        ("escape", "dismiss", "Close"),
    ]

    def compose(self) -> ComposeResult:
        controller = getattr(self.app, "controller", None)
        profile = controller.profile if controller else None

        gw = f"{profile.genre_weight:.2f}" if profile else "0.60"
        dw = f"{profile.director_weight:.2f}" if profile else "0.80"
        aw = f"{profile.actor_weight:.2f}" if profile else "0.50"
        dealbreakers_str = ", ".join(profile.dealbreakers) if profile and profile.dealbreakers else ""
        query_disabled = "query_history" in (profile.disabled_signals if profile else [])

        top_genres = ", ".join(profile.get_top_genres(5)) if profile else "None"
        top_directors = ", ".join(profile.get_top_directors(5)) if profile else "None"

        container = Vertical(
            VerticalScroll(
                Static(f"Active User: {profile.user_id if profile else 'N/A'}", id="pref-user-heading"),
                Rule(),
                Label("[ AFFINITIES ]"),
                Static(f"  Top Genres   : {top_genres}"),
                Static(f"  Top Directors: {top_directors}"),
                Rule(),
                Label("[ SENSITIVITY WEIGHTS ] (0.0 to 1.0)"),
                Horizontal(
                    Label("Genre:"), Input(value=gw, id="input-pref-gw", classes="compact-input"),
                    Label("Director:"), Input(value=dw, id="input-pref-dw", classes="compact-input"),
                    Label("Actor:"), Input(value=aw, id="input-pref-aw", classes="compact-input"),
                    id="pref-weights-row"
                ),
                Rule(),
                Label("[ DEALBREAKERS ] (comma-separated)"),
                Input(value=dealbreakers_str, id="input-pref-dealbreakers"),
                Rule(),
                Label("[ PRIVACY & DATA ]"),
                Checkbox("Disable search query history collection", value=query_disabled, id="chk-disable-query-history"),
                Button("Clear Search Query History", id="btn-clear-history", classes="-destructive"),
                id="pref-scroll-body"
            ),
            Horizontal(
                Button("SAVE PREFERENCES", id="btn-save-pref", classes="-filled"),
                Button("CLOSE", id="btn-close-pref", classes="-ghost"),
                id="pref-buttons"
            ),
            id="pref-container"
        )

        yield Container(
            container,
            id="modal-wrapper"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        controller = getattr(self.app, "controller", None)
        if event.button.id == "btn-close-pref":
            self.dismiss()
        elif event.button.id == "btn-clear-history":
            if controller:
                controller.clear_query_history()
                self.notify("Search query history cleared.", title="Data Controls")
        elif event.button.id == "btn-save-pref":
            self._save_preferences()

    def _save_preferences(self) -> None:
        controller = getattr(self.app, "controller", None)
        if not controller or not controller.profile:
            self.dismiss()
            return

        p = controller.profile

        # Save weights
        try:
            p.genre_weight = max(0.0, min(1.0, float(self.query_one("#input-pref-gw", Input).value)))
            p.director_weight = max(0.0, min(1.0, float(self.query_one("#input-pref-dw", Input).value)))
            p.actor_weight = max(0.0, min(1.0, float(self.query_one("#input-pref-aw", Input).value)))
        except ValueError:
            pass

        # Save dealbreakers
        raw_db = self.query_one("#input-pref-dealbreakers", Input).value
        p.dealbreakers = [d.strip().lower() for d in raw_db.split(",") if d.strip()]

        # Save privacy checkboxes
        chk_query = self.query_one("#chk-disable-query-history", Checkbox).value
        controller.set_signal_enabled("query_history", enabled=not chk_query)

        self.notify("Profile preferences saved.", title="Preferences Updated")
        self.dismiss()
