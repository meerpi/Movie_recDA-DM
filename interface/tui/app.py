"""
interface/tui/app.py — Main Textual Application Entry Point for CineVault
"""

import sys
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.theme import Theme

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interface.controller import CineVaultController
from interface.tui.screens.search import SearchScreen

PHOSPHOR_CONSOLE = Theme(
    name="phosphor-console",
    primary="#8AB4F8",         # Focus, selection, primary action
    secondary="#E8B339",      # Warning / caution only
    accent="#E8B339",         # Alias for warning
    foreground="#E3E3E6",     # on-surface
    background="#0B0B0D",     # App canvas
    surface="#141417",        # Level 1 panels
    panel="#1C1C20",          # Level 2 modals / raised
    success="#7CC894",        # Confirmations only
    warning="#E8B339",        # Caution only
    error="#E5534B",          # Destructive only
    dark=True,
)



class CineVaultApp(App):
    """
    Main Textual Application class for CineVault.
    Wires system controller, reactive TUI screens, and modal overlays.
    """

    TITLE = "CineVault — Neural Movie Search & Recommendation Engine"
    CSS_PATH = "cinevault.tcss"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit App", show=True),
        Binding("ctrl+s", "focus_search", "Focus Search", show=True),
        Binding("ctrl+o", "open_onboarding", "Onboarding", show=True),
        Binding("ctrl+p", "open_profile", "Profile", show=True),
    ]

    def __init__(self, user_id: str = "default_user", controller: Optional[CineVaultController] = None, **kwargs):
        super().__init__(**kwargs)
        self.user_id = user_id
        self.controller = controller or CineVaultController(user_id=self.user_id)

    def on_mount(self) -> None:
        self.register_theme(PHOSPHOR_CONSOLE)
        self.theme = "phosphor-console"
        self.push_screen(SearchScreen())


    def action_focus_search(self) -> None:
        if isinstance(self.screen, SearchScreen):
            self.screen.action_focus_search()

    def action_open_onboarding(self) -> None:
        from interface.tui.modals.onboarding import ColdStartOnboardingModal
        self.push_screen(ColdStartOnboardingModal())

    def action_open_profile(self) -> None:
        from interface.tui.screens.profile import ProfileScreen
        self.push_screen(ProfileScreen())


if __name__ == "__main__":
    app = CineVaultApp()
    app.run()
