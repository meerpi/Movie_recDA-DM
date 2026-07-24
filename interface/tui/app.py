"""
interface/tui/app.py — Main Textual Application Entry Point for CineVault
"""

import sys
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interface.controller import CineVaultController
from interface.tui.screens.search import SearchScreen


CSS_THEME = """
Screen {
    background: #0d1117;
    color: #c9d1d9;
}

#app-banner {
    background: #161b22;
    color: #58a6ff;
    text-align: center;
    text-style: bold;
    padding: 1;
    border-bottom: heavy #30363d;
}

#query-bar {
    padding: 1;
    height: 5;
    background: #161b22;
}

#query-input {
    width: 80%;
    border: tall #30363d;
    background: #0d1117;
    color: #f0f6fc;
}

#btn-search {
    width: 20%;
    margin-left: 1;
    background: #238636;
    color: #ffffff;
    text-style: bold;
}

#controls-bar {
    padding: 0 1;
    height: 3;
    background: #161b22;
    align: left middle;
}

.compact-input {
    width: 10;
    margin-right: 2;
    background: #0d1117;
    color: #58a6ff;
}

#user-badge {
    color: #a5d6ff;
    text-style: bold;
    margin-left: 2;
}

#status-bar {
    padding: 0 1;
    height: 1;
    background: #21262d;
    color: #8b949e;
    text-style: italic;
}

#results-table {
    height: 1fr;
    border: solid #30363d;
    background: #0d1117;
}

/* Modals */
#modal-wrapper {
    align: center middle;
}

#inspector-container, #review-container, #onboard-container {
    width: 85;
    height: 30;
    background: #161b22;
    border: heavy #58a6ff;
    padding: 1 2;
}

#modal-header, #review-header, #onboard-header {
    background: #1f6feb;
    color: #ffffff;
    text-align: center;
    text-style: bold;
    padding: 1;
    margin-bottom: 1;
}

#inspector-body, #review-scroll-body, #onboard-scroll-body {
    height: 1fr;
    margin-bottom: 1;
}

#inspector-buttons, #review-buttons, #onboard-buttons {
    height: 3;
    align: center middle;
}

Button {
    margin: 0 1;
}
"""


class CineVaultApp(App):
    """
    Main Textual Application class for CineVault.
    Wires system controller, reactive TUI screens, and modal overlays.
    """

    TITLE = "CineVault TUI — Neural Movie Search & Recommendation Engine"
    CSS = CSS_THEME

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit App", show=True),
        Binding("ctrl+s", "focus_search", "Focus Search", show=True),
        Binding("ctrl+o", "open_onboarding", "Onboarding", show=True),
    ]

    def __init__(self, user_id: str = "default_user", controller: Optional[CineVaultController] = None, **kwargs):
        super().__init__(**kwargs)
        self.user_id = user_id
        self.controller = controller or CineVaultController(user_id=self.user_id)

    def on_mount(self) -> None:
        self.push_screen(SearchScreen())

    def action_focus_search(self) -> None:
        if isinstance(self.screen, SearchScreen):
            self.screen.action_focus_search()

    def action_open_onboarding(self) -> None:
        from interface.tui.modals.onboarding import ColdStartOnboardingModal
        self.push_screen(ColdStartOnboardingModal())


if __name__ == "__main__":
    app = CineVaultApp()
    app.run()
