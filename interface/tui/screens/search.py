"""
interface/tui/screens/search.py — Unified Search Hub Screen for CineVault TUI
"""

import asyncio
from typing import Any, Dict, List
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Checkbox, DataTable, Header, Input, Label, Static


class SearchScreen(Screen):
    """
    Unified Search Hub Screen.
    Contains Query Bar, Personalization Dial (λ), Exclude Watched toggle, and Results DataTable.
    """

    BINDINGS = [
        ("ctrl+s", "focus_search", "Focus Search"),
        ("ctrl+o", "open_onboarding", "Onboarding"),
        ("enter", "inspect_selected", "Inspect Movie"),
        ("r", "review_selected", "Review Movie"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_results: List[Dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(
            Vertical(
                # Title Banner
                Static("🍿 CINEVAULT :: RETRO NEURAL RECOMMENDATION ENGINE", id="app-banner"),
                
                # Query Bar Container
                Horizontal(
                    Input(
                        placeholder="🔍 Enter natural language prompt (e.g. 'dark mind-bending sci-fi thriller')...",
                        id="query-input"
                    ),
                    Button("🔍 SEARCH", id="btn-search", variant="primary"),
                    id="query-bar"
                ),
                
                # Interactive Controls Bar
                Horizontal(
                    Label("λ Personalization Dial:"),
                    Input(value="0.7", id="input-lambda", classes="compact-input"),
                    Checkbox("Exclude Watched", value=True, id="chk-exclude-watched"),
                    Button("⚡ Onboarding Wizard", id="btn-onboard", variant="default"),
                    Static("👤 Active User: [default_user]", id="user-badge"),
                    id="controls-bar"
                ),

                # Loading Status Bar
                Static("Ready. Type a query above and press Enter to search.", id="status-bar"),

                # Live Results DataTable
                DataTable(id="results-table", cursor_type="row"),
                id="search-container"
            ),
            id="main-screen-wrapper"
        )

    def on_mount(self) -> None:
        table = self.query_one("#results-table", DataTable)
        table.add_column("#", key="rank")
        table.add_column("Title (Year)", key="title")
        table.add_column("IMDb Rating", key="rating")
        table.add_column("Genres", key="genres")
        table.add_column("Score", key="score")
        table.add_column("RRF Pool Rank", key="rrf_rank")
        
        # Default focus on query input
        self.query_one("#query-input", Input).focus()

    def action_focus_search(self) -> None:
        self.query_one("#query-input", Input).focus()

    def action_open_onboarding(self) -> None:
        from interface.tui.modals.onboarding import ColdStartOnboardingModal
        self.app.push_screen(ColdStartOnboardingModal())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-search":
            self.run_search()
        elif event.button.id == "btn-onboard":
            self.action_open_onboarding()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "query-input":
            self.run_search()
        elif event.input.id == "input-lambda":
            self.update_lambda_setting()

    def update_lambda_setting(self) -> None:
        val_str = self.query_one("#input-lambda", Input).value
        try:
            val = float(val_str)
            val = max(0.0, min(1.0, val))
            controller = getattr(self.app, "controller", None)
            if controller:
                controller.set_lambda(val)
                self.notify(f"Personalization λ set to {val:.2f}", title="Settings Updated")
        except ValueError:
            pass

    def run_search(self) -> None:
        query_text = self.query_one("#query-input", Input).value.strip()
        if not query_text:
            return

        status = self.query_one("#status-bar", Static)
        status.update(f"⏳ Searching CineVault for '{query_text}'...")

        self.update_lambda_setting()

        # Update exclude watched setting
        chk_watched = self.query_one("#chk-exclude-watched", Checkbox).value
        controller = getattr(self.app, "controller", None)
        if controller:
            controller.set_exclude_watched(chk_watched)

        # Run non-blocking search task
        self.run_worker(self._async_search_task(query_text), exclusive=True)

    async def _async_search_task(self, query_text: str) -> None:
        controller = getattr(self.app, "controller", None)
        if not controller:
            return

        t0 = asyncio.get_event_loop().time()
        results = await controller.search_async(query_text, top_k=10)
        elapsed_ms = (asyncio.get_event_loop().time() - t0) * 1000

        self.current_results = results
        self.populate_results(results, query_text, elapsed_ms)

    def populate_results(self, results: List[Dict[str, Any]], query_text: str, elapsed_ms: float) -> None:
        table = self.query_one("#results-table", DataTable)
        table.clear()

        status = self.query_one("#status-bar", Static)
        status.update(f"✓ Found {len(results)} recommendations for '{query_text}' in {elapsed_ms:.1f}ms.")

        for idx, item in enumerate(results, 1):
            final_rank = str(item.get("final_rank", idx))
            title = f"{item.get('title', 'Unknown')} ({item.get('year', '')})"
            rating = f"★ {item['avg_rating']:.2f}" if item.get("avg_rating") else "★ Unrated"
            genres = ", ".join(item.get("genres", [])[:3])
            score = f"{item.get('final_score', 0.0):.4f}"
            rrf_rank = f"#{item.get('rrf_rank', '-')}"

            table.add_row(final_rank, title, rating, genres, score, rrf_rank, key=str(item.get("movie_id", idx)))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.action_inspect_selected()

    def action_inspect_selected(self) -> None:
        selected_item = self.get_selected_movie_item()
        if selected_item:
            from interface.tui.modals.inspector import MovieInspectorModal
            self.app.push_screen(MovieInspectorModal(selected_item))

    def action_review_selected(self) -> None:
        selected_item = self.get_selected_movie_item()
        if selected_item:
            from interface.tui.modals.review import ReviewSubmissionModal
            self.app.push_screen(ReviewSubmissionModal(selected_item))

    def get_selected_movie_item(self) -> Optional[Dict[str, Any]]:
        table = self.query_one("#results-table", DataTable)
        if table.row_count == 0 or table.cursor_row < 0:
            return None

        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        movie_id = int(row_key.value) if str(row_key.value).isdigit() else None

        if movie_id:
            for item in self.current_results:
                if item.get("movie_id") == movie_id:
                    return item

        # Fallback to current index
        if 0 <= table.cursor_row < len(self.current_results):
            return self.current_results[table.cursor_row]

        return None
