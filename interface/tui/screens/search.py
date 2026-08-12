"""
interface/tui/screens/search.py — Search & Recommendation Screen for CineVault TUI
"""

import asyncio
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Header, Input, Rule, Static


class SearchScreen(Screen):

    BINDINGS = [
        ("ctrl+s", "focus_search", "Focus Search"),
        ("enter", "inspect_selected", "Inspect Movie"),
        ("r", "review_selected", "Review Movie"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_results = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(
            Vertical(
                # Title — plain text, no decorative border box
                Static("CINEVAULT", id="app-title"),
                Rule(id="app-title-rule"),

                # Query Bar
                Horizontal(
                    Input(
                        placeholder="Search for movies (e.g. 'dark slow tension horror', 'mind-bending sci-fi')...",
                        id="query-input"
                    ),
                    Button("SEARCH", id="btn-search"),
                    id="query-bar"
                ),

                # Status Bar
                Static("Ready.", id="status-bar"),

                # Recommendations DataTable
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

        # Default focus on query input
        self.query_one("#query-input", Input).focus()

        # Load default top 10 recommendations based on user profile on screen mount
        self.run_search(initial_load=True)

    def action_focus_search(self) -> None:
        self.query_one("#query-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-search":
            self.run_search()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "query-input":
            self.run_search()

    def run_search(self, initial_load: bool = False) -> None:
        query_text = self.query_one("#query-input", Input).value.strip()

        status = self.query_one("#status-bar", Static)
        if query_text:
            status.update(f"Searching CineVault for '{query_text}'...")
        else:
            status.update("Loading top 10 recommended movies based on your profile...")

        self.run_worker(self._async_search_task(query_text, initial_load), exclusive=True)

    async def _async_search_task(self, query_text: str, initial_load: bool = False) -> None:
        controller = getattr(self.app, "controller", None)
        if not controller:
            return

        t0 = asyncio.get_event_loop().time()
        results = await controller.search_async(query_text, top_k=10)
        elapsed_ms = (asyncio.get_event_loop().time() - t0) * 1000

        self.current_results = results
        self.populate_results(results, query_text, elapsed_ms, initial_load)

    def populate_results(
        self,
        results,
        query_text: str,
        elapsed_ms: float,
        initial_load: bool = False
    ) -> None:
        table = self.query_one("#results-table", DataTable)
        table.clear()

        status = self.query_one("#status-bar", Static)
        user_id = getattr(getattr(self.app, "controller", None), "user_id", "default_user")

        if query_text:
            status.update(f"Found {len(results)} recommendations for '{query_text}' in {elapsed_ms:.1f}ms.")
        else:
            status.update(f"Top 10 Recommended Movies for [{user_id}] (Profile-Based)")

        for idx, item in enumerate(results, 1):
            final_rank = str(item.get("final_rank", idx))
            title = f"{item.get('title', 'Unknown')} ({item.get('year', '')})"
            rating = f"★ {item['avg_rating']:.2f}" if item.get("avg_rating") else "Unrated"
            genres = ", ".join(item.get("genres", [])[:3])
            score = f"{item.get('final_score', 0.0):.4f}"

            table.add_row(final_rank, title, rating, genres, score, key=str(item.get("movie_id", idx)))

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

    def get_selected_movie_item(self):
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
