"""
interface/tui/screens/profile.py — User Profile Screen

Full screen (not a modal) reachable via ctrl+p.
Two sections:
  1. Personalization — per-signal weight segmented controls
  2. Memory — editable list of inferred insights
"""

from typing import Any, Dict, List

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Header, Input, Label, Rule, Static


SIGNAL_LEVELS = ["off", "light", "balanced", "strong"]
SIGNAL_LABELS = {
    "watch_history": "Watch history",
    "ratings": "Ratings",
    "reviews": "Written reviews",
}


class ProfileScreen(Screen):
    """
    User Profile Screen — Personalization and Memory.
    """

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(
            Vertical(
                Static("[ PROFILE ]", id="profile-title"),
                Rule(),
                VerticalScroll(
                    # ── Personalization Section ──
                    Label("PERSONALIZATION", classes="section-title"),
                    Static(
                        "Control how strongly each signal influences your recommendations.",
                        classes="signal-label",
                    ),
                    Rule(),
                    *self._build_signal_rows(),
                    Rule(),

                    # ── Memory Section ──
                    Label("MEMORY", classes="section-title"),
                    Static(
                        "What the system has inferred about your taste. Each entry is individually deletable.",
                        classes="signal-label",
                    ),
                    Rule(),
                    Vertical(id="memory-list"),
                    Horizontal(
                        Input(
                            placeholder="Add a new insight (e.g. 'Prefers slow-burn character studies')...",
                            id="memory-add-input",
                        ),
                        Button("[ ADD ]", id="btn-add-memory", classes="-outlined"),
                        id="memory-add-bar",
                    ),
                    Rule(),
                    Button(
                        "[ CLEAR ALL MEMORY ]",
                        id="btn-clear-memory",
                        classes="-destructive",
                    ),
                    id="profile-scroll-body",
                ),

                # Bottom buttons
                Horizontal(
                    Button("[ BACK ]", id="btn-back-profile", classes="-ghost"),
                    id="profile-buttons",
                ),
                id="profile-container",
            ),
            id="profile-wrapper",
        )

    def _build_signal_rows(self) -> list:
        """Build the signal weight rows."""
        controller = getattr(self.app, "controller", None)
        weights = controller.get_signal_weights() if controller else {}

        rows = []
        for signal_key, signal_label in SIGNAL_LABELS.items():
            current_level = weights.get(signal_key, "balanced")
            buttons = []
            for level in SIGNAL_LEVELS:
                btn_id = f"seg-{signal_key}-{level}"
                btn = Button(
                    level.capitalize(),
                    id=btn_id,
                    classes="seg-button" + (" -active" if level == current_level else ""),
                )
                buttons.append(btn)

            rows.append(
                Horizontal(
                    Label(signal_label, classes="signal-label"),
                    *buttons,
                    classes="signal-row",
                )
            )
        return rows

    def on_mount(self) -> None:
        self._refresh_memory_list()

    def _refresh_memory_list(self) -> None:
        """Refresh the memory entries list from the controller."""
        controller = getattr(self.app, "controller", None)
        memory_list = self.query_one("#memory-list", Vertical)
        memory_list.remove_children()

        if controller:
            entries = controller.get_memory_entries()
            if entries:
                for idx, entry in enumerate(entries):
                    memory_list.mount(
                        Horizontal(
                            Static(f"  {entry}", classes="memory-text"),
                            Button("[x]", id=f"btn-del-mem-{idx}", classes="memory-delete"),
                            classes="memory-item",
                        )
                    )
            else:
                memory_list.mount(
                    Static("  No insights recorded yet.", classes="memory-text")
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        controller = getattr(self.app, "controller", None)

        if bid == "btn-back-profile":
            self.app.pop_screen()
            return

        if bid == "btn-clear-memory":
            if controller:
                controller.clear_all_memory()
                self._refresh_memory_list()
                self.notify("All memory entries cleared.", title="Memory Cleared")
            return

        if bid == "btn-add-memory":
            text_input = self.query_one("#memory-add-input", Input)
            text = text_input.value.strip()
            if text and controller:
                controller.add_memory_entry(text)
                text_input.value = ""
                self._refresh_memory_list()
                self.notify(f"Added: {text}", title="Memory Updated")
            return

        # Handle delete buttons
        if bid and bid.startswith("btn-del-mem-"):
            try:
                idx = int(bid.replace("btn-del-mem-", ""))
                if controller:
                    controller.delete_memory_entry(idx)
                    self._refresh_memory_list()
                    self.notify("Entry removed.", title="Memory Updated")
            except (ValueError, IndexError):
                pass
            return

        # Handle segmented control buttons
        if bid and bid.startswith("seg-"):
            parts = bid.split("-", 2)  # seg-{signal_key}-{level}
            if len(parts) >= 3:
                signal_key = parts[1]
                # The level is everything after the second dash
                level = bid.split(f"seg-{signal_key}-", 1)[1]
                if controller and level in SIGNAL_LEVELS:
                    controller.set_signal_weight(signal_key, level)
                    # Update visual state: remove -active from siblings, add to this one
                    for l in SIGNAL_LEVELS:
                        sibling_id = f"seg-{signal_key}-{l}"
                        try:
                            sibling = self.query_one(f"#{sibling_id}", Button)
                            sibling.remove_class("-active")
                        except Exception:
                            pass
                    event.button.add_class("-active")
