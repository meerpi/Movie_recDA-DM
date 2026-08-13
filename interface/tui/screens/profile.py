"""
interface/tui/screens/profile.py — User Profile Screen for CineVault TUI
"""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Header, Input, Label, Rule, Static, Switch

from user_profile.schema import UserPreset


LAMBDA_LEVELS = ["off", "subtle", "balanced", "strong", "maximum"]
LAMBDA_MAP = UserPreset.LAMBDA_LEVELS  # label → float

SIGNAL_LABELS = {
    "watch_history": "Watch History",
    "ratings": "Ratings",
    "reviews": "Written Reviews",
}

# Unicode bar chart characters
_BAR_CHARS = " ▏▎▍▌▋▊▉█"


def _unicode_bar(value, max_value, width=12):
    """Render a unicode bar chart string of given character width."""
    if max_value <= 0:
        return " " * width
    ratio = max(0.0, min(1.0, value / max_value))
    full_blocks = int(ratio * width)
    remainder = (ratio * width) - full_blocks
    partial_idx = int(remainder * (len(_BAR_CHARS) - 1))
    bar = "█" * full_blocks
    if full_blocks < width:
        bar += _BAR_CHARS[partial_idx]
        bar += " " * (width - full_blocks - 1)
    return bar


class ProfileScreen(Screen):

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
                    # ── Section A: Identity ──
                    self._build_identity_section(),
                    Rule(),

                    # ── Section B: Personalization ──
                    Label("PERSONALIZATION", classes="section-title"),
                    Static(
                        "How much should your taste profile influence search results?",
                        classes="section-hint",
                    ),
                    Rule(),
                    # Global λ segmented control
                    Label("Influence Level", classes="signal-label"),
                    Horizontal(
                        *self._build_lambda_buttons(),
                        classes="signal-row",
                    ),
                    Rule(),
                    # Per-signal toggles
                    Label("Active Signals", classes="signal-label"),
                    Static(
                        "Toggle which data sources feed into your recommendations.",
                        classes="section-hint",
                    ),
                    *self._build_signal_toggles(),
                    Rule(),

                    # ── Section C: Presets ──
                    Label("PRESETS", classes="section-title"),
                    Static(
                        "Save and switch between different recommendation modes.",
                        classes="section-hint",
                    ),
                    Rule(),
                    Vertical(id="preset-list"),
                    Horizontal(
                        Input(
                            placeholder="New preset name (e.g. Arthouse Mood)...",
                            id="preset-name-input",
                        ),
                        Button("SAVE AS PRESET", id="btn-save-preset", classes="-outlined"),
                        Button("CLEAR PRESET", id="btn-clear-preset", classes="-ghost"),
                        id="preset-add-bar",
                    ),
                    Rule(),

                    # ── Section D: Learned Taste ──
                    Label("LEARNED TASTE", classes="section-title"),
                    Static(
                        "What the system has learned about your preferences (read-only).",
                        classes="section-hint",
                    ),
                    Rule(),
                    Horizontal(
                        Vertical(id="taste-genres", classes="taste-column"),
                        Vertical(id="taste-directors", classes="taste-column"),
                        Vertical(id="taste-actors", classes="taste-column"),
                        id="taste-columns",
                    ),
                    Rule(),

                    # ── Section E: Memory ──
                    Label("MEMORY", classes="section-title"),
                    Static(
                        "What the system has inferred about your taste. Each entry is individually deletable.",
                        classes="section-hint",
                    ),
                    Rule(),
                    Vertical(id="memory-list"),
                    Horizontal(
                        Input(
                            placeholder="Add a new insight (e.g. 'Prefers slow-burn character studies')...",
                            id="memory-add-input",
                        ),
                        Button("ADD", id="btn-add-memory", classes="-outlined"),
                        id="memory-add-bar",
                    ),
                    Rule(),

                    # ── Section F: Danger Zone ──
                    Label("DANGER ZONE", classes="section-title danger-title"),
                    Static(
                        "Irreversible actions that reset parts of your profile.",
                        classes="section-hint",
                    ),
                    Rule(),
                    Horizontal(
                        Button(
                            "CLEAR ALL MEMORY",
                            id="btn-clear-memory",
                            classes="-destructive",
                        ),
                        Button(
                            "CLEAR RATING HISTORY",
                            id="btn-clear-ratings",
                            classes="-destructive",
                        ),
                        id="danger-buttons",
                    ),
                    id="profile-scroll-body",
                ),

                # Bottom buttons
                Horizontal(
                    Button("BACK", id="btn-back-profile", classes="-ghost"),
                    id="profile-buttons",
                ),
                id="profile-container",
            ),
            id="profile-wrapper",
        )

    # ── Build helpers ──

    def _build_identity_section(self):
        controller = getattr(self.app, "controller", None)
        if controller:
            p = controller.profile
            num_rated = len(p.rating_log)
            watchlist_size = len(p.watchlist)
            sessions = p.session_count
            user_id = p.user_id
            preset_name = controller.active_preset_name
        else:
            num_rated = 0
            watchlist_size = 0
            sessions = 0
            user_id = "unknown"
            preset_name = "Default"

        identity_text = (
            f"  User: {user_id}      "
            f"Active Preset: {preset_name}\n"
            f"  Movies Rated: {num_rated}   "
            f"Watchlist: {watchlist_size}   "
            f"Sessions: {sessions}"
        )
        return Static(identity_text, id="identity-card")

    def _build_lambda_buttons(self):
        controller = getattr(self.app, "controller", None)
        current_lambda = controller.lambda_personalization if controller else 0.7
        current_label = UserPreset.lambda_to_label(current_lambda)

        buttons = []
        for level in LAMBDA_LEVELS:
            btn_id = f"lambda-{level}"
            btn = Button(
                level.capitalize(),
                id=btn_id,
                classes="seg-button" + (" -active" if level == current_label else ""),
            )
            buttons.append(btn)
        return buttons

    def _build_signal_toggles(self):
        controller = getattr(self.app, "controller", None)
        disabled_signals = controller.profile.disabled_signals if controller else []

        rows = []
        for signal_key, signal_label in SIGNAL_LABELS.items():
            is_on = signal_key not in disabled_signals
            row = Horizontal(
                Label(signal_label, classes="signal-label"),
                Switch(value=is_on, id=f"toggle-{signal_key}"),
                classes="signal-toggle-row",
            )
            rows.append(row)
        return rows

    # ── Lifecycle ──

    def on_mount(self) -> None:
        self._refresh_preset_list()
        self._refresh_taste_display()
        self._refresh_memory_list()

    # ── Preset list ──

    def _refresh_preset_list(self) -> None:
        controller = getattr(self.app, "controller", None)
        preset_list = self.query_one("#preset-list", Vertical)
        preset_list.remove_children()

        if not controller:
            return

        presets = controller.list_presets()
        if not presets:
            preset_list.mount(
                Static("  No presets saved yet. Current settings use profile defaults.", classes="preset-empty")
            )
            return

        for preset in presets:
            indicator = "►" if preset.is_active else " "
            signals_str = ", ".join(
                label for key, label in SIGNAL_LABELS.items()
                if preset.signals.get(key, True)
            ) or "None"
            lambda_label = UserPreset.lambda_to_label(preset.lambda_val).capitalize()

            info = f"  {indicator}  {preset.name:<20s}  │  λ={lambda_label}  │  Signals: {signals_str}"

            row = Horizontal(
                Static(info, classes="preset-row-text"),
                Button(
                    "ACTIVE" if preset.is_active else "USE",
                    id=f"btn-use-preset-{preset.name}",
                    classes="preset-use-btn" + (" -active-profile" if preset.is_active else ""),
                    disabled=preset.is_active,
                ),
                Button(
                    "DEL",
                    id=f"btn-del-preset-{preset.name}",
                    classes="memory-delete",
                ),
                classes="preset-row" + (" preset-row-active" if preset.is_active else ""),
            )
            preset_list.mount(row)

    # ── Learned taste display ──

    def _refresh_taste_display(self) -> None:
        controller = getattr(self.app, "controller", None)
        if not controller:
            return

        p = controller.profile
        self._populate_taste_column("#taste-genres", "Top Genres", p.genre_affinity)
        self._populate_taste_column("#taste-directors", "Top Directors", p.director_affinity)
        self._populate_taste_column("#taste-actors", "Top Actors", p.actor_affinity)

    def _populate_taste_column(self, container_id, title, affinity_dict, max_items=5):
        container = self.query_one(container_id, Vertical)
        container.remove_children()
        container.mount(Label(title, classes="taste-column-title"))

        if not affinity_dict:
            container.mount(Static("  No data yet.", classes="taste-empty"))
            return

        sorted_items = sorted(affinity_dict.items(), key=lambda x: x[1], reverse=True)[:max_items]
        max_val = max(abs(v) for _, v in sorted_items) if sorted_items else 1.0

        for name, val in sorted_items:
            display_name = name[:14] if len(name) > 14 else name
            bar = _unicode_bar(abs(val), max_val, width=10)
            sign = "+" if val >= 0 else "-"
            line = f"  {display_name:<14s} {bar} {sign}{abs(val):.1f}"
            container.mount(Static(line, classes="taste-entry"))

    # ── Memory list ──

    def _refresh_memory_list(self) -> None:
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

    # ── Event handlers ──

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        controller = getattr(self.app, "controller", None)

        if bid == "btn-back-profile":
            self.app.pop_screen()
            return

        # Lambda segmented control
        if bid and bid.startswith("lambda-"):
            level = bid.replace("lambda-", "")
            if level in LAMBDA_MAP and controller:
                new_lambda = LAMBDA_MAP[level]
                controller.set_lambda(new_lambda)
                # Update visual state
                for l in LAMBDA_LEVELS:
                    try:
                        sibling = self.query_one(f"#lambda-{l}", Button)
                        sibling.remove_class("-active")
                    except Exception:
                        pass
                event.button.add_class("-active")
                self._update_status_chip()
                self.notify(f"Influence set to {level.capitalize()} (λ={new_lambda})", title="Personalization")
            return

        # Memory controls
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

        # Clear rating history
        if bid == "btn-clear-ratings":
            if controller:
                controller.profile.rating_log.clear()
                controller.profile.watch_history.clear()
                controller.profile.highly_rated.clear()
                controller.profile.poorly_rated.clear()
                controller._save_profile_safe()
                # Refresh identity section
                try:
                    identity = self.query_one("#identity-card", Static)
                    identity.update(self._build_identity_text())
                except Exception:
                    pass
                self._refresh_taste_display()
                self.notify("Rating history cleared.", title="History Cleared")
            return

        # Preset controls
        if bid == "btn-save-preset":
            self._save_current_as_preset()
            return

        if bid == "btn-clear-preset":
            if controller:
                controller.deactivate_preset()
                self._refresh_preset_list()
                self._update_status_chip()
                # Refresh identity
                try:
                    identity = self.query_one("#identity-card", Static)
                    identity.update(self._build_identity_text())
                except Exception:
                    pass
                self.notify("Preset cleared. Using profile defaults.", title="Preset Cleared")
            return

        if bid and bid.startswith("btn-use-preset-"):
            name = bid.replace("btn-use-preset-", "")
            if controller:
                controller.activate_preset(name)
                self._refresh_preset_list()
                self._sync_ui_from_controller()
                self._update_status_chip()
                self.notify(f"Activated preset: {name}", title="Preset Activated")
            return

        if bid and bid.startswith("btn-del-preset-"):
            name = bid.replace("btn-del-preset-", "")
            if controller:
                controller.delete_preset(name)
                self._refresh_preset_list()
                self._update_status_chip()
                self.notify(f"Deleted preset: {name}", title="Preset Deleted")
            return

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle per-signal toggle changes."""
        controller = getattr(self.app, "controller", None)
        if not controller:
            return

        switch_id = event.switch.id
        if switch_id and switch_id.startswith("toggle-"):
            signal_key = switch_id.replace("toggle-", "")
            controller.set_signal_enabled(signal_key, event.value)

    # ── Preset save ──

    def _save_current_as_preset(self) -> None:
        controller = getattr(self.app, "controller", None)
        if not controller:
            return

        name_input = self.query_one("#preset-name-input", Input)
        name = name_input.value.strip()

        if not name:
            self.notify("Enter a preset name.", title="Error", severity="error")
            return

        # Build signals dict from current toggle states
        signals = {}
        for signal_key in SIGNAL_LABELS:
            try:
                toggle = self.query_one(f"#toggle-{signal_key}", Switch)
                signals[signal_key] = toggle.value
            except Exception:
                signals[signal_key] = True

        lambda_val = controller.lambda_personalization

        # Check if preset already exists — update it
        existing = [p.name for p in controller.list_presets()]
        try:
            if name in existing:
                controller.update_preset(name, lambda_val, signals)
                controller.activate_preset(name)
                self.notify(f"Updated and activated preset: {name}", title="Preset Updated")
            else:
                controller.create_preset(name, lambda_val, signals)
                controller.activate_preset(name)
                self.notify(f"Created and activated preset: {name}", title="Preset Created")
        except Exception as e:
            self.notify(f"Error saving preset: {e}", title="Error", severity="error")
            return

        name_input.value = ""
        self._refresh_preset_list()
        self._update_status_chip()
        # Refresh identity
        try:
            identity = self.query_one("#identity-card", Static)
            identity.update(self._build_identity_text())
        except Exception:
            pass

    # ── UI sync helpers ──

    def _sync_ui_from_controller(self) -> None:
        """Update the λ buttons and signal toggles to match the controller state."""
        controller = getattr(self.app, "controller", None)
        if not controller:
            return

        # Update λ buttons
        current_label = UserPreset.lambda_to_label(controller.lambda_personalization)
        for level in LAMBDA_LEVELS:
            try:
                btn = self.query_one(f"#lambda-{level}", Button)
                if level == current_label:
                    btn.add_class("-active")
                else:
                    btn.remove_class("-active")
            except Exception:
                pass

        # Update signal toggles
        for signal_key in SIGNAL_LABELS:
            try:
                toggle = self.query_one(f"#toggle-{signal_key}", Switch)
                toggle.value = signal_key not in controller.profile.disabled_signals
            except Exception:
                pass

    def _build_identity_text(self):
        controller = getattr(self.app, "controller", None)
        if controller:
            p = controller.profile
            return (
                f"  User: {p.user_id}      "
                f"Active Preset: {controller.active_preset_name}\n"
                f"  Movies Rated: {len(p.rating_log)}   "
                f"Watchlist: {len(p.watchlist)}   "
                f"Sessions: {p.session_count}"
            )
        return "  No profile loaded."

    def _update_status_chip(self) -> None:
        """Try to update the status chip on any visible screen."""
        try:
            chip = self.app.query_one("#status-chip", Static)
            controller = getattr(self.app, "controller", None)
            if controller and chip:
                chip.update(
                    f"{controller.user_id}  │  "
                    f"{controller.lambda_label.capitalize()}  │  "
                    f"{controller.active_preset_name}"
                )
        except Exception:
            pass
