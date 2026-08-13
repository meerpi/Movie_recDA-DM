"""
interface/tui/screens/profile_switcher.py — Profile Switcher Screen for CineVault TUI
"""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Header, Input, Label, Rule, Static


class ProfileSwitcherScreen(Screen):

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(
            Vertical(
                Static("[ PROFILES ]", id="switcher-title"),
                Rule(),
                VerticalScroll(
                    Label("SAVED PROFILES", classes="section-title"),
                    Static(
                        "Switch between user profiles or create a new one.",
                        classes="switcher-hint",
                    ),
                    Rule(),
                    Vertical(id="profile-list"),
                    Rule(),
                    # New profile creation area
                    Label("CREATE NEW PROFILE", classes="section-title"),
                    Horizontal(
                        Input(
                            placeholder="Enter a new profile name (e.g. alice, bob)...",
                            id="new-profile-input",
                        ),
                        Button("CREATE", id="btn-create-profile", classes="-outlined"),
                        id="new-profile-bar",
                    ),
                    id="switcher-scroll-body",
                ),
                # Bottom buttons
                Horizontal(
                    Button("BACK", id="btn-back-switcher", classes="-ghost"),
                    id="switcher-buttons",
                ),
                id="switcher-container",
            ),
            id="switcher-wrapper",
        )

    def on_mount(self) -> None:
        self._refresh_profile_list()
        self.query_one("#new-profile-input", Input).focus()

    def _refresh_profile_list(self) -> None:
        controller = getattr(self.app, "controller", None)
        profile_list = self.query_one("#profile-list", Vertical)
        profile_list.remove_children()

        if not controller:
            profile_list.mount(
                Static("  No controller available.", classes="switcher-empty")
            )
            return

        summaries = controller.get_user_summaries()

        if not summaries:
            profile_list.mount(
                Static("  No profiles found. Create one below.", classes="switcher-empty")
            )
            return

        for summary in summaries:
            uid = summary["user_id"]
            updated = summary.get("updated_at", "N/A")
            num_rated = summary.get("num_rated", 0)
            is_current = uid == controller.user_id

            # Format the display
            if isinstance(updated, str) and len(updated) > 16:
                updated = updated[:16]

            indicator = "►" if is_current else " "
            info_text = f"  {indicator}  {uid:<20s}  │  {num_rated:>3} rated  │  Last active: {updated}"

            row = Horizontal(
                Static(info_text, classes="profile-row-text"),
                Button(
                    "ACTIVE" if is_current else "SWITCH",
                    id=f"btn-switch-{uid}",
                    classes="profile-switch-btn" + (" -active-profile" if is_current else ""),
                    disabled=is_current,
                ),
                classes="profile-row" + (" profile-row-current" if is_current else ""),
            )
            profile_list.mount(row)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        controller = getattr(self.app, "controller", None)

        if bid == "btn-back-switcher":
            self.app.pop_screen()
            return

        if bid == "btn-create-profile":
            self._create_new_profile()
            return

        # Handle switch buttons
        if bid and bid.startswith("btn-switch-"):
            user_id = bid.replace("btn-switch-", "")
            if controller and user_id != controller.user_id:
                controller.switch_user(user_id)
                self.app.user_id = user_id
                self._refresh_profile_list()
                self._update_status_chip()
                self.notify(f"Switched to profile: {user_id}", title="Profile Switched")
            return

    def _create_new_profile(self) -> None:
        controller = getattr(self.app, "controller", None)
        if not controller:
            return

        text_input = self.query_one("#new-profile-input", Input)
        new_id = text_input.value.strip()

        if not new_id:
            self.notify("Enter a profile name.", title="Error", severity="error")
            return

        # Validate: alphanumeric, hyphens, underscores, 1-64 chars
        import re
        if not re.match(r'^[A-Za-z0-9_\-]{1,64}$', new_id):
            self.notify(
                "Profile name must be 1-64 chars: letters, digits, hyphens, underscores.",
                title="Invalid Name",
                severity="error",
            )
            return

        # Check if already exists
        existing = controller.store.list_users()
        if new_id in existing:
            self.notify(f"Profile '{new_id}' already exists.", title="Error", severity="error")
            return

        # Create and switch to the new profile
        controller.switch_user(new_id)
        controller.store.save_profile(controller.profile)
        self.app.user_id = new_id
        text_input.value = ""
        self._refresh_profile_list()
        self._update_status_chip()
        self.notify(f"Created and switched to profile: {new_id}", title="Profile Created")

    def _update_status_chip(self) -> None:
        """Try to update the status chip on any visible screen."""
        try:
            chip = self.app.query_one("#status-chip", Static)
            controller = getattr(self.app, "controller", None)
            if controller and chip:
                chip.update(self._format_chip(controller))
        except Exception:
            pass

    @staticmethod
    def _format_chip(controller):
        return f"{controller.user_id}  │  {controller.lambda_label.capitalize()}  │  {controller.active_preset_name}"
