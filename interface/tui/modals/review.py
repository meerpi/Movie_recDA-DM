"""
interface/tui/modals/review.py — Surgical Review Submission Modal Screen
"""

from typing import Any, Dict
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, RadioButton, RadioSet, Static


class ReviewSubmissionModal(ModalScreen[None]):
    """
    Surgical Review Submission Modal.
    Combines Star rating, Path A surgical checkboxes, and review text input.
    Integrates directly with local profile AND catalog database.
    """

    BINDINGS = [
        ("escape", "dismiss", "Cancel"),
    ]

    def __init__(self, item: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.item = item
        self.movie_id = item.get("movie_id")
        self.movie_title = item.get("title", f"Movie #{self.movie_id}")

    def compose(self) -> ComposeResult:
        container = Vertical(
            VerticalScroll(
                Label("1. Select Star Rating:"),
                RadioSet(
                    RadioButton("★ 5.0 - Masterpiece", id="star-5", value=True),
                    RadioButton("★ 4.0 - Great", id="star-4"),
                    RadioButton("★ 3.0 - Good / Average", id="star-3"),
                    RadioButton("★ 2.0 - Poor", id="star-2"),
                    RadioButton("★ 1.0 - Terrible", id="star-1"),
                    id="rating-radioset"
                ),
                Static("─" * 40),
                Label("2. Path A Surgical Aspects (Check all that applied):"),
                Horizontal(
                    Checkbox("Visuals / Cinematography", value=True, id="chk-visuals"),
                    Checkbox("Plot / Storytelling", value=True, id="chk-plot"),
                    Checkbox("Acting / Cast", value=True, id="chk-acting"),
                    id="checkboxes-row-1"
                ),
                Horizontal(
                    Checkbox("Pacing", value=False, id="chk-pacing"),
                    Checkbox("Soundtrack / Music", value=False, id="chk-music"),
                    Checkbox("Atmosphere / Tone", value=True, id="chk-tone"),
                    id="checkboxes-row-2"
                ),
                Static("─" * 40),
                Label("3. Write Review Text (Optional):"),
                Input(
                    placeholder="Enter your detailed movie review here...",
                    id="review-text-input"
                ),
                id="review-scroll-body"
            ),
            Horizontal(
                Button("SUBMIT REVIEW", id="btn-submit-review", variant="success"),
                Button("CANCEL [ESC]", id="btn-cancel-review", variant="error"),
                id="review-buttons"
            ),
            id="review-container"
        )
        container.border_title = f"▸ REVIEW · {self.movie_title.upper()}"
        container.border_subtitle = "rate & tag"

        yield Container(
            container,
            id="modal-wrapper"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel-review":
            self.dismiss()
        elif event.button.id == "btn-submit-review":
            self._process_submission()

    def _process_submission(self) -> None:
        # Determine selected star rating
        radioset = self.query_one("#rating-radioset", RadioSet)
        rating_val = 5.0
        if radioset.pressed_button:
            button_id = radioset.pressed_button.id
            if button_id == "star-5": rating_val = 5.0
            elif button_id == "star-4": rating_val = 4.0
            elif button_id == "star-3": rating_val = 3.0
            elif button_id == "star-2": rating_val = 2.0
            elif button_id == "star-1": rating_val = 1.0

        # Gather surgical checkboxes
        liked = []
        if self.query_one("#chk-visuals", Checkbox).value: liked.append("Visuals")
        if self.query_one("#chk-plot", Checkbox).value: liked.append("Plot")
        if self.query_one("#chk-acting", Checkbox).value: liked.append("Acting")
        if self.query_one("#chk-pacing", Checkbox).value: liked.append("Pacing")
        if self.query_one("#chk-music", Checkbox).value: liked.append("Soundtrack")
        if self.query_one("#chk-tone", Checkbox).value: liked.append("Atmosphere")

        review_text = self.query_one("#review-text-input", Input).value.strip()

        # Submit via controller
        controller = getattr(self.app, "controller", None)
        if controller:
            res = controller.submit_review(
                movie_id=self.movie_id,
                star_rating=rating_val,
                review_text=review_text if review_text else None,
                surgical_aspects={"liked": liked}
            )
            self.notify(f"✓ Review submitted & integrated into DB for '{self.movie_title}'!", title="Success")

        self.dismiss()
