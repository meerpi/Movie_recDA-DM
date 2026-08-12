"""
interface/tui/modals/review.py — Conversational Review Modal

Three-phase conversational flow:
  Phase 1  Pick which aspects to rate (click to toggle each chip)
  Phase 2  For each toggled aspect, answer a quality question (1–5)
  Phase 3  View computed average, write optional free-text review, submit
"""

from typing import Any, Dict, List, Optional
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, Rule, Static, TextArea


# Aspects the user can rate, with weights for the derived star score
ASPECTS = [
    ("Atmosphere / Tone",        "atmosphere", 1.0),
    ("Plot / Storytelling",      "plot",       1.0),
    ("Acting / Cast",            "acting",     1.0),
    ("Visuals / Cinematography", "visuals",    1.0),
    ("Pacing",                   "pacing",     0.6),
    ("Soundtrack",               "soundtrack", 0.6),
    ("Themes / Depth",           "themes",     0.8),
    ("Rewatchability",           "rewatch",    0.6),
]


class ReviewSubmissionModal(ModalScreen[None]):

    BINDINGS = [
        Binding("escape", "dismiss", "Cancel"),
    ]

    def __init__(self, item: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.item = item
        self.movie_id = item.get("movie_id")
        self.movie_title = item.get("title", f"Movie #{self.movie_id}")

        self._selected_aspects: Dict[str, bool] = {key: False for _, key, _ in ASPECTS}
        self._aspect_scores: Dict[str, Optional[int]] = {key: None for _, key, _ in ASPECTS}
        self._phase = 1
        self._scoring_index = 0
        self._scoring_queue: List[str] = []

    def compose(self) -> ComposeResult:
        container = Vertical(
            Static("", id="review-progress"),
            VerticalScroll(id="review-scroll-body"),
            Horizontal(
                Button("NEXT",   id="btn-next",   classes="-filled"),
                Button("CANCEL", id="btn-cancel", classes="-ghost"),
                id="review-buttons",
            ),
            id="review-container",
        )
        yield Container(container, id="modal-wrapper")

    def on_mount(self) -> None:
        container = self.query_one("#review-container")
        container.border_title = f"[ REVIEW ] {self.movie_title.upper()}"
        self._render_phase_1()

    def _update_progress(self) -> None:
        progress = self.query_one("#review-progress", Static)
        dots = []
        for i in range(1, 4):
            if i == self._phase:
                dots.append("●")
            else:
                dots.append("○")
        progress.update(f"  {' '.join(dots)}  Step {self._phase} of 3")

    def _render_phase_1(self) -> None:
        scroll = self.query_one("#review-scroll-body", VerticalScroll)
        scroll.remove_children()
        scroll.mount(Label("Which aspects of this film do you want to rate?"))
        scroll.mount(Static("Tap to select. Tap again to deselect."))
        scroll.mount(Rule())
        for label, key, _ in ASPECTS:
            scroll.mount(Button(f"  {label}", id=f"aspect-{key}", classes="aspect-chip"))
        self._phase = 1
        self._update_progress()
        self.query_one("#btn-next", Button).label = "NEXT"

    def _render_phase_2(self) -> None:
        key = self._scoring_queue[self._scoring_index]
        label = next(lbl for lbl, k, _ in ASPECTS if k == key)
        scroll = self.query_one("#review-scroll-body", VerticalScroll)
        scroll.remove_children()
        total = len(self._scoring_queue)
        current = self._scoring_index + 1
        scroll.mount(Label(f"({current}/{total})  How was the  {label.upper()}?"))
        scroll.mount(Rule())
        scroll.mount(
            RadioSet(
                RadioButton("5  —  Excellent",      id=f"{key}-5"),
                RadioButton("4  —  Good",            id=f"{key}-4"),
                RadioButton("3  —  Average",         id=f"{key}-3", value=True),
                RadioButton("2  —  Below average",   id=f"{key}-2"),
                RadioButton("1  —  Poor",            id=f"{key}-1"),
                id=f"radioset-{key}",
            )
        )
        self._phase = 2
        self._update_progress()
        self.query_one("#btn-next", Button).label = "NEXT"

    def _render_phase_3(self) -> None:
        scroll = self.query_one("#review-scroll-body", VerticalScroll)
        scroll.remove_children()
        derived = self._derived_star_rating()
        stars_filled = "★" * round(derived)
        stars_empty  = "☆" * (5 - round(derived))
        scroll.mount(Label(f"Your Score:  {stars_filled}{stars_empty}  ({derived:.1f} / 5.0)"))
        scroll.mount(Rule())
        scroll.mount(Label("What stood out to you? (optional)"))
        scroll.mount(
            TextArea(
                id="review-text-input",
            )
        )
        self._phase = 3
        self._update_progress()
        # Submit is the filled primary action; Back is ghost
        btn_next = self.query_one("#btn-next", Button)
        btn_next.label = "SUBMIT"
        btn_cancel = self.query_one("#btn-cancel", Button)
        btn_cancel.label = "BACK"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id

        if bid == "btn-cancel":
            if self._phase == 3:
                # Back goes to phase 2 (last aspect) or phase 1
                if self._scoring_queue:
                    self._scoring_index = len(self._scoring_queue) - 1
                    self._render_phase_2()
                    # Restore cancel label
                    self.query_one("#btn-cancel", Button).label = "CANCEL"
                else:
                    self._render_phase_1()
                    self.query_one("#btn-cancel", Button).label = "CANCEL"
            else:
                self.dismiss()
            return

        if bid == "btn-next":
            self._advance()
            return

        if self._phase == 1 and bid and bid.startswith("aspect-"):
            key = bid.replace("aspect-", "", 1)
            self._selected_aspects[key] = not self._selected_aspects[key]
            self._sync_chip_style(key)

    def _sync_chip_style(self, key: str) -> None:
        btn = self.query_one(f"#aspect-{key}", Button)
        label_text = next(lbl for lbl, k, _ in ASPECTS if k == key)
        if self._selected_aspects[key]:
            btn.label = f"✓ {label_text}"
            btn.add_class("-selected")
            btn.remove_class("aspect-chip")
            btn.add_class("aspect-chip")
        else:
            btn.label = f"  {label_text}"
            btn.remove_class("-selected")

    def _advance(self) -> None:
        if self._phase == 1:
            self._scoring_queue = [k for k, selected in self._selected_aspects.items() if selected]
            if not self._scoring_queue:
                self.notify("Select at least one aspect to rate.", title="Nothing selected", severity="warning")
                return
            self._scoring_index = 0
            self._render_phase_2()

        elif self._phase == 2:
            key = self._scoring_queue[self._scoring_index]
            radioset = self.query_one(f"#radioset-{key}", RadioSet)
            if radioset.pressed_button:
                score_str = radioset.pressed_button.id.split("-")[-1]
                self._aspect_scores[key] = int(score_str)
            self._scoring_index += 1
            if self._scoring_index < len(self._scoring_queue):
                self._render_phase_2()
            else:
                self._render_phase_3()

        elif self._phase == 3:
            self._submit()

    def _derived_star_rating(self) -> float:
        weight_map = {key: w for _, key, w in ASPECTS}
        total_w = 0.0
        weighted_sum = 0.0
        for key in self._scoring_queue:
            score = self._aspect_scores.get(key)
            if score is not None:
                w = weight_map.get(key, 1.0)
                weighted_sum += score * w
                total_w += w
        if total_w == 0:
            return 3.0
        return round(weighted_sum / total_w, 2)

    def _submit(self) -> None:
        star_rating = self._derived_star_rating()

        surgical_aspects = []
        for key in self._scoring_queue:
            score = self._aspect_scores.get(key)
            if score is not None:
                label = next(lbl for lbl, k, _ in ASPECTS if k == key)
                surgical_aspects.append(f"{label}:{score}")

        try:
            review_text = self.query_one("#review-text-input", TextArea).text.strip()
        except Exception:
            review_text = ""

        self.app.controller.submit_review(
            movie_id=self.movie_id,
            star_rating=star_rating,
            review_text=review_text or None,
            surgical_aspects=surgical_aspects,
        )

        stars = "★" * round(star_rating) + "☆" * (5 - round(star_rating))
        self.notify(
            f"{stars}  {star_rating:.1f}/5.0 — logged for '{self.movie_title}'",
            title="Review Saved",
        )
        self.dismiss()
