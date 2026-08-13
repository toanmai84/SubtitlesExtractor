"""[v3.23.110] Test khả năng THU GỌN (collapsible) của SectionCard."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PySide6.QtWidgets import QApplication, QLabel


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_non_collapsible_default(app) -> None:
    from subtitles_extractor.presentation.widgets.section_card import SectionCard
    card = SectionCard("Thường")
    card.add_widget(QLabel("a"))
    assert card.content_layout.count() == 1
    # Không có container thu gọn -> set_collapsed không gây lỗi và không làm gì.
    card.set_collapsed(True)


def test_collapsible_collapsed_hides_content(app) -> None:
    from subtitles_extractor.presentation.widgets.section_card import SectionCard
    card = SectionCard("Tuỳ chọn", collapsible=True, collapsed=True)
    card.add_widget(QLabel("x"))
    assert card._content_container is not None
    assert not card._content_container.isVisibleTo(card)  # thu gọn sẵn


def test_collapsible_expanded_shows_content(app) -> None:
    from subtitles_extractor.presentation.widgets.section_card import SectionCard
    card = SectionCard("Tuỳ chọn", collapsible=True, collapsed=False)
    assert card._content_container.isVisibleTo(card)


def test_toggle_flips_state(app) -> None:
    from subtitles_extractor.presentation.widgets.section_card import SectionCard
    card = SectionCard("Tuỳ chọn", collapsible=True, collapsed=True)
    assert not card._content_container.isVisibleTo(card)
    card.toggle_collapsed()
    assert card._content_container.isVisibleTo(card)
    card.toggle_collapsed()
    assert not card._content_container.isVisibleTo(card)
