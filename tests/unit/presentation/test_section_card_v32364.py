"""Test [v3.23.64] widget SectionCard tái sử dụng."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_card_has_title(app) -> None:
    from subtitles_extractor.presentation.widgets.section_card import SectionCard
    card = SectionCard("Tiêu đề thử")
    assert card.getTitle() == "Tiêu đề thử"


def test_add_widget(app) -> None:
    from subtitles_extractor.presentation.widgets.section_card import SectionCard
    card = SectionCard("X")
    card.add_widget(QLabel("a"))
    card.add_widget(QLabel("b"))
    assert card.content_layout.count() == 2


def test_add_layout(app) -> None:
    from subtitles_extractor.presentation.widgets.section_card import SectionCard
    card = SectionCard("X")
    row = QHBoxLayout()
    row.addWidget(QLabel("c"))
    card.add_layout(row)
    assert card.content_layout.count() == 1


def test_spacing_from_metrics(app) -> None:
    from subtitles_extractor.presentation.widgets.section_card import SectionCard
    from subtitles_extractor.presentation.theme import metrics
    card = SectionCard("X")
    # Khoảng cách nội dung lấy từ thang chuẩn (không hardcode).
    assert card.content_layout.spacing() == metrics.SPACING_SM
