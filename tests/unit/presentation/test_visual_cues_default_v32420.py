"""[v3.23.120] Test: đính video ngữ cảnh -> mặc định BẬT 'Phân tích hình ảnh'."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PySide6.QtWidgets import QApplication, QCheckBox, QPushButton


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class _Stub:
    """Stub tối thiểu các widget mà _on_attach_video_toggled đụng tới."""

    def __init__(self) -> None:
        self._btn_pick_video = QPushButton()
        self._video_stage_literal = QCheckBox()
        self._video_stage_style = QCheckBox()
        self._video_stage_localize = QCheckBox()
        self._visual_cues_check = QCheckBox()
        self._visual_cues_check.setEnabled(False)


def _handler():
    from subtitles_extractor.presentation.pages.translate_page import TranslatePage
    return TranslatePage._on_attach_video_toggled


def test_attach_video_enables_and_checks_image_analysis(app) -> None:
    handler = _handler()
    stub = _Stub()
    handler(stub, True)  # đính video
    assert stub._visual_cues_check.isEnabled()
    assert stub._visual_cues_check.isChecked()  # mặc định BẬT
    assert stub._btn_pick_video.isEnabled()


def test_detach_video_unchecks_image_analysis(app) -> None:
    handler = _handler()
    stub = _Stub()
    handler(stub, True)
    handler(stub, False)  # bỏ đính video
    assert not stub._visual_cues_check.isEnabled()
    assert not stub._visual_cues_check.isChecked()
