"""[v3.23.120] Test: đính video ngữ cảnh -> mặc định BẬT 'Phân tích hình ảnh'."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

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
        # [v3.23.553] Handler đụng cả ô "tag cảm xúc" từ v3.23.448 — stub thiếu ô này nên
        # test hỏng, nhưng bị che vì guard cũ importorskip("PyQt6") (app dùng PySide6).
        self._emotion_cues_check = QCheckBox()
        self._emotion_cues_check.setEnabled(False)

    def _set_video_controls_enabled(self, enabled: bool) -> None:
        # [v3.23.609] Handler bật/tắt các ô phụ thuộc video qua hàm dùng chung.
        from subtitles_extractor.presentation.pages.translate_page import TranslatePage

        TranslatePage._set_video_controls_enabled(self, enabled)  # type: ignore[arg-type]


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


def test_detach_video_also_clears_emotion_cues(app) -> None:
    """Tag cảm xúc cần video ngữ cảnh — bỏ đính video phải tắt và bỏ chọn nó."""
    handler = _handler()
    stub = _Stub()
    handler(stub, True)
    assert stub._emotion_cues_check.isEnabled()
    stub._emotion_cues_check.setChecked(True)
    handler(stub, False)
    assert not stub._emotion_cues_check.isEnabled()
    assert not stub._emotion_cues_check.isChecked()
