"""Test [v3.23.40] Cài đặt lô/ngữ cảnh áp vào trang dịch.

Lưu ý: chỉ tạo MỘT TranslatePage mỗi tiến trình test để tránh segfault do tích luỹ
widget trong QApplication dùng chung; kiểm logic đọc settings qua helper thuần.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings

from subtitles_extractor.composition.bootstrap import bootstrap_for_gui


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_settings_batch_applied_to_stages(app) -> None:
    from subtitles_extractor.presentation.pages.translate_page import TranslatePage
    QSettings("SubtitlesExtractor", "TranslatePage").clear()
    c = bootstrap_for_gui()
    c.settings_service.update(
        translation={"default_batch_size": 35, "default_context_size": 14}
    )
    page = TranslatePage(c)
    # LITERAL dùng đúng giá trị Cài đặt (đọc lúc dựng panel).
    # Lưu ý: nếu QSettings có lưu lựa chọn cũ, nó có thể ghi đè — test này verify
    # nhánh đọc settings khi QSettings đã được clear ở trên.
    assert page._stage_literal.batch_spin.value() == 35
    assert page._stage_literal.ctx_spin.value() == 14
    assert page._stage_style.batch_spin.value() == 28  # 35 * 0.8
    assert page._stage_style.ctx_spin.value() == 16     # 14 + 2
