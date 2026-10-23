"""[v3.23.614] Trang Nhật ký (và tệp app_log xuất từ đó) giữ NGUYÊN VĂN mọi dòng log.

Dòng log được chèn vào khung xem dạng HTML mà không thoát ký tự: mọi thứ sau dấu "<" bị
coi là thẻ HTML và biến mất, xuống dòng bị gộp thành dấu cách. app_log thật 14/10:
"Giãn giờ đọc: … khung OCR chớp nhoáng" (mất "< 0.3 s: 5 → 5 dòng"), "Flicker drop:
'JRAS' (dur=0.001s" (mất nửa sau). Traceback kèm lỗi (thêm ở 3.23.613) cũng bị nuốt từ
"in <module>" và dồn thành một dòng — chính thứ cần để chẩn đoán.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from subtitles_extractor.infrastructure.logging.qt_log_handler import (
    LogEntry,
    QtLogBridge,
)

MESSAGE = (
    "Giãn giờ đọc: 5 dòng; khung OCR chớp nhoáng < 0.3 s: 5 → 5 dòng.\n"
    "Traceback (most recent call last):\n"
    '  File "editor_page.py", line 2540, in <module>\n'
    "    self._do_export_clicked()\n"
    "ValueError: a  <b> & c"
)


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_export_text_keeps_every_character(app: QApplication) -> None:
    from subtitles_extractor.infrastructure.i18n.null_translator import NullTranslator
    from subtitles_extractor.presentation.pages.log_page import LogPage

    bridge = QtLogBridge()
    page = LogPage(bridge, NullTranslator())
    page._on_record(LogEntry("09:34:58.626", "INFO", "a.b.translate_page", MESSAGE))
    text = page._view.toPlainText()
    assert text.endswith(MESSAGE)
    assert "09:34:58.626" in text and "translate_page" in text
    page.close()
