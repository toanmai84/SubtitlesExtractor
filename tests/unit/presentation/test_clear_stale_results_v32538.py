"""[v3.23.138] Test: các trang XOÁ kết quả cũ khi nạp nguồn/dữ liệu mới.

Trước đây trang Dịch và trang TTS không xoá bảng kết quả khi phụ đề mới được ĐẨY sang
(vd từ Editor) -> vẫn hiển thị bản dịch/âm thanh của nguồn CŨ, gây hiểu nhầm là đã xử lý.

Gọi các phương thức UI ở dạng UNBOUND (bỏ qua __init__) với widget giả (MagicMock) để
kiểm hành vi mà không cần dựng toàn bộ giao diện.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from subtitles_extractor.presentation.pages.translate_page import TranslatePage
from subtitles_extractor.presentation.pages.tts_page import TTSPage


def test_translate_clear_results_wipes_table_and_details() -> None:
    page = TranslatePage.__new__(TranslatePage)
    page._table = MagicMock()
    page._detail_source = MagicMock()
    page._detail_translation = MagicMock()
    page._progress = MagicMock()

    page._clear_results()

    page._table.setRowCount.assert_called_once_with(0)
    page._detail_source.clear.assert_called_once()
    page._detail_translation.clear.assert_called_once()
    page._progress.setValue.assert_called_once_with(0)


def test_translate_source_changed_clears_old_results() -> None:
    page = TranslatePage.__new__(TranslatePage)
    page._table = MagicMock()
    page._detail_source = MagicMock()
    page._detail_translation = MagicMock()
    page._progress = MagicMock()
    # Các widget/nhánh khác mà _on_source_changed đụng tới khi count>0.
    page._lbl_source = MagicMock()
    page._stage_label = MagicMock()
    page._source_video_path = ""
    page._video_path = ""
    page._update_action_states = MagicMock()

    page._on_source_changed(5)  # nạp 5 dòng mới

    # Kết quả cũ PHẢI bị xoá ngay khi nguồn đổi.
    page._table.setRowCount.assert_called_once_with(0)
    page._detail_source.clear.assert_called_once()


def test_tts_source_changed_clears_old_results() -> None:
    page = TTSPage.__new__(TTSPage)
    page._table = MagicMock()
    page._lbl_summary = MagicMock()
    page._lbl_filter_count = MagicMock()
    page._btn_export = MagicMock()
    page._btn_open = MagicMock()
    page._btn_open_file = MagicMock()
    page._lbl_source = MagicMock()
    page._stage_lbl = MagicMock()
    page._update_action_states = MagicMock()
    page._last_results = ["cũ"]

    page._on_source_changed(3)

    # Bảng kết quả audio cũ bị xoá + danh sách kết quả nội bộ rỗng.
    page._table.setRowCount.assert_called_with(0)
    assert page._last_results == []
