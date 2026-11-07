"""[v3.23.689] Tách "Dịch cả bộ" khỏi translate_page.py (U6): hành vi không đổi, không phương thức nào bị che."""

from __future__ import annotations

import logging

from subtitles_extractor.presentation.pages import translate_batch, translate_page
from subtitles_extractor.presentation.pages.translate_batch import TranslateBatchMixin
from subtitles_extractor.presentation.pages.translate_page import TranslatePage

BATCH_METHODS = {
    "_on_batch_translate_clicked", "start_series_batch", "cancel_series_batch", "_batch_request_estimate",
    "_remaining_daily_requests", "_run_next_batch_translate", "_prepare_batch_episode", "_on_source_changed_for_batch",
    "_start_batch_episode", "_reset_batch_analysis", "_maybe_analyze_batch_episode", "_batch_analysis_done",
    "_on_error_for_batch", "_advance_batch_translate", "_finish_batch_translate", "_finish_batch_translate_report",
}


def test_batch_methods_live_in_mixin_and_are_not_shadowed() -> None:
    assert TranslateBatchMixin in TranslatePage.__mro__
    mixin_methods = {name for name in vars(TranslateBatchMixin) if not name.startswith("__")}
    assert mixin_methods == BATCH_METHODS
    # Không phương thức nào định nghĩa ở CẢ hai nơi (bản trên trang sẽ lặng lẽ che bản trong mixin).
    assert not mixin_methods & set(vars(TranslatePage))
    for name in BATCH_METHODS:
        assert getattr(TranslatePage, name) is vars(TranslateBatchMixin)[name]


def test_shared_names_and_logger_unchanged() -> None:
    assert translate_page.DEFAULT_ANALYSIS_MODEL == translate_batch.DEFAULT_ANALYSIS_MODEL == "gemini-3.8-flash"
    assert translate_page._BATCH_STOP_MARKERS is translate_batch._BATCH_STOP_MARKERS
    # Cùng tên logger như trước khi tách → bộ lọc / tệp log theo module vẫn bắt được log "Dịch cả bộ".
    assert translate_batch.logger is logging.getLogger(translate_page.__name__)
