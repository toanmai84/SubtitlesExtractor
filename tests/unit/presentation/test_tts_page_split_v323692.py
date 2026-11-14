"""[v3.23.692] Tách trang Lồng tiếng (U6): lồng tiếng cả bộ, cài GPU VieNeu, xuất chi tiết — hành vi không đổi."""

from __future__ import annotations

import logging

import pytest

from subtitles_extractor.presentation.pages import (
    tts_batch,
    tts_constants,
    tts_export_details,
    tts_gpu_install,
    tts_page,
)
from subtitles_extractor.presentation.pages.tts_batch import TTSBatchMixin
from subtitles_extractor.presentation.pages.tts_export_details import TTSExportDetailsMixin
from subtitles_extractor.presentation.pages.tts_gpu_install import TTSGpuInstallMixin
from subtitles_extractor.presentation.pages.tts_page import TTSPage

EXPECTED = {
    TTSBatchMixin: {
        "_on_batch_tts_clicked", "start_series_batch", "cancel_series_batch", "_run_next_batch_tts",
        "_on_source_changed_for_batch", "_generate_for_batch", "_on_error_for_batch", "_advance_batch_tts",
        "_finish_batch_tts", "_finish_batch_tts_report",
    },
    TTSGpuInstallMixin: {
        "_on_install_vieneu_gpu_clicked", "_on_gpu_install_progress", "_on_gpu_install_finished",
        "_on_gpu_install_failed", "_cleanup_gpu_install_thread", "_refresh_vieneu_gpu_note", "_probe_gpu_plan",
        "_on_gpu_plan_ready", "_apply_vieneu_gpu_note",
    },
    TTSExportDetailsMixin: {"_on_export_details", "_write_debug_config"},
}


@pytest.mark.parametrize("mixin", list(EXPECTED))
def test_mixin_methods_are_complete_and_not_shadowed(mixin: type) -> None:
    assert mixin in TTSPage.__mro__
    methods = {name for name in vars(mixin) if not name.startswith("__")}
    assert methods == EXPECTED[mixin]
    assert not methods & set(vars(TTSPage))
    for other in EXPECTED:
        if other is not mixin:
            assert not methods & set(vars(other))


def test_shared_constants_and_logger_unchanged() -> None:
    assert tts_page._ENGINE_VIENEU == tts_constants._ENGINE_VIENEU == "vieneu"
    assert tts_page._EDGE_ONLY_FIELDS is tts_constants._EDGE_ONLY_FIELDS
    assert "edge_concurrency" in tts_page._EDGE_ONLY_FIELDS
    for module in (tts_batch, tts_gpu_install, tts_export_details):
        assert module.logger is logging.getLogger(tts_page.__name__)
