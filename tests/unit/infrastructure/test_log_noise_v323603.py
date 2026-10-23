"""[v3.23.603] Nhật ký ứng dụng không còn DEBUG của filelock khi nạp model VieNeu.

app_log thật 11/10: 60/553 dòng là "Attempting to acquire lock …" / "Lock … released"
của filelock (huggingface_hub giữ khoá từng file model) — logger "filelock" (bản cũ ghi
dưới tên "filelock._api").
"""

from __future__ import annotations

import logging

from subtitles_extractor.infrastructure.logging.loguru_config import setup_loguru


def test_filelock_chi_con_canh_bao() -> None:
    setup_loguru(level="DEBUG")
    for name in ("filelock", "filelock._api"):
        logger = logging.getLogger(name)
        assert not logger.isEnabledFor(logging.DEBUG)
        assert logger.isEnabledFor(logging.WARNING)


def test_log_cua_ung_dung_van_giu_debug() -> None:
    setup_loguru(level="DEBUG")
    app_logger = logging.getLogger(
        "subtitles_extractor.infrastructure.tts.vieneu_tts_adapter"
    )
    assert app_logger.isEnabledFor(logging.DEBUG)
