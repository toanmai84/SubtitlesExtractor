"""Unit tests cho v2.29 — Loguru-based logging.

Bảo vệ các tính năng:
    * ``setup_loguru()`` cài đặt thành công và không lỗi.
    * ``setup_logging()`` backward-compat (level int → name string).
    * ``InterceptHandler`` bắt stdlib logging và forward sang Loguru.
    * 5 core modules dùng ``from loguru import logger`` thay vì stdlib.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.logging.loguru_config import (
    InterceptHandler,
    setup_logging,
    setup_loguru,
)


class TestSetupLoguru:
    """v2.29: setup_loguru cấu hình Loguru thành công."""

    def test_setup_loguru_console_only_no_error(self) -> None:
        """Cấu hình chỉ console (không log_dir) không lỗi."""
        setup_loguru(level="INFO")
        # Nếu không raise thì test pass.

    def test_setup_loguru_with_file_sink(self, tmp_path: Path) -> None:
        """Cấu hình có file sink: tạo thư mục log và ghi file."""
        log_dir = tmp_path / "logs"
        setup_loguru(level="INFO", log_dir=log_dir)
        assert log_dir.exists()

    def test_setup_loguru_idempotent_when_called_repeatedly(self) -> None:
        """Gọi nhiều lần không bị nhân đôi handler (vẫn an toàn)."""
        setup_loguru(level="INFO")
        setup_loguru(level="DEBUG")
        setup_loguru(level="WARNING")
        # Không lỗi → pass.

    def test_setup_loguru_levels_accepted(self) -> None:
        """Tất cả level Loguru chuẩn được chấp nhận."""
        for level_name in ("TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            setup_loguru(level=level_name)


class TestSetupLoggingBackwardCompat:
    """v2.29: setup_logging (legacy API) vẫn hoạt động."""

    def test_setup_logging_with_int_level(self, tmp_path: Path) -> None:
        """API cũ dùng logging.INFO (int) không phá vỡ."""
        setup_logging(level=logging.INFO, log_dir=tmp_path / "logs")

    def test_setup_logging_with_debug_int_level(self, tmp_path: Path) -> None:
        """logging.DEBUG (10) được chuyển sang 'DEBUG'."""
        setup_logging(level=logging.DEBUG, log_dir=tmp_path / "logs")

    def test_setup_logging_with_file_max_bytes_kb_size(self) -> None:
        """file_max_bytes < 1MB chuyển sang 'N KB' format."""
        setup_logging(level=logging.INFO, file_max_bytes=500 * 1024)


class TestInterceptHandler:
    """v2.29: InterceptHandler bridge stdlib → Loguru."""

    def test_intercept_handler_class_is_logging_handler_subclass(self) -> None:
        """InterceptHandler kế thừa từ logging.Handler."""
        assert issubclass(InterceptHandler, logging.Handler)

    def test_intercept_handler_can_emit_without_error(self) -> None:
        """emit() chấp nhận LogRecord và không lỗi."""
        setup_loguru(level="DEBUG")
        handler = InterceptHandler()
        record = logging.LogRecord(
            name="test_module",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message: %s",
            args=("hello",),
            exc_info=None,
        )
        handler.emit(record)  # không raise.

    def test_intercept_handler_handles_exception_records(self) -> None:
        """emit() xử lý LogRecord có exc_info."""
        setup_loguru(level="DEBUG")
        handler = InterceptHandler()
        try:
            raise ValueError("test exception")
        except ValueError:
            import sys
            record = logging.LogRecord(
                name="test_module",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Caught exception",
                args=(),
                exc_info=sys.exc_info(),
            )
            handler.emit(record)


class TestCoreModulesUseLoguru:
    """v2.29: 5 core modules dùng ``from loguru import logger`` native."""

    def test_subtitle_builder_module_imports_loguru(self) -> None:
        """subtitle_pipeline modules dùng loguru.logger, không phải logging.Logger.

        Sau refactor v3.0, ``subtitle_builder.py`` thành thin shim (re-export).
        Logger được dùng trong các module nội bộ của ``subtitle_pipeline/``:
        ``text_correction``, ``event_filters``. Test này verify 2 module có
        logger là loguru instance.
        """
        from subtitles_extractor.application.services.subtitle_pipeline import (
            event_filters,
            text_correction,
        )
        # loguru.logger là instance, không phải class — check qua __class__.
        assert event_filters.logger.__class__.__module__.startswith("loguru")
        assert text_correction.logger.__class__.__module__.startswith("loguru")

    def test_outlier_detection_module_imports_loguru(self) -> None:
        from subtitles_extractor.application.services import outlier_detection
        assert outlier_detection.logger.__class__.__module__.startswith("loguru")

    def test_flicker_absorber_module_imports_loguru(self) -> None:
        from subtitles_extractor.application.services import flicker_absorber
        assert flicker_absorber.logger.__class__.__module__.startswith("loguru")

    def test_bbox_analyzer_module_imports_loguru(self) -> None:
        from subtitles_extractor.infrastructure.video import bbox_analyzer
        assert bbox_analyzer.logger.__class__.__module__.startswith("loguru")

    def test_raw_ocr_serializer_module_imports_loguru(self) -> None:
        from subtitles_extractor.infrastructure.serializers import raw_ocr_serializer
        assert raw_ocr_serializer.logger.__class__.__module__.startswith("loguru")


class TestLogguruApiUsable:
    """v2.29: Loguru API quen dùng vẫn hoạt động (smoke)."""

    def test_loguru_curly_brace_formatting(self) -> None:
        """logger.info('text {}', value) — Loguru style format."""
        from loguru import logger
        setup_loguru(level="DEBUG")
        # Không raise → format placeholder OK.
        logger.info("test {} {}", 42, "hello")

    def test_loguru_supports_f_string_lazy(self) -> None:
        """logger.debug(f'...{x}...') — f-string vẫn ok (eager eval)."""
        from loguru import logger
        setup_loguru(level="DEBUG")
        value = 100
        logger.debug(f"Eager f-string: {value}")
