"""Test [v3.22.4] Dependency Doctor + fix logging handler torio."""

from __future__ import annotations

import logging

from subtitles_extractor.infrastructure.diagnostics.dependency_doctor import (
    DependencyReport,
    DependencyStatus,
    check_ffmpeg_cli,
    check_whisperx,
    install_package,
    run_full_diagnosis,
)


class TestDependencyDoctor:
    def test_ffmpeg_check_returns_report(self) -> None:
        report = check_ffmpeg_cli()
        assert isinstance(report, DependencyReport)
        # Sandbox có ffmpeg → OK; nếu không thì BROKEN_RUNTIME có hint.
        if not report.is_ok:
            assert report.install_hint

    def test_whisperx_missing_in_sandbox(self) -> None:
        report = check_whisperx()
        # Sandbox không cài whisperx → MISSING_PACKAGE + có lệnh pip.
        assert report.status == DependencyStatus.MISSING_PACKAGE
        assert report.pip_args == ("whisperx",)
        assert "whisperx" in report.install_hint.lower()

    def test_full_diagnosis_lists_all(self) -> None:
        reports = run_full_diagnosis()
        names = [r.name for r in reports]
        assert any("FFmpeg" in n for n in names)
        assert any("WhisperX" in n for n in names)

    def test_install_package_empty_args_fails_gracefully(self) -> None:
        ok, msg = install_package(())
        assert ok is False
        assert msg


class TestLoggingHandlerSurvivesTorioError:
    """[v3.22.4] InterceptHandler không crash khi torio log DEBUG kèm exc_info."""

    def test_noisy_debug_with_exception_does_not_raise(self) -> None:
        from subtitles_extractor.infrastructure.logging.loguru_config import (
            InterceptHandler,
        )

        handler = InterceptHandler()
        try:
            raise FileNotFoundError("libtorio_ffmpeg6.pyd not found")
        except FileNotFoundError:
            record = logging.LogRecord(
                name="torio._extension.utils",
                level=logging.DEBUG,
                pathname=__file__,
                lineno=1,
                msg="Failed to load FFmpeg6 extension.",
                args=(),
                exc_info=True,
            )
        # Không được ném bất kỳ exception nào.
        handler.emit(record)

    def test_noisy_prefixes_include_torio(self) -> None:
        from subtitles_extractor.infrastructure.logging.loguru_config import (
            InterceptHandler,
        )

        assert "torio" in InterceptHandler._NOISY_DEBUG_PREFIXES
