"""[v3.23.392] Test hợp đồng cho preflight_build_check (thuần, không phụ thuộc build_env)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[3] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import preflight_build_check as pf  # noqa: E402


def test_run_all_checks_covers_expected_modules() -> None:
    names = {r.name for r in pf.run_all_checks()}
    assert "setuptools.command.easy_install" in names
    assert "PyInstaller" in names
    assert "paddle" in names
    assert "paddleocr" in names
    assert "PySide6" in names


def test_critical_vs_warning_flags() -> None:
    by_name = {r.name: r for r in pf.run_all_checks()}
    # Các mục BẮT BUỘC.
    assert by_name["setuptools.command.easy_install"].critical is True
    assert by_name["PyInstaller"].critical is True
    assert by_name["PySide6"].critical is True
    # paddle chỉ cảnh báo (spec collect_all có try/except; build_env nên có nhưng không chặn).
    assert by_name["paddle"].critical is False
    assert by_name["paddleocr"].critical is False


def test_setuptools_check_matches_find_spec() -> None:
    """Kết quả mục setuptools phải khớp với find_spec thực tế của môi trường chạy test."""
    expected = importlib.util.find_spec("setuptools.command.easy_install") is not None
    result = pf._check_setuptools_easy_install()
    assert result.passed is expected
    assert result.critical is True
    assert "easy_install" in result.detail


def test_importable_check_true_for_stdlib() -> None:
    """Module chắc chắn có (json) phải PASS."""
    result = pf._check_importable("json", critical=True, hint="x")
    assert result.passed is True
    assert result.detail.endswith("OK.")


def test_importable_check_false_for_missing() -> None:
    result = pf._check_importable(
        "khong_ton_tai_module_xyz", critical=False, hint="gợi ý"
    )
    assert result.passed is False
    assert "gợi ý" in result.detail
