"""Test kiểm cú pháp tệp batch — v3.23.337.

SỰ CỐ THẬT ĐƯỢC CHỐNG: v3.23.333 sửa ``build_windows.bat`` bằng phép thay chuỗi, dùng
``s.index(":skip_whisperx")`` làm mốc kết thúc — nhưng chuỗi đó CŨNG nằm trong
``goto :skip_whisperx`` ở đầu khối. Kết quả chỉ thay phần đầu, khối cũ còn nguyên:

* nhãn ``:skip_whisperx`` bị định nghĩa HAI lần;
* lệnh ``pip install whisperx`` vào môi trường CHÍNH nằm SAU nhãn nên chạy vô điều kiện.

Hậu quả trên máy người dùng: ``huggingface-hub`` bị hạ 1.25.1 → 0.36.2, ``gradio`` hỏng.
Phép kiểm cũ (ngoặc cân bằng + goto có nhãn) không phát hiện gì.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _load_tool():
    import importlib.util

    import subtitles_extractor.domain.entities.project_record as anchor

    root = Path(anchor.__file__).resolve().parents[4]
    path = root / "tools" / "check_batch_syntax.py"
    if not path.is_file():
        pytest.skip("Không tìm thấy tools/check_batch_syntax.py")
    spec = importlib.util.spec_from_file_location("_batch_tool", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_batch_tool"] = module
    spec.loader.exec_module(module)
    return module


#: Tái hiện ĐÚNG tệp đã gây sự cố.
_BROKEN_BATCH = """@echo off
if /i not "%SUBEXT_ENABLE_WHISPERX%"=="1" goto :skip_whisperx
echo [INFO] tao moi truong RIENG...
python -m venv whisperx_env
whisperx_env\\Scripts\\python.exe -m pip install whisperx
:skip_whisperx
echo.
echo [INFO] cai WhisperX + torch...
"%PYEXE%" -m pip install whisperx --retries 3
:skip_whisperx
"""

_CLEAN_BATCH = """@echo off
if /i not "%FLAG%"=="1" goto :skip_thing
echo Lam viec...
python -m venv thing_env
thing_env\\Scripts\\python.exe -m pip install thing
:skip_thing
echo Xong.
"""


def test_detects_duplicate_label(tmp_path: Path) -> None:
    """Nhãn trùng = dấu hiệu còn sót khối cũ — chính là lỗi đã xảy ra."""
    tool = _load_tool()
    path = tmp_path / "bad.bat"
    path.write_text(_BROKEN_BATCH, encoding="utf-8")
    issues = tool.check_batch_file(path)
    assert any("đã được định nghĩa" in issue.message for issue in issues)


def test_detects_install_after_skip_label(tmp_path: Path) -> None:
    """Lệnh cài ngay sau nhãn ``skip_*`` sẽ chạy dù tính năng bị TẮT."""
    tool = _load_tool()
    path = tmp_path / "bad.bat"
    path.write_text(_BROKEN_BATCH, encoding="utf-8")
    issues = tool.check_batch_file(path)
    assert any("VÔ ĐIỀU KIỆN" in issue.message for issue in issues)


def test_clean_file_has_no_issues(tmp_path: Path) -> None:
    """KHÔNG được báo động giả với tệp đúng."""
    tool = _load_tool()
    path = tmp_path / "ok.bat"
    path.write_text(_CLEAN_BATCH, encoding="utf-8")
    assert tool.check_batch_file(path) == []


def test_detects_missing_goto_label(tmp_path: Path) -> None:
    tool = _load_tool()
    path = tmp_path / "bad.bat"
    path.write_text("@echo off\ngoto :khong_ton_tai\n", encoding="utf-8")
    issues = tool.check_batch_file(path)
    assert any("không có nhãn" in issue.message for issue in issues)


def test_goto_eof_is_allowed(tmp_path: Path) -> None:
    """``goto :eof`` là nhãn dựng sẵn của batch — không phải lỗi."""
    tool = _load_tool()
    path = tmp_path / "ok.bat"
    path.write_text("@echo off\ngoto :eof\n", encoding="utf-8")
    assert tool.check_batch_file(path) == []


def test_detects_unbalanced_parentheses(tmp_path: Path) -> None:
    tool = _load_tool()
    path = tmp_path / "bad.bat"
    path.write_text('@echo off\nif exist x (\n  echo co\n', encoding="utf-8")
    issues = tool.check_batch_file(path)
    assert any(")" in issue.message for issue in issues)


def test_comments_are_ignored(tmp_path: Path) -> None:
    """Nhãn/goto trong dòng REM không được tính."""
    tool = _load_tool()
    path = tmp_path / "ok.bat"
    path.write_text(
        "@echo off\nREM goto :khong_co_that\nREM :nhan_gia\necho xong\n",
        encoding="utf-8",
    )
    assert tool.check_batch_file(path) == []


# ── Tệp thật của dự án ───────────────────────────────────────────────────────
def test_project_batch_files_are_clean() -> None:
    """``build_windows.bat`` thật phải sạch — đây là canh giữ chính."""
    tool = _load_tool()
    import subtitles_extractor.domain.entities.project_record as anchor

    root = Path(anchor.__file__).resolve().parents[4]
    for path in sorted(root.glob("*.bat")):
        issues = tool.check_batch_file(path)
        assert not issues, f"{path.name}: {[i.message for i in issues]}"


def test_build_script_installs_whisperx_only_into_isolated_env() -> None:
    """Không được còn lệnh cài whisperx vào môi trường chính."""
    import subtitles_extractor.domain.entities.project_record as anchor

    root = Path(anchor.__file__).resolve().parents[4]
    script = root / "build_windows.bat"
    if not script.is_file():
        pytest.skip("Không tìm thấy build_windows.bat")
    text = script.read_text(encoding="utf-8", errors="replace")

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("REM", "::")) or "pip install" not in stripped:
            continue
        if "whisperx" in stripped or "torch" in stripped:
            assert "whisperx_env" in stripped, f"Cài vào môi trường chính: {stripped}"


def test_build_script_runs_both_checkers() -> None:
    """Cả hai công cụ kiểm phải chạy TRONG build, không phải nhớ chạy tay."""
    import subtitles_extractor.domain.entities.project_record as anchor

    root = Path(anchor.__file__).resolve().parents[4]
    script = root / "build_windows.bat"
    if not script.is_file():
        pytest.skip("Không tìm thấy build_windows.bat")
    text = script.read_text(encoding="utf-8", errors="replace")
    assert "check_batch_syntax.py" in text
    assert "check_dependency_conflicts.py" in text
