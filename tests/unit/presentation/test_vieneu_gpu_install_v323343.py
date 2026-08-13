"""Test nút tự cài VieNeu cho GPU — v3.23.343.

Thay vì bắt người dùng gõ lệnh tay, giờ có nút. An toàn vì cài vào môi trường RIÊNG
``whisperx_env`` — không đụng môi trường của ứng dụng (nơi torch sẽ xung đột CUDA với
paddle).

Bộ test cũng canh giữ hai lỗi đã sửa cùng bản:
    * ``build_windows.bat`` từng có HAI khối kiểm bundle, và một khối chú thích bị chèn
      vào GIỮA dòng ``if exist build rmdir /s /q build``.
    * Hai công cụ kiểm bundle trùng chức năng.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest


def _page_source() -> str:
    import subtitles_extractor.domain.entities.project_record as anchor

    root = Path(anchor.__file__).resolve().parents[2]
    return (root / "presentation" / "pages" / "tts_page.py").read_text(encoding="utf-8")


def _worker_source() -> str:
    """Đọc THẲNG tệp, không import.

    Import module này sẽ kéo theo PySide6 — thứ không có trong môi trường test không
    màn hình. Các test khác trong dự án cũng đọc trực tiếp vì lý do đó.
    """
    import subtitles_extractor.domain.entities.project_record as anchor

    root = Path(anchor.__file__).resolve().parents[2]
    path = root / "presentation" / "workers" / "install_vieneu_gpu_worker.py"
    return path.read_text(encoding="utf-8")


def _project_root() -> Path:
    import subtitles_extractor.domain.entities.project_record as anchor

    return Path(anchor.__file__).resolve().parents[4]


# ── Nút và nối dây ───────────────────────────────────────────────────────────
def test_install_button_exists_and_is_in_layout() -> None:
    """Tạo nút mà quên thêm vào layout thì nó không bao giờ hiện."""
    source = _page_source()
    assert "self._btn_install_gpu = PushButton" in source
    assert 'addRow("", self._btn_install_gpu)' in source


def test_progress_bar_is_in_layout() -> None:
    source = _page_source()
    assert 'addRow("", self._gpu_install_progress)' in source


def test_signals_use_queued_connection() -> None:
    """Worker ở luồng khác — tín hiệu phải xếp hàng về luồng giao diện."""
    source = _page_source()
    index = source.index("self._gpu_install_worker.progress.connect")
    window = source[index : index + 900]
    assert "QueuedConnection" in window


def test_thread_state_initialised() -> None:
    """Thiếu khởi tạo -> ``getattr`` trả None ngẫu nhiên, dễ chạy hai lần."""
    assert "self._gpu_install_thread = None" in _page_source()


def test_button_hidden_when_already_ready() -> None:
    """Đã đủ gói thì không cần hiện nút nữa."""
    assert "self._btn_install_gpu.setVisible(False)" in _page_source()


def test_unchecks_force_cpu_after_install() -> None:
    """Cài xong phải TỰ bỏ tick — nếu không thì cài rồi vẫn chạy CPU."""
    source = _page_source()
    index = source.index("def _on_gpu_install_finished")
    window = source[index : index + 900]
    assert "self._vieneu_force_cpu.setChecked(False)" in window


def test_thread_is_cleaned_up_even_on_failure() -> None:
    """``done`` luôn phát ở cuối; dọn luồng phải móc vào đó."""
    source = _page_source()
    assert "self._gpu_install_worker.done.connect" in source
    assert "def _cleanup_gpu_install_thread" in source


# ── An toàn: chỉ cài vào môi trường riêng ────────────────────────────────────
def test_worker_refuses_commands_outside_isolated_env() -> None:
    """BẤT BIẾN: lệnh không trỏ vào môi trường riêng phải bị từ chối.

    Cài torch/vieneu vào môi trường chính sẽ gây đúng xung đột CUDA mà cả thiết kế này
    ra đời để tránh.
    """
    source = _worker_source()
    assert "if command[0] != plan.python_exe:" in source
    assert "Từ chối lệnh cài không trỏ vào môi trường riêng" in source


def test_worker_verifies_result_instead_of_trusting_exit_code() -> None:
    """Mã thoát 0 chưa chắc là cài được — phải lập kế hoạch lại để xác nhận."""
    source = _worker_source()
    index = source.index("verified = build_gpu_tts_plan()")
    window = source[index : index + 400]
    assert "not verified.is_ready" in window


def test_worker_guides_when_no_environment() -> None:
    """Chưa có môi trường riêng -> chỉ tới nút cài WhisperX (nơi có torch)."""
    source = _worker_source()
    assert "Cài WhisperX tự động" in source


def test_worker_logs_system_errors_with_traceback() -> None:
    assert "logger.exception" in _worker_source()


def test_worker_module_is_valid() -> None:
    ast.parse(_worker_source())


# ── Hai lỗi build script đã sửa ──────────────────────────────────────────────
def test_only_one_bundle_check_in_build_script() -> None:
    """Từng có HAI khối kiểm bundle gọi hai công cụ trùng chức năng."""
    script = _project_root() / "build_windows.bat"
    if not script.is_file():
        pytest.skip("Không tìm thấy build_windows.bat")
    text = script.read_text(encoding="utf-8", errors="replace")
    assert text.count("check_bundle") == 1


def test_duplicate_bundle_tool_removed() -> None:
    """``check_bundle_contents.py`` đã được gộp vào ``check_bundle.py`` rồi xoá."""
    tools = _project_root() / "tools"
    if not tools.is_dir():
        pytest.skip("Không tìm thấy tools/")
    assert not (tools / "check_bundle_contents.py").exists()
    assert (tools / "check_bundle.py").is_file()


def test_build_cleanup_command_survived() -> None:
    """Dòng ``rmdir /s /q build`` từng bị một khối chú thích chèn vào giữa làm mất."""
    script = _project_root() / "build_windows.bat"
    if not script.is_file():
        pytest.skip("Không tìm thấy build_windows.bat")
    text = script.read_text(encoding="utf-8", errors="replace")
    assert "if exist build rmdir /s /q build" in text


def test_batch_checker_detects_statement_swallowed_by_comment(tmp_path: Path) -> None:
    """Phép kiểm mới: bắt lệnh điều kiện bị REM nuốt mất phần thân.

    Lỗi cũ (``if exist build REM ...``) là cú pháp HỢP LỆ nên các phép kiểm trước không
    thấy gì — chính vì thế mới cần phép kiểm riêng cho mẫu này.
    """
    import importlib.util

    path = _project_root() / "tools" / "check_batch_syntax.py"
    if not path.is_file():
        pytest.skip("Không tìm thấy check_batch_syntax.py")
    spec = importlib.util.spec_from_file_location("_bt", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_bt"] = module
    spec.loader.exec_module(module)

    bad = tmp_path / "bad.bat"
    bad.write_text("@echo off\nif exist build REM --- chú thích ---\n", encoding="utf-8")
    issues = module.check_batch_file(bad)
    assert any("REM" in issue.message for issue in issues)

    good = tmp_path / "ok.bat"
    good.write_text(
        "@echo off\nREM if exist build rmdir\nif exist build rmdir /s /q build\n",
        encoding="utf-8",
    )
    assert module.check_batch_file(good) == []
