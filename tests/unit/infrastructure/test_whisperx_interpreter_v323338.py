"""Canh giữ trình thông dịch chạy worker WhisperX — v3.23.338.

LỖI THẬT ĐƯỢC SỬA: người dùng thấy ``WhisperX lỗi: Tiến trình WhisperX thoát mã 2``.

Nguyên nhân: v3.23.333 chuyển WhisperX sang môi trường riêng và sửa nhánh **phiên âm**
dùng ``resolve_whisperx_python()``, nhưng BỎ SÓT nhánh **căn chỉnh** — nó vẫn dùng
``sys.executable``. Ở bản đóng gói, ``sys.executable`` chính là tệp ``.exe`` của ứng
dụng; chạy nó với ``--mode align`` thì app không hiểu tham số và argparse trả **mã 2**.

Bộ test này phân tích cú pháp mã nguồn để bắt đúng loại lỗi "sửa một nhánh, sót nhánh
kia" — thứ mà đọc bằng mắt rất dễ bỏ qua.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


def _adapter_source() -> str:
    import subtitles_extractor.infrastructure.stt.whisperx_adapter as module

    return Path(module.__file__).read_text(encoding="utf-8")


def _command_lists(source: str) -> list[ast.List]:
    """Mọi danh sách trông như một lệnh subprocess (phần tử đầu là trình thông dịch)."""
    tree = ast.parse(source)
    results: list[ast.List] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.List) or not node.elts:
            continue
        # Lệnh subprocess luôn có ít nhất [python, script, ...].
        if len(node.elts) >= 2:
            results.append(node)
    return results


def test_no_command_starts_with_sys_executable() -> None:
    """BẤT BIẾN CHÍNH: không lệnh nào được chạy bằng ``sys.executable``.

    Ở bản đóng gói đó là tệp .exe của app — chạy nó sẽ mở lại ứng dụng hoặc thoát mã 2.
    """
    offenders: list[int] = []
    for node in _command_lists(_adapter_source()):
        first = node.elts[0]
        if (
            isinstance(first, ast.Attribute)
            and first.attr == "executable"
            and isinstance(first.value, ast.Name)
            and first.value.id == "sys"
        ):
            offenders.append(node.lineno)
    assert not offenders, f"Lệnh dùng sys.executable ở dòng: {offenders}"


@pytest.mark.parametrize("marker", ['"--mode", "align"', '"--audio", str(audio_path)'])
def test_both_worker_branches_resolve_isolated_python(marker: str) -> None:
    """CẢ HAI nhánh (phiên âm và căn chỉnh) phải phân giải môi trường riêng."""
    source = _adapter_source()
    index = source.index(marker)
    window = source[max(0, index - 500) : index]
    assert "python_exe" in window, f"Nhánh {marker} không dùng python_exe"


def test_both_branches_fail_clearly_when_env_missing() -> None:
    """Thiếu môi trường -> báo hướng dẫn, KHÔNG chạy nhầm rồi thoát mã khó hiểu."""
    source = _adapter_source()
    assert source.count("raise SpeechToTextError(_MISSING_ENV_MESSAGE)") >= 2


# ── Chẩn đoán mã thoát 2 ─────────────────────────────────────────────────────
def test_exit_code_two_has_actionable_message() -> None:
    """Mã 2 mà không có dòng lỗi = chạy nhầm trình thông dịch. Phải nói rõ cách sửa."""
    source = _adapter_source()
    assert "return_code == 2 and not error_lines" in source
    index = source.index("return_code == 2 and not error_lines")
    window = source[index : index + 1200]
    assert "Cài WhisperX tự động" in window  # trỏ tới nút trong app
    assert "whisperx_env" in window
    assert "cu129" in window


def test_stderr_is_preferred_over_bare_exit_code() -> None:
    """Có dòng lỗi thật thì phải hiện nó, đừng chỉ báo mã thoát."""
    source = _adapter_source()
    assert '"; ".join(error_lines) or' in source


# ── Đường dẫn thư viện torch ─────────────────────────────────────────────────
def test_torch_lib_resolved_from_isolated_env() -> None:
    """torch nằm trong ``whisperx_env``, không phải môi trường chính.

    Lấy nhầm từ ``sysconfig`` của tiến trình chính sẽ không tìm thấy gì, khiến bước căn
    chỉnh cấp từ lỗi bộ nhớ vì thiếu cuDNN.
    """
    source = _adapter_source()
    index = source.index("_build_subprocess_env")
    window = source[index : index + 1600]
    assert "resolve_whisperx_python()" in window
    assert "site-packages" in window
    # Không được suy từ sysconfig của tiến trình chính nữa.
    assert "sysconfig.get_paths()" not in window


def test_adapter_module_is_valid() -> None:
    ast.parse(_adapter_source())
