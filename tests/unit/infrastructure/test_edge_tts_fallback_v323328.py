"""Test đường chạy dự phòng của Edge TTS — v3.23.328.

VẤN ĐỀ ĐƯỢC SỬA: thiết kế cũ CỐ Ý không đóng gói edge-tts (cách ly GPL) nên bản đóng
gói **bắt buộc phải có Python hệ thống** mới chạy được Edge TTS. Dự án nay là mã nguồn
mở và spec ĐÃ gom ``edge_tts`` vào bundle — nên khi không có Python ngoài, chạy thẳng
trong tiến trình là hợp lệ và tốt hơn nhiều so với báo thất bại.

Thứ tự ưu tiên: subprocess (cách ly tốt) → trong tiến trình (dự phòng) → báo lỗi rõ.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest


def _adapter_source() -> str:
    import subtitles_extractor.infrastructure.tts.edge_tts_adapter as module

    return Path(module.__file__).read_text(encoding="utf-8")


def _in_process_body() -> str:
    """Thân hàm ``_run_edge_in_process`` — cắt tới định nghĩa kế tiếp.

    Cắt theo độ dài cố định dễ hụt khi hàm dài ra; lấy tới ``def`` kế tiếp là chắc chắn.
    """
    source = _adapter_source()
    start = source.index("async def _run_edge_in_process")
    end = source.index("def _resolve_python_for_worker", start)
    return source[start:end]


# ── Cấu trúc rẽ nhánh ────────────────────────────────────────────────────────
def test_in_process_fallback_exists() -> None:
    assert "async def _run_edge_in_process" in _adapter_source()


def test_subprocess_is_still_preferred() -> None:
    """Subprocess phải được thử TRƯỚC — cách ly giúp edge-tts treo không sập app.

    Đo trong THÂN hàm gọi, không so vị trí định nghĩa hàm (chúng nằm khác chỗ trong tệp).
    """
    source = _adapter_source()
    start = source.index("worker = Path(__file__).with_name(\"edge_tts_subprocess.py\")")
    window = source[start : start + 900]
    assert window.index("_resolve_python_for_worker()") < window.index(
        "_run_edge_in_process("
    )


def test_fallback_triggers_when_no_external_python() -> None:
    """Không có Python ngoài -> phải lùi về trong tiến trình, KHÔNG trả False ngay."""
    source = _adapter_source()
    marker = source.index("python_exe = EdgeTTSAdapter._resolve_python_for_worker()")
    window = source[marker : marker + 900]
    assert "if python_exe is None" in window
    assert "_run_edge_in_process" in window


def test_missing_worker_file_also_falls_back() -> None:
    """Bản đóng gói thiếu tệp worker cũng phải lùi về trong tiến trình.

    [v3.23.342] Cắt cửa sổ theo ĐỘ DÀI CỐ ĐỊNH là sai: v3.23.342 chèn thêm phần kiểm
    cờ ``_prefer_in_process`` nên khối dài ra và ``worker.is_file()`` rơi ra ngoài 400
    ký tự — test báo hỏng dù code vẫn đúng. Nay cắt tới ``def`` kế tiếp.
    """
    source = _adapter_source()
    marker = source.index("python_exe = EdgeTTSAdapter._resolve_python_for_worker()")
    end = source.index("\n    def ", marker)
    window = source[marker:end]
    assert "worker.is_file()" in window


def test_in_process_reuses_worker_converter() -> None:
    """Phải tái dùng ``_mp3_to_wav`` của worker để hai đường cho kết quả y hệt."""
    body = _in_process_body()
    assert "edge_tts_subprocess import" in body
    assert "_mp3_to_wav" in body


def test_in_process_has_timeout_and_cleanup() -> None:
    """Mạng chậm không được treo vô hạn; tệp tạm phải được dọn."""
    body = _in_process_body()
    assert "timeout=30.0" in body
    assert "unlink(missing_ok=True)" in body


def test_in_process_reports_clearly_when_library_absent() -> None:
    """Thiếu cả Python ngoài lẫn thư viện -> phải nói rõ cách khắc phục."""
    body = _in_process_body()
    assert "pip install edge-tts" in body
    assert "VieNeu" in body  # gợi ý engine thay thế


def test_adapter_module_is_syntactically_valid() -> None:
    ast.parse(_adapter_source())


# ── Hợp đồng với thư viện edge-tts ───────────────────────────────────────────
edge_tts = pytest.importorskip("edge_tts", reason="Cần edge-tts để kiểm hợp đồng API")


def test_communicate_accepts_arguments_adapter_uses() -> None:
    """Adapter gọi ``Communicate(text, voice, rate=...)`` — API phải khớp."""
    signature = inspect.signature(edge_tts.Communicate.__init__)
    for name in ("text", "voice", "rate"):
        assert name in signature.parameters


def test_save_is_awaitable() -> None:
    """Adapter ``await communicate.save(...)`` — phải là coroutine."""
    assert inspect.iscoroutinefunction(edge_tts.Communicate.save)


def test_required_dependency_tabulate_is_importable() -> None:
    """``tabulate`` là phụ thuộc BẮT BUỘC của edge-tts, dễ bị bỏ sót khi đóng gói."""
    pytest.importorskip("tabulate")


# ── Khai báo phụ thuộc ───────────────────────────────────────────────────────
def _requirements_text() -> str:
    import subtitles_extractor.domain.entities.project_record as anchor

    root = Path(anchor.__file__).resolve().parents[4]
    path = root / "requirements-gpu.txt"
    if not path.is_file():
        pytest.skip("Không tìm thấy requirements-gpu.txt")
    return path.read_text(encoding="utf-8")


def _declares(text: str, package: str) -> bool:
    """Gói có được KHAI BÁO không — BỎ QUA dòng chú thích.

    Đây chính là chỗ v3.23.312 mắc lỗi: điều kiện bảo vệ khớp nhầm chữ trong chú thích
    nên khối thêm gói bị bỏ qua âm thầm.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.lower().split("=")[0].split(">")[0].split("<")[0].strip() == package.lower():
            return True
    return False


@pytest.mark.parametrize(
    "package", ["edge-tts", "pedalboard", "fastembed", "scikit-learn"]
)
def test_quality_packages_are_declared(package: str) -> None:
    """LỖI ĐÃ SỬA: 4 gói này bị bỏ sót suốt từ v3.23.312 do điều kiện bảo vệ sai."""
    assert _declares(_requirements_text(), package)


def test_comment_mentioning_package_does_not_count_as_declaration() -> None:
    """Canh giữ chính cái bẫy đã mắc: chú thích nhắc tên gói KHÔNG phải khai báo."""
    sample = "# librosa thay pedalboard GPL\nlibrosa\n"
    assert _declares(sample, "librosa")
    assert not _declares(sample, "pedalboard")
