"""Canh giữ: tác vụ nền phải GHI LOG khi thất bại — v3.23.340.

VẤN ĐỀ NGƯỜI DÙNG BÁO: *"Chạy bị lỗi mà nhật ký không thấy log gì."*

Các worker bắt lỗi rồi ``emit`` tín hiệu thất bại nhưng **không ghi log lần nào**. Lỗi
chỉ hiện trên màn hình vài giây rồi biến mất; trang Nhật ký trống trơn nên không có
cách nào chẩn đoán khi người dùng báo lại.

Rà soát thấy **10 chỗ** như vậy ở 4 tệp worker khác nhau — không phải trường hợp cá biệt.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_LOG_CALLS = ("logger.error", "logger.exception", "logger.warning")


def _workers_dir() -> Path:
    import subtitles_extractor.domain.entities.project_record as anchor

    root = Path(anchor.__file__).resolve().parents[2]
    path = root / "presentation" / "workers"
    if not path.is_dir():
        pytest.skip("Không tìm thấy thư mục workers/")
    return path


def _handlers_emitting_failure(source: str) -> list[tuple[int, str]]:
    """Các khối ``except`` có phát tín hiệu thất bại, kèm nội dung khối."""
    results: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ExceptHandler):
            continue
        body = ast.unparse(node)
        if ".failed" in body or "_failed.emit" in body:
            results.append((node.lineno, body))
    return results


def test_every_failure_handler_logs() -> None:
    """BẤT BIẾN CHÍNH: phát tín hiệu lỗi thì phải ghi log trước.

    Không ghi log = người dùng báo lỗi mà không ai chẩn đoán được.
    """
    offenders: list[str] = []
    for path in sorted(_workers_dir().glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for lineno, body in _handlers_emitting_failure(source):
            if not any(call in body for call in _LOG_CALLS):
                offenders.append(f"{path.name}:{lineno}")
    assert not offenders, f"Bắt lỗi mà không ghi log tại: {offenders}"


def test_workers_declare_a_logger() -> None:
    """Tệp có xử lý lỗi phải khai báo logger — nếu không sẽ NameError lúc chạy."""
    offenders: list[str] = []
    for path in sorted(_workers_dir().glob("*.py")):
        source = path.read_text(encoding="utf-8")
        if not _handlers_emitting_failure(source):
            continue
        if "logger = logging.getLogger(__name__)" not in source:
            offenders.append(path.name)
    assert not offenders, f"Thiếu khai báo logger: {offenders}"


def test_system_errors_use_exception_for_traceback() -> None:
    """Lỗi hệ thống (OSError/RuntimeError) nên dùng ``logger.exception``.

    ``exception`` ghi kèm traceback — thứ cần nhất để chẩn đoán lỗi không lường trước.
    """
    weak: list[str] = []
    for path in sorted(_workers_dir().glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ExceptHandler) or node.type is None:
                continue
            caught = ast.unparse(node.type)
            body = ast.unparse(node)
            if ".failed" not in body and "_failed.emit" not in body:
                continue
            is_system = any(k in caught for k in ("OSError", "RuntimeError", "Exception"))
            if is_system and "logger.exception" not in body:
                weak.append(f"{path.name}:{node.lineno}")
    assert not weak, f"Lỗi hệ thống nên dùng logger.exception tại: {weak}"


def test_transcribe_worker_logs_failures() -> None:
    """Kiểm riêng worker phiên âm — đúng nơi người dùng gặp sự cố."""
    source = (_workers_dir() / "transcribe_worker.py").read_text(encoding="utf-8")
    handlers = _handlers_emitting_failure(source)
    assert handlers
    for _lineno, body in handlers:
        assert any(call in body for call in _LOG_CALLS)


# ── Tệp worker phụ trợ phải nằm trong bản đóng gói ───────────────────────────
def test_subprocess_workers_are_bundled() -> None:
    """Worker chạy bằng ``python <script>`` PHẢI có trong bundle.

    Thiếu tệp thì Python báo "can't open file" và thoát **mã 2** — trùng mã của lỗi sai
    đối số, nên rất dễ chẩn đoán nhầm. Đây chính là sự cố đã xảy ra với WhisperX.
    """
    import subtitles_extractor.domain.entities.project_record as anchor

    root = Path(anchor.__file__).resolve().parents[4]
    spec = root / "SubtitlesExtractor.spec"
    if not spec.is_file():
        pytest.skip("Không tìm thấy SubtitlesExtractor.spec")
    text = spec.read_text(encoding="utf-8")

    for worker in ("whisperx_subprocess.py", "edge_tts_subprocess.py"):
        assert worker in text, f"{worker} chưa được nhúng vào bundle"


def test_adapter_checks_worker_script_exists() -> None:
    """Adapter phải kiểm tệp worker TỒN TẠI trước khi chạy, để báo lỗi rõ ràng."""
    import subtitles_extractor.infrastructure.stt.whisperx_adapter as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "_ensure_worker_script" in source
    # Cả hai nhánh (phiên âm + căn chỉnh) đều phải kiểm.
    assert source.count("_ensure_worker_script(worker_script)") >= 2
