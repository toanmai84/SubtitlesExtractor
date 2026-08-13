"""Test lùi về chạy trong tiến trình khi Python ngoài thiếu edge-tts — v3.23.342.

LỖI THẬT TỪ LOG NGƯỜI DÙNG::

    Edge TTS subprocess lỗi (mã 3): EDGE_TTS_MISSING: chưa cài edge-tts…
    Edge TTS probe lần 1/10: audio rỗng   … (lặp 10 lần, mất ~50 giây)
    TTSWorker gen error: Không thể kết nối Edge TTS sau 10 lần. Kiểm tra mạng.

Ba vấn đề trong một:

1. Mã 3 = Python NGOÀI thiếu thư viện. Nhưng bản đóng gói CÓ gom ``edge_tts``, nên chạy
   trong tiến trình là được — adapter lại chỉ trả ``False``.
2. Thử lại 10 lần cùng một cách chắc chắn thất bại, tốn ~50 giây.
3. Thông điệp cuối nói "Kiểm tra mạng" — SAI, mạng hoàn toàn bình thường.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _adapter_source() -> str:
    import subtitles_extractor.infrastructure.tts.edge_tts_adapter as module

    return Path(module.__file__).read_text(encoding="utf-8")


def _worker_source() -> str:
    import subtitles_extractor.infrastructure.tts.edge_tts_subprocess as module

    return Path(module.__file__).read_text(encoding="utf-8")


# ── Lùi về chạy trong tiến trình ─────────────────────────────────────────────
def test_missing_library_exit_code_is_named() -> None:
    """Mã 3 phải có hằng số đặt tên, không viết số cứng rải rác."""
    assert "_EXIT_CODE_LIBRARY_MISSING: int = 3" in _adapter_source()


def test_exit_code_three_falls_back_in_process() -> None:
    """Mã 3 -> chạy trong tiến trình, KHÔNG trả về thất bại."""
    source = _adapter_source()
    index = source.index("_EXIT_CODE_LIBRARY_MISSING:")
    window = source[index : index + 900]
    assert "_run_edge_in_process" in window


def test_remembers_to_skip_subprocess_next_time() -> None:
    """Biết rồi thì khỏi thử subprocess nữa — 55 câu × 2 giây là gần 2 phút."""
    source = _adapter_source()
    assert "_prefer_in_process: bool = False" in source
    assert "if EdgeTTSAdapter._prefer_in_process:" in source


def test_skip_happens_before_resolving_python() -> None:
    """Kiểm cờ TRƯỚC khi dựng lệnh, để không tốn công vô ích."""
    source = _adapter_source()
    start = source.index('worker = Path(__file__).with_name("edge_tts_subprocess.py")')
    window = source[start : start + 700]
    assert window.index("_prefer_in_process") < window.index("if python_exe is None")


# ── Thông điệp lỗi không được sai lệch ───────────────────────────────────────
def test_error_message_no_longer_blames_network_only() -> None:
    """Log thật chứng minh nguyên nhân là thiếu thư viện, không phải mạng."""
    source = _adapter_source()
    assert "Kiểm tra mạng." not in source


def test_error_message_lists_both_causes() -> None:
    """Phải nêu CẢ HAI khả năng và cách phân biệt chúng."""
    source = _adapter_source()
    index = source.index("Edge TTS không tạo được âm thanh sau")
    window = source[index : index + 900]
    assert "EDGE_TTS_MISSING" in window          # cách nhận biết lỗi thư viện
    assert "speech.platform.bing.com" in window  # cách nhận biết lỗi mạng
    assert "VieNeu" in window                    # gợi ý phương án offline


# ── Mã hoá thông điệp worker ─────────────────────────────────────────────────
def test_worker_forces_utf8_stderr() -> None:
    """Log thật hiện ``ch\\u01b0a c?i`` — chữ hỏng vì stderr Windows dùng cp1252."""
    source = _worker_source()
    assert 'reconfigure(encoding="utf-8"' in source


def test_worker_encoding_failure_is_tolerated() -> None:
    """Không phải nền tảng nào cũng cho ``reconfigure`` — không được vì thế mà sập."""
    source = _worker_source()
    index = source.index("reconfigure(encoding=")
    window = source[max(0, index - 200) : index + 200]
    assert "except" in window


@pytest.mark.parametrize(
    "marker", ["EDGE_TTS_MISSING", "EDGE_TTS_TIMEOUT", "EDGE_TTS_API_ERROR"]
)
def test_worker_uses_distinct_error_markers(marker: str) -> None:
    """Mỗi loại lỗi có dấu hiệu riêng để adapter phân biệt được nguyên nhân."""
    assert marker in _worker_source()
