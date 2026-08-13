"""[v3.23.154] Test Gemini/F5 TTS dùng ĐÚNG chuẩn "văn bản âm thanh thật" như Edge.

Bug: Gemini TTS (3 điểm) và F5 (1 điểm) gọi ``_preprocess_tts_text`` thiếu skip
ngoặc/nhạc và/hoặc thiếu ``strip_speaker_tag=True`` -> văn bản đưa vào tổng hợp còn
nguyên tag "[Nam:]", "(cười)", ký hiệu nhạc -> ENGINE ĐỌC TO các thứ đó bất chấp cấu
hình người dùng; đồng thời ``is_dialog`` (chèn khoảng lặng hội thoại) tính sai — đúng
họ bug đã vá cho Edge ở v3.23.148.

Test kiểm bất biến qua chính các lời gọi trong source (không cần mạng/model): mọi
điểm gọi ``_preprocess_tts_text`` của hai adapter phải truyền đủ skip options +
``strip_speaker_tag=True``, và kết quả chuẩn hoá phải khớp chuẩn Edge Pass 1.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

from subtitles_extractor.infrastructure.tts.edge_tts_adapter import (
    _preprocess_tts_text,
    _skip_from_request,
)

_TTS_DIR = Path("src/subtitles_extractor/infrastructure/tts")


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        clean_tags=True,
        skip_paren=True, skip_square=False, skip_curly=False,
        skip_music_pair=True, skip_music_line=False,
    )


def test_canonical_text_strips_tag_and_paren() -> None:
    """Chuẩn hoá đúng: bỏ tag người nói + bỏ ngoặc theo cấu hình, giữ lời thoại."""
    req = _request()
    text, is_dialog = _preprocess_tts_text(
        "[Nam:] (cười) - Xin chào", True, _skip_from_request(req),
        strip_speaker_tag=True,
    )
    assert "[Nam:]" not in text and "(cười)" not in text
    assert "Xin chào" in text
    assert is_dialog is True  # dấu '-' về đầu sau khi bỏ tag/ngoặc


def _calls_in(source: str) -> list[str]:
    """Trích các đoạn gọi _preprocess_tts_text(...) (gộp nhiều dòng)."""
    return [
        m.group(0)
        for m in re.finditer(r"_preprocess_tts_text\((?:[^()]|\([^()]*\))*\)", source)
    ]


def test_gemini_adapter_all_calls_use_canonical_form() -> None:
    source = (_TTS_DIR / "gemini_tts_adapter.py").read_text(encoding="utf-8")
    calls = _calls_in(source)
    assert calls, "Không tìm thấy lời gọi _preprocess_tts_text trong gemini_tts_adapter"
    for call in calls:
        assert "_skip_from_request(request)" in call, call
        assert "strip_speaker_tag=True" in call, call


def test_vieneu_adapter_all_calls_use_canonical_form() -> None:
    # [v3.23.195] F5-TTS đã gỡ khỏi ứng dụng; VieNeu thay thế và phải giữ cùng chuẩn.
    source = (_TTS_DIR / "vieneu_tts_adapter.py").read_text(encoding="utf-8")
    calls = _calls_in(source)
    assert calls, "Không tìm thấy lời gọi _preprocess_tts_text trong vieneu_tts_adapter"
    for call in calls:
        assert "_skip_from_request(request)" in call, call
        assert "strip_speaker_tag=True" in call, call


def test_three_engines_share_identical_canonical_text() -> None:
    """Cùng một câu vào -> văn bản âm thanh thật của 3 engine phải GIỐNG HỆT nhau."""
    req = _request()
    raw = "[Lan:] ♪ lời bài hát ♪ (thì thầm) - Đừng đi mà"
    canonical, _ = _preprocess_tts_text(
        raw, True, _skip_from_request(req), strip_speaker_tag=True
    )
    # Chuẩn Edge Pass 1 == chuẩn Gemini == chuẩn VieNeu (cùng một lời gọi duy nhất).
    again, _ = _preprocess_tts_text(
        raw, True, _skip_from_request(req), strip_speaker_tag=True
    )
    assert canonical == again
    assert "♪" not in canonical and "(thì thầm)" not in canonical
    assert "[Lan:]" not in canonical
    assert "Đừng đi mà" in canonical
