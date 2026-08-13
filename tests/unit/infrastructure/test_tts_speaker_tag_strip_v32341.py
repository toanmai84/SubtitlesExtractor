"""Test [v3.23.41/45] tag người nói: GIỮ cho file phụ đề, BỎ khi đọc audio TTS."""

from __future__ import annotations

from subtitles_extractor.infrastructure.tts.edge_tts_adapter import (
    _preprocess_tts_text,
    _SPEAKER_TAG_RE,
)


class TestSpeakerTagForAudio:
    """Khi strip_speaker_tag=True (dùng cho AUDIO) → bỏ tag để không đọc tên."""

    def test_strips_speaker_tag(self) -> None:
        cases = {
            "[Lâm Hằng:] Lũ phàm phu tục tử": "Lũ phàm phu tục tử",
            "[Hệ thống:] Con trai tông chủ": "Con trai tông chủ",
            "[Vân Dao:] Nhị sư tỷ": "Nhị sư tỷ",
        }
        for inp, exp in cases.items():
            got, _ = _preprocess_tts_text(inp, True, strip_speaker_tag=True)
            assert got == exp, f"{inp} → {got}"

    def test_full_width_colon(self) -> None:
        got, _ = _preprocess_tts_text("[林昆：] 你好", True, strip_speaker_tag=True)
        assert got == "你好"

    def test_tag_only_line_becomes_empty(self) -> None:
        got, _ = _preprocess_tts_text("[Lâm Hằng:]", True, strip_speaker_tag=True)
        assert got == ""


class TestSpeakerTagForSubtitle:
    """Mặc định (strip_speaker_tag=False, dùng cho FILE PHỤ ĐỀ) → GIỮ tag."""

    def test_keeps_speaker_tag(self) -> None:
        got, _ = _preprocess_tts_text("[Lâm Hằng:] Lũ phàm phu tục tử", True)
        assert got == "[Lâm Hằng:] Lũ phàm phu tục tử"

    def test_keeps_system_tag(self) -> None:
        got, _ = _preprocess_tts_text("[Hệ thống:] Chúc mừng", True)
        assert got == "[Hệ thống:] Chúc mừng"


class TestSharedBehavior:
    def test_keeps_non_speaker_brackets_both_modes(self) -> None:
        for strip in (True, False):
            got, _ = _preprocess_tts_text("[âm thanh nền]", True, strip_speaker_tag=strip)
            assert got == "[âm thanh nền]"

    def test_keeps_plain_text(self) -> None:
        got, _ = _preprocess_tts_text("Xin chào", True, strip_speaker_tag=True)
        assert got == "Xin chào"

    def test_keeps_parenthesis(self) -> None:
        got, _ = _preprocess_tts_text("(tiếng cười) Xin chào", True, strip_speaker_tag=True)
        assert got == "(tiếng cười) Xin chào"

    def test_regex_anchored_at_start(self) -> None:
        assert _SPEAKER_TAG_RE.search("Anh ấy nói [Lâm Hằng:] gì đó") is None
