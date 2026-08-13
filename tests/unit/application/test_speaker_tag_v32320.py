"""Test [v3.23.20] tag tên người nói: chỉ khi ĐỔI người, bỏ SFX & N/A."""

from __future__ import annotations

from dataclasses import replace

from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationLine,
)


def _ctx(tags: bool = True, desc: bool = False) -> TranslationContext:
    base = TranslationContext(characters="", overview="", source_lang="en", target_lang="vi")
    return replace(base, enable_tags=tags, include_desc=desc)


def _line(idx: int, text: str, speaker: str = "", desc: str = "") -> TranslationLine:
    return TranslationLine(index=idx, start_ms=0, end_ms=1, text=text,
                           speaker=speaker, description=desc)


class TestSpeakerTagging:
    def test_tag_only_on_speaker_change(self) -> None:
        ctx = _ctx()
        comp = TranslateSubtitlesUseCase._compose_display_text
        t1, s1 = comp(_line(1, "Câu A", "An"), "Câu A", ctx, "")
        t2, s2 = comp(_line(2, "Câu B", "An"), "Câu B", ctx, s1)
        t3, s3 = comp(_line(3, "Câu C", "Bình"), "Câu C", ctx, s2)
        assert t1 == "[An:] Câu A"   # đổi (từ rỗng) → tag
        assert t2 == "Câu B"          # cùng An → không tag
        assert t3 == "[Bình:] Câu C"  # đổi → tag

    def test_sound_effect_never_tagged(self) -> None:
        ctx = _ctx()
        comp = TranslateSubtitlesUseCase._compose_display_text
        t, s = comp(_line(1, "(nhạc kịch tính)", "Người dẫn"), "(nhạc kịch tính)", ctx, "")
        assert t == "(nhạc kịch tính)"
        assert s == ""  # reset người nói

    def test_sfx_resets_speaker(self) -> None:
        ctx = _ctx()
        comp = TranslateSubtitlesUseCase._compose_display_text
        _, s1 = comp(_line(1, "Xin chào", "An"), "Xin chào", ctx, "")
        _, s2 = comp(_line(2, "(tiếng cười)", "An"), "(tiếng cười)", ctx, s1)
        t3, _ = comp(_line(3, "Tiếp tục", "An"), "Tiếp tục", ctx, s2)
        assert t3 == "[An:] Tiếp tục"  # sau SFX → tag lại

    def test_na_speaker_not_tagged(self) -> None:
        ctx = _ctx()
        comp = TranslateSubtitlesUseCase._compose_display_text
        for bad in ("N/A", "", "unknown", "không rõ", "?"):
            t, _ = comp(_line(1, "Câu", bad), "Câu", ctx, "")
            assert t == "Câu", f"speaker '{bad}' không được tag"

    def test_tags_disabled(self) -> None:
        ctx = _ctx(tags=False)
        comp = TranslateSubtitlesUseCase._compose_display_text
        t, _ = comp(_line(1, "Câu", "An"), "Câu", ctx, "")
        assert t == "Câu"

    def test_description_kept_for_sfx(self) -> None:
        ctx = _ctx(desc=True)
        comp = TranslateSubtitlesUseCase._compose_display_text
        t, _ = comp(_line(1, "(nhạc)", "An", "nền"), "(nhạc)", ctx, "")
        assert "(nền)" in t and "[An:]" not in t

    def test_is_sound_effect(self) -> None:
        f = TranslateSubtitlesUseCase._is_sound_effect
        assert f("(nhạc kịch tính)") is True
        assert f("[âm thanh nền]") is True
        assert f("♪ la la la ♪") is True
        assert f("Xin chào mọi người") is False
        assert f("(một phần) câu nói") is False  # không bao toàn bộ


class TestLocalizeSpeaker:
    def test_generic_translated(self) -> None:
        loc = TranslateSubtitlesUseCase._localize_speaker
        assert loc("Woman") == "Người phụ nữ"
        assert loc("Man") == "Người đàn ông"
        assert loc("Narrator") == "Người dẫn chuyện"
        assert loc("Crowd") == "Đám đông"

    def test_proper_names_kept(self) -> None:
        loc = TranslateSubtitlesUseCase._localize_speaker
        assert loc("John Teven") == "John Teven"
        assert loc("Tiến sĩ Jonh Teven") == "Tiến sĩ Jonh Teven"
        assert loc("Vera Rubin") == "Vera Rubin"

    def test_case_insensitive(self) -> None:
        loc = TranslateSubtitlesUseCase._localize_speaker
        assert loc("VOICE") == "Giọng nói"
        assert loc("narrator") == "Người dẫn chuyện"

    def test_tag_uses_localized(self) -> None:
        ctx = _ctx()
        comp = TranslateSubtitlesUseCase._compose_display_text
        t, _ = comp(_line(1, "Xin chào", "Woman"), "Xin chào", ctx, "")
        assert t == "[Người phụ nữ:] Xin chào"


class TestCjkRomanization:
    def test_roster_map_extraction(self) -> None:
        m = TranslateSubtitlesUseCase._build_roster_pronunciation_map(
            "林昆 (Lâm Côn), 王语嫣 (Vương Ngữ Yên), John Teven")
        assert m["林昆"] == "Lâm Côn"
        assert m["王语嫣"] == "Vương Ngữ Yên"

    def test_cjk_name_romanized(self) -> None:
        m = {"林昆": "Lâm Côn"}
        assert TranslateSubtitlesUseCase._localize_speaker("林昆", m) == "Lâm Côn"

    def test_cjk_without_roster_kept(self) -> None:
        # Không có roster → giữ nguyên CJK (model dịch xử lý ở câu thoại).
        assert TranslateSubtitlesUseCase._localize_speaker("林昆", {}) == "林昆"

    def test_latin_kept_even_with_roster(self) -> None:
        m = {"林昆": "Lâm Côn"}
        assert TranslateSubtitlesUseCase._localize_speaker("John Teven", m) == "John Teven"

    def test_generic_still_translated_with_roster(self) -> None:
        m = {"林昆": "Lâm Côn"}
        assert TranslateSubtitlesUseCase._localize_speaker("Woman", m) == "Người phụ nữ"

    def test_has_cjk_detection(self) -> None:
        assert TranslateSubtitlesUseCase._has_cjk("林昆") is True
        assert TranslateSubtitlesUseCase._has_cjk("さとう") is True  # hiragana
        assert TranslateSubtitlesUseCase._has_cjk("John") is False


class TestSpeakerChannelNotes:
    def test_man_on_computer(self) -> None:
        loc = TranslateSubtitlesUseCase._localize_speaker
        assert loc("MAN (on computer)", {}) == "Người đàn ông (trên máy tính)"

    def test_astronaut_on_radio(self) -> None:
        loc = TranslateSubtitlesUseCase._localize_speaker
        assert loc("ASTRONAUT (on radio)", {}) == "Phi hành gia (qua radio)"

    def test_nasa_announcer(self) -> None:
        loc = TranslateSubtitlesUseCase._localize_speaker
        assert loc("NASA ANNOUNCER", {}) == "Phát thanh viên NASA"

    def test_proper_name_with_channel_kept(self) -> None:
        loc = TranslateSubtitlesUseCase._localize_speaker
        assert loc("Swati Mohan (on radio)", {}) == "Swati Mohan (qua radio)"

    def test_cjk_with_channel(self) -> None:
        loc = TranslateSubtitlesUseCase._localize_speaker
        assert loc("林昆 (on phone)", {"林昆": "Lâm Côn"}) == "Lâm Côn (qua điện thoại)"

    def test_unknown_channel_note_kept(self) -> None:
        loc = TranslateSubtitlesUseCase._localize_speaker
        # Chú thích lạ → giữ nguyên, tên vẫn dịch.
        assert loc("WOMAN (whispering)", {}) == "Người phụ nữ (whispering)"
