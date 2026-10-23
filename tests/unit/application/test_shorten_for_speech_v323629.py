"""[v3.23.629] Dịch ↔ TTS: tìm dòng không vừa khung giọng đọc và rút gọn bằng AI (có kiểm tra)."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from subtitles_extractor.application.use_cases.shorten_for_speech import (
    ShortenForSpeechUseCase,
    validate_shortened,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import SubtitleTranslationError
from subtitles_extractor.infrastructure.tts.tts_fit import (
    REASON_FAST,
    REASON_PREDICTED,
    REASON_TRUNCATED,
    LineFit,
    measured_misfits,
    merge_misfits,
    natural_seconds,
    predict_misfits,
    target_syllables_for,
)


def _ev(start: float, end: float, text: str):
    return NS(start_sec=start, end_sec=end, text=text)


def _res(index: int, speed: float = 1.0, truncated: bool = False, overlap: float = 0.0,
         skipped: bool = False):
    return NS(event_index=index, speed_used=speed, was_truncated=truncated, overlap_s=overlap,
              has_audible_overlap=overlap > 0.001, was_skipped=skipped)


class TestFitModel:
    def test_natural_duration_matches_the_measured_voice(self) -> None:
        # Đo 141 dòng thật: 0,312 + 0,177 x âm tiết.
        assert natural_seconds(5) == pytest.approx(1.197)
        assert target_syllables_for(1.2, 1.0) == 5
        assert target_syllables_for(0.2) == 1

    def test_predict_flags_only_clearly_too_long_lines(self) -> None:
        events = [
            _ev(0.0, 0.5, "Nghịch đồ vô liêm sỉ lắm"),     # 6 âm tiết trong 0,5 s, sát câu sau → nén ~2,7×
            _ev(0.6, 3.0, "Ngươi đi đi"),                    # thoải mái
            _ev(3.0, 3.5, "Ừ"),                              # quá ngắn để đụng
            _ev(5.0, 5.4, "Nhiệm vụ thứ hai"),               # 4 âm tiết nhưng khung + gap rộng
        ]
        fits = predict_misfits(events)
        assert [f.position for f in fits] == [0]
        assert fits[0].reason == REASON_PREDICTED and fits[0].target_syllables < fits[0].syllables

    def test_never_asks_for_more_than_a_40_percent_cut(self) -> None:
        # 10 âm tiết trong 0,5 s: mục tiêu thô ~2 → kẹp lên 6 (60%).
        fit = predict_misfits([_ev(0.0, 0.5, "một hai ba bốn năm sáu bảy tám chín mười"),
                               _ev(0.6, 1.0, "x")])[0]
        assert fit.target_syllables == 6

    def test_short_lines_are_left_alone(self) -> None:
        assert predict_misfits([_ev(0.0, 0.2, "đi ngay đi"), _ev(0.3, 0.5, "x")]) == []

    def test_measured_uses_real_speed_truncation_and_overlap(self) -> None:
        events = [_ev(0, 0.95, "một hai ba bốn năm sáu"), _ev(1, 1.95, "bảy tám chín mười mười một"),
                  _ev(2, 2.95, "ừ ờ à"), _ev(3, 3.95, "a b c d e f g h")]
        results = [_res(0, speed=1.9), _res(1, truncated=True), _res(2, speed=1.0),
                   _res(3, overlap=0.6)]
        fits = measured_misfits(events, results)
        assert [(f.position, f.reason) for f in fits] == [
            (0, REASON_FAST), (1, REASON_TRUNCATED), (3, "overlap")]
        assert measured_misfits(events, [_res(0, speed=1.3)]) == []          # nén nhẹ: không đụng
        assert measured_misfits(events, [_res(0, speed=2.5, skipped=True)]) == []

    def test_merge_prefers_later_groups_and_sorts(self) -> None:
        a = LineFit(2, 6, 4, 1.0, REASON_PREDICTED)
        b = LineFit(2, 6, 4, 1.0, REASON_FAST, 1.9)
        c = LineFit(0, 5, 3, 1.0, REASON_PREDICTED)
        assert merge_misfits([a, c], [b]) == [c, b]


class TestValidation:
    FIT = LineFit(0, 7, 5, 1.0, REASON_FAST, 1.9)

    def test_accepts_a_faithful_shortening(self) -> None:
        assert validate_shortened(self.FIT, "Đều có thể giải trừ một phần", "Đều giải trừ một phần") \
            == "Đều giải trừ một phần"

    def test_rejects_content_from_a_neighbouring_line(self) -> None:
        """Ca thật: "Hiện giải phóng cho ký chủ" → "Mở toàn lực một ngày"."""
        assert validate_shortened(self.FIT, "Hiện giải phóng cho ký chủ", "Mở toàn lực một ngày") == ""

    def test_rejects_invented_words_and_synonym_swaps(self) -> None:
        assert validate_shortened(self.FIT, "Chỉ có sư thúc là người luôn", "Sư thúc luôn chăm sóc") == ""
        assert validate_shortened(
            self.FIT, "Không màng tới ý kiến của chúng", "Mặc kệ ý họ") == ""

    def test_rejects_unchanged_longer_empty_and_over_cut(self) -> None:
        before = "Nhưng sư huynh đây đã không còn nữa rồi"
        assert validate_shortened(self.FIT, before, before) == ""
        assert validate_shortened(self.FIT, before, "") == ""
        assert validate_shortened(self.FIT, before, before + " thật đấy") == ""
        assert validate_shortened(self.FIT, before, "Nhưng") == ""

    def test_keeps_the_speaker_tag_and_blocks_new_cjk(self) -> None:
        out = validate_shortened(self.FIT, "[Diệp Nhiên:] Đều có thể giải trừ một phần",
                                 "Đều giải trừ một phần")
        assert out.startswith("[Diệp Nhiên:] ")
        assert validate_shortened(self.FIT, "Đều có thể giải trừ một phần", "Đều giải 解除 một phần") == ""


class TestUseCase:
    def _texts(self) -> list[str]:
        return ["Xin chào", "Đều có thể giải trừ một phần", "Hiện giải phóng cho ký chủ", "Tạm biệt"]

    def test_sends_context_and_keeps_only_valid_answers(self) -> None:
        seen: dict = {}

        class _Translator:
            def shorten_for_speech(self, items, lang, model, cancel_cb=None):
                seen.update(items=items, lang=lang, model=model)
                return {2: "Đều giải trừ một phần", 3: "Mở toàn lực một ngày", 99: "x"}

        fits = [LineFit(1, 7, 5, 1.0, REASON_FAST, 1.9), LineFit(2, 6, 5, 1.0, REASON_FAST, 1.8)]
        out = ShortenForSpeechUseCase(_Translator()).execute(
            fits, self._texts(), "Vietnamese", "m", source_texts=["", "都可以解除一部分", "现在释放"],
            speakers=["", "叶然", ""], addressees=["", "瑶瑶", ""],
        )
        assert [(s.position, s.after) for s in out] == [(1, "Đều giải trừ một phần")]
        assert out[0].syllables_before == 7 and out[0].syllables_after == 5
        first = seen["items"][0]
        assert first["id"] == 2 and first["toi_da_am_tiet"] == 5
        assert first["nguon"] == "都可以解除一部分" and first["nguoi_noi"] == "叶然"
        assert first["nguoi_nghe"] == "瑶瑶" and first["truoc"] == ["Xin chào"]
        assert first["sau"] == ["Hiện giải phóng cho ký chủ", "Tạm biệt"]

    def test_unsupported_translator_and_no_fits(self) -> None:
        with pytest.raises(SubtitleTranslationError):
            ShortenForSpeechUseCase(object()).execute([LineFit(0, 5, 3, 1, REASON_FAST)], ["a b c d e"], "vi", "m")
        assert ShortenForSpeechUseCase(NS(shorten_for_speech=lambda *a, **k: {})).execute(
            [], [], "vi", "m") == []
