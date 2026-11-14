"""[v3.23.702] Dịch ↔ TTS: ngân sách số TIẾNG theo nhịp đọc VieNeu + lượt rút gọn tự động sau khi dịch.

Lượt chạy thật phim 86 tập (3.5-flash-lite): 500/3664 dòng TTS phải đọc nhanh > 1,5× vì ngân sách cũ tính theo nhịp
Gemini TTS ở 1,9× (≈ 2× với VieNeu) và prompt dặn "cứ vượt, phần mềm tự nén".
"""

from __future__ import annotations

from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleTranslationError,
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.infrastructure.tts.tts_fit import translation_word_budget
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import GeminiSubtitleTranslator


def test_word_budget_follows_vieneu_at_1_5x() -> None:
    # 0,72 s (câu thật "mà đã bị cái hệ thống", 6 tiếng, VieNeu phải đọc 1,82×) → tối đa 3 tiếng.
    assert translation_word_budget(0.72) == 3
    assert translation_word_budget(2.0) == 11
    assert translation_word_budget(0.1) == 2          # sàn: không ép câu về 0-1 tiếng
    assert translation_word_budget(0.0) == 2


def _line(index: int, start: float, end: float, text: str) -> TranslationLine:
    return TranslationLine(index=index, start_ms=int(start * 1000), end_ms=int(end * 1000), text=text)


def test_payload_sends_max_words_not_max_chars() -> None:
    translator = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    translator._reading_cps_budget = 0.0
    data = translator._line_to_payload(_line(1, 0.0, 0.72, "就被系统"), next_start_ms=720)
    assert data["max_words"] == 3 and "max_chars" not in data
    # Câu cuối (không biết câu sau): chỉ khung gốc 2 s → 11 tiếng.
    assert translator.line_word_budgets([_line(1, 0.0, 0.72, "a"), _line(2, 0.72, 2.72, "b")]) == {1: 3, 2: 11}


class _Translator:
    def __init__(self, answers: dict[int, str] | None = None, fail: bool = False) -> None:
        self.answers = answers or {}
        self.fail = fail
        self.items: list[dict] = []

    def line_word_budgets(self, lines):
        return {line.index: 3 for line in lines}

    def shorten_for_speech(self, items, lang, model, cancel_cb=None):
        if self.fail:
            raise SubtitleTranslationError("503")
        self.items = items
        return self.answers


def _run(translator, lines):
    use_case = TranslateSubtitlesUseCase.__new__(TranslateSubtitlesUseCase)
    use_case._translator = translator
    request = TranslateSubtitlesRequest(
        events=[], context=TranslationContext(target_lang="Vietnamese"),
        stages=[TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="gemini-3.5-flash-lite")],
    )
    source = [_line(ln.index, 0, 1, "源") for ln in lines]
    return use_case._fit_lines_for_speech(lines, source, request, None)


def test_only_lines_over_budget_by_more_than_one_word_are_shortened() -> None:
    lines = [_line(1, 0, 1, "Ngươi đi đi"), _line(2, 1, 2, "mà đã bị cái hệ thống này"),
             _line(3, 2, 3, "ta đi đây nhé")]       # 4 tiếng = ngân sách + 1 → để yên
    translator = _Translator({2: "đã bị hệ thống này"})
    out = _run(translator, lines)
    assert [item["id"] for item in translator.items] == [2]
    assert [ln.text for ln in out] == ["Ngươi đi đi", "đã bị hệ thống này", "ta đi đây nhé"]


def test_invalid_shortening_and_api_errors_keep_the_translation() -> None:
    lines = [_line(1, 0, 1, "Nhưng sư huynh đây đã không còn nữa rồi")]
    assert _run(_Translator({1: "Mở toàn lực một ngày"}), lines)[0].text == lines[0].text   # bịa câu khác
    assert _run(_Translator(fail=True), lines)[0].text == lines[0].text


def test_thinks_only_at_high_models() -> None:
    from subtitles_extractor.infrastructure.translation.thinking_policy import thinks_only_at_high

    assert thinks_only_at_high("gemini-3.5-flash-lite")
    assert thinks_only_at_high("gemini-3.5-flash-lite-preview-11-2026")
    assert not thinks_only_at_high("gemini-3.1-flash-lite")
    assert not thinks_only_at_high("gemini-3.8-flash")
    assert not thinks_only_at_high("")


def test_shortening_that_flips_meaning_markers_is_rejected() -> None:
    """Ca thật A/B 361 câu: bộ kiểm cũ cho qua cả bốn câu dưới."""
    from subtitles_extractor.application.use_cases.shorten_for_speech import validate_shortened
    from subtitles_extractor.infrastructure.tts.tts_fit import REASON_PREDICTED, LineFit

    fit = LineFit(0, 11, 5, 1.0, REASON_PREDICTED)
    assert validate_shortened(fit, "thậm chí một đạo lôi kiếp cũng không qua nổi.", "qua nổi một kiếp.") == ""
    assert validate_shortened(fit, "Không lẽ ngay cả đạo lôi kiếp đầu tiên", "Đạo lôi kiếp đầu à?") == ""
    assert validate_shortened(fit, "Diệp Nhiên là đồ nhi của ta, thật đấy", "Diệp Nhiên là đồ tôi.") == ""
    assert validate_shortened(fit, "Thiên Thanh sư huynh, huynh điên rồi sao?", "Sư huynh điên rồi sao?") == ""
    # Rút gọn đúng vẫn qua.
    assert validate_shortened(fit, "mà đã tăng lên năm mươi đạo lôi kiếp rồi.", "đã tăng năm mươi đạo rồi.") \
        == "đã tăng năm mươi đạo rồi."


def test_analysis_level_upgrades_only_for_models_that_think_only_at_high() -> None:
    translator = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    translator._analysis_thinking_level = "medium"
    assert translator._analysis_level_for("gemini-3.5-flash-lite") == "high"
    assert translator._analysis_level_for("gemini-3.1-flash-lite") == "medium"
    translator._analysis_thinking_level = "minimal"          # người dùng chủ động tắt suy nghĩ
    assert translator._analysis_level_for("gemini-3.5-flash-lite") == "minimal"


def test_two_char_name_with_misread_surname_is_fixed() -> None:
    """A/B phân tích thật (3.5-flash-lite, mức high): ``叶然 (Lâm Nhiên)`` — "Lâm" là âm của 林."""
    from subtitles_extractor.application.services.name_reading_check import NameReadings, correct_context_names

    readings = NameReadings.from_encoded({"叶": "hiệp,diệp", "然": "nhiên", "林": "lâm", "胖": "bàn", "虎": "hổ"})
    context = TranslationContext(
        target_lang="Vietnamese",
        characters="叶然 (Lâm Nhiên) [nam] — nhân vật chính, sư phụ gọi Lâm Nhiên\n胖虎 (Hổ Béo) [nam] — bạn",
    )
    fixed, fixes = correct_context_names(context, readings)
    assert "Lâm Nhiên" not in fixed.characters and fixed.characters.count("Diệp Nhiên") == 2
    assert "Hổ Béo" in fixed.characters          # tên Việt hoá có chủ ý: không đụng
    assert [f.char for f in fixes] == ["叶"]


def test_non_vietnamese_target_keeps_char_budget_and_skips_fit_pass() -> None:
    from subtitles_extractor.infrastructure.tts.tts_fit import is_vietnamese_target

    assert is_vietnamese_target("Vietnamese") and is_vietnamese_target("Tiếng Việt") and is_vietnamese_target("vi")
    assert not is_vietnamese_target("English") and not is_vietnamese_target("")
    translator = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    translator._reading_cps_budget = 0.0
    translator._speech_words = False
    data = translator._line_to_payload(_line(1, 0.0, 1.0, "你好"), next_start_ms=1000)
    assert "max_chars" in data and "max_words" not in data
    use_case = TranslateSubtitlesUseCase.__new__(TranslateSubtitlesUseCase)
    use_case._translator = _Translator({1: "x"})
    request = TranslateSubtitlesRequest(events=[], context=TranslationContext(target_lang="English"), stages=[
        TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="m")])
    lines = [_line(1, 0, 1, "one two three four five six seven")]
    assert use_case._fit_lines_for_speech(lines, lines, request, None) == lines


def test_prompt_length_rules_follow_target_language() -> None:
    translator = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    stage = TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="m")
    vi = translator._system_instruction(stage, TranslationContext(target_lang="Vietnamese"), False)
    en = translator._system_instruction(stage, TranslationContext(target_lang="English"), False)
    assert "số TIẾNG" in vi and "max_words" in vi
    assert "max_words" not in en and "số KÝ TỰ" in en


# ── Rà độc lập (v3.23.702) ───────────────────────────────────────────────────────────────────────


def test_quota_error_in_fit_pass_keeps_the_finished_translation() -> None:
    from subtitles_extractor.infrastructure.translation.gemini_quota_manager import QuotaExhaustedError

    class _Quota(_Translator):
        def shorten_for_speech(self, items, lang, model, cancel_cb=None):
            raise QuotaExhaustedError("hết hạn mức ngày")

    lines = [_line(1, 0, 1, "Nhưng sư huynh đây đã không còn nữa rồi")]
    assert _run(_Quota(), lines)[0].text == lines[0].text


def test_subset_retranslation_skips_fit_pass() -> None:
    use_case = TranslateSubtitlesUseCase.__new__(TranslateSubtitlesUseCase)
    translator = _Translator({1: "x"})
    use_case._translator = translator
    request = TranslateSubtitlesRequest(events=[], context=TranslationContext(target_lang="Vietnamese"), stages=[
        TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="m")], contiguous=False)
    lines = [_line(1, 0, 1, "Nhưng sư huynh đây đã không còn nữa rồi")]
    assert use_case._fit_lines_for_speech(lines, lines, request, None) == lines and translator.items == []


def test_shortened_line_keeps_final_punctuation() -> None:
    lines = [_line(1, 0, 1, "mà đã tăng lên năm mươi đạo lôi kiếp rồi.")]
    out = _run(_Translator({1: "đã tăng năm mươi đạo rồi"}), lines)
    assert out[0].text == "đã tăng năm mươi đạo rồi."


def test_surname_rule_leaves_kinship_titles_and_place_names() -> None:
    from subtitles_extractor.application.services.name_reading_check import NameReadings, correct_context_names

    readings = NameReadings.from_encoded({"长": "trường,trưởng", "兄": "huynh", "安": "an", "叶": "hiệp,diệp",
                                          "子": "tử", "西": "tây"})
    context = TranslationContext(target_lang="Vietnamese",
                                 characters="长兄 (Đại Huynh) [nam] — anh cả\n长安 (Tây An) — kinh thành")
    fixed, _ = correct_context_names(context, readings)
    assert "Đại Huynh" in fixed.characters and "Trưởng" not in fixed.characters


def test_sentence_start_capital_is_not_treated_as_a_name() -> None:
    from subtitles_extractor.application.use_cases.shorten_for_speech import _changes_meaning_markers

    assert not _changes_meaning_markers("Đi thôi. Nhanh lên kẻo muộn mất rồi.", "Đi thôi, nhanh kẻo muộn.")
    assert not _changes_meaning_markers("Sư Phụ, người đi đâu thế?", "Sư phụ, đi đâu thế?")
    assert _changes_meaning_markers("Thiên Thanh sư huynh, huynh điên rồi sao?", "Sư huynh điên rồi sao?")


def test_language_and_model_name_variants() -> None:
    from subtitles_extractor.infrastructure.translation.thinking_policy import thinks_only_at_high
    from subtitles_extractor.infrastructure.tts.tts_fit import is_vietnamese_target

    assert is_vietnamese_target("vi_VN") and is_vietnamese_target("Việt")
    assert thinks_only_at_high("models/gemini-3.5-flash-lite")
