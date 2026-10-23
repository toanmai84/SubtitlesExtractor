"""[v3.23.554] Hồi quy cho lưới ÉP XƯNG HÔ theo bảng nhân vật (chuẩn đoán thật v3.23.553).

Dữ liệu thật (phim 3902 dòng, roster 119 quy tắc xưng hô): 51 dòng xưng sai quy tắc đã ghi
rõ, trong đó **50 dòng sai ngay từ giai đoạn dịch thô** — gắn quy tắc vào lô dịch
(v3.23.551) đã tới hạn, phải ép bằng mã.

Mỗi lớp test dưới đây tương ứng một chốt an toàn, dựng từ ca thật đã soi tay.
"""

from __future__ import annotations

import json

from subtitles_extractor.application.services.address_rule_guard import (
    apply_address_fixes,
    find_address_rule_fixes,
    find_address_rule_violations,
    is_possessive_position,
    unconditional_self_form,
)

CHARACTERS = (
    "宋钱 (Tống Tiền) [nam, ~22] — nhân vật chính; xưng hô: với Từ Tuấn Hào tự xưng "
    "'tao' gọi 'mày'; với Mẹ Tống tự xưng 'con' gọi 'mẹ'; với Liễu Mộ Vân trước xưng "
    "'tôi' gọi 'cô Liễu' (sau xưng 'em' gọi 'chị Mộ Vân'); với Trương Y Bối tự xưng "
    "'anh/tôi' gọi 'em'\n"
    "徐俊豪 (Từ Tuấn Hào) [nam, ~25] — phản diện\n"
    "宋母 (Mẹ Tống) [nữ, ~50] — mẹ Tống Tiền\n"
    "柳慕云 (Liễu Mộ Vân) [nữ, ~28] — nữ tổng tài\n"
    "张依贝 (Trương Y Bối) [nữ, ~20] — hàng xóm\n"
)


def _cues(pairs: dict[int, tuple[str, str]]) -> str:
    return json.dumps(
        [{"id": index, "spk": spk, "to": to} for index, (spk, to) in pairs.items()],
        ensure_ascii=False,
    )


def _scene(
    speaker: str, addressee: str, first: int, count: int
) -> dict[int, tuple[str, str]]:
    return {first + step: (speaker, addressee) for step in range(count)}


class TestRealCases:
    def test_self_pronoun_forced_to_rule(self) -> None:
        """Ca thật dòng 2035: 宋钱 với Từ Tuấn Hào phải xưng 'tao', bản dịch ra 'tôi'."""
        cues = _cues(_scene("宋钱", "徐俊豪", 2030, 6))
        sources = {i: "我一定送上一份" for i in range(2030, 2036)}
        translations = {i: "tao nhất định sẽ tặng một phần" for i in range(2030, 2035)}
        translations[2035] = "tôi nhất định sẽ tặng cho hai người"
        fixes = find_address_rule_fixes(sources, translations, cues, CHARACTERS)
        assert [(f.index, f.pronoun_found, f.pronoun_expected) for f in fixes] == [
            (2035, "tôi", "tao")
        ]
        assert fixes[0].text_after == "tao nhất định sẽ tặng cho hai người"

    def test_capitalization_is_kept(self) -> None:
        """Ca thật dòng 509: 'Tôi đã khôi phục' → 'Con…', giữ chữ hoa đầu câu."""
        cues = _cues(_scene("宋钱", "宋母", 500, 6))
        sources = {i: "如今我恢复了正常" for i in range(500, 506)}
        translations = {i: "Con đã khôi phục bình thường" for i in range(500, 505)}
        translations[505] = "Tôi đã khôi phục bình thường,"
        fixes = find_address_rule_fixes(sources, translations, cues, CHARACTERS)
        assert fixes[0].text_after == "Con đã khôi phục bình thường,"

    def test_applying_fixes_returns_copy(self) -> None:
        cues = _cues(_scene("宋钱", "宋母", 500, 6))
        sources = {i: "如今我恢复了正常" for i in range(500, 506)}
        translations = {i: "Con đã khôi phục bình thường" for i in range(500, 505)}
        translations[505] = "Tôi đã khôi phục bình thường,"
        fixes = find_address_rule_fixes(sources, translations, cues, CHARACTERS)
        patched = apply_address_fixes(translations, fixes)
        assert patched[505] == "Con đã khôi phục bình thường,"
        assert translations[505] == "Tôi đã khôi phục bình thường,"  # bản gốc KHÔNG đổi


class TestSafetyGuards:
    def test_conditional_rule_is_never_forced(self) -> None:
        """Roster ghi 'trước xưng tôi… sau xưng em' — hai cách đều hợp lệ."""
        cues = _cues(_scene("宋钱", "柳慕云", 1400, 6))
        sources = {i: "不然我" for i in range(1400, 1406)}
        translations = {i: "nếu không thì em" for i in range(1400, 1406)}
        assert find_address_rule_fixes(sources, translations, cues, CHARACTERS) == []

    def test_multi_option_rule_is_never_forced(self) -> None:
        cues = _cues(_scene("宋钱", "张依贝", 600, 6))
        sources = {i: "我知道" for i in range(600, 606)}
        translations = {i: "tôi biết" for i in range(600, 606)}
        assert find_address_rule_fixes(sources, translations, cues, CHARACTERS) == []

    def test_kinship_pronoun_is_not_replaced(self) -> None:
        """'em', 'anh', 'con' cũng là danh từ — thay máy móc dễ hỏng ("sinh con")."""
        cues = _cues(_scene("宋钱", "徐俊豪", 2030, 6))
        sources = {i: "我一定送上一份" for i in range(2030, 2036)}
        translations = {i: "tao nhất định sẽ tặng một phần" for i in range(2030, 2035)}
        translations[2035] = "em nhất định sẽ tặng một phần"
        fixes = find_address_rule_fixes(sources, translations, cues, CHARACTERS)
        assert fixes == []
        reasons = {
            v.index: v.reason
            for v in find_address_rule_violations(sources, translations, cues, CHARACTERS)
        }
        assert "danh từ chỉ người" in reasons[2035]

    def test_singular_possessive_follows_the_speaker(self) -> None:
        """Ca thật dòng 2937: 'mẹ tôi' giữa cảnh 'mẹ tao' → sửa thành 'mẹ tao'.

        [v3.23.562] Trước đây chốt sở hữu chặn cả ca này với lý do "đổi nghĩa". Soi lại
        trên dữ liệu thật: ở dòng TỰ XƯNG (gốc có 我, không có 你), "mẹ tôi" chỉ có thể là
        mẹ của người nói — đổi đại từ theo cách xưng của họ giữ nguyên nghĩa, chỉ đổi sắc
        thái cho khớp cả cảnh. Ca thật sự đổi nghĩa ("anh tôi" → "anh em", "mẹ tôi" → "mẹ
        con") vẫn bị chặn — xem test_address_possessive_v323562.py.
        """
        cues = _cues(_scene("宋钱", "徐俊豪", 2930, 6))
        sources = {i: "把我妈的腿打断" for i in range(2930, 2936)}
        translations = {i: "đánh gãy chân mẹ tao," for i in range(2930, 2935)}
        translations[2935] = "đánh gãy chân mẹ tôi,"
        fixes = find_address_rule_fixes(sources, translations, cues, CHARACTERS)
        assert [(f.index, f.text_after) for f in fixes] == [
            (2935, "đánh gãy chân mẹ tao,")
        ]

    def test_proper_noun_before_pronoun_is_not_replaced(self) -> None:
        """Ca thật dòng 2258: 'nhà họ Liễu ta' — sở hữu tập thể, không phải tự xưng."""
        assert is_possessive_position("Tốt tốt, nhà họ Liễu ta đúng là", "ta") is True
        assert is_possessive_position("Nay tôi đã khỏi", "tôi") is False

    def test_plural_source_is_ignored(self) -> None:
        """Ca thật v3.23.549: '咱俩' = HAI ta — số nhiều, không phải ngôi nhất số ít."""
        cues = _cues(_scene("宋钱", "徐俊豪", 700, 6))
        sources = {i: "谁让咱俩是一个村的" for i in range(700, 706)}
        translations = {
            i: "Dù sao hai ta cũng là người một làng," for i in range(700, 706)
        }
        assert find_address_rule_fixes(sources, translations, cues, CHARACTERS) == []

    def test_second_person_source_is_ignored(self) -> None:
        """Câu có 你: đại từ trong bản dịch có thể là CÁCH GỌI, không phải tự xưng."""
        cues = _cues(_scene("宋钱", "徐俊豪", 2030, 6))
        sources = {i: "你说我怎么办" for i in range(2030, 2036)}
        translations = {i: "cậu bảo tôi phải làm sao" for i in range(2030, 2036)}
        assert find_address_rule_fixes(sources, translations, cues, CHARACTERS) == []

    def test_scene_that_switched_register_is_left_alone(self) -> None:
        """Ca thật: An Tuyết Mạn đang 'tao/mày' chuyển hẳn sang 'tôi' khi quỳ xin lỗi.

        [v3.23.557] Máy vẫn KHÔNG đè bản dịch. Khác biệt là chỗ BÁO: cả phim (ở đây là
        toàn bộ 8 dòng của cặp) dùng cách khác thì đó là quy tắc roster lỗi thời — báo một
        dòng đề xuất sửa bảng nhân vật, thay vì liệt kê 8 dòng như lỗi dịch.
        """
        from subtitles_extractor.application.services.address_rule_guard import (
            find_outdated_rules,
        )

        cues = _cues(_scene("宋钱", "徐俊豪", 3200, 8))
        sources = {i: "之前是我错了啊" for i in range(3200, 3208)}
        translations = {i: "trước đây là tôi sai rồi," for i in range(3200, 3208)}
        assert find_address_rule_fixes(sources, translations, cues, CHARACTERS) == []
        assert find_address_rule_violations(sources, translations, cues, CHARACTERS) == []
        outdated = find_outdated_rules(sources, translations, cues, CHARACTERS)
        assert [(r.expected, r.observed) for r in outdated] == [("tao", "tôi")]

    def test_a_few_deviating_lines_stay_in_the_per_line_list(self) -> None:
        """Lệch lác đác (không phải cả phim) vẫn phải hiện ra từng dòng để soi tay."""
        cues = _cues(_scene("宋钱", "徐俊豪", 3200, 8))
        sources = {i: "之前是我错了啊" for i in range(3200, 3208)}
        translations = {i: "trước đây là tao sai rồi," for i in range(3200, 3208)}
        translations[3203] = "trước đây là em sai rồi,"
        violations = find_address_rule_violations(sources, translations, cues, CHARACTERS)
        assert [v.index for v in violations] == [3203]


class TestRuleParsing:
    def test_unconditional_single_pronoun(self) -> None:
        assert unconditional_self_form("tự xưng 'tao' gọi 'mày'") == "tao"

    def test_conditional_rule_returns_empty(self) -> None:
        assert unconditional_self_form("trước xưng 'tôi' gọi 'cô Liễu'") == ""
        assert unconditional_self_form("khi nhận đệ tử xưng 'sư phụ' gọi 'con'") == ""

    def test_multiple_options_return_empty(self) -> None:
        assert unconditional_self_form("tự xưng 'tôi/chị' gọi 'cậu'") == ""

    def test_title_instead_of_pronoun_returns_empty(self) -> None:
        assert unconditional_self_form("tự xưng 'Tống thần y' gọi 'quý khách'") == ""


class TestUseCaseWiring:
    def test_stage_output_is_patched(self) -> None:
        from subtitles_extractor.application.use_cases.translate_subtitles import (
            TranslateSubtitlesUseCase,
        )
        from subtitles_extractor.domain.ports.subtitle_translator_port import (
            TranslationContext,
            TranslationLine,
        )

        def line(index: int, text: str, original: str) -> TranslationLine:
            return TranslationLine(
                index=index, start_ms=index * 1000, end_ms=index * 1000 + 900,
                text=text, original_text=original,
            )

        source = [line(i, "", "我一定送上一份") for i in range(2030, 2036)]
        stage = [line(i, "tao nhất định sẽ tặng một phần", "") for i in range(2030, 2035)]
        stage.append(line(2035, "tôi nhất định sẽ tặng cho hai người", ""))
        context = TranslationContext(
            target_lang="Vietnamese",
            characters=CHARACTERS,
            visual_cues=_cues(_scene("宋钱", "徐俊豪", 2030, 6)),
        )
        result = TranslateSubtitlesUseCase._enforce_address_rules(
            stage, source, context, "Dịch thô"
        )
        assert result[-1].text == "tao nhất định sẽ tặng cho hai người"
        assert result[0].text == "tao nhất định sẽ tặng một phần"

    def test_no_cues_means_no_change(self) -> None:
        from subtitles_extractor.application.use_cases.translate_subtitles import (
            TranslateSubtitlesUseCase,
        )
        from subtitles_extractor.domain.ports.subtitle_translator_port import (
            TranslationContext,
        )

        context = TranslationContext(target_lang="Vietnamese", characters=CHARACTERS)
        stage: list = []
        assert TranslateSubtitlesUseCase._enforce_address_rules(
            stage, [], context, "Dịch thô"
        ) is stage


class TestDiagnosticsFlag:
    def test_unfixable_violations_are_reported(self) -> None:
        """Phần máy không dám sửa vẫn phải hiện ra chuẩn đoán để người dùng tự quyết."""
        from dataclasses import dataclass

        from subtitles_extractor.application.services.translation_diagnostics import (
            detect_quality_flags,
        )

        @dataclass
        class _E:
            index: int
            text: str
            start_sec: float = 0.0
            end_sec: float = 3.0

        indices = range(2030, 2036)
        sources = [_E(i, "我一定送上一份") for i in indices]
        translated = [_E(i, "tao nhất định sẽ tặng một phần") for i in indices][:-1]
        translated.append(_E(2035, "em nhất định sẽ tặng một phần"))
        flags = detect_quality_flags(
            sources, translated,
            characters=CHARACTERS,
            visual_cues=_cues(_scene("宋钱", "徐俊豪", 2030, 6)),
        )
        assert flags["counts"]["address_rule_violations"] == 1
        finding = flags["address_rule_violations"][0]
        assert finding["index"] == 2035
        assert finding["expected"] == "tao" and finding["found"] == "em"

    def test_fixed_lines_are_not_reported(self) -> None:
        """Dòng lưới TỰ SỬA được thì trong bản dịch cuối đã đúng → không còn cờ."""
        from dataclasses import dataclass

        from subtitles_extractor.application.services.translation_diagnostics import (
            detect_quality_flags,
        )

        @dataclass
        class _E:
            index: int
            text: str
            start_sec: float = 0.0
            end_sec: float = 3.0

        indices = range(2030, 2036)
        sources = [_E(i, "我一定送上一份") for i in indices]
        translated = [_E(i, "tao nhất định sẽ tặng một phần") for i in indices]
        flags = detect_quality_flags(
            sources, translated,
            characters=CHARACTERS,
            visual_cues=_cues(_scene("宋钱", "徐俊豪", 2030, 6)),
        )
        assert flags["counts"]["address_rule_violations"] == 0
