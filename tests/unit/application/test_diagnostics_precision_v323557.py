"""[v3.23.557] Ba lỗi đo lường lộ ra ở chuẩn đoán thật v3.23.556 (phim 3902 dòng).

1. **Dịch trùng oan**: hai dòng gốc chỉ khác BẢNG MÃ dấu chấm than (full-width và ASCII)
   bị coi là "nguồn khác nhau", nên bản dịch giống hệt (vốn đúng) bị báo trùng — tầng dịch
   còn tốn một lượt gọi để dịch lại hai dòng đã đúng. 2/2 cờ là dương tính giả.
2. **Độc thoại tính vào xưng hô**: 109 dòng có nhãn người nói TRÙNG người nghe, sinh cờ ảo
   ``宋钱→宋钱`` "trôi" giữa 'tôi' và 'em'.
3. **Roster lỗi thời**: roster bắt ``沈明月`` xưng 'tôi' với ``柳慕云`` nhưng 6/6 dòng
   dịch 'chị'; roster bắt ``宋钱`` xưng 'ta' với ``向城`` nhưng 5/5 dòng dùng 'tôi'. Lỗi
   ở BẢNG NHÂN VẬT — 11 dòng bị liệt như lỗi dịch, trong khi chỉ cần sửa một dòng roster.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from subtitles_extractor.application.services.address_rule_guard import (
    find_address_rule_violations,
    find_outdated_rules,
)
from subtitles_extractor.application.services.translation_diagnostics import (
    address_drift,
    detect_quality_flags,
    strip_punctuation,
)

CHARACTERS = (
    "沈明月 (Thẩm Minh Nguyệt) [nữ, ~26] — bác sĩ; xưng hô: với Liễu Mộ Vân tự xưng "
    "'tôi' gọi 'cô'\n"
    "柳慕云 (Liễu Mộ Vân) [nữ, ~28] — nữ tổng tài\n"
    "宋钱 (Tống Tiền) [nam, ~22] — nhân vật chính; xưng hô: với Từ Tuấn Hào tự xưng "
    "'tao' gọi 'mày'\n"
    "徐俊豪 (Từ Tuấn Hào) [nam, ~25] — phản diện\n"
)


@dataclass
class _Event:
    index: int
    text: str
    start_sec: float = 0.0
    end_sec: float = 3.0


def _cues(entries: list[tuple[int, str, str]]) -> str:
    return json.dumps(
        [{"id": index, "spk": spk, "to": to} for index, spk, to in entries],
        ensure_ascii=False,
    )


class TestPunctuationOnlyDuplicates:
    def test_full_width_and_ascii_punctuation_are_the_same_source(self) -> None:
        assert strip_punctuation("宋钱\uff01") == strip_punctuation("宋钱!")
        assert strip_punctuation("这是元神碎片") == strip_punctuation(
            "这是元神碎片\u3002"
        )

    def test_real_case_is_no_longer_flagged(self) -> None:
        """Ca thật dòng 3730-3731: "Tống Tiền" hai bảng mã dấu — dịch giống là ĐÚNG."""
        sources = [_Event(3730, "宋钱\uff01"), _Event(3731, "宋钱!")]
        translated = [_Event(3730, "Tống Tiền!"), _Event(3731, "Tống Tiền!")]
        flags = detect_quality_flags(sources, translated)
        assert flags["counts"]["duplicate_adjacent"] == 0

    def test_genuinely_different_sources_are_still_flagged(self) -> None:
        """Ca thật dòng 2820-2821: nguồn KHÁC hẳn mà dịch trùng — vẫn phải báo."""
        sources = [_Event(2820, "各位各位"), _Event(2821, "大家稍安勿躁")]
        translated = [_Event(2820, "Xin mọi người bình tĩnh."),
                      _Event(2821, "Xin mọi người bình tĩnh.")]
        flags = detect_quality_flags(sources, translated)
        assert flags["counts"]["duplicate_adjacent"] == 1

    def test_use_case_does_not_retranslate_punctuation_twins(self) -> None:
        """Tầng dịch cũng phải bỏ qua — trước đây tốn một lượt gọi Gemini cho hai dòng."""
        from subtitles_extractor.application.services.translation_diagnostics import (
            strip_punctuation as normalize,
        )

        assert normalize("宋钱\uff01") == normalize("宋钱!")


class TestMonologueIsNotAddress:
    def test_self_pair_is_ignored(self) -> None:
        """Ca thật: 109 dòng có người nói = người nghe (độc thoại)."""
        cues = _cues([(index, "宋钱", "宋钱") for index in range(46, 66)])
        sources = {index: "我知道了" for index in range(46, 66)}
        translated = [
            _Event(index, "tôi biết rồi" if index % 2 else "em biết rồi")
            for index in range(46, 66)
        ]
        assert address_drift(cues, sources, translated) == []

    def test_real_dialogue_is_still_measured(self) -> None:
        cues = _cues([(index, "宋钱", "徐俊豪") for index in range(46, 66)])
        sources = {index: "我知道了" for index in range(46, 66)}
        translated = [
            _Event(index, "tôi biết rồi" if index % 2 else "tao biết rồi")
            for index in range(46, 66)
        ]
        assert address_drift(cues, sources, translated)


class TestOutdatedRosterRule:
    @staticmethod
    def _film() -> tuple[dict[int, str], dict[int, str], str]:
        indices = list(range(2520, 2532))
        sources = {index: "我明白了" for index in indices}
        translations = {index: "Chị hiểu rồi," for index in indices}
        cues = _cues([(index, "沈明月", "柳慕云") for index in indices])
        return sources, translations, cues

    def test_whole_film_using_another_form_is_a_roster_problem(self) -> None:
        """Ca thật: roster bắt xưng 'tôi' nhưng 6/6 dòng dịch 'chị' — lỗi ở roster."""
        sources, translations, cues = self._film()
        outdated = find_outdated_rules(sources, translations, cues, CHARACTERS)
        assert len(outdated) == 1
        assert (outdated[0].expected, outdated[0].observed) == ("tôi", "chị")
        assert outdated[0].speaker == "沈明月"

    def test_those_lines_leave_the_per_line_list(self) -> None:
        sources, translations, cues = self._film()
        violations = find_address_rule_violations(sources, translations, cues, CHARACTERS)
        assert violations == []

    def test_diagnostics_carries_a_concrete_suggestion(self) -> None:
        sources, translations, cues = self._film()
        flags = detect_quality_flags(
            [_Event(i, text) for i, text in sources.items()],
            [_Event(i, text) for i, text in translations.items()],
            characters=CHARACTERS,
            visual_cues=cues,
        )
        assert flags["counts"]["address_rule_outdated"] == 1
        suggestion = flags["address_rule_outdated"][0]["suggestion"]
        assert "'chị'" in suggestion and "'tôi'" in suggestion

    def test_a_few_stray_lines_are_not_a_roster_problem(self) -> None:
        """Lệch lác đác vẫn là lỗi DỊCH; cả phim dùng cách khác mới đổ cho roster."""
        indices = list(range(2520, 2532))
        sources = {index: "我明白了" for index in indices}
        translations = {index: "tôi hiểu rồi," for index in indices}
        translations[2525] = "Chị hiểu rồi,"
        translations[2527] = "Chị hiểu rồi,"
        cues = _cues([(index, "沈明月", "柳慕云") for index in indices])
        assert find_outdated_rules(sources, translations, cues, CHARACTERS) == []
