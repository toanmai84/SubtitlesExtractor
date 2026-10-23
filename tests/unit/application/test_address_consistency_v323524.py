"""[v3.23.524] Đo tính NHẤT QUÁN của xưng hô theo từng cặp nhân vật.

Tiếng Việt không có đại từ trung tính như "you" — mỗi cặp quan hệ có một cặp xưng hô riêng. Model
dịch theo từng lô nên dễ đổi cách gọi giữa chừng. Đo trên phim thật: cặp 春花 → 宋钱 dùng lẫn lộn
"chị" 16 lượt, "em" 15, "cậu" 12 — người xem thấy quan hệ nhân vật nhảy loạn, mà KHÔNG cờ nào hiện
có bắt được.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from subtitles_extractor.application.services.translation_diagnostics import (
    address_inconsistencies,
    detect_quality_flags,
)


@dataclass
class _Event:
    index: int
    text: str


def _cues(pairs: list[tuple[int, str, str]]) -> str:
    return json.dumps(
        [{"id": index, "spk": speaker, "to": addressee} for index, speaker, addressee in pairs]
    )


class TestDetection:
    def test_mixed_pronouns_flagged(self) -> None:
        cues = _cues([(i, "春花", "宋钱") for i in range(1, 13)])
        translated = [
            _Event(1, "Chị nói cho em biết"), _Event(2, "Chị lo cho em"),
            _Event(3, "Chị thấy em ổn"), _Event(4, "Cậu đi đâu đấy"),
            _Event(5, "Cậu ăn chưa"), _Event(6, "Cậu về đi"),
            _Event(7, "Em nghe chị nói"), _Event(8, "Em ở lại"), _Event(9, "Em hiểu rồi"),
            _Event(10, "Anh thấy sao"), _Event(11, "Anh nói đi"), _Event(12, "Anh ở đây"),
        ]
        findings = address_inconsistencies(cues, translated)
        assert findings and findings[0]["speaker"] == "春花"
        # [v3.23.536] Ngưỡng là QUÁ 3 kiểu — cặp xưng hô nhất quán đã chiếm 2 kiểu.
        assert len(findings[0]["variants"]) > 3

    def test_consistent_pair_not_flagged(self) -> None:
        cues = _cues([(i, "春花", "宋钱") for i in range(1, 13)])
        translated = [_Event(i, "Chị bảo em nghe này") for i in range(1, 10)]
        assert address_inconsistencies(cues, translated) == []

    def test_too_few_mentions_ignored(self) -> None:
        """Ít lượt quá thì chưa đủ căn cứ — không báo bừa."""
        cues = _cues([(1, "春花", "宋钱"), (2, "春花", "宋钱")])
        translated = [_Event(1, "Chị đi"), _Event(2, "Cậu về")]
        assert address_inconsistencies(cues, translated) == []

    def test_rare_variant_not_counted(self) -> None:
        """Một lần lỡ dùng khác không phải là lộn xộn."""
        cues = _cues([(i, "春花", "宋钱") for i in range(1, 21)])
        translated = [_Event(i, "Chị bảo em nghe này") for i in range(1, 20)]
        translated.append(_Event(20, "Cậu nghe đây"))
        assert address_inconsistencies(cues, translated) == []


class TestSafety:
    def test_no_cues_returns_empty(self) -> None:
        assert address_inconsistencies("", [_Event(1, "Chị đi")]) == []

    def test_malformed_cues_safe(self) -> None:
        assert address_inconsistencies("không phải json", [_Event(1, "Chị đi")]) == []
        assert address_inconsistencies('["rác"]', [_Event(1, "Chị đi")]) == []

    def test_wrapped_cues_supported(self) -> None:
        cues = json.dumps({"cues": [{"id": i, "spk": "A", "to": "B"} for i in range(1, 13)]})
        # [v3.23.536] Ngưỡng nay là QUÁ 3 kiểu — vì một cặp xưng hô nhất quán đã chiếm 2 kiểu.
        translated = [
            _Event(1, "Chị đi"), _Event(2, "Chị về"), _Event(3, "Chị ăn"),
            _Event(4, "Cậu đi"), _Event(5, "Cậu về"), _Event(6, "Cậu ăn"),
            _Event(7, "Em đi"), _Event(8, "Em về"), _Event(9, "Em ăn"),
            _Event(10, "Anh đi"), _Event(11, "Anh về"), _Event(12, "Anh ăn"),
        ]
        assert address_inconsistencies(cues, translated) != []


class TestReportIntegration:
    def test_flag_exposed_in_report(self) -> None:
        cues = _cues([(i, "春花", "宋钱") for i in range(1, 13)])
        # [v3.23.536] Ngưỡng nay là QUÁ 3 kiểu — vì một cặp xưng hô nhất quán đã chiếm 2 kiểu.
        translated = [
            _Event(1, "Chị đi"), _Event(2, "Chị về"), _Event(3, "Chị ăn"),
            _Event(4, "Cậu đi"), _Event(5, "Cậu về"), _Event(6, "Cậu ăn"),
            _Event(7, "Em đi"), _Event(8, "Em về"), _Event(9, "Em ăn"),
            _Event(10, "Anh đi"), _Event(11, "Anh về"), _Event(12, "Anh ăn"),
        ]
        source = [_Event(i, f"句{i}") for i in range(1, 10)]
        flags = detect_quality_flags(source, translated, visual_cues=cues)
        assert flags["counts"]["address_inconsistency"] == 1
        assert flags["address_inconsistency"][0]["addressee"] == "宋钱"

    def test_absent_without_cues(self) -> None:
        flags = detect_quality_flags([_Event(1, "句")], [_Event(1, "Câu")])
        assert flags["counts"]["address_inconsistency"] == 0


class TestPromptRules:
    def test_analysis_pins_address_convention(self) -> None:
        import inspect

        from subtitles_extractor.infrastructure.translation import (
            gemini_translation_adapter as module,
        )

        source = inspect.getsource(module)
        assert "BẮT BUỘC chốt CÁCH XƯNG HÔ" in source
        assert "BÁM BẢNG XƯNG HÔ ĐÃ CHỐT" in source


class TestPairsExamined:
    """[v3.23.525] Báo cáo phải kèm MẪU SỐ — số cặp đủ dữ liệu để xét.

    Độ phủ Visual Cues dao động rất mạnh giữa các lần phân tích (đo thật: 58,5% xuống 16,9% khi đổi
    model), kéo số cặp đo được từ 49 xuống 16. Không có mẫu số thì "12 cặp lộn xộn" và "2 cặp lộn
    xộn" trông như cải thiện 6 lần, trong khi thực tế là 24% xuống 12%.
    """

    def _flags(self, cues: str, translated: list[_Event]) -> dict:
        source = [_Event(event.index, f"句{event.index}") for event in translated]
        return detect_quality_flags(source, translated, visual_cues=cues)

    def test_counts_measurable_pairs(self) -> None:
        cues = _cues(
            [(i, "A", "B") for i in range(1, 10)] + [(i, "C", "D") for i in range(10, 19)]
        )
        translated = [_Event(i, "Chị bảo em nghe này") for i in range(1, 19)]
        assert self._flags(cues, translated)["counts"]["address_pairs_examined"] == 2

    def test_pair_below_threshold_not_counted(self) -> None:
        cues = _cues([(1, "A", "B"), (2, "A", "B")])
        translated = [_Event(1, "Chị đi"), _Event(2, "Em về")]
        assert self._flags(cues, translated)["counts"]["address_pairs_examined"] == 0

    def test_zero_without_cues(self) -> None:
        flags = detect_quality_flags([_Event(1, "句")], [_Event(1, "Câu")])
        assert flags["counts"]["address_pairs_examined"] == 0

    def test_inconsistent_pair_also_counted_in_denominator(self) -> None:
        cues = _cues([(i, "春花", "宋钱") for i in range(1, 13)])
        # [v3.23.536] Ngưỡng nay là QUÁ 3 kiểu — vì một cặp xưng hô nhất quán đã chiếm 2 kiểu.
        translated = [
            _Event(1, "Chị đi"), _Event(2, "Chị về"), _Event(3, "Chị ăn"),
            _Event(4, "Cậu đi"), _Event(5, "Cậu về"), _Event(6, "Cậu ăn"),
            _Event(7, "Em đi"), _Event(8, "Em về"), _Event(9, "Em ăn"),
            _Event(10, "Anh đi"), _Event(11, "Anh về"), _Event(12, "Anh ăn"),
        ]
        counts = self._flags(cues, translated)["counts"]
        assert counts["address_inconsistency"] == 1
        assert counts["address_pairs_examined"] == 1
