"""[v3.23.490] Khoá các yêu cầu chất lượng trong prompt phân tích ngữ cảnh.

Ba vấn đề người dùng báo: (1) roster thiếu nhân vật phụ có tên thật (vd 王梅/Vương Mai);
(2) glossary quá ít (9 mục cho 5540 dòng); (3) tên trong Visual Cues lúc tiếng đích lúc tiếng gốc.
Prompt cũng phải NHẤT QUÁN với schema mảng đã đổi ở v3.23.488.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_ADAPTER = Path(
    "src/subtitles_extractor/infrastructure/translation/gemini_translation_adapter.py"
).read_text(encoding="utf-8")


class TestSchemaPromptConsistency:
    """Prompt phải mô tả MẢNG đối tượng, khớp schema (không còn mô tả chuỗi tự do)."""

    @pytest.mark.parametrize(
        "needle",
        ["characters: MẢNG đối tượng", "glossary: MẢNG đối tượng"],
    )
    def test_prompt_describes_arrays(self, needle: str) -> None:
        assert needle in _ADAPTER

    def test_old_freeform_line_format_removed(self) -> None:
        # Định dạng chuỗi cũ "'Tên gốc (phiên âm...) [nam/nữ, ~tuổi]: vai trò'" không còn được
        # mô tả cho characters — nếu còn, model sẽ trả chuỗi thay vì mảng.
        assert "Định dạng SONG NGỮ cho TỪNG nhân vật trên mỗi dòng" not in _ADAPTER


class TestRosterCompleteness:
    """Roster phải gồm cả nhân vật phụ / chỉ được nhắc tên (fix thiếu 王梅)."""

    @pytest.mark.parametrize(
        "needle",
        [
            "LIỆT KÊ ĐẦY ĐỦ",
            "chỉ được NHẮC TÊN trong lời thoại",
            "Thà thừa còn hơn thiếu",
        ],
    )
    def test_completeness_instructions(self, needle: str) -> None:
        assert needle in _ADAPTER


class TestGlossaryBreadth:
    """Glossary phải phủ nhiều nhóm, không chỉ vài thuật ngữ đặc thù."""

    @pytest.mark.parametrize(
        "needle",
        [
            "ĐỊA DANH & CƠ SỞ",
            "TỔ CHỨC & CHỨC DANH",
            "CHUYÊN NGÀNH",
            "TIỀN TỆ & ĐƠN VỊ",
            "DANH NGÔN",
            "BIỆT DANH & TIẾNG LÓNG",
            "HÀNG CHỤC mục",
        ],
    )
    def test_categories_present(self, needle: str) -> None:
        assert needle in _ADAPTER


class TestCueNameConsistency:
    """spk/to phải dùng MỘT ngôn ngữ (tên gốc) để khớp máy móc với roster."""

    def test_requires_original_name_matching_roster(self) -> None:
        assert "TRÙNG KHỚP TUYỆT ĐỐI với trường" in _ADAPTER
        assert "QUY TẮC TÊN (BẮT BUỘC, không được lẫn lộn)" in _ADAPTER

    def test_old_transliteration_rule_removed(self) -> None:
        # Quy tắc cũ "Dùng phiên âm {target_lang} cho tên CJK" ở phần cues chính là nguyên nhân
        # lẫn lộn ngôn ngữ — phải bị gỡ.
        assert "Dùng phiên âm {target_lang} cho tên CJK; tên Latin giữ nguyên" not in _ADAPTER


class TestNameMappingDirective:
    """[v3.23.491] Prompt dịch phải có BẢNG QUY ĐỔI tên gốc → tên đích.

    Từ v3.23.490, nhãn spk/to trong Visual Cues là TÊN GỐC. Nếu prompt dịch chỉ đưa danh sách tên
    tiếng đích, model phải tự đoán '林逸' ứng với 'Lâm Dật' → gọi sai tên/đảo nhân vật.
    """

    ROSTER = (
        "林逸 (Lâm Dật) [nam, ~24 tuổi] — nhân vật chính\n"
        "宋欣爽 (Tống Hân Sảng) [nữ, ~26 tuổi] — vợ\n"
        "John Smith [nam] — bác sĩ"
    )

    def _directive(self) -> str:
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            GeminiSubtitleTranslator,
        )
        return GeminiSubtitleTranslator._canonical_names_directive(self.ROSTER)

    def test_maps_source_to_target_with_traits(self) -> None:
        directive = self._directive()
        assert "林逸 → Lâm Dật [nam, ~24 tuổi]" in directive
        assert "宋欣爽 → Tống Hân Sảng [nữ, ~26 tuổi]" in directive

    def test_latin_name_has_no_arrow(self) -> None:
        assert "John Smith [nam]" in self._directive()

    def test_explains_speaker_labels_are_source_names(self) -> None:
        directive = self._directive()
        assert "nhãn speaker/addressee kèm mỗi dòng được ghi bằng TÊN GỐC" in directive
        assert "không để lọt chữ gốc vào bản dịch" in directive

    def test_empty_roster_returns_empty(self) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            GeminiSubtitleTranslator,
        )
        assert GeminiSubtitleTranslator._canonical_names_directive("") == ""

    def test_legacy_roster_still_works(self) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            GeminiSubtitleTranslator,
        )
        directive = GeminiSubtitleTranslator._canonical_names_directive(
            "林逸 (Lâm Dật): chính, mạnh mẽ. 宋欣爽 (Tống Hân Sảng): vợ."
        )
        assert "林逸" in directive and "宋欣爽" in directive


class TestSpeakerLabelCleaning:
    """[v3.23.493] Lớp phòng vệ: nhãn spk/to phải về TÊN GỐC thuần để khớp bảng quy đổi tên,
    kể cả khi model lỡ kèm phiên âm/chức danh (dù prompt đã cấm)."""

    def _cues(self, items):
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            GeminiSubtitleTranslator,
        )
        return GeminiSubtitleTranslator._parse_visual_cue_items(items)

    def test_strips_transliteration_in_parentheses(self) -> None:
        cues = self._cues([{"id": 1, "spk": "林逸（Lâm Dật）", "to": "宋欣爽 (Tống Hân Sảng)"}])
        assert cues[0].speaker == "林逸"
        assert cues[0].addressee == "宋欣爽"

    def test_plain_name_unchanged(self) -> None:
        assert self._cues([{"id": 1, "spk": "林逸"}])[0].speaker == "林逸"

    def test_whitespace_and_punctuation_trimmed(self) -> None:
        assert self._cues([{"id": 1, "spk": "  林逸 -"}])[0].speaker == "林逸"

    def test_invalid_ids_dropped(self) -> None:
        cues = self._cues([{"id": 0, "spk": "X"}, {"spk": "Y"}, {"id": -3, "spk": "Z"},
                           {"id": 5, "spk": "林逸"}])
        assert [c.line_no for c in cues] == [5]


class TestRosterPurityRules:
    """[v3.23.494] Roster chỉ chứa CON NGƯỜI; bí danh của cùng một người phải gộp một mục."""

    @pytest.mark.parametrize(
        "needle",
        [
            "CHỈ LIỆT KÊ CON NGƯỜI",
            "những thứ đó thuộc",
            "GỘP BÍ DANH",
            "KHÔNG tạo hai phần tử cho cùng",
        ],
    )
    def test_rules_present(self, needle: str) -> None:
        assert needle in _ADAPTER


class TestCueLineBounds:
    """[v3.23.496] Cue trỏ dòng KHÔNG tồn tại (id vượt số dòng thật) phải bị loại tại nguồn."""

    def _cues(self, items, max_line_no=0):
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            GeminiSubtitleTranslator,
        )
        return GeminiSubtitleTranslator._parse_visual_cue_items(items, max_line_no=max_line_no)

    def test_drops_out_of_range_ids(self) -> None:
        items = [{"id": 1}, {"id": 5540}, {"id": 5541}, {"id": 9999}]
        assert [c.line_no for c in self._cues(items, 5540)] == [1, 5540]

    def test_no_limit_keeps_all_positive_ids(self) -> None:
        items = [{"id": 1}, {"id": 9999}]
        assert [c.line_no for c in self._cues(items)] == [1, 9999]
