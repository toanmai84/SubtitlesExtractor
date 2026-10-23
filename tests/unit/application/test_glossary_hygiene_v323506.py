"""[v3.23.506] Vệ sinh bảng thuật ngữ bằng CODE, không dựa vào chỉ thị.

Hai vấn đề đo được trên dữ liệu thật:
1. 9 cặp thuật ngữ là BIẾN THỂ cùng nghĩa (骨髓配对/骨髓配型) → khâu dịch nhận chỉ thị mâu thuẫn.
   Prompt đã nhắc nhưng model vẫn sinh trùng.
2. Mục ``小可爱`` bị vi phạm 37/37 lần — bản dịch NHẤT QUÁN chọn cách khác, tức lỗi nằm ở MỤC
   glossary chứ không phải ở khâu dịch. Gộp chung khiến con số mất ý nghĩa.
"""
from __future__ import annotations

from dataclasses import dataclass

from subtitles_extractor.application.services.roster_format import (
    drop_variant_glossary_entries,
    parse_glossary,
    render_glossary,
)
from subtitles_extractor.application.services.translation_diagnostics import (
    detect_quality_flags,
    split_suspect_entries,
)


@dataclass
class _Event:
    index: int
    text: str


def _clean(glossary: str) -> str:
    return render_glossary(drop_variant_glossary_entries(parse_glossary(glossary)))


class TestVariantRemoval:
    def test_same_meaning_variant_dropped(self) -> None:
        cleaned = _clean("骨髓配对 = đối chiếu tủy\n骨髓配型 = đối chiếu tủy")
        assert cleaned == "骨髓配对 = đối chiếu tủy"

    def test_first_entry_kept(self) -> None:
        cleaned = _clean("满月宴 = tiệc đầy tháng\n满月酒 = tiệc đầy tháng")
        assert cleaned.startswith("满月宴")

    def test_opposite_meaning_both_kept(self) -> None:
        # Khác một ký tự nhưng NGƯỢC nghĩa — phải giữ cả hai.
        cleaned = _clean("装货 = chất hàng\n卸货 = dỡ hàng")
        assert len(cleaned.splitlines()) == 2

    def test_unrelated_terms_kept(self) -> None:
        cleaned = _clean("宠妻系统 = Hệ thống\n黑金卡 = thẻ đen")
        assert len(cleaned.splitlines()) == 2

    def test_substring_translation_treated_as_same(self) -> None:
        cleaned = _clean("补偿金 = tiền bồi thường\n补偿款 = tiền bồi thường thêm")
        assert len(cleaned.splitlines()) == 1


class TestSuspectEntrySplit:
    def test_always_violated_entry_is_suspect(self) -> None:
        real, suspect = split_suspect_entries(
            [{"term": "小可爱", "expected": "tiểu khả ái", "occurrences": 37, "violations": 37}]
        )
        assert real == []
        assert suspect[0]["term"] == "小可爱"
        assert "gượng ép" in suspect[0]["hint"]

    def test_partial_violation_is_real(self) -> None:
        real, suspect = split_suspect_entries(
            [{"term": "奔驰", "expected": "Mercedes-Benz", "occurrences": 9, "violations": 7}]
        )
        assert suspect == []
        assert real[0]["term"] == "奔驰"

    def test_rare_term_not_marked_suspect(self) -> None:
        # Ít lần xuất hiện → chưa đủ căn cứ kết luận mục sai.
        real, suspect = split_suspect_entries(
            [{"term": "绿本", "expected": "giấy tờ xe", "occurrences": 2, "violations": 2}]
        )
        assert suspect == [] and len(real) == 1

    def test_report_exposes_both_buckets(self) -> None:
        glossary = "小可爱 = tiểu khả ái\n奔驰 = Mercedes-Benz"
        source = [_Event(i, "小可爱") for i in range(1, 7)] + [_Event(7, "奔驰")]
        translated = [_Event(i, "cục cưng") for i in range(1, 7)] + [
            _Event(7, "xe Mercedes-Benz")
        ]
        flags = detect_quality_flags(source, translated, glossary=glossary)
        assert flags["counts"]["suspect_glossary_entries"] == 1
        assert flags["counts"]["glossary_violations"] == 0


class TestCompoundNameShortForm:
    """[v3.23.507] Dạng rút gọn hợp lệ của tên ghép bằng gạch nối phải được chấp nhận.

    Bảng ghi "Mercedes-Benz" nhưng bản dịch dùng "Mercedes" là hoàn toàn đúng; trước đây bị tính
    vi phạm 9/9 và còn bị gắn nhãn "mục glossary đáng ngờ" oan.
    """

    GLOSSARY = "奔驰 = Mercedes-Benz\n宠妻系统 = Hệ thống Cưng Chiều Vợ"

    def test_short_form_accepted(self) -> None:
        flags = detect_quality_flags(
            [_Event(1, "奔驰接送")], [_Event(1, "Mercedes đón đưa,")], glossary=self.GLOSSARY
        )
        assert flags["counts"]["glossary_violations"] == 0

    def test_full_form_still_accepted(self) -> None:
        flags = detect_quality_flags(
            [_Event(1, "奔驰接送")], [_Event(1, "Xe Mercedes-Benz đón đưa")], glossary=self.GLOSSARY
        )
        assert flags["counts"]["glossary_violations"] == 0

    def test_unrelated_translation_still_violation(self) -> None:
        flags = detect_quality_flags(
            [_Event(1, "奔驰接送")], [_Event(1, "tài xế riêng đón đưa")], glossary=self.GLOSSARY
        )
        assert flags["counts"]["glossary_violations"] == 1

    def test_short_core_not_used_as_shortcut(self) -> None:
        # Phần trước gạch nối quá ngắn (<4 ký tự) thì KHÔNG dùng làm dạng rút gọn.
        flags = detect_quality_flags(
            [_Event(1, "A话")], [_Event(1, "chỉ có A thôi")], glossary="A话 = A-B-C"
        )
        assert flags["counts"]["glossary_violations"] == 1


class TestAliasTermMatching:
    """[v3.23.508] Vế TRÁI có thể gộp nhiều bí danh — phải tìm TỪNG bí danh trong phụ đề.

    Trước đây bộ đo tìm nguyên chuỗi "满月宴 / 满月酒" nên KHÔNG BAO GIỜ khớp; toàn bộ mục dạng
    này (21 mục trên dữ liệu thật) bị bỏ qua khi đo tuân thủ — im lặng, không ai biết.
    """

    GLOSSARY = "满月宴 / 满月酒 = tiệc đầy tháng"

    def test_alias_occurrences_counted(self) -> None:
        flags = detect_quality_flags(
            [_Event(1, "参加满月酒"), _Event(2, "参加满月宴")],
            [_Event(1, "đi dự bữa nhậu"), _Event(2, "đi dự bữa nhậu")],
            glossary=self.GLOSSARY,
        )
        # [v3.23.550] Số chính ``counts.glossary_violations`` nay chỉ đếm TÊN RIÊNG; "tiệc đầy
        # tháng" là biểu đạt nên đếm ở ``glossary_expression_deviations``. Bất biến cần khoá vẫn
        # là: CẢ HAI bí danh đều được tìm thấy và tính lượt.
        assert sum(item["violations"] for item in flags["glossary_violations"]) == 2
        assert flags["counts"]["glossary_expression_deviations"] == 2

    def test_correct_translation_no_violation(self) -> None:
        flags = detect_quality_flags(
            [_Event(1, "参加满月酒")], [_Event(1, "đi dự tiệc đầy tháng")], glossary=self.GLOSSARY
        )
        assert flags["counts"]["glossary_violations"] == 0

    def test_unrelated_source_not_counted(self) -> None:
        flags = detect_quality_flags(
            [_Event(1, "今天天气不错")], [_Event(1, "hôm nay trời đẹp")], glossary=self.GLOSSARY
        )
        assert flags["glossary_violations"] == []


class TestSubsumedEntryRemoval:
    """[v3.23.508] Model vừa ghi mục gộp bí danh vừa ghi mục lẻ → 21/287 mục thừa."""

    def test_subset_entry_dropped(self) -> None:
        from subtitles_extractor.application.services.roster_format import (
            drop_subsumed_glossary_entries,
        )
        entries = parse_glossary("满月宴 / 满月酒 = tiệc đầy tháng\n满月宴 = tiệc đầy tháng")
        kept = drop_subsumed_glossary_entries(entries)
        assert len(kept) == 1
        assert "满月酒" in kept[0].term

    def test_richer_alias_entry_kept(self) -> None:
        from subtitles_extractor.application.services.roster_format import (
            drop_subsumed_glossary_entries,
        )
        entries = parse_glossary(
            "绿毛龟 = bị cắm sừng\n绿毛龟 / 绿帽子 = bị cắm sừng\n绿帽子 = bị cắm sừng"
        )
        kept = drop_subsumed_glossary_entries(entries)
        assert [e.term for e in kept] == ["绿毛龟 / 绿帽子"]

    def test_independent_entries_kept(self) -> None:
        from subtitles_extractor.application.services.roster_format import (
            drop_subsumed_glossary_entries,
        )
        entries = parse_glossary("宠妻系统 = Hệ thống\n黑金卡 = thẻ đen")
        assert len(drop_subsumed_glossary_entries(entries)) == 2


class TestSuspectThreshold:
    """[v3.23.514] Ngưỡng 0.8: mục quy định CỤM DÀI bị vi phạm ~86% là lỗi của MỤC, không phải dịch.

    Dữ liệu thật: ``到账`` quy định "tiền vào tài khoản (chuyển khoản thành công)" bị vi phạm
    18/21 = 86%; các bản dịch đều hợp lý ("hai vạn tệ đã vào tài khoản") nhưng không chứa trọn cụm.
    Ngưỡng 0.9 cũ để lọt nhóm này và thổi phồng số vi phạm.
    """

    def test_86_percent_is_suspect_entry(self) -> None:
        real, suspect = split_suspect_entries(
            [{"term": "到账", "expected": "tiền vào tài khoản",
              "occurrences": 21, "violations": 18}]
        )
        assert real == []
        assert suspect[0]["term"] == "到账"

    def test_below_threshold_still_real_violation(self) -> None:
        # 7/10 = 70% → vẫn là vi phạm THẬT, đáng sửa ở khâu dịch.
        real, suspect = split_suspect_entries(
            [{"term": "奔驰", "expected": "Mercedes-Benz",
              "occurrences": 10, "violations": 7}]
        )
        assert suspect == []
        assert real[0]["term"] == "奔驰"

    def test_rare_term_still_needs_min_occurrences(self) -> None:
        real, suspect = split_suspect_entries(
            [{"term": "绿本", "expected": "giấy tờ xe", "occurrences": 4, "violations": 4}]
        )
        assert suspect == [] and len(real) == 1
