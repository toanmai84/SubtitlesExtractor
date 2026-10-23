"""[v3.23.501] Dọn chữ Hán còn sót bằng bảng tên/thuật ngữ ĐÃ DUYỆT.

Dữ liệu thật (5540 dòng) còn 19 dòng lẫn chữ Hán kiểu dịch NỬA VỜI: "Lâm先生", "Cường哥",
"ủy屈", và tên bỏ nguyên "秦雅". Dịch lại cả dòng thường cho ra đúng kết quả cũ nên vòng dọn hiện
có không cứu được; thay theo bảng đã duyệt thì tất định và không thể bịa nội dung.
"""
from __future__ import annotations

import pytest

from subtitles_extractor.application.services.roster_format import (
    cjk_substitution_map,
    substitute_known_cjk,
)

ROSTER = (
    "林逸 (Lâm Dật) [nam] — nhân vật chính\n"
    "秦雅 (Tần Nhã) [nữ] — nhân viên\n"
    "贺强 (Hạ Cường) [nam] — bạn học\n"
    "胖子 / 三胖子 (Mập) [nam] — bạn thân"
)
GLOSSARY = "奔驰 = Mercedes-Benz\n心动值 = chỉ số rung động\n身份证 = thẻ căn cước / CMND"


@pytest.fixture
def mapping() -> dict[str, str]:
    return cjk_substitution_map(ROSTER, GLOSSARY)


class TestSubstitution:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Lâm先生,", "Lâm tiên sinh,"),
            ("xin lỗi Lâm先生 đi!", "xin lỗi Lâm tiên sinh đi!"),
            ("Nhưng mà Cường哥,", "Nhưng mà Cường ca,"),
            ("秦雅,", "Tần Nhã,"),
            ("đạt 心动值 tối đa", "đạt chỉ số rung động tối đa"),
            ("Xe 奔驰 mới", "Xe Mercedes-Benz mới"),
        ],
    )
    def test_known_fragments_replaced(self, mapping, text: str, expected: str) -> None:
        assert substitute_known_cjk(text, mapping) == expected

    def test_alias_maps_to_target_name(self, mapping) -> None:
        assert substitute_known_cjk("三胖子 nói", mapping) == "Mập nói"

    def test_longer_term_wins_over_substring(self) -> None:
        # "董事长" phải thay trọn, không để "长" cắt mất.
        result = substitute_known_cjk("Ông 董事长 tới", cjk_substitution_map("", ""))
        assert "董事" not in result and "chủ tịch" in result

    def test_glossary_first_alternative_used(self, mapping) -> None:
        # "thẻ căn cước / CMND" → lấy phương án ĐẦU cho nhất quán.
        assert substitute_known_cjk("mang 身份证 theo", mapping) == "mang thẻ căn cước theo"

    def test_unknown_cjk_left_untouched(self, mapping) -> None:
        # Không có trong bảng → GIỮ NGUYÊN, để vòng dịch lại xử lý (không bịa nội dung).
        assert "陌生词" in substitute_known_cjk("có 陌生词 đây", mapping)

    def test_clean_text_unchanged(self, mapping) -> None:
        assert substitute_known_cjk("Câu đã sạch hoàn toàn.", mapping) == "Câu đã sạch hoàn toàn."

    def test_empty_input(self, mapping) -> None:
        assert substitute_known_cjk("", mapping) == ""


class TestSubstitutionMap:
    def test_includes_roster_aliases(self, mapping) -> None:
        assert mapping["三胖子"] == "Mập"
        assert mapping["秦雅"] == "Tần Nhã"

    def test_includes_glossary_terms(self, mapping) -> None:
        assert mapping["奔驰"] == "Mercedes-Benz"

    def test_includes_common_honorifics_by_default(self) -> None:
        assert "先生" in cjk_substitution_map("", "")

    def test_user_data_overrides_default_honorific(self) -> None:
        # Bảng thuật ngữ của người dùng được ưu tiên hơn mặc định.
        custom = cjk_substitution_map("", "先生 = ngài")
        assert custom["先生"] == "ngài"
