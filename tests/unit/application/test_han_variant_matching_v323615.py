"""[v3.23.615] So tên/thuật ngữ không phân biệt chữ Hán phồn/giản.

Dữ liệu OCR thật 14/10 (phim "天骄女友退婚" tập 1, dòng 50): câu "楚瑶已筑基圆满" có 10/12
khung đọc thành "楚瑤" (瑤 phồn thể) và phụ đề giữ "楚瑤"; bảng nhân vật ghi "楚瑶". Chẩn
đoán ngữ cảnh xếp nữ chính vào "identified_from_video" (không được gọi tên trong thoại).
"""

from __future__ import annotations

from types import SimpleNamespace

from subtitles_extractor.application.services.context_diagnostics import (
    _count_name_mentions,
)
from subtitles_extractor.application.services.han_variants import fold_han_variants
from subtitles_extractor.application.services.roster_format import (
    substitute_known_cjk,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    _batch_glossary_block,
)

#: Các dòng nguồn thật quanh dòng 50.
SUBTITLES = "\n".join(["多谢师兄挂念", "楚瑤", "已筑基圆满", "离金丹一步之遥"])


def test_fold_keeps_length_and_latin() -> None:
    assert fold_han_variants("楚瑤 Sở Dao") == "楚瑶 Sở Dao"
    assert fold_han_variants("") == ""


def test_heroine_is_mentioned_in_subtitles() -> None:
    mentioned, video_only, _missing = _count_name_mentions(
        ["叶然", "楚瑶"], SUBTITLES, {"楚瑶", "叶然"}
    )
    assert mentioned == ["楚瑶"]
    assert video_only == ["叶然"]  # mẫu không có câu nào gọi tên 叶然


def test_batch_glossary_reminds_terms_written_in_the_other_script() -> None:
    batch = [SimpleNamespace(text="楚瑤"), SimpleNamespace(text="已筑基圆满")]
    hits = _batch_glossary_block(batch, "楚瑶 = Sở Dao\n筑基 = Trúc Cơ")
    assert [hit["term"] for hit in hits] == ["楚瑶", "筑基"]


def test_leftover_name_in_other_script_is_substituted() -> None:
    assert substitute_known_cjk("楚瑤 muội", {"楚瑶": "Sở Dao"}) == "Sở Dao muội"
