"""[v3.23.645] Biến thể tên tông môn (灵霄宗 / 凌霄宗) nhận cùng bản dịch."""

from __future__ import annotations

from types import SimpleNamespace

from subtitles_extractor.application.services.term_variants import find_variants, is_organisation_term
from subtitles_extractor.application.services.translation_diagnostics import _glossary_violations
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    _batch_glossary_block,
)

GLOSSARY = "灵霄宗 = Linh Tiêu Tông\n叶然 = Diệp Nhiên"


def _batch(*texts: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(text=t) for t in texts]


def test_organisation_term_shape() -> None:
    assert is_organisation_term("灵霄宗")
    assert not is_organisation_term("叶然")
    assert not is_organisation_term("宗")
    assert not is_organisation_term("Linh宗")


def test_find_variants_one_char_in_name_only() -> None:
    assert find_variants("灵霄宗", "他是凌霄宗的人，灵霄宗也在") == ["凌霄宗"]
    assert find_variants("灵霄宗", "灵霄门的弟子") == []  # đổi hậu tố = tổ chức khác
    assert find_variants("灵霄宗", "凌云宗") == []  # lệch hai chữ


def test_variant_in_other_glossary_entry_is_not_a_variant() -> None:
    assert find_variants("灵霄宗", "凌霄宗", frozenset({"凌霄宗"})) == []


def test_batch_block_hints_variant_with_same_rendering() -> None:
    hits = _batch_glossary_block(_batch("凌霄宗的弟子来了"), GLOSSARY)
    assert hits == [
        {"term": "凌霄宗", "must_translate_as": "Linh Tiêu Tông", "variant_of": "灵霄宗"}
    ]


def test_batch_block_with_both_forms_lists_both() -> None:
    hits = _batch_glossary_block(_batch("灵霄宗", "凌霄宗"), GLOSSARY)
    assert [h["term"] for h in hits] == ["灵霄宗", "凌霄宗"]
    assert "variant_of" not in hits[0] and hits[1]["variant_of"] == "灵霄宗"
    assert {h["must_translate_as"] for h in hits} == {"Linh Tiêu Tông"}


def test_batch_block_unchanged_without_variant() -> None:
    assert _batch_glossary_block(_batch("叶然来了"), GLOSSARY) == [
        {"term": "叶然", "must_translate_as": "Diệp Nhiên"}
    ]


def test_diagnostics_counts_variant_inconsistency() -> None:
    source = {1: "灵霄宗的人", 20: "凌霄宗的人"}
    translated = [
        SimpleNamespace(index=1, text="người của Linh Tiêu Tông"),
        SimpleNamespace(index=20, text="người của Thiên Kiếm Môn"),
    ]
    findings = _glossary_violations(source, translated, GLOSSARY)  # type: ignore[arg-type]
    assert len(findings) == 1
    assert findings[0]["occurrences"] == 2
    assert findings[0]["violations"] == 1
    assert findings[0]["example_indices"] == [20]


class TestReviewFixes:
    """[v3.23.645] Ca biên do rà soát độc lập tìm ra."""

    def test_common_function_characters_are_not_organisation_suffixes(self) -> None:
        assert find_variants("天地会", "明天才会到。今天也会来。") == []
        assert find_variants("斧头帮", "回头帮你") == []
        assert find_variants("青龙会", "那条龙会喷火") == []

    def test_variants_do_not_displace_real_terms_and_are_not_duplicated(self) -> None:
        glossary = "灵霄宗 = Linh Tiêu Tông\n灵云宗 = Linh Vân Tông\n" + "\n".join(
            f"人物{chr(0x4e00 + i)}名 = Nhân {i}" for i in range(30)
        )
        text = "灵山宗 " + " ".join(f"人物{chr(0x4e00 + i)}名" for i in range(30))
        hits = _batch_glossary_block(_batch(text), glossary)
        assert len(hits) == 25
        assert [h["term"] for h in hits].count("灵山宗") <= 1
        assert all("variant_of" not in h for h in hits[:25] if h["term"] != "灵山宗")

    def test_duplicate_variant_listed_once(self) -> None:
        glossary = "灵霄宗 = Linh Tiêu Tông\n灵云宗 = Linh Vân Tông"
        hits = _batch_glossary_block(_batch("灵山宗的人"), glossary)
        assert [h["term"] for h in hits] == ["灵山宗"]

    def test_diagnostics_fold_traditional_source(self) -> None:
        glossary = "灵霄宗 = Thiên Kiếm Phái\n凌霄宗 = Bạch Vân Các"
        source = {1: "靈霄宗的人"}
        translated = [SimpleNamespace(index=1, text="người của Thiên Kiếm Phái")]
        assert _glossary_violations(source, translated, glossary) == []  # type: ignore[arg-type]

    def test_index_is_fast_with_hundreds_of_organisations(self) -> None:
        import time

        from subtitles_extractor.application.services.term_variants import VariantIndex

        terms = [f"{chr(0x4e00 + i)}{chr(0x4f00 + i)}宗" for i in range(400)]
        text = "".join(f"{chr(0x4e00 + i)}{chr(0x5000 + i)}宗说话" for i in range(400)) * 8
        started = time.perf_counter()
        index = VariantIndex(text)
        for term in terms:
            index.variants_of(term)
        assert time.perf_counter() - started < 1.0
