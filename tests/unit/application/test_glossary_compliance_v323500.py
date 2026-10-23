"""[v3.23.500] Bám bảng thuật ngữ: nhắc theo LÔ khi dịch + ĐO mức tuân thủ ở bản dịch cuối.

Đo trên một bản dịch thật: 72/113 thuật ngữ bị dịch khác quy định (173 lượt), có ca mất hẳn nội
dung (奔驰 = Mercedes-Benz → "tài xế riêng"). Nguyên nhân: cả bảng nằm trong system_instruction nên
model phân tán chú ý.
"""
from __future__ import annotations

from dataclasses import dataclass

from subtitles_extractor.application.services.translation_diagnostics import (
    detect_quality_flags,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    _MAX_BATCH_GLOSSARY_TERMS,
    _batch_glossary_block,
)

GLOSSARY = (
    "福安私立医院 = Bệnh viện Tư nhân Phúc An\n"
    "宠妻系统 = Hệ thống Cưng Chiều Vợ\n"
    "奔驰 = Mercedes-Benz\n"
    "心动值 = chỉ số rung động\n"
    "身份证 = thẻ căn cước / CMND"
)


@dataclass
class _Line:
    text: str


@dataclass
class _Event:
    index: int
    text: str


class TestBatchGlossaryBlock:
    """Chỉ nhắc thuật ngữ CÓ trong lô — ngắn, sát ngữ cảnh, không tốn token thừa."""

    def test_only_terms_present_in_batch(self) -> None:
        block = _batch_glossary_block([_Line("奔驰接送"), _Line("心动值达百分之百")], GLOSSARY)
        assert {item["term"] for item in block} == {"奔驰", "心动值"}

    def test_expected_translation_included(self) -> None:
        block = _batch_glossary_block([_Line("奔驰接送")], GLOSSARY)
        assert block[0]["must_translate_as"] == "Mercedes-Benz"

    def test_batch_without_terms_is_empty(self) -> None:
        assert _batch_glossary_block([_Line("你好")], GLOSSARY) == []

    def test_empty_glossary_or_batch(self) -> None:
        assert _batch_glossary_block([_Line("奔驰")], "") == []
        assert _batch_glossary_block([], GLOSSARY) == []

    def test_capped_to_avoid_prompt_bloat(self) -> None:
        many = "\n".join(f"术语{i:03d} = thuật ngữ {i}" for i in range(100))
        batch = [_Line("".join(f"术语{i:03d}" for i in range(100)))]
        assert len(_batch_glossary_block(batch, many)) <= _MAX_BATCH_GLOSSARY_TERMS


class TestGlossaryViolationFlag:
    """Cờ đo mức tuân thủ ở bản dịch cuối — không đo thì không biết để sửa."""

    def test_detects_violation(self) -> None:
        flags = detect_quality_flags(
            [_Event(1, "奔驰接送")], [_Event(1, "tài xế riêng.")], glossary=GLOSSARY
        )
        assert flags["counts"]["glossary_violations"] == 1
        assert flags["glossary_violations"][0]["term"] == "奔驰"
        assert flags["glossary_violations"][0]["example_indices"] == [1]

    def test_compliant_translation_not_flagged(self) -> None:
        flags = detect_quality_flags(
            [_Event(1, "奔驰接送")], [_Event(1, "Xe Mercedes-Benz đưa đón.")], glossary=GLOSSARY
        )
        assert flags["counts"]["glossary_violations"] == 0

    def test_any_alternative_accepted(self) -> None:
        # "thẻ căn cước / CMND": dùng phương án nào cũng đạt.
        flags = detect_quality_flags(
            [_Event(1, "带上身份证"), _Event(2, "带上身份证")],
            [_Event(1, "mang thẻ căn cước"), _Event(2, "mang CMND")],
            glossary=GLOSSARY,
        )
        assert flags["counts"]["glossary_violations"] == 0

    def test_case_insensitive(self) -> None:
        flags = detect_quality_flags(
            [_Event(1, "奔驰")], [_Event(1, "xe MERCEDES-BENZ")], glossary=GLOSSARY
        )
        assert flags["counts"]["glossary_violations"] == 0

    def test_no_glossary_keeps_flag_empty(self) -> None:
        flags = detect_quality_flags([_Event(1, "奔驰")], [_Event(1, "xe hơi")])
        assert flags["glossary_violations"] == []
        assert flags["counts"]["glossary_violations"] == 0
