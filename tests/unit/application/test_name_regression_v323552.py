"""[v3.23.552] Hồi quy cho chuẩn đoán thật (phim 3902 dòng, v3.23.551).

1. Giai đoạn TINH CHỈNH làm mất tên riêng đã dịch đúng: ``桃花嫂子`` "chị dâu Đào Hoa" → "vợ Đào Hoa".
2. "G-Class" (dạng rút gọn hợp lệ của "xe Mercedes G-Class") bị tính vi phạm tên riêng oan.
3. Bằng chứng lệch KHỐI chỉ ghi dòng đầu/cuối khối — không kiểm lại được phần giữa.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from subtitles_extractor.application.services.name_regression_guard import (
    find_name_regressions,
)
from subtitles_extractor.application.services.translation_diagnostics import (
    detect_quality_flags,
    glossary_alternatives,
)

GLOSSARY = (
    "桃花嫂子 = chị dâu Đào Hoa\n"
    "大G = xe Mercedes G-Class (Đại G)\n"
    "作威作福 = tác oai tác quái"
)


# ── 1. Lưới chặn mất tên riêng ───────────────────────────────────────────────
class TestNameRegressionGuard:
    def test_real_case_detected(self) -> None:
        sources = {3267: "桃花嫂子的元神碎片"}
        previous = {3267: "mảnh nguyên thần của chị dâu Đào Hoa,"}
        current = {3267: "mảnh nguyên thần của vợ Đào Hoa,"}
        assert find_name_regressions(sources, previous, current, GLOSSARY) == [
            (3267, "桃花嫂子"),
        ]

    def test_name_moved_to_neighbour_line_is_fine(self) -> None:
        """Tinh chỉnh được phép dời tên sang dòng kề (dòng phụ đề là mảnh câu)."""
        sources = {1: "桃花嫂子的", 2: "元神碎片"}
        previous = {1: "của chị dâu Đào Hoa", 2: "mảnh nguyên thần"}
        current = {1: "mảnh nguyên thần", 2: "của chị dâu Đào Hoa"}
        assert find_name_regressions(sources, previous, current, GLOSSARY) == []

    def test_expression_rewording_is_allowed(self) -> None:
        """Thành ngữ được diễn đạt lại khi tinh chỉnh — KHÔNG phải hồi quy."""
        sources = {1: "他一向作威作福"}
        previous = {1: "Hắn vốn tác oai tác quái."}
        current = {1: "Hắn vốn hống hách bá đạo."}
        assert find_name_regressions(sources, previous, current, GLOSSARY) == []

    def test_previous_already_wrong_is_not_a_regression(self) -> None:
        sources = {1: "桃花嫂子来了"}
        previous = {1: "Vợ Đào Hoa đến rồi."}
        current = {1: "Vợ Đào Hoa tới rồi."}
        assert find_name_regressions(sources, previous, current, GLOSSARY) == []

    def test_use_case_reverts_only_regressed_line(self) -> None:
        from subtitles_extractor.application.use_cases.translate_subtitles import (
            TranslateSubtitlesUseCase,
        )
        from subtitles_extractor.domain.ports.subtitle_translator_port import (
            TranslationLine,
        )

        def line(index: int, text: str) -> TranslationLine:
            return TranslationLine(index=index, start_ms=0, end_ms=1000, text=text)

        source = [line(1, "桃花嫂子的元神碎片"), line(2, "你好")]
        previous = [line(1, "mảnh nguyên thần của chị dâu Đào Hoa,"), line(2, "Chào.")]
        current = [line(1, "mảnh nguyên thần của vợ Đào Hoa,"), line(2, "Xin chào.")]
        result = TranslateSubtitlesUseCase._revert_name_regressions(
            current, previous, source, GLOSSARY, "Tinh chỉnh"
        )
        assert result[0].text == "mảnh nguyên thần của chị dâu Đào Hoa,"
        assert result[1].text == "Xin chào."  # dòng lành GIỮ bản tinh chỉnh

    def test_use_case_skips_without_glossary(self) -> None:
        from subtitles_extractor.application.use_cases.translate_subtitles import (
            TranslateSubtitlesUseCase,
        )

        stage = [SimpleNamespace(index=1, text="x")]
        assert TranslateSubtitlesUseCase._revert_name_regressions(
            stage, [], [], "", "Tinh chỉnh"
        ) is stage


# ── 2. Từ ghép gạch nối ──────────────────────────────────────────────────────
class TestHyphenCompound:
    def test_hyphen_word_becomes_alternative(self) -> None:
        assert "g-class" in glossary_alternatives("xe Mercedes G-Class (Đại G)")

    def test_real_case_no_longer_flagged(self) -> None:
        @dataclass
        class _E:
            index: int
            text: str
            start_sec: float = 0.0
            end_sec: float = 3.0

        flags = detect_quality_flags(
            [_E(1, "大G的车钥匙呢")],
            [_E(1, "có chìa khóa xe G-Class chứ?")],
            glossary="大G = xe Mercedes G-Class (Đại G)",
        )
        assert flags["glossary_violations"] == []


# ── 3. Bằng chứng khối lệch ──────────────────────────────────────────────────
class TestBlockEvidence:
    def test_block_focus_covers_whole_block(self) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            _misalignment_focus_lines,
        )

        focus = _misalignment_focus_lines([
            {"index": 2129, "belongs_to_index": 2129, "shift": 0,
             "block_lines": 15, "min_correlation": -0.04},
            {"index": 2143, "belongs_to_index": 2143, "shift": 0},
        ])
        assert focus == list(range(2129, 2144))

    def test_block_focus_is_capped(self) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            MAX_BLOCK_EVIDENCE_LINES,
            _misalignment_focus_lines,
        )

        focus = _misalignment_focus_lines([
            {"index": 1, "belongs_to_index": 1, "shift": 0,
             "block_lines": 80, "min_correlation": -0.5},
            {"index": 80, "belongs_to_index": 80, "shift": 0},
        ])
        assert len(focus) == MAX_BLOCK_EVIDENCE_LINES

    def test_anchor_focus_unchanged(self) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            _misalignment_focus_lines,
        )

        assert _misalignment_focus_lines([
            {"index": 5, "belongs_to_index": 7, "shift": 2},
            {"index": 7, "belongs_to_index": 5, "shift": -2},
        ]) == [5, 7]

    def test_sample_records_more_than_twelve_lines_for_block(self) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            _batch_content_sample,
        )

        batch = [SimpleNamespace(index=i, original_text=f"src{i}", text="") for i in range(1, 61)]
        payload = {"subtitles": [{"line_no": i, "text": f"t{i}"} for i in range(1, 61)]}
        sample = _batch_content_sample(batch, payload, list(range(20, 40)))
        assert len(sample) == 22  # 20 dòng khối + 1 dòng mỗi bên
        assert sample[0]["line_no"] == 19 and sample[-1]["line_no"] == 40
