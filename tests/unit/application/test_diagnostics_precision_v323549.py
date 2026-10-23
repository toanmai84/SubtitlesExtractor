"""[v3.23.549] Độ CHÍNH XÁC của chẩn đoán dịch — dựng từ phim thật 3902 dòng.

Soi file chuẩn đoán thật cho thấy các cờ quan trọng nhất báo SAI theo cả hai chiều:

* ``shifted_lines`` báo 5 dòng — cả 5 đều DỊCH ĐÚNG. Trong khi đó khối 2684-2701 sai liền 18 dòng
  (model gộp hai dòng ngắn rồi đẩy lệch cả đoạn) thì KHÔNG cờ nào báo.
* ``glossary_violations`` báo 42 lượt — 37 lượt là bản dịch đúng, bị tính sai vì thuật ngữ rơi sang
  dòng kề (phụ đề là mảnh câu) hoặc đổi loại từ ("kẻ" / "gã" / "tên" quê mùa).
* Mục thuật ngữ ``生死道消 = "thân tử đạo消"`` dịch dở dang lọt qua mọi bộ kiểm rồi vào prompt.
* ``process_log`` rỗng ở lần chạy sạch — không phân biệt được với cơ chế ghi hỏng.
"""

from __future__ import annotations

from dataclasses import dataclass

from subtitles_extractor.application.services.alignment_check import (
    corroborated_findings,
    drop_unreliable_anchors,
    find_low_correlation_blocks,
)
from subtitles_extractor.application.services.roster_format import (
    GlossaryEntry,
    is_half_translated,
    split_half_translated_entries,
)
from subtitles_extractor.application.services.translation_diagnostics import (
    _glossary_violations,
    _matches_expected_term,
    detect_quality_flags,
)


@dataclass
class _Event:
    index: int
    text: str
    start_sec: float = 0.0
    end_sec: float = 1.0


# Nguồn có độ dài dao động rõ (như phụ đề thật) — cần biến thiên để tương quan có nghĩa.
_SOURCE_LENGTHS = [3, 9, 5, 12, 4, 10, 6, 14, 3, 8, 11, 5, 13, 4, 9, 7, 12, 3, 10, 6] * 3


def _aligned_corpus(total: int = 60) -> tuple[dict[int, str], dict[int, str]]:
    """Nguồn + bản dịch CĂN ĐÚNG: bản dịch dài ~4 lần nguồn, đúng tỉ lệ đo được trên phim thật."""
    sources: dict[int, str] = {}
    translations: dict[int, str] = {}
    for index in range(total):
        length = _SOURCE_LENGTHS[index % len(_SOURCE_LENGTHS)]
        sources[index] = "字" * length
        translations[index] = "x" * (length * 4)
    return sources, translations


# ── Lệch theo KHỐI ────────────────────────────────────────────────────────────
class TestLowCorrelationBlocks:
    def test_aligned_translation_has_no_block(self) -> None:
        sources, translations = _aligned_corpus()
        assert find_low_correlation_blocks(sources, translations) == []

    def test_detects_merge_then_shift_block(self) -> None:
        """Tái hiện ca thật: model gộp hai dòng, 18 dòng sau mang bản dịch của dòng KẾ TIẾP."""
        sources, translations = _aligned_corpus()
        shifted = dict(translations)
        for index in range(20, 38):
            shifted[index] = translations[index + 1]
        blocks = find_low_correlation_blocks(sources, shifted)
        assert len(blocks) == 1
        block = blocks[0]
        assert block["from_index"] <= 22 and block["to_index"] >= 35
        assert block["min_correlation"] < 0.2

    def test_short_local_dip_is_ignored(self) -> None:
        """Dao động cục bộ ngắn (1-4 cửa sổ) là nhiễu lành tính — không được báo."""
        sources, translations = _aligned_corpus()
        noisy = dict(translations)
        noisy[30], noisy[31] = translations[31], translations[30]
        assert find_low_correlation_blocks(sources, noisy) == []

    def test_too_short_input_is_ignored(self) -> None:
        sources, translations = _aligned_corpus(total=12)
        assert find_low_correlation_blocks(sources, translations) == []

    def test_diagnostics_report_blocks(self) -> None:
        sources, translations = _aligned_corpus()
        for index in range(20, 38):
            translations[index] = "x" * (_SOURCE_LENGTHS[(index + 1) % 20] * 4)
        flags = detect_quality_flags(
            [_Event(i, t) for i, t in sources.items()],
            [_Event(i, t) for i, t in translations.items()],
        )
        assert flags["counts"]["misaligned_blocks"] == 1
        assert flags["counts"]["misaligned_block_lines"] >= 14


# ── Mốc neo: chống dương tính giả ─────────────────────────────────────────────
class TestAnchorFalsePositives:
    def test_target_more_common_than_source_is_dropped(self) -> None:
        """Ca thật ``医生`` → "bác sĩ": gốc 11 dòng, dịch 16 dòng — model thêm hô ngữ."""
        sources = {1: "医生来了", 2: "你怎么知道的"}
        translations = {1: "Bác sĩ đến rồi.", 2: "Bác sĩ, sao ông biết?"}
        assert drop_unreliable_anchors([("医生", "bác sĩ")], sources, translations) == []

    def test_one_to_one_anchor_is_kept(self) -> None:
        sources = {1: "医生来了", 2: "你好"}
        translations = {1: "Bác sĩ đến rồi.", 2: "Xin chào."}
        anchors = [("医生", "bác sĩ")]
        assert drop_unreliable_anchors(anchors, sources, translations) == anchors

    def test_absent_source_term_is_dropped(self) -> None:
        assert drop_unreliable_anchors([("医生", "bác sĩ")], {1: "你好"}, {1: "Chào"}) == []

    def test_isolated_findings_with_distinct_shifts_are_dropped(self) -> None:
        """Đúng dạng 5 dương tính giả thật: mỗi dòng một độ lệch, cách xa nhau."""
        findings = [
            {"index": 642, "belongs_to_index": 641, "shift": -1},
            {"index": 696, "belongs_to_index": 698, "shift": 2},
            {"index": 1128, "belongs_to_index": 1125, "shift": -3},
            {"index": 2664, "belongs_to_index": 2667, "shift": 3},
            {"index": 2899, "belongs_to_index": 2901, "shift": 2},
        ]
        assert corroborated_findings(findings) == []

    def test_run_with_same_shift_is_kept(self) -> None:
        findings = [
            {"index": 10, "belongs_to_index": 11, "shift": 1},
            {"index": 12, "belongs_to_index": 13, "shift": 1},
        ]
        assert [item["index"] for item in corroborated_findings(findings)] == [10, 12]

    def test_mutual_swap_is_kept(self) -> None:
        findings = [
            {"index": 5, "belongs_to_index": 7, "shift": 2},
            {"index": 7, "belongs_to_index": 5, "shift": -2},
        ]
        assert [item["index"] for item in corroborated_findings(findings)] == [5, 7]

    def test_no_duplicates_when_swap_and_run_overlap(self) -> None:
        findings = [
            {"index": 5, "belongs_to_index": 6, "shift": 1},
            {"index": 6, "belongs_to_index": 5, "shift": -1},
            {"index": 8, "belongs_to_index": 9, "shift": 1},
        ]
        indices = [item["index"] for item in corroborated_findings(findings)]
        assert indices == sorted(set(indices))


# ── Vi phạm thuật ngữ ─────────────────────────────────────────────────────────
class TestGlossaryViolationPrecision:
    def test_term_on_neighbour_line_is_not_a_violation(self) -> None:
        """Ca thật: ``和江城医院的`` → "cùng với Viện trưởng Tôn", thuật ngữ nằm ở dòng kế."""
        sources = {1: "和江城医院的", 2: "孙院长"}
        translated = [
            _Event(1, "cùng với Viện trưởng Tôn"),
            _Event(2, "của Bệnh viện Giang Thành"),
        ]
        glossary = "江城医院 = Bệnh viện Giang Thành"
        assert _glossary_violations(sources, translated, glossary) == []

    def test_classifier_variation_is_accepted(self) -> None:
        """"kẻ quê mùa" / "gã quê mùa" / "tên quê mùa" đều đúng."""
        for variant in ("gã quê mùa", "tên quê mùa", "thằng quê mùa"):
            assert _matches_expected_term(["kẻ quê mùa"], variant)

    def test_dropped_leading_classifier_is_accepted(self) -> None:
        assert _matches_expected_term(["tiền sính lễ"], "mẹ cũng nhận sính lễ rồi")
        assert _matches_expected_term(["đôi cẩu nam nữ"], "hai kẻ cẩu nam nữ đó")

    def test_parenthetical_is_a_valid_alternative(self) -> None:
        """Mục ``大G = "xe Mercedes G-Class (Đại G)"``: dùng "Đại G" là đúng."""
        sources = {1: "我靠最新款奔驰大G"}
        translated = [_Event(1, "Mẹ kiếp xe Đại G mới nhất đấy!")]
        glossary = "大G = xe Mercedes G-Class (Đại G)"
        assert _glossary_violations(sources, translated, glossary) == []

    def test_genuinely_different_wording_is_still_flagged(self) -> None:
        """Nới lỏng không được che lỗi thật: dịch sai hẳn nghĩa vẫn phải bị bắt."""
        sources = {1: "这玉佩虽然不值钱", 2: "你好", 3: "再见"}
        translated = [
            _Event(1, "cái vòng này tuy không đáng tiền"),
            _Event(2, "Xin chào"),
            _Event(3, "Tạm biệt"),
        ]
        findings = _glossary_violations(sources, translated, "玉佩 = ngọc bội")
        assert findings and findings[0]["term"] == "玉佩"

    def test_neighbour_window_does_not_reach_two_lines_away(self) -> None:
        sources = {1: "玉佩", 2: "a", 3: "b"}
        translated = [_Event(1, "x"), _Event(2, "y"), _Event(3, "ngọc bội")]
        assert _glossary_violations(sources, translated, "玉佩 = ngọc bội")


# ── Mục thuật ngữ dịch dở ─────────────────────────────────────────────────────
class TestHalfTranslatedGlossary:
    def test_real_case_detected(self) -> None:
        assert is_half_translated("thân tử đạo消")

    def test_healthy_entries_pass(self) -> None:
        for value in ("mảnh nguyên thần", "xe Mercedes G-Class (Đại G)", ""):
            assert not is_half_translated(value)

    def test_fully_cjk_target_is_legitimate(self) -> None:
        """Ngôn ngữ đích có thể CHÍNH LÀ tiếng Trung — không được loại oan."""
        assert not is_half_translated("生死道消")

    def test_split_keeps_order(self) -> None:
        entries = [
            GlossaryEntry("元神碎片", "mảnh nguyên thần"),
            GlossaryEntry("生死道消", "thân tử đạo消"),
            GlossaryEntry("识海", "Thức hải"),
        ]
        healthy, broken = split_half_translated_entries(entries)
        assert [e.term for e in healthy] == ["元神碎片", "识海"]
        assert [e.term for e in broken] == ["生死道消"]

    def test_normalize_glossary_drops_broken_entry(self) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            _normalize_glossary,
        )

        normalized = _normalize_glossary(
            "元神碎片 = mảnh nguyên thần\n生死道消 = thân tử đạo消"
        )
        assert "元神碎片" in normalized
        assert "生死道消" not in normalized


# ── Thông điệp thử lại ────────────────────────────────────────────────────────
class TestMisalignmentMessage:
    def test_block_message_names_range_not_self_pointer(self) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            _misalignment_message,
        )

        message = _misalignment_message([
            {"index": 20, "belongs_to_index": 20, "shift": 0,
             "block_lines": 18, "min_correlation": -0.6},
            {"index": 37, "belongs_to_index": 37, "shift": 0},
        ])
        assert "KHỐI" in message and "20-37" in message
        assert "mang nội dung của dòng 20" not in message

    def test_anchor_message_unchanged(self) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            _misalignment_message,
        )

        message = _misalignment_message([{"index": 5, "belongs_to_index": 7, "shift": 2}])
        assert "dòng 5 mang nội dung của dòng 7" in message


# ── Nhật ký diễn biến ─────────────────────────────────────────────────────────
class TestProcessLogOnCleanRun:
    def test_stage_completion_is_recorded(self) -> None:
        from types import SimpleNamespace

        from subtitles_extractor.application.services.translation_process_log import (
            get_process_log,
        )
        from subtitles_extractor.application.use_cases.translate_subtitles import (
            TranslateSubtitlesUseCase,
        )

        log = get_process_log()
        log.clear()
        stage = SimpleNamespace(kind=SimpleNamespace(value="literal"), model_name="m")
        TranslateSubtitlesUseCase._record_stage_completed(stage, "Dịch thô", [1, 2, 3])
        events = log.snapshot()["events"]
        assert events and events[-1]["event"] == "stage_completed"
        assert events[-1]["lines"] == 3
        assert events[-1]["stage"] == "literal"
        log.clear()
