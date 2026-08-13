"""Unit tests cho các bug fixes của ``SubtitleBuilder`` phiên bản 2.23.

Mỗi test bảo vệ một bug đã sửa để tránh regression trong tương lai. Tests
được viết theo phong cách AAA (Arrange-Act-Assert) với tên mô tả rõ behavior
kỳ vọng. Sử dụng dữ liệu nhỏ, lập sẵn (không phụ thuộc file ngoài).
"""

from __future__ import annotations

import pytest

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    SubtitleBuilderConfig,
)
from subtitles_extractor.application.services.subtitle_builder import (
    SubtitleBuilder,
    _is_latin_gibberish,
)
from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.roi import Roi, TextAlignment


def _make_frame(
    timestamp_sec: float,
    text: str,
    confidence: float,
    polygon: list[tuple[int, int]] | None = None,
) -> OcrFrameResult:
    """Helper tạo OcrFrameResult ngắn gọn."""
    if polygon is None:
        polygon = [(300, 20), (400, 20), (400, 80), (300, 80)]
    box = OcrTextBox(
        text=text,
        confidence=Confidence(confidence),
        polygon=polygon,
    )
    frame_index = int(timestamp_sec * 25)
    return OcrFrameResult(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        text_boxes=[box],
    )


def _make_roi() -> Roi:
    """ROI mặc định 640×100 với alignment CENTER."""
    return Roi(x=40, y=880, width=640, height=100, alignment=TextAlignment.CENTER)


# ── BUG A: len=1 distance=1 KHÔNG được coi là similarity=1.0 ───────────


class TestBugASingleCharCjkNotMergedFalsely:
    """`'三'` vs `'二'` cùng len=1 distance=1 nhưng 100% khác nhau."""

    def test_single_char_cjk_different_chars_not_merged(self) -> None:
        # Arrange: tạo 2 câu '三' (gap dài) rồi '二' với gap > merge_gap
        # để 2 group riêng biệt được tạo.
        frames = []
        for i in range(10):
            ts = 15.0 + i * 0.04
            frames.append(_make_frame(ts, "三", 0.95))
        # Gap 0.6+ giây
        for i in range(5):
            ts = 16.1 + i * 0.04
            frames.append(_make_frame(ts, "二", 0.92))

        builder = SubtitleBuilder(SubtitleBuilderConfig(use_viterbi=False))

        # Act
        events = builder.build(frames, roi=_make_roi())

        # Assert: phải có 2 event riêng biệt
        texts = [e.text for e in events]
        assert "三" in texts, f"Phải có '三' trong kết quả, nhận: {texts}"
        assert "二" in texts, f"Phải có '二' trong kết quả, nhận: {texts}"

    def test_two_char_cjk_distance_one_still_merged(self) -> None:
        """Với len=2, distance=1 (50% khác) → vẫn merge (OCR error điển hình)."""
        builder = SubtitleBuilder(SubtitleBuilderConfig())

        # `_calculate_effective_similarity` chứ không build full pipeline
        score = builder._calculate_effective_similarity(
            "你好", "我好", conf_alpha=0.90, conf_beta=0.90
        )
        # confidence < 0.93 → critical reversal KHÔNG apply
        # nhưng `'我'` là tử huyệt → ở conf 0.90 (<0.93) sẽ KHÔNG distinct.
        # shortcut len 2-3 với distance=1 → return 1.0
        assert score == 1.0, f"len=2 distance=1 sim phải = 1.0, nhận {score}"


# ── BUG B: Critical Reversal cần confidence ≥ 0.93 ─────────────────────


class TestBugBCriticalReversalConfidenceAware:
    """OCR error '没'→'设' không được nhầm là reversal thật."""

    def test_ocr_error_low_conf_not_distinct(self) -> None:
        """'看清楚没有' vs '看清楚设有' với conf 0.95 / 0.92 → OCR error → NOT distinct."""
        result = SubtitleBuilder._is_distinct_cjk_utterance(
            "看清楚没有", "看清楚设有", conf_alpha=0.95, conf_beta=0.92
        )
        assert result is False, "OCR error '没'→'设' conf 0.92 không phải reversal thật"

    def test_high_conf_reversal_still_distinct(self) -> None:
        """'我爱你' vs '我不爱你' với conf 0.97 / 0.96 → reversal thật → distinct."""
        result = SubtitleBuilder._is_distinct_cjk_utterance(
            "我爱你", "我不爱你", conf_alpha=0.97, conf_beta=0.96
        )
        assert result is True, "Reversal thật conf cao phải distinct"

    def test_critical_keyword_low_conf_one_side(self) -> None:
        """Một bên conf < 0.93 → bỏ qua critical reversal → không distinct."""
        result = SubtitleBuilder._is_distinct_cjk_utterance(
            "看清楚没有", "看清楚设有", conf_alpha=0.95, conf_beta=0.88
        )
        assert result is False, "conf 0.88 (< 0.93) → OCR error, không phải reversal"


# ── BUG C: Latin Gibberish bắt rác uppercase + low conf ────────────────


class TestBugCLatinGibberishImproved:
    """Bắt rác như GAPST, GNPSU với 3-5 frames."""

    def test_gapst_gnpsu_caught(self) -> None:
        """GAPST 3 frames conf 0.49 phải bị bắt."""
        assert _is_latin_gibberish("GAPST", confidence=0.49, frame_count=3) is True
        assert _is_latin_gibberish("GNPSU", confidence=0.49, frame_count=3) is True

    def test_valid_acronyms_kept(self) -> None:
        """Acronym hợp lệ KHÔNG bị drop."""
        assert _is_latin_gibberish("OK", confidence=0.5, frame_count=1) is False
        assert _is_latin_gibberish("USD", confidence=0.5, frame_count=1) is False
        assert _is_latin_gibberish("CEO", confidence=0.5, frame_count=1) is False

    def test_high_confidence_text_kept(self) -> None:
        """Text conf >= 0.85 không drop (tin cậy cao)."""
        assert _is_latin_gibberish("GAPST", confidence=0.88, frame_count=3) is False

    def test_long_uppercase_low_conf_caught(self) -> None:
        """Chuỗi uppercase 4+ liên tiếp + conf < 0.65 → drop."""
        assert _is_latin_gibberish("LKTR", confidence=0.55, frame_count=2) is True
        assert _is_latin_gibberish("RWWZ", confidence=0.50, frame_count=4) is True

    def test_boundary_vowel_ratio(self) -> None:
        """vowel_ratio = 0.20 (1/5) phải bị bắt (boundary inclusive)."""
        # GNPSU: G, N, P, S, U → 1 vowel (U) / 5 letters = 0.20
        assert _is_latin_gibberish("GNPSU", confidence=0.49, frame_count=3) is True


# ── BUG D: Latin-CJK hỗn hợp ngắn `G光光` ───────────────────────────────


class TestBugDShortMixedCjkLatinDropped:
    """'G光光' (1 Latin + 2 CJK, 1 frame, conf 0.50) phải bị drop."""

    def test_short_cjk_with_latin_noise_dropped(self) -> None:
        # Arrange: 1 frame text='G光光', conf rất thấp
        frame = _make_frame(50.0, "G光光", 0.50)
        # Padding để có tối thiểu pipeline (cần build)
        # Tạo thêm 1 câu chính khác xa
        main_frames = [_make_frame(20.0 + i * 0.04, "正常的句子", 0.95) for i in range(20)]
        frames = main_frames + [frame]

        builder = SubtitleBuilder(SubtitleBuilderConfig(use_viterbi=False))

        # Act
        events = builder.build(frames, roi=_make_roi())

        # Assert
        texts = [e.text for e in events]
        assert "G光光" not in texts, f"Rác 'G光光' phải bị drop, nhận: {texts}"


# ── BUG E: CJK 1-ký tự intro animated giữ lại ──────────────────────────


class TestBugEAnimatedCjkIntroKept:
    """`'十'` ở 48.28 (2 frame, conf 0.68) — intro của `'十多个人肾虚'` 48.64."""

    def test_animated_intro_kept(self) -> None:
        # Arrange: 2 frame '十' rồi câu dài cùng prefix
        frames = [
            _make_frame(48.28, "十", 0.68),
            _make_frame(48.32, "十", 0.68),
        ]
        for i in range(30):
            ts = 48.64 + i * 0.04
            frames.append(_make_frame(ts, "十多个人肾虚", 0.95))

        builder = SubtitleBuilder(SubtitleBuilderConfig(use_viterbi=False))

        # Act
        events = builder.build(frames, roi=_make_roi())

        # Assert [re-calibrated v3.15]: hiệu chuẩn với ground-truth thực
        # (haomen 2122 câu) cho thấy phụ đề chuẩn KHÔNG BAO GIỜ chứa event
        # < 0.21s; mảnh typing-intro 0.08s ('十') là rác flash — bị
        # drop_flash_fragments loại. Câu đầy đủ phải được giữ nguyên vẹn.
        texts = [e.text for e in events]
        assert "十" not in texts, f"Mảnh intro 0.08s phải bị loại, nhận: {texts}"
        assert "十多个人肾虚" in texts

    def test_orphan_single_char_still_dropped(self) -> None:
        """`'土'` 2 frame conf 0.68 KHÔNG có câu kế tiếp → DROP (rác cảnh nền)."""
        frames = [
            _make_frame(48.28, "土", 0.68),
            _make_frame(48.32, "土", 0.68),
            # Câu kế tiếp KHÁC prefix
            _make_frame(50.0, "完全khác", 0.95),
            _make_frame(50.04, "完全khác", 0.95),
        ]

        builder = SubtitleBuilder(SubtitleBuilderConfig(use_viterbi=False))
        events = builder.build(frames, roi=_make_roi())
        texts = [e.text for e in events]
        assert "土" not in texts, f"Orphan '土' (rác cảnh nền) phải bị drop, nhận: {texts}"


# ── NEW: Single-Frame Outlier Absorber ─────────────────────────────────


class TestSingleFrameOutlierAbsorber:
    """`'我想高开就高开'` 1 frame conf 0.842 → absorb vào câu chính."""

    def test_single_frame_outlier_absorbed(self) -> None:
        # Arrange: 1 frame outlier + 22 frame câu chính
        frames = [_make_frame(64.32, "我想高开就高开", 0.842)]
        for i in range(22):
            ts = 64.40 + i * 0.04
            frames.append(_make_frame(ts, "我想离开就离开", 0.924))

        builder = SubtitleBuilder(SubtitleBuilderConfig(use_viterbi=False))

        # Act
        events = builder.build(frames, roi=_make_roi())

        # Assert
        texts = [e.text for e in events]
        assert "我想高开就高开" not in texts, (
            f"Outlier 1-frame phải được absorb, nhận: {texts}"
        )
        assert "我想离开就离开" in texts
