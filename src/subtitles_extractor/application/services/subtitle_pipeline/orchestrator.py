"""Orchestrator chính: lớp :class:`SubtitleBuilder` mỏng điều phối pipeline.

Sau refactor v3.0, lớp này chỉ là **thin orchestrator** ~120 dòng — toàn
bộ logic được tách ra các module SRP trong package
:mod:`subtitle_pipeline`. So với bản v2.36 (1901 dòng, 27 method trong
một class), bản này dễ test, dễ maintain, và performance tương đương
hoặc tốt hơn (greedy mặc định, không Viterbi).

Thứ tự pipeline:

    1. Pre-filter garbage boxes (box-level OCR rác).
    2. Per-frame spatial cleanup (ROI alignment).
    3. Confidence filter (frame-level).
    4. Cross-frame Y outlier filter.
    5. Group thành :class:`FrameGroup` (greedy/Viterbi).
    6. Merge adjacent duplicates.
    7. Echo trail filter (cấp group, lặp đến hội tụ tối đa 3 lần).
    8. Convert → :class:`SubtitleEvent`.
    9. Filter short duration (flicker absorber).
    10. Post-merge duplicates (bao gồm extended-gap cho identical text).
    11. Filter short text events (rác).
    12. Echo trail filter (cấp event, lần cuối).
    13. Reindex.
    13b. Strip Latin watermark prefix từ CJK multi-line events (v3.6+).
    13c. Strip persistent watermark lines theo tần suất (v3.6+).
    13e. Mốc kết thúc = lúc chữ biến mất (độ giữ khung, v3.23.615).
    14. Reindex (final).
"""

from __future__ import annotations

import logging
from dataclasses import replace
import re
from collections.abc import Sequence
from functools import lru_cache

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    SubtitleBuilderConfig,
)
from subtitles_extractor.application.services.subtitle_pipeline.boundary_refinement import (
    pixel_confirmed_blanks,
    refine_event_boundaries,
)
from subtitles_extractor.application.services.subtitle_pipeline.box_filters import (
    pre_filter_garbage_boxes,
)
from subtitles_extractor.application.services.subtitle_pipeline.box_geometry import (
    drop_drifting_events,
    drop_implausible_boxes,
)
from subtitles_extractor.application.services.subtitle_pipeline.event_filters import (
    convert_groups_to_events,
    extend_ends_by_display_hold,
    extend_group_timing_from_soft_drops,
    filter_echo_trail_events,
    filter_echo_trail_groups,
    filter_short_duration_events,
    filter_short_text_events,
    frame_display_hold,
    merge_adjacent_duplicates,
    post_merge_duplicates,
    reindex_events,
    split_repeat_gap_groups,
    split_stable_variant_groups,
    strip_latin_watermark_prefix_from_cjk_events,
    strip_persistent_watermark_lines,
    textful_timestamps,
)
from subtitles_extractor.application.services.subtitle_pipeline.frame_group import (
    FrameGroup,
)
from subtitles_extractor.application.services.subtitle_pipeline.grouping import (
    group_using_greedy,
    group_using_viterbi,
)
from subtitles_extractor.application.services.subtitle_pipeline.spatial_filters import (
    clean_spatial_outliers,
    filter_cross_frame_spatial_outliers,
    filter_noise_by_confidence,
)
from subtitles_extractor.application.services.subtitle_pipeline.text_correction import (
    centering_reference,
    collect_restored_texts,
)
from subtitles_extractor.application.services.subtitle_pipeline.voting import (
    clear_frame_build_cache,
    prewarm_frame_cache,
)
from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.boundary_track import BoundaryTrack
from subtitles_extractor.domain.value_objects.roi import (
    Roi,
    TextAlignment,
    TextOrientation,
)

# Số vòng lặp tối đa cho echo trail filter (hội tụ thường <= 2 vòng).
_ECHO_TRAIL_MAX_ITERATIONS: int = 3


@lru_cache(maxsize=8192)
def text_similarity_cached(text_left: str, text_right: str) -> float:
    """LRU-cached wrapper quanh :func:`text_similarity` ở module gốc.

    Cache giúp tránh tính lại similarity cho cùng cặp text trong vòng đời
    của một lần xây dựng phụ đề (đặc biệt khi merge multi-pass).

    Args:
        text_left: Text 1.
        text_right: Text 2.

    Returns:
        Similarity ∈ [0.0, 1.0].
    """
    from subtitles_extractor.application.services.text_similarity import (
        text_similarity as _actual_text_similarity,
    )

    return _actual_text_similarity(text_left, text_right)


def _frame_boxes(frame: OcrFrameResult) -> list[tuple[int, int, int, int]]:
    """Bbox các ô chữ có polygon của một khung (hàm thuần)."""
    return [
        box.bounding_box for box in frame.text_boxes if box.bounding_box is not None
    ]



logger = logging.getLogger(__name__)

class SubtitleBuilder:
    """Service xây dựng :class:`SubtitleEvent` từ kết quả OCR theo frame.

    Đây là lớp facade duy nhất mà bên ngoài (presentation/composition layer)
    cần biết. Toàn bộ logic phức tạp được uỷ quyền cho các function thuần
    trong package :mod:`subtitle_pipeline`.

    Args:
        config: Cấu hình builder (ngưỡng similarity, gap, ROI, v.v.).
    """

    def __init__(self, config: SubtitleBuilderConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Backward-compat staticmethod shims (deprecated).
    # ------------------------------------------------------------------
    # Các method dưới đây tồn tại để các unit test v2.x hiện hữu tiếp tục
    # work mà không cần sửa import. Mã nguồn mới nên import trực tiếp từ
    # các module trong package :mod:`subtitle_pipeline`.

    @staticmethod
    def _restore_dropped_yi_prefix(
        voted_text: str,
        texts_with_confidences: list[tuple[str, float]],
    ) -> str:
        """Backward-compat shim — gọi :func:`text_correction.restore_dropped_yi_prefix`."""
        from subtitles_extractor.application.services.subtitle_pipeline.text_correction import (
            restore_dropped_yi_prefix as _restore,
        )

        return _restore(voted_text, texts_with_confidences)

    @staticmethod
    def _restore_dropped_space(
        voted_text: str,
        texts_with_confidences: list[tuple[str, float]],
    ) -> str:
        """Backward-compat shim — gọi :func:`text_correction.restore_dropped_space`."""
        from subtitles_extractor.application.services.subtitle_pipeline.text_correction import (
            restore_dropped_space as _restore,
        )

        return _restore(voted_text, texts_with_confidences)

    @staticmethod
    def _is_distinct_cjk_utterance(
        text_alpha: str,
        text_beta: str,
        conf_alpha: float,
        conf_beta: float,
        time_gap_seconds: float = 0.0,
    ) -> bool:
        """Backward-compat shim — gọi :func:`voting.is_distinct_cjk_utterance`."""
        from subtitles_extractor.application.services.subtitle_pipeline.voting import (
            is_distinct_cjk_utterance as _check,
        )

        return _check(text_alpha, text_beta, conf_alpha, conf_beta, time_gap_seconds)

    @staticmethod
    def _is_superset_within_one_char(text_a: str, text_b: str) -> bool:
        """Backward-compat shim — gọi :func:`event_filters.is_superset_within_one_char`."""
        from subtitles_extractor.application.services.subtitle_pipeline.event_filters import (
            is_superset_within_one_char as _check,
        )

        return _check(text_a, text_b)

    @staticmethod
    def _select_anchor_text(texts_with_confidences: list[tuple[str, float]]) -> str:
        """Backward-compat shim — gọi :func:`voting.select_anchor_text`."""
        from subtitles_extractor.application.services.subtitle_pipeline.voting import (
            select_anchor_text as _select,
        )

        return _select(texts_with_confidences)

    @staticmethod
    def _calculate_effective_similarity(
        text_alpha: str,
        text_beta: str,
        conf_alpha: float,
        conf_beta: float,
        time_gap_seconds: float = 0.0,
    ) -> float:
        """Backward-compat shim — gọi :func:`voting.calculate_effective_similarity`."""
        from subtitles_extractor.application.services.subtitle_pipeline.voting import (
            calculate_effective_similarity as _calc,
        )

        return _calc(text_alpha, text_beta, conf_alpha, conf_beta, time_gap_seconds)

    def _clean_spatial_outliers(
        self,
        frame_result: OcrFrameResult,
        roi: Roi | None,
    ) -> OcrFrameResult:
        """Backward-compat shim — gọi :func:`spatial_filters.clean_spatial_outliers`."""
        from subtitles_extractor.application.services.subtitle_pipeline.spatial_filters import (
            clean_spatial_outliers as _clean,
        )

        return _clean(frame_result, roi, self._config)

    def _convert_groups_to_events(
        self,
        assembled_frame_groups: list[FrameGroup],
    ) -> list[SubtitleEvent]:
        """Backward-compat shim — gọi :func:`event_filters.convert_groups_to_events`."""
        from subtitles_extractor.application.services.subtitle_pipeline.event_filters import (
            convert_groups_to_events as _convert,
        )

        return _convert(assembled_frame_groups, self._config)

    def build(
        self,
        frame_results: Sequence[OcrFrameResult],
        roi: Roi | None = None,
        boundary_track: BoundaryTrack | None = None,
    ) -> list[SubtitleEvent]:
        """Dựng danh sách :class:`SubtitleEvent` từ chuỗi kết quả OCR.

        Args:
            frame_results: Chuỗi frame OCR (sẽ được sort theo timestamp).
            roi: ROI căn lề + kích thước (None = bỏ qua spatial filter).
            boundary_track: [v3.23.680] Dải ranh giới của lượt giải mã → đặt mốc câu đúng khung (None = như cũ).

        Returns:
            Danh sách :class:`SubtitleEvent` đã hoàn chỉnh, index 1-based.
        """
        if not frame_results:
            return []

        # ── Stage 1: Box-level OCR garbage filter ──
        purified_frames = pre_filter_garbage_boxes(frame_results)
        # [v3.23.682] Hộp sai hình dạng so với cỡ chữ phụ đề của chính phim (vật trong cảnh bị đọc thành chữ).
        if roi is None or roi.orientation != TextOrientation.VERTICAL:
            purified_frames, dropped_boxes = drop_implausible_boxes(purified_frames)
            if dropped_boxes:
                logger.info("Bỏ %d hộp chữ sai hình dạng so với cỡ chữ phụ đề (vật trong cảnh).", dropped_boxes)
        if not purified_frames:
            return []

        # ── Stage 2: Per-frame spatial cleanup (alignment-aware) ──
        # [v3.7] Thu thập timestamp frame bị lọc nhưng có nội dung (soft-drop)
        # để dùng phục hồi timing biên sau giai đoạn grouping.
        soft_dropped_timestamps: list[float] = []
        # [v3.23.616] Vị trí chữ của khung soft-drop: chỉ nới biên câu bằng khung có chữ
        # nằm cùng chỗ với câu (xem ``extend_group_timing_from_soft_drops``).
        soft_dropped_boxes: dict[float, list[tuple[int, int, int, int]]] = {}
        spatially_cleaned_frames: list[OcrFrameResult] = []
        for current_frame in purified_frames:
            cleaned_frame = clean_spatial_outliers(current_frame, roi, self._config)
            if not cleaned_frame.is_empty:
                spatially_cleaned_frames.append(cleaned_frame)
            elif not current_frame.is_empty:
                # Frame có nội dung nhưng spatial cleanup loại hết boxes → ghi ts
                soft_dropped_timestamps.append(current_frame.timestamp_sec)
                soft_dropped_boxes[current_frame.timestamp_sec] = _frame_boxes(
                    current_frame
                )

        # ── Stage 3: Confidence-based frame filter ──
        confident_frames = filter_noise_by_confidence(
            spatially_cleaned_frames, self._config.min_confidence
        )
        if not confident_frames:
            return []

        # Frames đã qua spatial nhưng bị lọc bởi confidence → cũng là soft-drop
        confident_ts_set = {f.timestamp_sec for f in confident_frames}
        for f in spatially_cleaned_frames:
            if f.timestamp_sec not in confident_ts_set:
                soft_dropped_timestamps.append(f.timestamp_sec)
                soft_dropped_boxes[f.timestamp_sec] = _frame_boxes(f)

        soft_dropped_timestamps.sort()

        # ── Stage 4: Cross-frame Y-outlier filter ──
        confident_frames = filter_cross_frame_spatial_outliers(confident_frames, roi=roi)
        if not confident_frames:
            return []

        # ── Stage 5: Group thành FrameGroup ──
        text_similarity_cached.cache_clear()
        # [v3.23.655] Thu các câu được khôi phục ký tự (Yi-Restorer) để đánh dấu "Nghi ngờ OCR".
        restored_scope = collect_restored_texts()
        restored_texts = restored_scope.__enter__()
        # [v3.23.657] Phụ đề căn giữa: tâm ROI (toạ độ ô chữ tính trong ROI) để bộ khôi phục ký tự kiểm căn giữa.
        # Chỉ phụ đề NGANG: phụ đề dọc thì bề rộng hộp ≈ một chữ, ngưỡng "0,25 chữ" thành ~1 px → loại nhầm.
        centered = (
            roi is not None and roi.alignment in (TextAlignment.CENTER, TextAlignment.UNKNOWN)
            and roi.orientation == TextOrientation.HORIZONTAL
        )
        center_scope = centering_reference(roi.width / 2.0 if centered and roi is not None else None)
        center_scope.__enter__()
        try:
            # Làm nóng cache frame text + confidence một lần O(N) trước khi
            # bước vào greedy/Viterbi loop O(N²) để tránh tính lại thừa.
            prewarm_frame_cache(
                confident_frames,
                self._config.y_clustering_tolerance_ratio,
                self._config.y_clustering_tolerance_min_px,
            )
            if self._config.use_viterbi:
                grouped_frames = group_using_viterbi(confident_frames, self._config)
            else:
                grouped_frames = group_using_greedy(confident_frames, self._config)

            # ── Stage 6: Merge group liền kề ──
            # [v3.23.672] Mốc các khung OCR CÓ chữ (trước mọi bộ lọc): bằng chứng "trống thật"
            # cho quy tắc câu lặp — khung bị lọc vì mờ/lệch không tính là trống.
            textful = textful_timestamps(frame_results)
            # [v3.23.683] Khung mẫu mà điểm ảnh xác nhận trống chữ → câu lặp cách nhau 1 khung OCR trống vẫn tách được.
            confirmed_blanks = pixel_confirmed_blanks(boundary_track)
            merged_groups = merge_adjacent_duplicates(grouped_frames, self._config, textful, confirmed_blanks)

            # ── Stage 6.5: Phục hồi timing biên từ soft-drop timestamps ──
            # Các stage 2+3 loại bỏ frame có nội dung nhưng kém chất lượng
            # (confidence thấp khi text mới xuất hiện/mờ dần, hoặc lệch vị trí).
            # Điều này làm start/end của group bị lệch ≈1 step (40ms). Ta phục
            # hồi bằng cách kiểm tra soft-drop timestamps liền kề biên group.
            if soft_dropped_timestamps:
                merged_groups = extend_group_timing_from_soft_drops(
                    merged_groups,
                    soft_dropped_timestamps,
                    self._config.sample_step_sec,
                    soft_dropped_boxes,
                )

            # ── Stage 7: Echo trail filter (group-level), lặp đến hội tụ ──
            for _iteration_index in range(_ECHO_TRAIL_MAX_ITERATIONS):
                previous_group_count = len(merged_groups)
                merged_groups = filter_echo_trail_groups(merged_groups, self._config)
                if len(merged_groups) == previous_group_count:
                    break

            # ── Stage 7.5: Tách group chứa nhiều khối text ổn định ──
            # (progressive utterance bị merge nhầm, vd ``舒服吗`` → ``舒服``).
            merged_groups = split_stable_variant_groups(merged_groups)
            # [v3.23.672] Câu lặp cách nhau khoảng trống ngắn (Viterbi nối qua khung trống).
            merged_groups = split_repeat_gap_groups(
                merged_groups, float(self._config.sample_step_sec or 0.0), textful, confirmed_blanks,
            )

            # ── Stage 8-12: Group → Event và các filter cuối ──
            subtitle_events = convert_groups_to_events(merged_groups, self._config)
            subtitle_events = filter_short_duration_events(
                subtitle_events, self._config
            )
            subtitle_events = post_merge_duplicates(subtitle_events, self._config, textful, confirmed_blanks)
            subtitle_events = filter_short_text_events(subtitle_events, self._config)
            subtitle_events = filter_echo_trail_events(subtitle_events, self._config)

            # ── Stage 13b: Strip Latin watermark prefix từ CJK multi-line events ──
            subtitle_events = strip_latin_watermark_prefix_from_cjk_events(
                subtitle_events
            )

            # ── Stage 13c: Strip persistent watermark lines ──
            subtitle_events = strip_persistent_watermark_lines(subtitle_events)

            # ── Stage 13d: Tinh lọc cuối (watermark Latin + mảnh flash) ──
            from subtitles_extractor.application.services.subtitle_pipeline.event_refinement import (
                refine_final_events,
            )
            subtitle_events = refine_final_events(
                subtitle_events, float(self._config.sample_step_sec or 0.0),
            )
            # [v3.23.685] Chữ phụ đề đứng yên; hộp chữ trôi ngang đều qua khung là vật trong cảnh ("00" trên đai lưng).
            if roi is None or roi.orientation != TextOrientation.VERTICAL:
                subtitle_events, drifting = drop_drifting_events(
                    subtitle_events, sorted(frame_results, key=lambda f: f.timestamp_sec),
                    float(self._config.sample_step_sec or 0.0),
                )
                if drifting:
                    logger.info("Bỏ %d câu có hộp chữ trôi theo cảnh (không phải phụ đề).", drifting)

            # ── Stage 13d': [v3.23.654] Gộp phụ đề "hiện dần" (tuỳ chọn) ──
            if getattr(self._config, "merge_progressive", False):
                from subtitles_extractor.application.services.subtitle_pipeline.event_refinement import (
                    merge_progressive_events,
                )

                subtitle_events, absorbed = merge_progressive_events(subtitle_events)
                if absorbed:
                    logger.info("Gộp %d phụ đề hiện dần vào câu đầy đủ ngay sau.", absorbed)

            # ── Stage 13e: Mốc kết thúc = lúc chữ biến mất (độ giữ khung) ──
            # Làm SAU mọi bộ lọc theo thời lượng để không đổi câu nào bị giữ/bỏ.
            subtitle_events = extend_ends_by_display_hold(
                subtitle_events,
                frame_display_hold([frame.timestamp_sec for frame in frame_results]),
            )
            # ── Stage 13f: [v3.23.680] Mốc đúng khung từ dải ranh giới (khung không OCR) ──
            if boundary_track is not None:
                subtitle_events = refine_event_boundaries(subtitle_events, boundary_track)

            mark_restored_events(subtitle_events, restored_texts)

            # ── Stage 13g: [v3.23.682] Phim giản thể: chữ phồn lẻ tẻ do OCR đọc lệch → giản thể ──
            from subtitles_extractor.application.services.subtitle_pipeline.text_correction import (
                unify_minority_traditional,
            )

            for position, text in unify_minority_traditional([e.text for e in subtitle_events]).items():
                subtitle_events[position] = replace(subtitle_events[position], text=text)

            # ── Stage 14: Reindex ──
            return reindex_events(subtitle_events)
        finally:
            center_scope.__exit__(None, None, None)
            restored_scope.__exit__(None, None, None)
            text_similarity_cached.cache_clear()
            clear_frame_build_cache()


#: [v3.23.655] Ghi chú ``SubtitleEvent.ocr_notes``: câu có ký tự OCR làm rơi được khôi phục theo số đông khung.
OCR_NOTE_RESTORED = "restored_char"


def mark_restored_events(events: Sequence[SubtitleEvent], restored_texts: set[str]) -> int:
    """[v3.23.655] Gắn :data:`OCR_NOTE_RESTORED` cho câu chứa một câu đã được khôi phục ký tự. Trả số câu gắn."""
    keys = {re.sub(r"\s+", "", text) for text in restored_texts}
    keys = {key for key in keys if len(key) >= 2}
    if not keys:
        return 0
    marked = 0
    for event in events:
        # Khớp NGUYÊN câu (hoặc nguyên một dòng của câu nhiều dòng) — khớp chuỗi con thì câu khôi phục ngắn
        # (vd "一个") đánh dấu oan mọi câu đúng có chứa nó.
        parts = {re.sub(r"\s+", "", event.text or "")}
        parts.update(re.sub(r"\s+", "", line) for line in (event.text or "").splitlines())
        if parts & keys and OCR_NOTE_RESTORED not in event.ocr_notes:
            event.ocr_notes = (*event.ocr_notes, OCR_NOTE_RESTORED)
            marked += 1
    return marked


__all__ = ["OCR_NOTE_RESTORED", "SubtitleBuilder", "mark_restored_events", "text_similarity_cached"]
