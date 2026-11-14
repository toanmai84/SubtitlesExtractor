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

import bisect
import dataclasses
import itertools
import logging
import math
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
    drop_mixed_noise_fragments,
    extend_ends_by_display_hold,
    frame_text_scope,
    extend_group_timing_from_soft_drops,
    filter_echo_trail_events,
    filter_echo_trail_groups,
    filter_short_duration_events,
    filter_short_text_events,
    frame_display_hold,
    merge_adjacent_duplicates,
    merge_misread_substitution_events,
    sampled_frame_times,
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
from subtitles_extractor.application.services.subtitle_pipeline.read_fusion import (
    fuse_same_pixel_reads,
    measure_char_pitch,
    measure_text_height,
    measure_text_band,
    snap_inflated_boxes_to_band,
    tighten_inflated_boxes,
    unify_confusable_variants,
)
from subtitles_extractor.application.services.subtitle_pipeline.spatial_filters import (
    measure_subtitle_center,
    clean_spatial_outliers,
    filter_cross_frame_spatial_outliers,
    filter_noise_by_confidence,
    strip_static_overlays,
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


def measure_sample_step(frames: Sequence[OcrFrameResult]) -> float | None:
    """[v3.23.715] Khoảng cách trung vị giữa hai khung OCR liền nhau (giây); chưa đủ 20 khung → ``None``."""
    times = sorted(frame.timestamp_sec for frame in frames)
    gaps = sorted(b - a for a, b in itertools.pairwise(times) if b - a > 1e-6)
    if len(gaps) < 20:
        return None
    return round(gaps[len(gaps) // 2], 4)


#: [v3.23.718] Phát hiện ĐỔI BỐ CỤC phụ đề giữa phim (phim nối nhiều tập, mỗi tập đặt phụ đề một chỗ): so số đo theo
#: từng cửa sổ thời gian; tâm x hoặc mép trên / dưới dời ≥ chừng này lần cỡ chữ là bố cục khác.
LAYOUT_WINDOW_SEC: float = 120.0
LAYOUT_MIN_WINDOW_SAMPLES: int = 30
LAYOUT_SHIFT_HEIGHTS: float = 0.35
#: Mẫu (khung một hộp ≥ 3 chữ, tin cậy ≥ 0,9) "thuộc" một bố cục khi lệch số đo của bố cục đó dưới chừng này cỡ chữ.
LAYOUT_MATCH_HEIGHTS: float = 0.2


def _layout_samples(
    frames: Sequence[OcrFrameResult], use_center: bool,
) -> list[tuple[float, float, float, float]]:
    """``(mốc, tâm x, mép trên, mép dưới)`` của các khung chắc chắn là phụ đề (một hộp ≥ 3 chữ, tin cậy ≥ 0,9)."""
    samples: list[tuple[float, float, float, float]] = []
    for frame in frames:
        if len(frame.text_boxes) != 1:
            continue
        box = frame.text_boxes[0]
        if len(box.text.strip()) < 3 or float(box.confidence) < 0.9 or not box.polygon:
            continue
        xs = [float(point[0]) for point in box.polygon]
        ys = [float(point[1]) for point in box.polygon]
        samples.append((frame.timestamp_sec, (min(xs) + max(xs)) / 2.0 if use_center else 0.0, min(ys), max(ys)))
    return samples


def find_layout_cuts(frames: Sequence[OcrFrameResult], roi: Roi | None) -> list[float]:
    """[v3.23.718] Mốc cắt phim thành các đoạn có bố cục phụ đề khác nhau (rỗng = một bố cục).

    Tâm phụ đề, dải phụ đề, cỡ chữ đều được đo MỘT lần cho cả phim; phim nối nhiều tập mà nửa sau đặt phụ đề cao hơn
    25 px (hoặc lệch ngang 40 px) thì các luật dựa vào số đo đó sai ở nửa sau — thử sức trên OCR thật mất câu một chữ
    "你", "那". Mốc cắt đặt giữa khoảng trống chữ dài nhất nằm giữa câu cuối của bố cục cũ và câu đầu của bố cục mới,
    nên không cắt ngang câu nào.
    """
    if roi is None or roi.orientation != TextOrientation.HORIZONTAL:
        return []
    use_center = roi.alignment in (TextAlignment.CENTER, TextAlignment.UNKNOWN)
    samples = _layout_samples(frames, use_center)
    if len(samples) < 2 * LAYOUT_MIN_WINDOW_SAMPLES:
        return []
    heights = sorted(bottom - top for _ts, _cx, top, bottom in samples)
    height = heights[len(heights) // 2]
    if height <= 0:
        return []

    def stats(chunk: Sequence[tuple[float, float, float, float]]) -> tuple[float, float, float]:
        def median(values: list[float]) -> float:
            values.sort()
            return values[len(values) // 2]

        return median([s[1] for s in chunk]), median([s[2] for s in chunk]), median([s[3] for s in chunk])

    def distance(sample_or_stats: Sequence[float], layout: tuple[float, float, float]) -> float:
        cx, top, bottom = sample_or_stats[-3:]
        return max(abs(cx - layout[0]), abs(top - layout[1]), abs(bottom - layout[2]))

    windows: list[list[tuple[float, float, float, float]]] = []
    start = samples[0][0]
    for sample in samples:
        if not windows or sample[0] - start >= LAYOUT_WINDOW_SEC:
            windows.append([])
            start = sample[0]
        windows[-1].append(sample)
    windows = [window for window in windows if len(window) >= LAYOUT_MIN_WINDOW_SAMPLES]
    if len(windows) < 2:
        return []
    textful = sorted(frame.timestamp_sec for frame in frames if frame.text_boxes)
    cuts: list[float] = []
    current = stats(windows[0])
    for window in windows[1:]:
        candidate = stats(window)
        if distance(candidate, current) < LAYOUT_SHIFT_HEIGHTS * height:
            continue
        # Mẫu cuối của bố cục cũ và mẫu đầu của bố cục mới, trong phạm vi hai cửa sổ.
        span_start = window[0][0] - LAYOUT_WINDOW_SEC
        nearby = [s for s in samples if span_start <= s[0] <= window[-1][0]]
        first_new = next(
            (s[0] for index, s in enumerate(nearby)
             if distance(s, candidate) < LAYOUT_MATCH_HEIGHTS * height
             and all(distance(t, candidate) < LAYOUT_MATCH_HEIGHTS * height for t in nearby[index:index + 5])),
            None,
        )
        if first_new is None:
            continue
        last_old = max(
            (s[0] for s in nearby if s[0] < first_new and distance(s, current) < LAYOUT_MATCH_HEIGHTS * height),
            default=None,
        )
        if last_old is None:
            continue
        lo = bisect.bisect_right(textful, last_old)
        hi = bisect.bisect_left(textful, first_new)
        points = [last_old, *textful[lo:hi], first_new]
        gap_start, gap_end = max(itertools.pairwise(points), key=lambda pair: pair[1] - pair[0])
        cuts.append((gap_start + gap_end) / 2.0)
        current = candidate
    return cuts


#: [v3.23.718] Khoảng cách tới khung trước vượt chừng này lần bước lấy mẫu = có khung bị mất giữa hai khung.
SAME_PIXEL_MAX_GAP_STEPS: float = 1.5


def _frame_signature(frame: OcrFrameResult) -> tuple[tuple[str, float, tuple[tuple[float, float], ...]], ...]:
    return tuple(
        (box.text, float(box.confidence), tuple((float(x), float(y)) for x, y in box.polygon))
        for box in frame.text_boxes
    )


def drop_duplicate_samples(frames: Sequence[OcrFrameResult]) -> tuple[list[OcrFrameResult], int]:
    """[v3.23.719] Sắp theo mốc và bỏ khung lặp y hệt khung đứng trước (cùng mốc ±1 µs, cùng hộp chữ).

    Thử sức ``dupts`` trên OCR thật 豪门Z外: "混蛋龌龊的家伙" có bản đọc kèm / không kèm dấu cách gần hoà phiếu; 2 % khung
    bị lặp làm câu ra "混蛋 龌龊的家伙". Khung cùng mốc nhưng KHÁC nội dung vẫn giữ (không biết bản nào đúng).

    Returns:
        ``(frames đã sắp và khử trùng, số khung đã bỏ)``.
    """
    ordered = sorted(frames, key=lambda frame: frame.timestamp_sec)
    kept: list[OcrFrameResult] = []
    group_start = 0     # vị trí trong ``kept`` của khung đầu tiên mang mốc hiện tại
    same_time_signatures: set[tuple[tuple[str, float, tuple[tuple[float, float], ...]], ...]] | None = None
    for frame in ordered:
        if not kept or abs(frame.timestamp_sec - kept[-1].timestamp_sec) > 1e-6:
            kept.append(frame)
            group_start = len(kept) - 1
            same_time_signatures = None     # chữ ký chỉ tính khi thật sự có mốc trùng (hiếm)
            continue
        if same_time_signatures is None:
            same_time_signatures = {_frame_signature(other) for other in kept[group_start:]}
        signature = _frame_signature(frame)
        if signature in same_time_signatures:
            continue
        same_time_signatures.add(signature)
        kept.append(frame)
    return kept, len(ordered) - len(kept)


def break_same_pixel_chains_at_gaps(
    frames: Sequence[OcrFrameResult], sample_step_sec: float,
) -> tuple[list[OcrFrameResult], int]:
    """[v3.23.718] Bỏ cờ "cùng điểm ảnh khung trước" ở khung đứng ngay sau một chỗ thiếu khung.

    Cờ khẳng định khung giống khung lấy mẫu NGAY TRƯỚC nó. Khi khung đó không có trong dữ liệu (bộ giải mã bỏ khung,
    lượt chạy tiếp từ checkpoint), cờ trỏ vào một hình không còn: thử sức trên OCR thật (bỏ 3 % khung) — mất khung đổi
    hình ngay sau "瑶瑶" thì các khung trống sau đó bị coi là bản chép của "瑶瑶", bước gộp lần đọc lấp chữ vào và câu
    kéo dài 0,36 s; 74 câu lệch mốc > 0,1 s. Trả ``(khung, số chuỗi bị cắt)``.
    """
    if sample_step_sec <= 0 or len(frames) < 2:
        return list(frames), 0
    limit = SAME_PIXEL_MAX_GAP_STEPS * sample_step_sec
    result = [frames[0]]
    broken = 0
    for previous, frame in itertools.pairwise(frames):
        if frame.continues_previous and frame.timestamp_sec - previous.timestamp_sec > limit:
            frame = dataclasses.replace(frame, continues_previous=False)
            broken += 1
        result.append(frame)
    return result, broken


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
        #: [v3.23.718] Bộ dựng của MỘT đoạn bố cục (không dò đổi bố cục lần nữa).
        self._layout_segment = False

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
        # [v3.23.719] Khung trùng (cùng mốc, cùng nội dung — bộ giải mã trả lặp, lượt chạy tiếp từ checkpoint ghi lại)
        # là MỘT mẫu: đếm hai lần thì lệch phiếu bầu bản đọc khi hai bản gần hoà nhau.
        frame_results, duplicate_count = drop_duplicate_samples(frame_results)
        if duplicate_count:
            logger.info("Bỏ %d khung trùng mốc và nội dung.", duplicate_count)

        # [v3.23.715] Bước lấy mẫu THẬT đo từ mốc các khung: người dùng đặt 0,03 s nhưng phim 25 fps chỉ có khung mỗi
        # 0,04 s — các luật đếm "số khung trống" (câu lặp, mảnh chớp) chia cho bước cấu hình nên đếm 2 khung trống
        # thành 3 và coi chỗ OCR hụt là câu lặp. Dựng lại bằng cấu hình mang bước thật.
        measured_step = measure_sample_step(frame_results)
        configured_step = float(self._config.sample_step_sec or 0.0)
        if measured_step is not None and configured_step > 0 and measured_step > configured_step * 1.15:
            logger.info(
                "Bước lấy mẫu thật %.3f s (cấu hình %.3f s) — dựng phụ đề theo bước thật.",
                measured_step, configured_step,
            )
            effective = replace(self._config, sample_step_sec=measured_step)
            return SubtitleBuilder(effective).build(frame_results, roi=roi, boundary_track=boundary_track)

        # [v3.23.718] Phim đổi bố cục phụ đề giữa chừng → dựng từng đoạn với số đo (tâm, dải, cỡ chữ) của chính nó.
        if not self._layout_segment:
            ordered = sorted(frame_results, key=lambda frame: frame.timestamp_sec)
            cuts = find_layout_cuts(ordered, roi)
            if cuts:
                logger.info(
                    "Bố cục phụ đề đổi giữa phim — dựng %d đoạn riêng (cắt tại %s s).",
                    len(cuts) + 1, ", ".join(f"{cut:.2f}" for cut in cuts),
                )
                events: list[SubtitleEvent] = []
                bounds = [-math.inf, *cuts, math.inf]
                for lo, hi in itertools.pairwise(bounds):
                    part = [frame for frame in ordered if lo <= frame.timestamp_sec < hi]
                    if not part:
                        continue
                    segment_builder = SubtitleBuilder(self._config)
                    segment_builder._layout_segment = True
                    events.extend(segment_builder.build(part, roi=roi, boundary_track=boundary_track))
                return reindex_events(sorted(events, key=lambda event: event.start_sec))

        # ── Stage 0: [v3.23.711] gộp các lần đọc của cùng một câu (cùng điểm ảnh chữ / cặp chữ dễ lẫn hình) ──
        frame_results = sorted(frame_results, key=lambda frame: frame.timestamp_sec)
        frame_results, broken_chains = break_same_pixel_chains_at_gaps(
            frame_results, configured_step if configured_step > 0 else (measured_step or 0.0),
        )
        if broken_chains:
            logger.info("Cắt %d chuỗi khung cùng điểm ảnh tại chỗ thiếu khung (mốc hụt).", broken_chains)
        # [v3.23.718] Lớp phủ tĩnh (logo kênh) có ở mọi khung làm tắt các luật "khung một hộp" → gỡ trước mọi bước.
        frame_results, overlay_texts = strip_static_overlays(
            frame_results, configured_step if configured_step > 0 else (measured_step or 0.0),
            float(self._config.max_duration_sec or 7.0),
        )
        if overlay_texts:
            logger.info("Gỡ lớp phủ tĩnh khỏi vùng chữ: %s.", ", ".join(repr(text) for text in overlay_texts))
        # Tâm phụ đề (đo từ phim, không đủ dữ liệu thì tâm vùng chữ) — chỉ với phụ đề NGANG CĂN GIỮA; ngoài ra ``None``:
        # luật chữ "一" dựa vào tâm nên không áp dụng, để bộ khôi phục ký tự sẵn có xử lý như trước.
        fusion_center: float | None = None
        if (
            roi is not None and roi.orientation == TextOrientation.HORIZONTAL
            and roi.alignment in (TextAlignment.CENTER, TextAlignment.UNKNOWN)
        ):
            fusion_center = measure_subtitle_center(frame_results, roi)
            if fusion_center is None:
                fusion_center = roi.width / 2.0
        char_pitch = measure_char_pitch(frame_results) if fusion_center is not None else None
        text_height = measure_text_height(frame_results) if fusion_center is not None else None
        # [v3.23.716] Đưa hộp phình về dải phụ đề TRƯỚC khi gộp lần đọc: câu "是" (2654,64 s, lượt 3.23.714) có 12 khung
        # hộp cao hết vùng chữ rồi 6 khung CÙNG điểm ảnh đọc rỗng — bước gộp chỉ lấp khung rỗng bằng lần đọc "đúng cỡ
        # phụ đề" nên bỏ qua hộp phình, câu kết thúc sớm 0,24 s.
        text_band = measure_text_band(frame_results) if fusion_center is not None else None
        frame_results, presnapped_frames = snap_inflated_boxes_to_band(
            frame_results, fusion_center, char_pitch, text_height, text_band,
        )
        frame_results, fused_frames, disputed_texts = fuse_same_pixel_reads(
            frame_results, fusion_center, char_pitch, text_height,
        )
        # [v3.23.714] Hộp phình của cùng một chữ ở khung liền nhau → hộp gọn (không cần cùng chuỗi điểm ảnh).
        frame_results, tightened_frames = tighten_inflated_boxes(frame_results)
        # [v3.23.715] Cả lượt hiện chữ đều hộp phình (không có hộp gọn để mượn) → đưa về dải phụ đề của phim.
        frame_results, snapped_frames = snap_inflated_boxes_to_band(
            frame_results, fusion_center, char_pitch, text_height, text_band,
        )
        tightened_frames += snapped_frames + presnapped_frames
        frame_results, confusable_texts = unify_confusable_variants(frame_results, fusion_center, char_pitch)
        confusable_texts |= disputed_texts
        if fused_frames or confusable_texts or tightened_frames:
            logger.info(
                "Gộp lần đọc của cùng một câu: sửa %d khung (đọc rỗng / rơi chữ 一), %d câu có hai bản đọc lẫn chữ gần "
                "giống, %d khung hộp phình nhận hộp gọn.",
                fused_frames, len(confusable_texts), tightened_frames,
            )

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
        # [v3.23.715] Chữ của khung soft-drop: khung đọc ra chữ không liên quan tới câu không được nới biên câu.
        soft_dropped_texts: dict[float, str] = {}
        spatially_cleaned_frames: list[OcrFrameResult] = []
        # [v3.23.710] Tâm phụ đề đo từ chính phim: vùng chữ vẽ lệch / toạ độ thô bản cũ bị co không còn làm lệch phép
        # kiểm căn giữa (bộ lọc từng khung và bộ khôi phục ký tự bên dưới dùng chung một tâm).
        measured_center = measure_subtitle_center(purified_frames, roi)
        if measured_center is not None and roi is not None and abs(measured_center - roi.width / 2.0) > 8.0:
            logger.info(
                "Tâm phụ đề đo được %.0f px (tâm vùng chữ %.0f px) — kiểm căn giữa theo tâm đo được.",
                measured_center, roi.width / 2.0,
            )
        for current_frame in purified_frames:
            cleaned_frame = clean_spatial_outliers(current_frame, roi, self._config, measured_center)
            if not cleaned_frame.is_empty:
                spatially_cleaned_frames.append(cleaned_frame)
            elif not current_frame.is_empty:
                # Frame có nội dung nhưng spatial cleanup loại hết boxes → ghi ts
                soft_dropped_timestamps.append(current_frame.timestamp_sec)
                soft_dropped_boxes[current_frame.timestamp_sec] = _frame_boxes(
                    current_frame
                )
                soft_dropped_texts[current_frame.timestamp_sec] = "".join(
                    box.text for box in current_frame.text_boxes
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
                soft_dropped_texts[f.timestamp_sec] = "".join(box.text for box in f.text_boxes)

        soft_dropped_timestamps.sort()

        # ── Stage 4: Cross-frame Y-outlier filter ──
        before_y_filter = confident_frames
        confident_frames = filter_cross_frame_spatial_outliers(confident_frames, roi=roi)
        if not confident_frames:
            return []
        # [v3.23.715] Khung bị bộ lọc lệch dọc loại HẾT hộp cũng là khung "có chữ nhưng bị lọc": khung đầu của "哼"
        # (1386,84 s) đọc thành "敬哼" với hộp cao hết vùng chữ → bị loại ở đây mà không được ghi nhận, câu bắt đầu trễ
        # một khung. Bước nới biên chỉ dùng khung chung ký tự với câu nên vật trong cảnh không được hưởng.
        surviving_ts = {f.timestamp_sec for f in confident_frames}
        y_dropped = [f for f in before_y_filter if f.timestamp_sec not in surviving_ts]
        if y_dropped:
            for f in y_dropped:
                soft_dropped_timestamps.append(f.timestamp_sec)
                soft_dropped_boxes[f.timestamp_sec] = _frame_boxes(f)
                soft_dropped_texts[f.timestamp_sec] = "".join(box.text for box in f.text_boxes)
            soft_dropped_timestamps.sort()
        # [v3.23.716] Kiểm căn lề LẠI cho khung vừa bị bộ lọc lệch dọc bớt hộp. Stage 2 xét cả dòng (mọi hộp chồng dọc
        # nhau): cảnh có 6 hộp chữ dọc trải khắp bề ngang (2547,0 s: "M向H图|肃疏|C气|肖|流气气|下") cho một "dòng" có
        # tâm đúng giữa → hợp lệ; sau đó bộ lọc lệch dọc bỏ bớt hộp, còn "肃疏|C气" ở mép trái (tâm x ≈ 99 so với 360)
        # mà không ai kiểm lại → câu rác "肃疏C气".
        boxes_before = {f.timestamp_sec: len(f.text_boxes) for f in before_y_filter}
        revalidated: list[OcrFrameResult] = []
        misaligned = 0
        for f in confident_frames:
            if len(f.text_boxes) == boxes_before.get(f.timestamp_sec, len(f.text_boxes)):
                revalidated.append(f)
                continue
            cleaned = clean_spatial_outliers(f, roi, self._config, measured_center)
            if not cleaned.is_empty:
                revalidated.append(cleaned)
                continue
            misaligned += 1
            soft_dropped_timestamps.append(f.timestamp_sec)
            soft_dropped_boxes[f.timestamp_sec] = _frame_boxes(f)
            soft_dropped_texts[f.timestamp_sec] = "".join(box.text for box in f.text_boxes)
        if misaligned:
            soft_dropped_timestamps.sort()
            logger.info("Kiểm căn lề lại sau bộ lọc lệch dọc: bỏ %d khung chỉ còn chữ lệch tâm.", misaligned)
            confident_frames = revalidated
            if not confident_frames:
                return []
        elif len(revalidated) == len(confident_frames):
            confident_frames = revalidated

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
        roi_center = roi.width / 2.0 if centered and roi is not None else None
        center_scope = centering_reference(
            measured_center if roi_center is not None and measured_center is not None else roi_center
        )
        center_scope.__enter__()
        # [v3.23.718] Các luật đếm khung trống chỉ đếm khung THỰC SỰ có trong dữ liệu (khung mất ≠ khung trống).
        sampled_scope = sampled_frame_times(sorted(frame.timestamp_sec for frame in frame_results))
        sampled_scope.__enter__()
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
                    soft_dropped_texts,
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
            # [v3.23.720] Chữ từng khung (đúng thứ khâu nhóm thấy) để bước gộp câu chọn chữ theo số khung thật.
            with frame_text_scope(
                [(frame.timestamp_sec, frame.get_joined_text(
                    self._config.y_clustering_tolerance_ratio, self._config.y_clustering_tolerance_min_px,
                ), min((float(box.confidence) for box in frame.text_boxes), default=0.0)) for frame in confident_frames]
            ):
                subtitle_events = post_merge_duplicates(subtitle_events, self._config, textful, confirmed_blanks)
            # [v3.23.717] Lần đọc sai MỘT chữ (tin cậy thấp rõ rệt) của câu liền kề → cùng một câu.
            subtitle_events, misread_merges = merge_misread_substitution_events(
                subtitle_events, float(self._config.sample_step_sec or 0.0),
            )
            if misread_merges:
                logger.info("Gộp %d cặp câu liền nhau chỉ khác một chữ do một lần đọc sai.", misread_merges)
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
            # [v3.23.699] Mảnh rác chữ Latin/số/ký hiệu tin cậy thấp trong phim phụ đề chữ Hán ("色6%").
            subtitle_events, noise = drop_mixed_noise_fragments(subtitle_events)
            if noise:
                logger.info("Bỏ %d câu rác OCR (mảnh chữ/số tin cậy thấp).", noise)

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
            # [v3.23.713] Câu có hai bản đọc mâu thuẫn: cờ riêng để vòng tự soát đọc lại nhiều khung và phân xử.
            if confusable_texts:
                from subtitles_extractor.application.services.subtitle_pipeline.geometry_check import (
                    OCR_NOTE_DISPUTED,
                )

                mark_restored_events(subtitle_events, confusable_texts, OCR_NOTE_DISPUTED)
            # [v3.23.708] Hộp chữ không khớp số chữ đọc được → nghi OCR đọc thiếu / thừa một chữ (để soát, OCR lại).
            if roi is None or roi.orientation != TextOrientation.VERTICAL:
                from subtitles_extractor.application.services.subtitle_pipeline.geometry_check import (
                    drop_small_non_han_events,
                    mark_geometry_suspects,
                )

                ordered_frames = sorted(frame_results, key=lambda f: f.timestamp_sec)
                # [v3.23.710] Câu không chữ Hán có chữ nhỏ hơn hẳn chữ phụ đề ("000" tin cậy 0,98) là vật trong cảnh.
                subtitle_events, small = drop_small_non_han_events(subtitle_events, ordered_frames)
                if small:
                    logger.info("Bỏ %d câu không chữ Hán có hộp chữ nhỏ hoặc lệch tâm so với phụ đề (vật trong cảnh).", small)
                suspects = mark_geometry_suspects(subtitle_events, ordered_frames)
                if suspects:
                    logger.info("Hình học hộp chữ: %d câu nghi đọc thiếu/thừa một chữ.", suspects)

            # ── Stage 13g: [v3.23.682] Phim giản thể: chữ phồn lẻ tẻ do OCR đọc lệch → giản thể ──
            from subtitles_extractor.application.services.subtitle_pipeline.text_correction import (
                unify_minority_traditional,
            )

            for position, text in unify_minority_traditional([e.text for e in subtitle_events]).items():
                subtitle_events[position] = replace(subtitle_events[position], text=text)

            # ── Stage 14: Reindex ──
            return reindex_events(subtitle_events)
        finally:
            sampled_scope.__exit__(None, None, None)
            center_scope.__exit__(None, None, None)
            restored_scope.__exit__(None, None, None)
            text_similarity_cached.cache_clear()
            clear_frame_build_cache()


#: [v3.23.655] Ghi chú ``SubtitleEvent.ocr_notes``: câu có ký tự OCR làm rơi được khôi phục theo số đông khung.
OCR_NOTE_RESTORED = "restored_char"


def mark_restored_events(
    events: Sequence[SubtitleEvent], restored_texts: set[str], note: str = OCR_NOTE_RESTORED,
) -> int:
    """[v3.23.655] Gắn ``note`` (mặc định :data:`OCR_NOTE_RESTORED`) cho câu chứa một câu trong ``restored_texts``.

    Trả số câu gắn. [v3.23.713] ``note`` khác dùng cho câu có hai bản đọc mâu thuẫn (``OCR_NOTE_DISPUTED``).
    """
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
        if parts & keys and note not in event.ocr_notes:
            event.ocr_notes = (*event.ocr_notes, note)
            marked += 1
    return marked


__all__ = ["OCR_NOTE_RESTORED", "SubtitleBuilder", "mark_restored_events", "text_similarity_cached"]
