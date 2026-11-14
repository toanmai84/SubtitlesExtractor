"""Hậu xử lý các :class:`FrameGroup` và :class:`SubtitleEvent`.

Pipeline post-processing 6 stage:

1. :func:`merge_adjacent_duplicates` — gộp 2 group liền kề có text tương đồng.
2. :func:`filter_echo_trail_groups` — loại "echo" OCR (group chớp nhoáng
   conf thấp giống neighbor conf cao).
3. :func:`convert_groups_to_events` — chuyển FrameGroup → SubtitleEvent.
4. :func:`filter_short_duration_events` — hấp thụ flicker (events ngắn).
5. :func:`filter_short_text_events` — lọc events có text rác/ngắn.
6. :func:`post_merge_duplicates` — merge lần cuối sau khi đã filter rác.
7. :func:`filter_echo_trail_events` — lớp echo filter cuối ở cấp event.
"""

from __future__ import annotations

import bisect
import contextlib
import contextvars
import dataclasses
import re
from typing import Any
from collections import Counter
from collections.abc import Iterator, Sequence
from itertools import pairwise

import rapidfuzz.fuzz as fuzz
from loguru import logger

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    SubtitleBuilderConfig,
)
from subtitles_extractor.application.services.cjk_utils import (
    adaptive_min_text_chars,
    cjk_char_count,
    contains_cjk,
    is_predominantly_cjk,
)
from subtitles_extractor.application.services.flicker_absorber import absorb_flickers
from subtitles_extractor.application.services.han_variants import fold_han_variants
from subtitles_extractor.application.services.subtitle_pipeline.box_filters import (
    is_latin_gibberish,
)
from subtitles_extractor.application.services.subtitle_pipeline.constants import (
    LATIN_VALID_SHORT_TOKENS,
    NON_WORD_CHARACTER_REGEX,
)
from subtitles_extractor.application.services.subtitle_pipeline.frame_group import (
    FrameGroup,
    normalize_cjk_punctuation,
)
from subtitles_extractor.application.services.subtitle_pipeline.voting import (
    calculate_effective_similarity,
    evaluate_multi_line_similarity,
)
from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

# ---------------------------------------------------------------------------
# Pass 1: Merge adjacent duplicate groups
# ---------------------------------------------------------------------------


#: [Stable Variant Split] Một "khối ổn định" cần tối thiểu bấy nhiêu frame...
#: [v3.23.672] 6 → 3: điều kiện thời lượng (>= 0,30s) mới là chính — 6 khung ở bước lấy mẫu
#: mặc định 0,12s là 0,72s, bỏ lọt câu ngắn thật ("师叔" 0,68s); ở bước 0,04s thì 0,30s vẫn
#: đòi >= 8 khung như trước. (Bản sao thứ hai của khối hằng số này đã bị xoá.)
_VARIANT_MIN_FRAMES: int = 3
#: ...và kéo dài tối thiểu bấy nhiêu giây.
_VARIANT_MIN_SPAN_SEC: float = 0.30
#: Run nhiễu (<= số frame này) được hấp thụ vào khối lớn đứng trước.
_VARIANT_NOISE_RUN_FRAMES: int = 2


def _normalize_for_variant(text: str) -> str:
    """Chuẩn hoá text để so khối: bỏ whitespace (OCR dao động space vô nghĩa).

    [v3.23.616] Gộp cả chữ phồn/giản: khung đọc "瑤瑤" và khung đọc "瑶瑶" là cùng một
    câu, không phải hai khối ổn định của hai câu thật.
    """
    return fold_han_variants("".join(text.split()))


def _is_cjk_char(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
        or 0xF900 <= cp <= 0xFAFF or 0x3040 <= cp <= 0x30FF
        or 0xAC00 <= cp <= 0xD7AF
    )


#: [v3.23.682] Hai khối cùng đứng yên ≥ ngưỡng này (giây) mà chữ khác nhau là hai câu thật (xem ``_should_split_boundary``).
_STABLE_DISTINCT_SPAN_SEC: float = 0.45
#: [v3.23.682] Mất/thêm đúng 1 chữ ở ĐẦU cần đứng yên lâu hơn mới coi là 2 câu: "就可以实现" (0,5 s, OCR làm rơi "不")
#: rồi "不就可以实现" là MỘT câu; "你不要抛弃我" → "不要抛弃我" (mỗi câu 0,84 s) là hai câu.
_STABLE_LEADING_DROP_SPAN_SEC: float = 0.7
#: [v3.23.682] Nét mảnh OCR hay đánh rơi ở đầu câu — chỉ những chữ này mới coi "mất 1 chữ đầu" là lỗi đọc.
_THIN_LEADING_CHARS = frozenset("一丨丶丿乙亅二")


def _should_split_boundary(
    text_before: str, text_after: str, span_before: float = 0.0, span_after: float = 0.0,
) -> bool:
    """Quyết định ranh giới 2 khối ổn định có phải 2 CÂU THẬT khác nhau.

    Quy tắc (rút từ dữ liệu thực):
      * Chênh ở CUỐI (prefix quan hệ) + delta toàn CJK >= 1 ký tự → 2 câu
        (progressive utterance: ``舒服吗``→``舒服``; ``不过``→``不过什么``).
      * Chênh ở ĐẦU (suffix quan hệ) → chỉ là 2 câu nếu delta toàn CJK và
        >= 2 ký tự; delta 1 ký tự đầu (vd ``一``) là OCR mất nét mảnh.
      * Delta chứa Latin → biến thể watermark dính (``BOR``/``CIACA``), KHÔNG tách.
      * Không prefix/suffix: similarity < 0.5 → 2 câu khác hẳn.
      * [v3.23.682] Cả hai khối đứng yên ≥ 0,45 s (``span_*``) và phần khác nhau toàn chữ Hán → 2 câu, kể cả khi chỉ
        khác 1 chữ. Phim nối 86 tập: "我去哪"→"你去哪" (mỗi câu 0,5 s), "我不闭关了"→"好不闭关了",
        "那师兄大可以试试"(1,3 s)→"那师兄今天就试试" từng bị gộp thành một câu (mất lời thoại). Lỗi đọc OCR chỉ là khối
        1–2 khung hoặc nhảy qua lại A-B-A (đã chặn ở nơi gọi).
      * Mất 1 chữ ở đầu: hai khối cùng đứng yên ≥ 0,7 s và chữ đó không phải nét mảnh (一, 丨…) → 2 câu
        ("你不要抛弃我"→"不要抛弃我"); ngắn hơn là OCR làm rơi chữ ("就可以实现" 0,5 s → "不就可以实现").
    """
    a = _normalize_for_variant(text_before)
    b = _normalize_for_variant(text_after)
    if not a or not b or a == b:
        return False

    both_stable = min(span_before, span_after) >= _STABLE_DISTINCT_SPAN_SEC
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    if longer.startswith(shorter):
        delta = longer[len(shorter):]
        return bool(delta) and all(_is_cjk_char(c) for c in delta)
    if longer.endswith(shorter):
        delta = longer[: len(longer) - len(shorter)]
        if not all(_is_cjk_char(c) for c in delta):
            return False
        leading_stable = min(span_before, span_after) >= _STABLE_LEADING_DROP_SPAN_SEC
        return len(delta) >= 2 or (leading_stable and delta not in _THIN_LEADING_CHARS)

    if fuzz.ratio(a, b) / 100.0 < 0.5:
        return True
    if both_stable:
        from subtitles_extractor.application.services.subtitle_pipeline.text_correction import (
            normalize_traditional_to_simplified,
        )

        # Cùng chữ, khác cách viết phồn/giản ("想自毀灵台" 0,52 s rồi "想自毁灵台") là MỘT câu.
        if normalize_traditional_to_simplified(a) == normalize_traditional_to_simplified(b):
            return False
        # Phần khác nhau (bỏ đầu/đuôi chung) phải toàn chữ Hán — chữ Latin chen vào là watermark dính.
        head = 0
        while head < min(len(a), len(b)) and a[head] == b[head]:
            head += 1
        tail = 0
        while tail < min(len(a), len(b)) - head and a[-1 - tail] == b[-1 - tail]:
            tail += 1
        diff = a[head:len(a) - tail] + b[head:len(b) - tail]
        return bool(diff) and all(_is_cjk_char(c) for c in diff)
    return False




def split_stable_variant_groups(
    source_groups: list[FrameGroup],
) -> list[FrameGroup]:
    """Tách group chứa NHIỀU khối text ổn định là các CÂU THẬT khác nhau.

    Failure mode khắc phục: "progressive utterance" — 2+ câu thoại thật liên
    tiếp na ná nhau (``舒服吗``→``舒服``; ``不过``→``不过什么``) bị greedy merge
    thành 1 group. Đặc trưng phân biệt với biến thể lỗi OCR: mỗi câu thật chiếm
    một KHỐI frame thuần nhất dài (>= 6 frame, >= 0.30s); lỗi OCR chỉ là run
    1-2 frame hoặc biến thể space/Latin/mất-1-ký-tự-đầu (xem
    :func:`_should_split_boundary`).

    Args:
        source_groups: Groups sau merge/echo-trail.

    Returns:
        Danh sách groups; group đa-câu đã được tách tại ranh giới hợp lệ.
    """
    result_groups: list[FrameGroup] = []
    for group in source_groups:
        member_texts = getattr(group, "member_texts", None) or []
        if len(member_texts) < 2 * _VARIANT_MIN_FRAMES:
            result_groups.append(group)
            continue

        # Nén run-length theo text đã chuẩn hoá space.
        runs: list[dict[str, Any]] = []
        for ts, text in member_texts:
            key = _normalize_for_variant(text)
            if runs and runs[-1]["key"] == key:
                runs[-1]["end"] = ts
                runs[-1]["count"] += 1
            else:
                runs.append({"key": key, "text": text, "start": ts, "end": ts, "count": 1})

        # Hấp thụ run nhiễu (1-2 frame) vào run đứng trước.
        absorbed: list[dict[str, Any]] = []
        for run in runs:
            if absorbed and (
                run["count"] <= _VARIANT_NOISE_RUN_FRAMES
                or run["key"] == absorbed[-1]["key"]
            ):
                absorbed[-1]["end"] = run["end"]
                absorbed[-1]["count"] += run["count"]
            else:
                absorbed.append(dict(run))

        stable_runs = [
            r for r in absorbed
            if r["count"] >= _VARIANT_MIN_FRAMES
            and (r["end"] - r["start"]) >= _VARIANT_MIN_SPAN_SEC
            and len(r["key"]) >= 2
        ]
        if len(stable_runs) < 2:
            result_groups.append(group)
            continue

        # Chống OCR dao động khối lớn: không tách nếu key lặp lại (A-B-A).
        keys = [r["key"] for r in stable_runs]
        if len(set(keys)) != len(keys):
            result_groups.append(group)
            continue

        # [v3.23.717] Chuỗi bản đọc THÔ (trước khi hấp thụ run 1–2 khung) cho biết hai khối có XEN KẼ không: "癞蛤蟆"
        # 13 khung → "痴蛤蟆" 20 khung → "癞蛤蟆" 2 khung. Chữ quay lại bản cũ = cùng một hình đọc hai kiểu, không phải
        # hai câu thật khác nhau một chữ ("任务一" → "任务二" không bao giờ quay lại).
        raw_keys = [run["key"] for run in runs]

        # Gom các khối thành segment; chỉ cắt tại ranh giới "2 câu thật".
        segments: list[list[dict[str, Any]]] = [[stable_runs[0]]]
        for prev_run, next_run in zip(stable_runs, stable_runs[1:], strict=False):
            if _reads_alternate(prev_run["key"], next_run["key"], raw_keys) or _is_low_confidence_misread(
                prev_run["text"], next_run["text"], getattr(group, "variant_evidence", None) or {},
            ):
                segments[-1].append(next_run)
            elif _should_split_boundary(
                prev_run["text"], next_run["text"],
                prev_run["end"] - prev_run["start"], next_run["end"] - next_run["start"],
            ) and not _is_fade_residue(
                prev_run["text"], next_run["text"],
                prev_run["end"] - prev_run["start"], next_run["end"] - next_run["start"],
            ):
                segments.append([next_run])
            else:
                segments[-1].append(next_run)

        if len(segments) < 2:
            result_groups.append(group)
            continue

        per_frame_confidence = (
            group.accumulated_confidence / group.total_frames_count
            if group.total_frames_count > 0 else 0.0
        )
        for seg_index, segment in enumerate(segments):
            best_run = max(segment, key=lambda r: r["count"])
            # [v3.23.676] Mép NGOÀI giữ mốc của group (đã nới theo khung bị lọc mềm ở bước trước), như
            # split_repeat_gap_groups — trước đây lấy mốc khung đầu/cuối, làm mất phần nới.
            seg_start = group.start_timestamp_sec if seg_index == 0 else segment[0]["start"]
            seg_end = group.end_timestamp_sec if seg_index == len(segments) - 1 else segment[-1]["end"]
            seg_frames = sum(r["count"] for r in segment)
            result_groups.append(
                FrameGroup(
                    reconstructed_text=best_run["text"],
                    start_timestamp_sec=seg_start,
                    end_timestamp_sec=seg_end,
                    accumulated_confidence=per_frame_confidence * seg_frames,
                    total_frames_count=seg_frames,
                    primary_center_position=group.primary_center_position,
                    aggregated_bounding_box=group.aggregated_bounding_box,
                    member_texts=[
                        (ts, tx) for ts, tx in member_texts
                        if seg_start <= ts <= seg_end
                    ],
                )
            )
        logger.debug(
            "Stable-variant split: tách group {!r} thành {} câu.",
            group.reconstructed_text, len(segments),
        )
    return result_groups


#: [v3.23.672] Câu LẶP (cùng chữ) được giữ riêng khi giữa hai lần hiện có tối thiểu bấy nhiêu
#: khung lấy mẫu TRỐNG liên tiếp và khoảng trống >= ``_REPEAT_MIN_BLANK_SEC``. Đo 12 tập phim
#: mẫu (bước 0,04s): hai lần "师叔"/"卑鄙" thật cách nhau 3-8 khung trống; OCR lỡ nhận một câu
#: đang hiện chỉ làm trống ĐÚNG 1 khung ("啊" 108 khung, trống 1, rồi 3 khung). Ở bước thưa
#: (0,12s) một khung trống không phân biệt được với một khung OCR lỡ → không tách (2 khung
#: trống ở 0,12s đã là >= 0,36s, tức ngưỡng cũ).
_REPEAT_MIN_BLANK_SAMPLES: int = 2
_REPEAT_MIN_BLANK_SEC: float = 0.10
#: [v3.23.676] Ở bước lấy mẫu DÀY (≤ 0,05s) cần ≥ 3 khung trống: OCR lỡ 2 khung liền (chữ mờ/nhoè chuyển động) chỉ là
#: 80 ms và từng cắt một câu dài thành hai câu trùng (dịch, lồng tiếng hai lần). Câu lặp thật cách nhau 3-8 khung.
_REPEAT_MIN_BLANK_SAMPLES_DENSE: int = 3
_REPEAT_DENSE_STEP_SEC: float = 0.06
#: Ngưỡng cũ (khoảng trống đủ dài thì luôn là câu lặp, bất kể bước lấy mẫu).
_REPEAT_LONG_GAP_SEC: float = 0.35
#: Phần dư khi chữ hiện/tắt dần: chênh 1 ký tự ở mép, khối ngắn hiện dưới bấy nhiêu giây.
_FADE_RESIDUE_MAX_SEC: float = 0.50


#: [v3.23.718] Mốc MỌI khung có trong dữ liệu của lần dựng đang chạy (``None`` = không biết → đếm theo thời gian).
_SAMPLED_TIMES: contextvars.ContextVar[Sequence[float] | None] = contextvars.ContextVar("sampled_times", default=None)


@contextlib.contextmanager
def sampled_frame_times(times: Sequence[float] | None) -> Iterator[None]:
    """[v3.23.718] Cho các luật "đếm khung trống" biết khung nào THỰC SỰ có trong dữ liệu (mốc đã sắp xếp)."""
    token = _SAMPLED_TIMES.set(times)
    try:
        yield
    finally:
        _SAMPLED_TIMES.reset(token)


#: [v3.23.720] Chữ của từng khung (đã qua bộ lọc không gian, đúng thứ khâu nhóm nhìn thấy):
#: ``(mốc đã sắp, chữ bỏ dấu cách, độ tin thấp nhất của khung)``.
_FRAME_TEXTS: contextvars.ContextVar[tuple[Sequence[float], Sequence[str], Sequence[float], float] | None] = (
    contextvars.ContextVar("frame_texts", default=None)
)


@contextlib.contextmanager
def frame_text_scope(entries: Sequence[tuple[float, str, float]] | None) -> Iterator[None]:
    """[v3.23.720] Cho bước gộp câu đếm số khung đọc ra MỖI bản chữ trên cả dòng (``(mốc, chữ, độ tin)``)."""
    if entries is None:
        token = _FRAME_TEXTS.set(None)
    else:
        ordered = sorted(entries, key=lambda entry: entry[0])
        times = [entry[0] for entry in ordered]
        gaps = sorted(b - a for a, b in pairwise(times) if b - a > 1e-6)
        token = _FRAME_TEXTS.set((
            times, ["".join(entry[1].split()) for entry in ordered], [float(entry[2]) for entry in ordered],
            gaps[len(gaps) // 2] if gaps else 0.0,     # bước lấy mẫu (tính một lần cho cả phim)
        ))
    try:
        yield
    finally:
        _FRAME_TEXTS.reset(token)


#: Khi mở rộng khoảng một dòng: cho lọt tối đa chừng này khung đọc ra chữ khác liên tiếp; hở mốc quá chừng này bước thì dừng.
_LINE_SPAN_MAX_FOREIGN = 2
_LINE_SPAN_MAX_GAP_STEPS = 2.5


def line_reads(variants: Sequence[str], start_sec: float, end_sec: float) -> dict[str, tuple[int, float]] | None:
    """[v3.23.720] Số khung và độ tin trung bình của từng bản trong ``variants`` trên CẢ DÒNG chứa ``[start, end]``.

    Khoảng được mở rộng ra hai phía chừng nào các khung liền kề còn đọc ra một trong các bản (cho lọt tối đa
    ``_LINE_SPAN_MAX_FOREIGN`` khung đọc chữ khác, dừng ở chỗ hở mốc): bước gộp câu chạy dần từng cặp, khoảng tạm thời
    của câu gộp chưa chứa các khung phía sau → tỉ lệ méo ("你" / "你人" 3 / 3 ở bước đầu, cả dòng 8 / 3).
    Không có dữ liệu → ``None``.
    """
    data = _FRAME_TEXTS.get()
    if data is None:
        return None
    times, texts, confs, step = data
    if not times:
        return None
    wanted = {"".join(text.split()) for text in variants}
    max_gap = step * _LINE_SPAN_MAX_GAP_STEPS if step > 0 else float("inf")
    lo = bisect.bisect_left(times, start_sec - 1e-6)
    hi = bisect.bisect_right(times, end_sec + 1e-6)

    def extend(index: int, direction: int, limit: int) -> int:
        foreign, last_inside = 0, index
        position = index + direction
        while 0 <= position < limit and abs(times[position] - times[position - direction]) <= max_gap:
            if texts[position] in wanted:
                foreign, last_inside = 0, position
            else:
                foreign += 1
                if foreign > _LINE_SPAN_MAX_FOREIGN or not texts[position]:
                    break
            position += direction
        return last_inside

    if lo < hi:
        lo = extend(lo, -1, len(times))
        hi = extend(hi - 1, 1, len(times)) + 1
    tallies: dict[str, list[float]] = {text: [] for text in wanted}
    for index in range(lo, hi):
        if texts[index] in tallies:
            tallies[texts[index]].append(confs[index])
    return {text: (len(values), sum(values) / len(values) if values else 0.0) for text, values in tallies.items()}


def observed_frames_between(after_sec: float, before_sec: float) -> int | None:
    """[v3.23.718] Số khung có trong dữ liệu nằm GIỮA hai mốc; không biết dữ liệu thì ``None``.

    Khoảng hở chia bước lấy mẫu chỉ đúng khi không thiếu khung: bộ giải mã bỏ khung / lượt chạy tiếp từ checkpoint làm
    ba khung MẤT bị đếm thành ba khung TRỐNG → câu đang hiện bị coi là nói lại và tách đôi ("一天都拦不住", thử sức bỏ 3 %
    khung trên OCR thật).
    """
    times = _SAMPLED_TIMES.get()
    if times is None:
        return None
    return max(0, bisect.bisect_left(times, before_sec - 1e-6) - bisect.bisect_right(times, after_sec + 1e-6))


def is_repeat_utterance(
    gap_sec: float, sample_step_sec: float, span_before_sec: float, span_after_sec: float,
    *, pixel_blank: bool = False, observed_blank_samples: int | None = None,
) -> bool:
    """[v3.23.672] Hai lần hiện CÙNG chữ là hai câu thật (nhân vật nói lặp), không phải OCR chớp.

    Args:
        gap_sec: Mốc bắt đầu lần sau − mốc kết thúc lần trước (theo khung có chữ).
        sample_step_sec: Bước lấy mẫu khung (0 = không rõ → chỉ dùng ngưỡng 0,35s cũ).
        span_before_sec / span_after_sec: Thời lượng mỗi lần hiện.
        pixel_blank: [v3.23.683] Dải ranh giới xác nhận màn hình trống chữ trong khoảng trống
            (:func:`confirmed_blank_between`) → MỘT khung OCR trống là đủ, vì OCR lỡ không làm mất điểm chữ sáng.
            Thời lượng mỗi lần hiện tính cả khung cuối (+1 bước): "散了吧" 3 khung = 0,36 s trên màn hình.
        observed_blank_samples: [v3.23.718] Số khung trống THỰC SỰ có trong dữ liệu giữa hai lần hiện
            (:func:`observed_frames_between`); khung bị mất không được tính là trống. ``None`` = không biết.

    Người gọi phải bảo đảm khoảng trống là trống THẬT (khung OCR không có chữ nào — xem
    :func:`blank_between`), không phải khung bị bộ lọc độ tin cậy/vị trí loại.
    """
    if observed_blank_samples is not None and observed_blank_samples <= 0:
        return False                                  # không thấy khung trống nào: không có bằng chứng nói lại
    if pixel_blank and sample_step_sec > 0:
        shown = min(span_before_sec, span_after_sec) + sample_step_sec
        return (
            shown >= _VARIANT_MIN_SPAN_SEC and gap_sec >= _REPEAT_MIN_BLANK_SEC
            and round(gap_sec / sample_step_sec) - 1 >= 1
        ) or is_repeat_utterance(
            gap_sec, sample_step_sec, span_before_sec, span_after_sec,
            observed_blank_samples=observed_blank_samples,
        )
    if span_before_sec < _VARIANT_MIN_SPAN_SEC or span_after_sec < _VARIANT_MIN_SPAN_SEC:
        return False
    if gap_sec >= _REPEAT_LONG_GAP_SEC:
        return True
    if sample_step_sec <= 0:
        return False
    blank_samples = round(gap_sec / sample_step_sec) - 1
    if observed_blank_samples is not None:
        # Đếm thẳng khung trống có trong dữ liệu: khung mất không tính, và không phụ thuộc làm tròn khoảng hở / bước
        # (mốc VFR rung ± 8 ms từng đổi 4 khung trống thành 3 → gộp nhầm câu lặp "我走着走着").
        blank_samples = observed_blank_samples
    min_blank = (
        _REPEAT_MIN_BLANK_SAMPLES_DENSE if sample_step_sec < _REPEAT_DENSE_STEP_SEC
        else _REPEAT_MIN_BLANK_SAMPLES
    )
    return blank_samples >= min_blank and gap_sec >= _REPEAT_MIN_BLANK_SEC


def textful_timestamps(frames: Sequence[OcrFrameResult]) -> list[float]:
    """[v3.23.672] Mốc các khung OCR đọc ra CÓ chữ (trước mọi bộ lọc), đã sắp xếp."""
    return sorted(f.timestamp_sec for f in frames if f.text_boxes)


def blank_between(textful: Sequence[float] | None, after_sec: float, before_sec: float) -> bool:
    """[v3.23.672] Không khung nào CÓ chữ nằm giữa hai mốc (khoảng trống thật).

    Khung bị loại vì độ tin cậy thấp / lệch vị trí / lệch dòng vẫn là khung có chữ — một câu dài
    có hai khung mờ không bị cắt làm đôi. ``textful`` rỗng/``None`` = không có bằng chứng.
    """
    if not textful:
        return False
    lo = bisect.bisect_right(textful, after_sec + 1e-6)
    hi = bisect.bisect_left(textful, before_sec - 1e-6)
    return lo >= hi


def confirmed_blank_between(
    confirmed: Sequence[float] | None, after_sec: float, before_sec: float,
) -> bool:
    """[v3.23.683] Có khung mẫu nằm giữa hai mốc mà điểm ảnh xác nhận trống chữ (xem
    :func:`~.boundary_refinement.pixel_confirmed_blanks`). ``confirmed`` rỗng/``None`` = không có bằng chứng."""
    if not confirmed:
        return False
    lo = bisect.bisect_right(confirmed, after_sec + 1e-6)
    hi = bisect.bisect_left(confirmed, before_sec - 1e-6)
    return lo < hi


def last_textful_at_or_before(
    textful: Sequence[float] | None, start_sec: float, limit_sec: float,
) -> float | None:
    """[v3.23.672] Mốc khung có chữ cuối cùng trong ``[start_sec, limit_sec)`` (bỏ phần đệm cuối)."""
    if not textful:
        return None
    idx = bisect.bisect_left(textful, limit_sec - 1e-6) - 1
    if idx >= 0 and textful[idx] >= start_sec - 1e-6:
        return textful[idx]
    return None


def _is_fade_residue(text_a: str, text_b: str, span_a: float, span_b: float) -> bool:
    """[v3.23.672] Mảnh chữ hiện/tắt dần: chênh đúng 1 ký tự ở mép, khối ngắn < 0,5s.

    "他就是掌门" rồi "他就是掌" 0,4s khi chữ mờ dần là MỘT câu; "舒服吗" rồi "舒服" (mỗi câu 1s)
    là hai câu thật.
    """
    a, b = _normalize_for_variant(text_a), _normalize_for_variant(text_b)
    if abs(len(a) - len(b)) != 1:
        return False
    short, long_, short_span = (a, b, span_a) if len(a) < len(b) else (b, a, span_b)
    return (
        (long_.startswith(short) or long_.endswith(short))
        and short_span < _FADE_RESIDUE_MAX_SEC
    )


def is_stable_distinct_pair(
    text_before: str, text_after: str,
    span_before_sec: float, span_after_sec: float,
    frames_before: int, frames_after: int,
) -> bool:
    """[v3.23.672] Hai câu LIỀN NHAU, mỗi câu ổn định trên màn hình, là hai câu thật — không gộp.

    Dùng chung tiêu chí với :func:`split_stable_variant_groups` (ranh giới
    :func:`_should_split_boundary`) cho các pass gộp liền kề. Ca thật (tập 5 phim mẫu): "师叔"
    (0,76s) rồi ngay "哎呦师叔" (0,88s), không có khung trống ở giữa — câu ngắn nằm trong câu dài
    nên điểm giống nhau cao, và trước đây chỉ cặp chênh ĐÚNG MỘT ký tự mới được giữ riêng.
    """
    return (
        span_before_sec >= _VARIANT_MIN_SPAN_SEC and span_after_sec >= _VARIANT_MIN_SPAN_SEC
        and frames_before >= _VARIANT_MIN_FRAMES and frames_after >= _VARIANT_MIN_FRAMES
        # Như pass tách: khối 1 chữ ("我" → "我只是") giữ hành vi cũ (tuỳ chọn gộp hiện dần).
        and len(_normalize_for_variant(text_before)) >= 2
        and len(_normalize_for_variant(text_after)) >= 2
        and not _is_fade_residue(text_before, text_after, span_before_sec, span_after_sec)
        and _should_split_boundary(text_before, text_after, span_before_sec, span_after_sec)
    )


def _majority_text(part: Sequence[tuple[float, str]]) -> str:
    counts = Counter(_normalize_for_variant(text) for _ts, text in part)
    key = counts.most_common(1)[0][0]
    return next(text for _ts, text in part if _normalize_for_variant(text) == key)


def _is_voted_variant(voted: str, majority: str) -> bool:
    """[v3.23.676] Chữ đã bầu của group là bản (đã sửa) của chính câu đa số: cùng độ dài ±1, giống ≥ 60%."""
    if not voted or not majority:
        return False
    a, b = _normalize_for_variant(voted), _normalize_for_variant(majority)
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b) and len(a) <= 2:
        # Câu ngắn ("师叔", "卑鄙" — ca câu lặp chính): cùng độ dài, khác đúng 1 chữ.
        return sum(x != y for x, y in zip(a, b, strict=True)) <= 1
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio() >= 0.6


def split_repeat_gap_groups(
    source_groups: list[FrameGroup], sample_step_sec: float,
    textful: Sequence[float] | None = None,
    confirmed_blanks: Sequence[float] | None = None,
) -> list[FrameGroup]:
    """[v3.23.672] Tách group gồm HAI lần hiện cùng một câu cách nhau một khoảng trống thật.

    Bộ gom Viterbi chỉ thấy các khung CÓ chữ nên nối liền hai lần "师叔" (tập 2 phim mẫu: 0,72s,
    trống 3 khung, 0,72s — phụ đề gốc là hai dòng). Chỉ cắt khi: khoảng trống là trống thật
    (:func:`blank_between` theo ``textful``), :func:`is_repeat_utterance` đúng, và CHỮ đa số hai
    phía giống nhau. Mỗi phần mang chữ đa số của chính nó; mép ngoài giữ mốc của group (đã nới
    theo khung bị lọc mềm).
    """
    result: list[FrameGroup] = []
    for group in source_groups:
        members = sorted(getattr(group, "member_texts", None) or [], key=lambda m: m[0])
        if len(members) < 2 * _VARIANT_MIN_FRAMES or not textful:
            result.append(group)
            continue
        # Ranh giới các đoạn liên tục (khoảng cách hai khung > 1,5 bước).
        breaks = [
            i for i in range(1, len(members))
            if members[i][0] - members[i - 1][0] > max(sample_step_sec, 1e-6) * 1.5
        ]
        if not breaks:
            result.append(group)
            continue
        cuts: list[int] = []
        seg_start = 0
        edges = [*breaks, len(members)]
        for k, i in enumerate(breaks):
            before, after = members[seg_start:i], members[i:edges[k + 1]]
            gap = members[i][0] - members[i - 1][0]
            if (
                len(before) >= _VARIANT_MIN_FRAMES and len(after) >= _VARIANT_MIN_FRAMES
                and blank_between(textful, members[i - 1][0], members[i][0])
                and is_repeat_utterance(
                    gap, sample_step_sec,
                    before[-1][0] - before[0][0], after[-1][0] - after[0][0],
                    pixel_blank=confirmed_blank_between(confirmed_blanks, members[i - 1][0], members[i][0]),
                    observed_blank_samples=observed_frames_between(members[i - 1][0], members[i][0]),
                )
                and _normalize_for_variant(_majority_text(before))
                == _normalize_for_variant(_majority_text(after))
            ):
                cuts.append(i)
                seg_start = i
        if not cuts:
            result.append(group)
            continue
        per_frame_confidence = (
            group.accumulated_confidence / group.total_frames_count
            if group.total_frames_count > 0 else 0.0
        )
        bounds = [0, *cuts, len(members)]
        # [v3.23.676] Hai phía đã cùng chữ đa số (đã chuẩn hoá) → dùng chữ đã BẦU của cả group (ROVER/phục hồi)
        # thay cho chuỗi thô xuất hiện nhiều nhất (có thể là biến thể nhiễu, vd "掌鬥" thay "掌门").
        voted = group.reconstructed_text or ""
        for n, (lo, hi) in enumerate(zip(bounds, bounds[1:], strict=False)):
            part = members[lo:hi]
            majority = _majority_text(part)
            text = voted if _is_voted_variant(voted, majority) else majority
            result.append(FrameGroup(
                reconstructed_text=text,
                start_timestamp_sec=group.start_timestamp_sec if n == 0 else part[0][0],
                end_timestamp_sec=(
                    group.end_timestamp_sec if hi == len(members) else part[-1][0]
                ),
                accumulated_confidence=per_frame_confidence * len(part),
                total_frames_count=len(part),
                primary_center_position=group.primary_center_position,
                aggregated_bounding_box=group.aggregated_bounding_box,
                member_texts=list(part),
            ))
    return result


def _restore_across_merged(best_text: str, evidence: dict[str, list[tuple[int, float]]]) -> str:
    """[v3.23.707] Khôi phục chữ bị rơi (``一``, ``三``, kết tử…) bằng bằng chứng của MỌI group đã gộp.

    Bộ gom tách các khung đọc xen kẽ "天都拦不住"(12) / "一天都拦不住"(7) / "天都拦不住"(22) / "一天都拦不住"(4) thành
    bốn group THUẦN NHẤT; mỗi group tự bầu nên bộ khôi phục chữ (:func:`~.text_correction.apply_yi_suffix_restore`)
    không bao giờ thấy bản có ``一``, rồi pass gộp chọn theo số đông → mất ``一`` dù 11/45 khung đọc đúng. Đo trên lượt
    chạy thật phim 86 tập (208 752 khung): 4 trong 10 câu sai còn lại thuộc dạng này.

    Dùng lại đúng ngưỡng bằng chứng của bộ khôi phục chữ trong một group (không đặt luật mới).
    """
    if len(evidence) < 2:
        return best_text
    from subtitles_extractor.application.services.subtitle_pipeline.text_correction import (
        restore_dropped_yi_prefix,
    )

    candidates = [
        (text, confidence)
        for text, parts in evidence.items()
        for count, confidence in parts
        for _ in range(max(0, int(count)))
    ]
    return restore_dropped_yi_prefix(best_text, candidates)


#: [v3.23.707] Gạch ngang đứng ngay trước chữ Hán ở đầu dòng: OCR đọc nét ``一`` thành dấu gạch.
_LEADING_DASH_RE = re.compile(r"^[-—–－_]+(?=[\u4e00-\u9fff])")
_DASH_YI_MIN_FRAMES: int = 3
_DASH_YI_MIN_SHARE: float = 0.15


#: [v3.23.718] Khi biết tâm phụ đề: hộp câu (thiếu ``一`` đầu câu) phải lệch PHẢI ≥ chừng này bề rộng một chữ.
_DASH_YI_MIN_RIGHT_SHIFT_CHARS: float = 0.25


def _restore_dash_as_yi(
    best_text: str, member_texts: Sequence[tuple[float, str]],
    group_center_x: float | None = None, group_box: tuple[int, int, int, int] | None = None,
) -> str:
    """[v3.23.707] ``一`` đầu câu bị đọc thành dấu gạch ở một phần khung, phần còn lại rơi mất hẳn.

    Ca thật (phim 86 tập): "一天呐" — 8 khung đọc "-天呐", 17 khung đọc "天呐" → ra "天呐". Trên cả 208 752 khung chỉ
    có 3 câu mang gạch ngang liền trước chữ Hán và cả 3 đều là ``一`` ("一天呐", "一晚上", "一码是一码").

    Chỉ sửa khi CẢ HAI dạng cùng có mặt trong một câu: phần lớn khung không có gạch (chữ đã bầu) và ≥ 3 khung, ≥ 15 %
    có gạch. Phim dùng gạch đầu dòng hội thoại ("-你好") thì mọi khung đều đọc ra gạch nên không bị đụng tới.

    [v3.23.718] Biết tâm phụ đề thì đòi thêm bằng chứng hình học: chữ ``一`` rơi mất thật thì hộp câu lệch PHẢI nửa chữ
    (đo thật: "天呐" thiếu ``一`` lệch +20 px ở bộ 3.23.690, +26 px ở bộ 3.23.711, chữ ≈ 60 px; câu "天呐" thật lệch
    ≤ 1 px). OCR chỉ thêm dấu gạch vào một lần đọc của câu đúng tâm thì không lệch — thử sức (gạch đầu câu ở 1 % lần
    đọc) từng thêm ``一`` vào 58 câu.
    """
    if not best_text or "\n" in best_text or _LEADING_DASH_RE.match(best_text) or best_text.startswith("一"):
        return best_text
    target = _normalize_for_variant(best_text)
    if len(target) < 2 or not all(_is_cjk_char(ch) for ch in target):
        return best_text
    total = dashed = 0
    for _ts, text in member_texts:
        stripped = (text or "").strip()
        if not stripped:
            continue
        total += 1
        match = _LEADING_DASH_RE.match(stripped)
        if match and _normalize_for_variant(stripped[match.end():]) == target:
            dashed += 1
    if not (dashed >= _DASH_YI_MIN_FRAMES and dashed >= _DASH_YI_MIN_SHARE * total and dashed < total):
        return best_text
    center_x = _subtitle_center_x()
    if center_x is not None and group_center_x is not None and group_box is not None:
        pitch = (group_box[2] - group_box[0]) / (len(target) + 1)
        if group_center_x - center_x < _DASH_YI_MIN_RIGHT_SHIFT_CHARS * pitch:
            return best_text
    return "一" + best_text


def _subtitle_center_x() -> float | None:
    """Tâm phụ đề của lần dựng đang chạy (``centering_reference``); ``None`` khi không áp kiểm căn giữa."""
    from subtitles_extractor.application.services.subtitle_pipeline.text_correction import (
        current_centering_reference,
    )

    return current_centering_reference()


def merge_adjacent_duplicates(
    source_frame_groups: list[FrameGroup],
    config: SubtitleBuilderConfig,
    textful: Sequence[float] | None = None,
    confirmed_blanks: Sequence[float] | None = None,
) -> list[FrameGroup]:
    """Gộp 2 group liền kề có text tương đồng (sau greedy/Viterbi grouping).

    Lý do tồn tại: thuật toán greedy/Viterbi có thể chia 1 câu thành 2
    group nếu OCR có biến động giữa chừng (vd 1-2 frame mờ). Pass này
    quét lại và gộp.

    Args:
        source_frame_groups: Danh sách FrameGroup từ grouper.
        config: Cấu hình builder.

    Returns:
        Danh sách FrameGroup đã gộp.
    """
    if not source_frame_groups:
        return source_frame_groups

    successfully_merged_results: list[FrameGroup] = [source_frame_groups[0]]
    relaxed_similarity_threshold = config.similarity_threshold
    multi_line_similarity_threshold = config.line_similarity_threshold
    maximum_allowed_gap_duration = config.merge_gap_sec
    #: [v3.23.699] Điểm theo TỪNG CHỮ của mỗi group đã gộp (id group → {chữ: tổng confidence}).
    text_scores: dict[int, dict[str, float]] = {}
    #: [v3.23.707] Bằng chứng theo TỪNG CHỮ của mỗi group đã gộp (id group → {chữ: [(số khung, confidence TB)]}).
    text_evidence: dict[int, dict[str, list[tuple[int, float]]]] = {}
    #: [v3.23.718] Tâm x của từng BẢN ĐỌC trong group đã gộp (bằng chứng hình học cho luật khôi phục "一" đầu câu).
    text_centers: dict[int, dict[str, float]] = {}

    for current_frame_group in source_frame_groups[1:]:
        previous_frame_group = successfully_merged_results[-1]
        time_gap_duration_seconds = (
            current_frame_group.start_timestamp_sec
            - previous_frame_group.end_timestamp_sec
        )

        # [v3.23.718] So chữ sau khi quy phồn → giản: OCR đọc lệch một chữ sang phồn thể ("出來了是吧" giữa "出来了是吧")
        # từng tách câu làm hai và cắt mốc 0,44 s; bước đổi phồn → giản chạy ở cuối khâu dựng, quá muộn để gộp.
        normalized_previous_text = fold_han_variants(normalize_cjk_punctuation(
            previous_frame_group.reconstructed_text
        ))
        normalized_current_text = fold_han_variants(normalize_cjk_punctuation(
            current_frame_group.reconstructed_text
        ))

        # [v3.23.717] CÙNG chữ, bước lấy mẫu dày, khoảng hở dưới ngưỡng "nói lại" (< 3 mẫu) là OCR hụt giữa câu chứ không
        # phải câu lặp — luật câu lặp vốn đã coi vậy, nhưng khoảng gộp đặt nhỏ (0,1 s ở bước 0,04 s) chặn mất: thử sức
        # trên OCR thật (bỏ ngẫu nhiên 2 khung liền ở 1 % vị trí) tách đôi 19 câu, mốc cuối sớm tới 3,8 s.
        allowed_gap_for_pair = maximum_allowed_gap_duration
        dense_step = float(getattr(config, "sample_step_sec", 0.0) or 0.0)
        if (
            normalized_previous_text == normalized_current_text and normalized_current_text
            and 0.0 < dense_step < _REPEAT_DENSE_STEP_SEC
        ):
            allowed_gap_for_pair = max(
                allowed_gap_for_pair, dense_step * _REPEAT_MIN_BLANK_SAMPLES_DENSE + 1e-6,
            )
            # [v3.23.718] Khoảng hở do MẤT khung (dữ liệu không có khung nào ở đó) chỉ tính các khung thật sự có.
            observed_gap = observed_frames_between(
                previous_frame_group.end_timestamp_sec, current_frame_group.start_timestamp_sec,
            )
            if observed_gap is not None and observed_gap < _REPEAT_MIN_BLANK_SAMPLES_DENSE:
                allowed_gap_for_pair = max(allowed_gap_for_pair, min(time_gap_duration_seconds, _REPEAT_LONG_GAP_SEC))
        if time_gap_duration_seconds > allowed_gap_for_pair:
            successfully_merged_results.append(current_frame_group)
            continue

        # [Repeat Utterance Guard] Hai group GIỐNG HỆT text, mỗi bên ổn định
        # >= 0.30s, cách nhau >= 0.35s frame trống → câu lặp, KHÔNG gộp.
        # [v3.23.672] Khoảng trống ngắn hơn 0,35s vẫn là câu lặp khi đủ khung trống liên tiếp.
        span_prev = previous_frame_group.end_timestamp_sec - previous_frame_group.start_timestamp_sec
        span_curr = current_frame_group.end_timestamp_sec - current_frame_group.start_timestamp_sec
        if normalized_previous_text == normalized_current_text and (
            # Ngưỡng cũ: trống >= 0,35s, cả hai ổn định >= 0,30s.
            (time_gap_duration_seconds >= _REPEAT_LONG_GAP_SEC
             and span_prev >= _VARIANT_MIN_SPAN_SEC and span_curr >= _VARIANT_MIN_SPAN_SEC)
            # [v3.23.672] Khoảng trống ngắn hơn nhưng đủ khung TRỐNG THẬT liên tiếp.
            or (
                blank_between(
                    textful, previous_frame_group.end_timestamp_sec,
                    current_frame_group.start_timestamp_sec)
                and is_repeat_utterance(
                    time_gap_duration_seconds,
                    float(getattr(config, "sample_step_sec", 0.0) or 0.0),
                    span_prev, span_curr,
                    pixel_blank=confirmed_blank_between(
                        confirmed_blanks, previous_frame_group.end_timestamp_sec,
                        current_frame_group.start_timestamp_sec),
                    observed_blank_samples=observed_frames_between(
                        previous_frame_group.end_timestamp_sec, current_frame_group.start_timestamp_sec))
            )
        ):
            successfully_merged_results.append(current_frame_group)
            continue

        calculated_prev_confidence = previous_frame_group.calculate_mean_confidence()
        calculated_curr_confidence = current_frame_group.calculate_mean_confidence()

        evaluated_overall_similarity = calculate_effective_similarity(
            normalized_previous_text,
            normalized_current_text,
            calculated_prev_confidence,
            calculated_curr_confidence,
            time_gap_duration_seconds,
        )

        is_lines_substantially_similar = False
        if multi_line_similarity_threshold > 0.0:
            is_lines_substantially_similar = evaluate_multi_line_similarity(
                normalized_previous_text,
                normalized_current_text,
                calculated_prev_confidence,
                calculated_curr_confidence,
                multi_line_similarity_threshold,
                time_gap_duration_seconds,
            )

        if (
            evaluated_overall_similarity >= relaxed_similarity_threshold
            or is_lines_substantially_similar
        ):
            previous_group_score = previous_frame_group.total_score
            current_group_score = current_frame_group.total_score
            scores = text_scores.setdefault(
                id(previous_frame_group), {previous_frame_group.reconstructed_text: previous_group_score},
            )
            scores[current_frame_group.reconstructed_text] = (
                scores.get(current_frame_group.reconstructed_text, 0.0) + current_group_score
            )

            evidence = text_evidence.setdefault(
                id(previous_frame_group),
                {previous_frame_group.reconstructed_text: [
                    (previous_frame_group.total_frames_count, calculated_prev_confidence)]},
            )
            evidence.setdefault(current_frame_group.reconstructed_text, []).append(
                (current_frame_group.total_frames_count, calculated_curr_confidence)
            )
            centers = text_centers.setdefault(id(previous_frame_group), {})
            for group in (previous_frame_group, current_frame_group):
                if group.primary_center_position is not None:
                    centers.setdefault(group.reconstructed_text, float(group.primary_center_position[0]))

            previous_frame_group.end_timestamp_sec = current_frame_group.end_timestamp_sec
            # [Stable Variant Split] Nối chuỗi text per-frame để pass tách sau dùng.
            previous_frame_group.member_texts.extend(current_frame_group.member_texts)
            previous_frame_group.accumulated_confidence += (
                current_frame_group.accumulated_confidence
            )
            previous_frame_group.total_frames_count += (
                current_frame_group.total_frames_count
            )

            # [v3.23.699] Chọn chữ có TỔNG điểm cao nhất qua mọi lần gộp. Trước đây so điểm cả group đã gộp với group
            # mới: "十不存一"(7 khung) + "十不存"(8) → "十不存", rồi group đã gộp (15 khung, gồm cả 7 khung "十不存一")
            # thắng "十不存一"(14) → giữ "十不存" dù "十不存一" có 21/29 khung (lượt trích xuất thật GPU, phim 86 tập).
            best_text = max(scores, key=lambda text: (scores[text], text == previous_frame_group.reconstructed_text))
            previous_frame_group.reconstructed_text = best_text
            continue

        successfully_merged_results.append(current_frame_group)

    # [v3.23.707] Khôi phục chữ bị rơi SAU khi gộp xong: sửa ngay trong vòng lặp sẽ đổi chữ dùng để so với group kế
    # tiếp ("一天呐" so với "天呐" bị coi là hai câu khác nhau).
    for merged_group in successfully_merged_results:
        merged_evidence = text_evidence.get(id(merged_group))
        if merged_evidence is None:
            continue
        merged_group.variant_evidence = merged_evidence
        merged_group.reconstructed_text = _restore_dash_as_yi(
            _restore_across_merged(
                _prefer_confident_substitution(merged_group.reconstructed_text, merged_evidence), merged_evidence,
            ),
            merged_group.member_texts,
            text_centers.get(id(merged_group), {}).get(
                merged_group.reconstructed_text,
                merged_group.primary_center_position[0] if merged_group.primary_center_position else None,
            ),
            merged_group.aggregated_bounding_box,
        )

    return successfully_merged_results


#: [v3.23.717] Hai bản đọc cùng số chữ, khác ≤ bấy nhiêu vị trí, là hai cách đọc của MỘT hình (như bước gộp lần đọc).
_SUBSTITUTION_MAX_DIFFS: int = 2
#: Bản đọc thay thế phải có ít nhất chừng này khung mới được xét (một khung đọc lạc không đủ làm bằng chứng).
_SUBSTITUTION_MIN_FRAMES: int = 3
#: Ngữ liệu không phân xử được thì tin cậy chỉ lật lựa chọn khi chênh ≥ chừng này VÀ bản kia có ≥ nửa số khung. Ca thật
#: đọc sai mà tự tin hơn: "拉不住" 0,99 / 7 khung so với "扛不住" 0,95 / 28 khung; "清清自白" 0,96 so với "清清白白" 0,95.
_CONFIDENCE_FLIP_MARGIN: float = 0.03
_CONFIDENCE_FLIP_MIN_SHARE: float = 0.5


def _pick_reading(left: str, left_stats: tuple[int, float], right: str, right_stats: tuple[int, float]) -> str:
    """[v3.23.717] Chọn giữa hai bản đọc lẫn chữ gần giống của CÙNG một câu từ ``(số khung, tin cậy trung bình)``.

    Bản nặng hơn (khung × tin cậy) thắng; bản nhẹ hơn chỉ thắng khi có ≥ nửa số khung VÀ tin cậy cao hơn ≥ 0,03. Đo thật:
    "癞蛤蟆" 15 khung 0,983 thắng "痴蛤蟆" 20 khung 0,942 (lượt 3.23.714 khi thiếu cờ cùng điểm ảnh); "拉不住" 7 khung 0,99
    KHÔNG thắng "扛不住" 28 khung 0,95 (bộ 3.23.690); "清清自白" 0,96 không thắng "清清白白" 0,95. Không dùng ngữ liệu
    phim ở đây: thử sức cho thấy bản đọc sai hay tình cờ thành cụm thông dụng ("你要不", "让你这").
    """
    (left_frames, left_conf), (right_frames, right_conf) = left_stats, right_stats
    heavy, light = (
        ((left, left_frames, left_conf), (right, right_frames, right_conf))
        if left_frames * left_conf >= right_frames * right_conf
        else ((right, right_frames, right_conf), (left, left_frames, left_conf))
    )
    if light[1] >= _CONFIDENCE_FLIP_MIN_SHARE * heavy[1] and light[2] >= heavy[2] + _CONFIDENCE_FLIP_MARGIN:
        return light[0]
    return heavy[0]


def _is_substitution_variant(left: str, right: str) -> bool:
    """Cùng số chữ (≥ 2), khác nhau từ 1 tới :data:`_SUBSTITUTION_MAX_DIFFS` vị trí."""
    if len(left) != len(right) or len(left) < 2:
        return False
    differing = sum(1 for a, b in zip(left, right, strict=True) if a != b)
    return 0 < differing <= _SUBSTITUTION_MAX_DIFFS


def _reads_alternate(first_key: str, second_key: str, raw_keys: Sequence[str]) -> bool:
    """[v3.23.717] Hai bản đọc lẫn chữ gần giống XEN KẼ nhau (A → B → A) trong chuỗi bản đọc thô của một group."""
    if not _is_substitution_variant(first_key, second_key):
        return False
    only = [key for key in raw_keys if key in (first_key, second_key)]
    collapsed = [key for index, key in enumerate(only) if index == 0 or only[index - 1] != key]
    return len(collapsed) >= 3


def _is_low_confidence_misread(left: str, right: str, evidence: dict[str, list[tuple[int, float]]]) -> bool:
    """[v3.23.717] Hai khối chỉ khác MỘT chữ (câu ≥ 3 chữ) mà một khối tin cậy thấp rõ rệt → một lần đọc sai, không tách.

    Cùng ngưỡng với :func:`merge_misread_substitution_events`; thiếu bằng chứng tin cậy thì coi là hai câu thật.
    """
    if len(left) != len(right) or len(left) < MISREAD_MIN_CHARS:
        return False
    if sum(1 for a, b in zip(left, right, strict=True) if a != b) != 1:
        return False

    def mean_confidence(text: str) -> float | None:
        runs = evidence.get(text) or []
        frames = sum(count for count, _conf in runs)
        return sum(count * conf for count, conf in runs) / frames if frames else None

    left_conf, right_conf = mean_confidence(left), mean_confidence(right)
    if left_conf is None or right_conf is None:
        return False
    low, high = sorted((left_conf, right_conf))
    return low < MISREAD_MAX_CONFIDENCE and high - low >= MISREAD_CONFIDENCE_MARGIN


def _prefer_confident_substitution(text: str, evidence: dict[str, list[tuple[int, float]]]) -> str:
    """[v3.23.717] Trong một câu đã gộp, phân xử lại giữa chữ đang chọn và các bản đọc lẫn chữ gần giống của nó.

    Bước gộp lần đọc (``fuse_same_pixel_reads``) đã phân xử khi có cờ "cùng điểm ảnh"; thiếu cờ (dữ liệu bản cũ, bộ giải
    mã khác) thì trước giờ chỉ đếm khung × tin cậy: "这不是癞蛤蟆吗" 15 khung (0,99) thua "这不是痴蛤蟆吗" 20 khung
    (0,93). Quy tắc chọn: :func:`_pick_reading`.
    """
    def stats(reading: str) -> tuple[int, float]:
        runs = evidence.get(reading) or []
        frames = sum(count for count, _conf in runs)
        return frames, (sum(count * conf for count, conf in runs) / frames if frames else 0.0)

    best = text
    for candidate in evidence:
        if candidate == best or not _is_substitution_variant(candidate, text):
            continue
        candidate_stats = stats(candidate)
        if candidate_stats[0] < _SUBSTITUTION_MIN_FRAMES:
            continue
        best = _pick_reading(best, stats(best), candidate, candidate_stats)
    return best


# ---------------------------------------------------------------------------
# Pass 2: Echo trail filter (cấp FrameGroup)
# ---------------------------------------------------------------------------


def filter_echo_trail_groups(
    source_frame_groups: list[FrameGroup],
    config: SubtitleBuilderConfig,
) -> list[FrameGroup]:
    """Loại bỏ "echo trail" — rác OCR còn lại sau khi text thật fade out.

    Hiện tượng: sau câu '吩咐下去' (conf 0.9+), OCR engine vẫn output các
    text "biến thái" trong 1-3 khung kế tiếp: '小分休下去', '分时子去',
    … (conf 0.5-0.6). Chúng không đủ giống để merge nhưng cũng không phải
    câu thực sự — chỉ là dư âm.

    Tiêu chí echo:
        * Group có ``frame_count <= 3``.
        * ``mean_confidence < 0.65``.
        * Group LIỀN KỀ có ``mean_confidence >= 0.80``.
        * Raw fuzz ratio giữa hai group >= 0.20.

    Args:
        source_frame_groups: Danh sách FrameGroup sau merge_adjacent_duplicates.
        config: Cấu hình builder.

    Returns:
        Danh sách FrameGroup mới, đã loại bỏ echo trail.
    """
    if len(source_frame_groups) <= 1:
        return source_frame_groups

    max_neighbor_gap_seconds = config.merge_gap_sec * 1.5
    minimum_echo_ratio = 0.20
    kept_groups: list[FrameGroup] = []
    dropped_echo_count = 0

    for group_index, current_group in enumerate(source_frame_groups):
        current_mean_conf = current_group.calculate_mean_confidence()
        is_echo_candidate = (
            current_group.total_frames_count <= 3 and current_mean_conf < 0.65
        )

        if not is_echo_candidate:
            kept_groups.append(current_group)
            continue

        current_normalized_text = normalize_cjk_punctuation(
            current_group.reconstructed_text
        )
        if not current_normalized_text:
            kept_groups.append(current_group)
            continue

        absorbed_by_previous = False
        absorbed_by_next = False

        if kept_groups:
            prev_group = kept_groups[-1]
            gap_to_prev = (
                current_group.start_timestamp_sec - prev_group.end_timestamp_sec
            )
            prev_mean_conf = prev_group.calculate_mean_confidence()
            if (
                0.0 <= gap_to_prev <= max_neighbor_gap_seconds
                and prev_mean_conf >= 0.80
            ):
                prev_normalized_text = normalize_cjk_punctuation(
                    prev_group.reconstructed_text
                )
                if prev_normalized_text:
                    echo_ratio = (
                        fuzz.ratio(prev_normalized_text, current_normalized_text) / 100.0
                    )
                    if echo_ratio >= minimum_echo_ratio:
                        prev_group.end_timestamp_sec = max(
                            prev_group.end_timestamp_sec,
                            current_group.end_timestamp_sec,
                        )
                        absorbed_by_previous = True

        if not absorbed_by_previous and group_index + 1 < len(source_frame_groups):
            next_group = source_frame_groups[group_index + 1]
            gap_to_next = (
                next_group.start_timestamp_sec - current_group.end_timestamp_sec
            )
            next_mean_conf = next_group.calculate_mean_confidence()
            if (
                0.0 <= gap_to_next <= max_neighbor_gap_seconds
                and next_mean_conf >= 0.80
            ):
                next_normalized_text = normalize_cjk_punctuation(
                    next_group.reconstructed_text
                )
                if next_normalized_text:
                    echo_ratio = (
                        fuzz.ratio(next_normalized_text, current_normalized_text) / 100.0
                    )
                    if echo_ratio >= minimum_echo_ratio:
                        absorbed_by_next = True

        if absorbed_by_previous or absorbed_by_next:
            dropped_echo_count += 1
            continue

        kept_groups.append(current_group)

    if dropped_echo_count:
        logger.debug("Echo trail filter đã loại {} group dư âm OCR.", dropped_echo_count)

    return kept_groups


# ---------------------------------------------------------------------------
# Pass 3: Convert groups → events
# ---------------------------------------------------------------------------


def convert_groups_to_events(
    assembled_frame_groups: list[FrameGroup],
    config: SubtitleBuilderConfig,
) -> list[SubtitleEvent]:
    """Chuyển FrameGroup → SubtitleEvent với temporal padding hợp lý.

    Padding cuối thêm ``temporal_padding_sec`` (giúp xem có chút overlap
    với câu kế nếu cần) nhưng KHÔNG vượt qua start của event kế. Đảm bảo
    không có overlap thời gian giữa các event.

    Args:
        assembled_frame_groups: Danh sách FrameGroup đã hậu xử lý.
        config: Cấu hình builder.

    Returns:
        Danh sách SubtitleEvent có index 1-based.
    """
    final_generated_events: list[SubtitleEvent] = []
    configured_padding_seconds = config.temporal_padding_sec
    minimum_gap_buffer_seconds = 0.001

    for group_index, current_group in enumerate(assembled_frame_groups):
        original_start_time = current_group.start_timestamp_sec
        original_end_time = current_group.end_timestamp_sec

        padded_start_time = max(0.0, original_start_time)
        padded_end_time = original_end_time + configured_padding_seconds

        if final_generated_events:
            previous_event_end = final_generated_events[-1].end_sec
            if padded_start_time <= previous_event_end:
                padded_start_time = previous_event_end + minimum_gap_buffer_seconds

        if group_index + 1 < len(assembled_frame_groups):
            next_group_original_start = assembled_frame_groups[
                group_index + 1
            ].start_timestamp_sec
            if padded_end_time >= next_group_original_start:
                padded_end_time = next_group_original_start - minimum_gap_buffer_seconds

        # [v3.23.674] Phần ĐỆM không kéo dòng quá "Thời lượng tối đa" (không cắt thời lượng gốc).
        padded_end_time = min(
            padded_end_time,
            max(original_end_time, padded_start_time + config.max_duration_sec),
        )

        if padded_end_time <= padded_start_time:
            padded_end_time = padded_start_time + minimum_gap_buffer_seconds

        final_generated_events.append(
            SubtitleEvent(
                index=group_index + 1,
                text=current_group.reconstructed_text,
                interval=TimeInterval(
                    start_sec=max(0.0, padded_start_time),
                    end_sec=max(0.001, padded_end_time),
                ),
                confidence=Confidence(current_group.calculate_mean_confidence()),
                frame_count=current_group.total_frames_count,
                position=current_group.primary_center_position,
                bounding_box=current_group.aggregated_bounding_box,
            )
        )
    return final_generated_events


# ---------------------------------------------------------------------------
# Pass 4: Flicker absorption (events quá ngắn được hấp thụ vào láng giềng)
# ---------------------------------------------------------------------------


def filter_short_duration_events(
    subtitles_events_list: list[SubtitleEvent],
    config: SubtitleBuilderConfig,
) -> list[SubtitleEvent]:
    """Hấp thụ flicker events ngắn (thin wrapper quanh :func:`absorb_flickers`).

    Args:
        subtitles_events_list: Danh sách events.
        config: Cấu hình builder.

    Returns:
        Danh sách events sau khi hấp thụ flicker.
    """
    return absorb_flickers(
        events=subtitles_events_list,
        min_duration_sec=config.min_duration_sec,
        similarity_threshold=config.similarity_threshold,
        merge_gap_sec=config.merge_gap_sec,
        max_duration_sec=config.max_duration_sec,
    )


# ---------------------------------------------------------------------------
# Pass 5: Filter short text events (rác sau khi đã absorb flicker)
# ---------------------------------------------------------------------------


def _is_cjk_edge_fragment(
    current_event: SubtitleEvent,
    event_index: int,
    all_events: list[SubtitleEvent],
) -> bool:
    """Kiểm tra event có phải là edge-fragment của subtitle animated hợp lệ.

    "Edge-fragment" = 1-2 frame đầu/cuối của subtitle xuất hiện dần dần
    (animated text entrance/exit). Cần giữ lại để không mất timing chính xác.

    Điều kiện edge-fragment:
        * Prev hoặc next event trong vòng 0.4s có text chứa/bắt đầu bằng
          text hiện tại (text hiện tại là subset của subtitle hợp lệ).
        * Dùng ``fuzz.partial_ratio`` >= 80 để chịu được OCR noise nhỏ.

    Args:
        current_event: Event đang kiểm tra.
        event_index: Vị trí trong ``all_events``.
        all_events: Danh sách tất cả events (chưa filter ở pass này).

    Returns:
        ``True`` nếu là edge-fragment hợp lệ, ``False`` nếu là rác.
    """
    total = len(all_events)
    curr_text = current_event.text.replace(" ", "")
    _EDGE_GAP_SEC: float = 0.40

    for offset in (-1, 1):
        neighbor_idx = event_index + offset
        if not (0 <= neighbor_idx < total):
            continue
        neighbor = all_events[neighbor_idx]
        gap = (
            current_event.start_sec - neighbor.end_sec
            if offset == -1
            else neighbor.start_sec - current_event.end_sec
        )
        if gap > _EDGE_GAP_SEC:
            continue
        nbr_text = neighbor.text.replace(" ", "")
        # Edge-fragment: text hiện tại là tiền tố/substring của neighbor
        if not nbr_text:
            continue
        if nbr_text.startswith(curr_text) or curr_text in nbr_text:
            return True
        # partial_ratio: tìm sự hiện diện của curr_text trong nbr_text
        if fuzz.partial_ratio(curr_text, nbr_text) >= 80:
            return True

    return False


#: [v3.23.616] Câu 2 chữ Hán ĐỨNG YÊN ít nhất chừng này giây và chừng này khung...
_STABLE_TWO_CJK_MIN_DURATION_SEC: float = 0.40
_STABLE_TWO_CJK_MIN_FRAMES: int = 5
#: ...thì chỉ cần conf trung bình chừng này (câu ngắn hơn vẫn cần 0.65 như cũ).
_STABLE_TWO_CJK_MIN_CONFIDENCE: float = 0.55
_TWO_CJK_MIN_CONFIDENCE: float = 0.65


def _two_cjk_min_confidence(event: SubtitleEvent) -> float:
    """[v3.23.616] Ngưỡng conf của câu đúng 2 chữ Hán, theo độ ổn định (hàm thuần).

    Đo trên OCR thật 14/10 (phim "天骄女友退婚" tập 1, cả 3 lần trích): câu gọi tên
    "瑶瑶" hiện rõ 0.44s (12 khung, cùng một vị trí, đọc giống nhau) trên nền áo sáng,
    PaddleOCR chấm 0.56-0.73/khung, trung bình 0.62-0.63 → bị ngưỡng cứng 0.65 loại; cả
    hai lần người dùng phải tự thêm lại câu này bằng tay. Rác OCR 2 chữ là mảnh chớp
    1-4 khung; câu đứng yên đủ lâu là bằng chứng mạnh hơn điểm từng khung — cùng tinh
    thần ngưỡng theo số khung của câu 1 chữ ở trên.
    """
    if (
        event.frame_count >= _STABLE_TWO_CJK_MIN_FRAMES
        and event.duration_sec >= _STABLE_TWO_CJK_MIN_DURATION_SEC
    ):
        return _STABLE_TWO_CJK_MIN_CONFIDENCE
    return _TWO_CJK_MIN_CONFIDENCE


#: [v3.23.682] Câu một chữ Hán ≤ 2 khung vẫn giữ nếu hiện ≥ 0,2 s với tin cậy ≥ 0,95.
_SINGLE_CJK_MIN_EVIDENCE_SEC = 0.2
_SINGLE_CJK_SHORT_MIN_CONF = 0.95
#: [v3.23.682] Câu một chữ đứng yên ≥ 0,6 s chỉ cần tin cậy ≥ 0,55.
_SINGLE_CJK_STABLE_SEC = 0.6
_SINGLE_CJK_STABLE_MIN_CONF = 0.55


#: [v3.23.710] Câu MỘT chữ Hán nằm đúng chỗ phụ đề (tâm và chiều cao hộp chữ khớp phim) được giữ với độ tin cậy thấp hơn.
_SINGLE_CJK_GEOMETRY_MIN_CONF: float = 0.50
_SINGLE_CJK_GEOMETRY_MIN_FRAMES: int = 5
_GEOMETRY_REFERENCE_MIN_EVENTS: int = 30
_GEOMETRY_CENTER_TOLERANCE: float = 0.25      # theo chiều cao chữ
_GEOMETRY_HEIGHT_RANGE: tuple[float, float] = (0.80, 1.30)


def _subtitle_geometry_reference(events: Sequence[SubtitleEvent]) -> tuple[float, float] | None:
    """[v3.23.710] ``(tâm x, chiều cao)`` trung vị hộp chữ của các câu ≥ 3 chữ Hán; ``None`` nếu chưa đủ câu / không căn giữa."""
    import statistics

    centers: list[float] = []
    heights: list[float] = []
    for event in events:
        box = event.bounding_box
        text = (event.text or "").strip()
        if box is None or "\n" in text or cjk_char_count(text) < 3:
            continue
        centers.append((box[0] + box[2]) / 2.0)
        heights.append(float(box[3] - box[1]))
    if len(centers) < _GEOMETRY_REFERENCE_MIN_EVENTS:
        return None
    center, height = statistics.median(centers), statistics.median(heights)
    if height <= 0:
        return None
    spread = sorted(abs(value - center) for value in centers)[int(0.9 * (len(centers) - 1))]
    if spread > _GEOMETRY_CENTER_TOLERANCE * height:
        return None            # phụ đề không căn giữa → không có "đúng chỗ" để đối chiếu
    return center, height


def _sits_on_subtitle_line(event: SubtitleEvent, reference: tuple[float, float] | None) -> bool:
    """[v3.23.710] Hộp chữ của câu nằm đúng tâm và đúng cỡ chữ phụ đề của phim."""
    box = event.bounding_box
    if reference is None or box is None:
        return False
    center, height = reference
    low, high = _GEOMETRY_HEIGHT_RANGE
    return (
        abs((box[0] + box[2]) / 2.0 - center) <= _GEOMETRY_CENTER_TOLERANCE * height
        and low * height <= float(box[3] - box[1]) <= high * height
    )


def filter_short_text_events(
    subtitles_events_list: list[SubtitleEvent],
    config: SubtitleBuilderConfig,
) -> list[SubtitleEvent]:
    """Lọc events có text rác/ngắn theo các tiêu chí frame_count + confidence.

    Phân nhánh theo loại text:

    * **CJK-dominant**:
        - Lai Latin-CJK ngắn (1-2 CJK + >= 2 Latin), conf < 0.85 → drop.
        - CJK 1-3 chars + >= 1 Latin/digit, frame_count <= 2, conf < 0.75 → drop.
        - CJK đơn lẻ (vd '土', '士'): drop trừ khi là tiền tố animated của
          event kế tiếp (gap <= 0.5s, conf >= 0.60).
        - CJK 2 chars + frame_count <= 1 hoặc conf < 0.65 → drop.

    * **Pure Latin/digit**:
        - Latin gibberish detection (vd 'COZA', 'AAR') → drop.
        - <= 2 chars, frame_count <= 2, conf < 0.90 → drop.

    Cuối cùng kiểm tra adaptive_min_text_chars (CJK=1, Latin tuỳ config).

    Args:
        subtitles_events_list: Danh sách events.
        config: Cấu hình builder.

    Returns:
        Danh sách events đã lọc.
    """
    kept_valid_events: list[SubtitleEvent] = []
    total_events_count = len(subtitles_events_list)
    geometry_reference = _subtitle_geometry_reference(subtitles_events_list)   # [v3.23.710]

    for event_index, current_event in enumerate(subtitles_events_list):
        original_text = current_event.text.strip()
        text_cleaned_for_length_check = NON_WORD_CHARACTER_REGEX.sub(
            "", original_text
        ).replace("_", "")

        if not text_cleaned_for_length_check:
            continue

        recorded_confidence = float(current_event.confidence)
        recorded_frame_count = current_event.frame_count
        total_clean_char_count = len(text_cleaned_for_length_check)

        if contains_cjk(text_cleaned_for_length_check):
            cjk_only_char_count = cjk_char_count(text_cleaned_for_length_check)
            non_cjk_residue_count = total_clean_char_count - cjk_only_char_count

            # CJK garble cực dài conf thấp fc nhỏ — credit screen overlap nhiều
            # dòng text bị OCR đọc chồng lên thành 1 chuỗi rác dài.
            # Quan sát thực nghiệm trên test2: các event 40+ chars CJK conf
            # 0.68-0.75 fc 1-15 đều là rác.
            if (
                cjk_only_char_count >= 25
                and recorded_confidence < 0.80
                and recorded_frame_count <= 15
            ):
                continue

            is_latin_dominant_mixed = (
                cjk_only_char_count <= 2
                and non_cjk_residue_count >= cjk_only_char_count
                and non_cjk_residue_count >= 2
            )
            if is_latin_dominant_mixed and recorded_confidence < 0.85:
                continue

            is_short_cjk_with_latin_noise = (
                1 <= cjk_only_char_count <= 3
                and non_cjk_residue_count >= 1
                and recorded_frame_count <= 2
                and recorded_confidence < 0.75
            )
            if is_short_cjk_with_latin_noise:
                continue

            if cjk_only_char_count == 1 and non_cjk_residue_count == 0:
                # CJK 1 ký tự: kiểm tra animated intro của event kế tiếp.
                is_animated_intro_of_next_event = False
                if event_index + 1 < total_events_count:
                    next_event = subtitles_events_list[event_index + 1]
                    next_text = next_event.text.strip()
                    gap_to_next = next_event.start_sec - current_event.end_sec
                    if (
                        0.0 <= gap_to_next <= 0.5
                        and next_text
                        and next_text[0] == text_cleaned_for_length_check[0]
                        and len(next_text) >= 3
                        and recorded_confidence >= 0.60
                    ):
                        is_animated_intro_of_next_event = True

                if not is_animated_intro_of_next_event:
                    # [v3.23.682] Theo THỜI GIAN chứ không theo số khung: 2 khung ở bước 0,04 s (0,08 s) là chớp nhiễu,
                    # nhưng ở bước mặc định 0,12 s là 0,24 s — đúng độ dài câu một chữ thật (我/当/切…, tin cậy 1,0).
                    evidence_sec = recorded_frame_count * float(getattr(config, "sample_step_sec", 0.0) or 0.0)
                    if recorded_frame_count <= 2 and not (
                        evidence_sec >= _SINGLE_CJK_MIN_EVIDENCE_SEC
                        and recorded_confidence >= _SINGLE_CJK_SHORT_MIN_CONF
                    ):
                        continue
                    # Adaptive threshold theo độ ổn định (frame_count):
                    #   fc >= 5: cluster ổn định >= 0.4s → 0.72 đủ (bắt
                    #            interjection có conf trung bình như '埃'
                    #            xuất hiện 8 frame conf 0.74).
                    #   fc 3-4:  cluster yếu hơn → cần 0.75.
                    # Lý do chọn 0.72: thực nghiệm 11 file cho thấy 0.72 cứu
                    # được '埃' (chinese_vid2 conf 0.745) và '哦/啊/阿' trong
                    # file test, ít gây EXTRA hơn so với 0.68.
                    minimum_required_conf_for_single_cjk = (
                        0.72 if recorded_frame_count >= 5 else 0.75
                    )
                    # [v3.23.682] Chữ đứng yên ≥ 0,6 s: bộ nhận dạng dao động giữa hai cách viết (别/別) nên tin cậy trung
                    # bình thấp (0,67) dù chữ rõ. Rác hình dạng lạ đã bị lọc theo hình học hộp chữ (box_geometry).
                    if evidence_sec >= _SINGLE_CJK_STABLE_SEC:
                        minimum_required_conf_for_single_cjk = _SINGLE_CJK_STABLE_MIN_CONF
                    # [v3.23.710] Câu một chữ đứng ĐÚNG chỗ phụ đề (căn giữa, đúng cỡ chữ) trong ≥ 5 khung: rác trong cảnh
                    # không trùng cả tâm lẫn cỡ chữ. Lượt chạy thật: "哼" (11 và 15 khung, tin cậy 0,55 / 0,68) bị bỏ.
                    if (
                        recorded_frame_count >= _SINGLE_CJK_GEOMETRY_MIN_FRAMES
                        and _sits_on_subtitle_line(current_event, geometry_reference)
                    ):
                        minimum_required_conf_for_single_cjk = min(
                            minimum_required_conf_for_single_cjk, _SINGLE_CJK_GEOMETRY_MIN_CONF
                        )
                    if recorded_confidence < minimum_required_conf_for_single_cjk:
                        continue
            elif cjk_only_char_count == 2 and non_cjk_residue_count == 0:
                if recorded_frame_count <= 1:
                    continue
                if recorded_confidence < _two_cjk_min_confidence(current_event):
                    continue

            # ── v2.9+: Flash-noise guard — CJK 3-24 chars, fc≤2, conf<0.78 ──
            # Phát hiện thực nghiệm: 155/276 extra trong test3, ~21/60 trong
            # test2 là garbled OCR 1-2 frame từ credits/transition screen.
            if (
                3 <= cjk_only_char_count <= 24
                and non_cjk_residue_count == 0
                and recorded_frame_count <= 2
                and recorded_confidence < 0.78
            ):
                if not _is_cjk_edge_fragment(
                    current_event, event_index, subtitles_events_list
                ):
                    continue

            # ── v2.9+: Mixed-Latin flash guard — unconditional drop ──
            # Latin chars trong OCR result CJK với fc<=2 LUÔN là artifact.
            # Không cần edge-fragment check vì `partial_ratio` có thể
            # false-positive với CJK suffix (vd 'caanfhittnng三千就三千'
            # có partial_ratio=100% với '三千就三千').
            if (
                cjk_only_char_count >= 3
                and non_cjk_residue_count >= 1
                and recorded_frame_count <= 2
                and recorded_confidence < 0.82
            ):
                continue  # Drop vô điều kiện

            # ── v2.9+: Persistent garbled guard — CJK 4-24 chars, fc 3-15 ──
            # Credits/transition text xuất hiện nhiều frame liên tiếp nhưng
            # conf thấp (0.65-0.75). Phân biệt với dialogue bằng edge-fragment
            # check (dialogue hợp lệ luôn có neighbor text tương đồng).
            if (
                4 <= cjk_only_char_count <= 24
                and non_cjk_residue_count == 0
                and 3 <= recorded_frame_count <= 15
                and recorded_confidence < 0.76
                and current_event.duration_sec < 0.80
            ):
                if not _is_cjk_edge_fragment(
                    current_event, event_index, subtitles_events_list
                ):
                    continue
        else:
            if is_latin_gibberish(
                text_cleaned_for_length_check,
                recorded_confidence,
                recorded_frame_count,
            ):
                continue
            if recorded_frame_count <= 2 and total_clean_char_count <= 2:
                if recorded_confidence < 0.90:
                    continue
            # ── v2.9+: Pure non-CJK flash noise — fc≤2, conf<0.80, len≤10 ──
            # Bắt: 'CuC=2', 'CCA', 'BYL2', '000-00-00' v.v.
            # is_latin_gibberish không bắt được vì dựa vào vowel-ratio.
            if (
                recorded_frame_count <= 2
                and recorded_confidence < 0.80
                and total_clean_char_count <= 10
                and text_cleaned_for_length_check.upper()
                not in LATIN_VALID_SHORT_TOKENS
            ):
                continue

        configured_minimum_characters = config.min_text_chars
        if configured_minimum_characters > 0:
            effective_minimum_required = adaptive_min_text_chars(
                text_cleaned_for_length_check,
                latin_min=configured_minimum_characters,
            )
            if total_clean_char_count < effective_minimum_required:
                continue

        # Tạo event mới chỉ với text đã strip — giữ nguyên các trường khác.
        verified_safe_event = SubtitleEvent(
            index=current_event.index,
            text=original_text,
            interval=current_event.interval,
            confidence=current_event.confidence,
            frame_count=current_event.frame_count,
            position=current_event.position,
            bounding_box=current_event.bounding_box,
            uid=current_event.uid,
        )
        kept_valid_events.append(verified_safe_event)

    return kept_valid_events


# ---------------------------------------------------------------------------
# Pass 6: Post-merge duplicates (sau khi đã filter rác)
# ---------------------------------------------------------------------------


def is_superset_within_one_char(text_a: str, text_b: str) -> bool:
    """Kiểm tra 1 text có là siêu chuỗi của cái kia với chỉ 1 ký tự thêm.

    Args:
        text_a: Text candidate 1.
        text_b: Text candidate 2.

    Returns:
        ``True`` nếu cả 2 đều CJK-dominant, length chênh đúng 1, và bỏ 1
        ký tự CJK trong text dài hơn thì khớp text ngắn hơn.
    """
    if not text_a or not text_b:
        return False
    if not is_predominantly_cjk(text_a) or not is_predominantly_cjk(text_b):
        return False
    if abs(len(text_a) - len(text_b)) != 1:
        return False

    longer_text, shorter_text = (
        (text_a, text_b) if len(text_a) > len(text_b) else (text_b, text_a)
    )
    if len(shorter_text) < 2:
        return False

    for skip_position in range(len(longer_text)):
        removed_char = longer_text[skip_position]
        if not (
            "\u4e00" <= removed_char <= "\u9fff"
            or "\u3400" <= removed_char <= "\u4dbf"
        ):
            continue
        stripped_result = longer_text[:skip_position] + longer_text[skip_position + 1 :]
        if stripped_result == shorter_text:
            return True
    return False


def _is_stutter_superset(text_a: str, text_b: str) -> bool:
    """[v3.23.699] Bản dài hơn chỉ thêm MỘT chữ lặp lại chữ liền kề ("别闹" → "别别闹") — kiểu OCR đọc THỪA (lắp), không
    phải đọc sót. Khi đó không được mặc định giữ bản dài."""
    short, long_ = (text_a, text_b) if len(text_a) < len(text_b) else (text_b, text_a)
    if len(long_) != len(short) + 1:
        return False
    for i, ch in enumerate(long_):
        if long_[:i] + long_[i + 1:] == short:
            return (i > 0 and long_[i - 1] == ch) or (i + 1 < len(long_) and long_[i + 1] == ch)
    return False


def _weighted_text(first: SubtitleEvent, second: SubtitleEvent) -> str:
    """Chữ của event nặng ký hơn (confidence × số khung); hoà thì giữ ``first``."""
    first_weight = float(first.confidence) * first.frame_count
    second_weight = float(second.confidence) * second.frame_count
    return first.text if first_weight >= second_weight else second.text


def _execute_superset_merge(
    target_result_list: list[SubtitleEvent],
    base_anchor_event: SubtitleEvent,
    event_to_merge_in: SubtitleEvent,
) -> None:
    """Merge cặp superset/subset — giữ text DÀI hơn (OCR drop > add ký tự)."""
    recalculated_end_time = max(base_anchor_event.end_sec, event_to_merge_in.end_sec)
    combined_total_frames = (
        base_anchor_event.frame_count + event_to_merge_in.frame_count
    )
    weighted_confidence_score = (
        float(base_anchor_event.confidence) * base_anchor_event.frame_count
        + float(event_to_merge_in.confidence) * event_to_merge_in.frame_count
    ) / max(1, combined_total_frames)

    longer_text = (
        base_anchor_event.text
        if len(base_anchor_event.text) >= len(event_to_merge_in.text)
        else event_to_merge_in.text
    )
    if _is_stutter_superset(base_anchor_event.text, event_to_merge_in.text):
        longer_text = _weighted_text(base_anchor_event, event_to_merge_in)   # [v3.23.699]

    target_result_list[-1] = SubtitleEvent(
        index=base_anchor_event.index,
        text=longer_text,
        interval=TimeInterval(base_anchor_event.start_sec, recalculated_end_time),
        confidence=Confidence(weighted_confidence_score),
        frame_count=combined_total_frames,
        position=base_anchor_event.position,
        bounding_box=base_anchor_event.bounding_box,
        uid=base_anchor_event.uid,
    )


def _execute_merge_onto_last(
    target_result_list: list[SubtitleEvent],
    base_anchor_event: SubtitleEvent,
    event_to_merge_in: SubtitleEvent,
) -> None:
    """Merge ``event_to_merge_in`` vào ``base_anchor_event`` (đã ở cuối list).

    Chọn text có weight (conf × frame_count) cao hơn. Tuy nhiên nếu là
    cặp superset chênh 1 ký tự CJK VÀ 1 cluster nhỏ (<=4 frames), ưu tiên
    text dài hơn (do OCR thường drop nhiều hơn add).
    """
    recalculated_end_time = max(base_anchor_event.end_sec, event_to_merge_in.end_sec)
    combined_total_frames = (
        base_anchor_event.frame_count + event_to_merge_in.frame_count
    )
    weighted_confidence_score = (
        float(base_anchor_event.confidence) * base_anchor_event.frame_count
        + float(event_to_merge_in.confidence) * event_to_merge_in.frame_count
    ) / max(1, combined_total_frames)

    is_one_cluster_small_enough_for_superset_override = (
        min(base_anchor_event.frame_count, event_to_merge_in.frame_count) <= 4
    )
    if is_one_cluster_small_enough_for_superset_override and is_superset_within_one_char(
        base_anchor_event.text, event_to_merge_in.text
    ) and not _is_stutter_superset(base_anchor_event.text, event_to_merge_in.text):   # [v3.23.699]
        optimal_text_choice = (
            base_anchor_event.text
            if len(base_anchor_event.text) >= len(event_to_merge_in.text)
            else event_to_merge_in.text
        )
    else:
        base_event_weight_score = (
            float(base_anchor_event.confidence) * base_anchor_event.frame_count
        )
        merge_event_weight_score = (
            float(event_to_merge_in.confidence) * event_to_merge_in.frame_count
        )
        optimal_text_choice = (
            base_anchor_event.text
            if base_event_weight_score >= merge_event_weight_score
            else event_to_merge_in.text
        )

    # [v3.23.720] Hai bản đọc hơn kém đúng MỘT chữ Hán ở mép: quyết theo số khung OCR đọc ra mỗi bản trên CẢ DÒNG (không
    # theo trọng số dồn — câu đã gộp mang chữ một bản nhưng ``frame_count`` cộng cả khung bản kia). Bản dài thắng khi đủ
    # khung (≥ 3, ≥ 0,6 lần bản ngắn) và đọc chắc (tin cậy trung bình ≥ 0,8; OCR làm rơi chữ nhiều hơn thêm chữ):
    # test3 "瑶儿" 12 / "瑶" 7, "岁月" 4 / "岁" 4, "我来了" 16 / "我来" 4. Bản ngắn được nhiều khung hơn thì bản ngắn: "好" 6
    # / "好子" 2 (tin 0,76), "你" 8 / "你人" 3 (tin 0,51). Cặp khác chữ giữ luật cũ (đếm nguyên câu cho "你去 / 你法",
    # "阵 / 车" làm hỏng: các biến thể chia nhau phiếu).
    if base_anchor_event.text != event_to_merge_in.text and _is_edge_variant(base_anchor_event.text, event_to_merge_in.text):
        longer = max(base_anchor_event.text, event_to_merge_in.text, key=len)
        shorter = event_to_merge_in.text if longer == base_anchor_event.text else base_anchor_event.text
        reads = line_reads(
            (longer, shorter), min(base_anchor_event.start_sec, event_to_merge_in.start_sec), recalculated_end_time,
        )
        if reads is not None:
            (long_count, long_conf), (short_count, _) = reads["".join(longer.split())], reads["".join(shorter.split())]
            if (
                long_count >= _EDGE_VARIANT_MIN_READS and long_count >= _EDGE_VARIANT_MIN_SHARE * short_count
                and long_conf >= _EDGE_VARIANT_MIN_CONFIDENCE
            ):
                optimal_text_choice = longer
            elif short_count > long_count:
                optimal_text_choice = shorter

    target_result_list[-1] = SubtitleEvent(
        index=base_anchor_event.index,
        text=optimal_text_choice,
        interval=TimeInterval(base_anchor_event.start_sec, recalculated_end_time),
        confidence=Confidence(weighted_confidence_score),
        frame_count=combined_total_frames,
        position=base_anchor_event.position,
        bounding_box=base_anchor_event.bounding_box,
        uid=base_anchor_event.uid,
    )


def _is_repeat_event_pair(
    prev: SubtitleEvent, curr: SubtitleEvent, config: SubtitleBuilderConfig,
    textful: Sequence[float] | None,
    confirmed_blanks: Sequence[float] | None = None,
) -> bool:
    """[v3.23.672] Hai event CÙNG chữ cách nhau khoảng trống thật ngắn (câu lặp).

    Mốc kết thúc của event có thể đã được đệm (``temporal_padding_sec``) tới sát câu sau → lấy
    mốc khung CÓ chữ cuối cùng của event trước làm điểm kết thúc thật.
    """
    if prev.frame_count < _VARIANT_MIN_FRAMES or curr.frame_count < _VARIANT_MIN_FRAMES:
        return False
    true_end = last_textful_at_or_before(textful, prev.start_sec, curr.start_sec)
    if true_end is None:
        return False
    return blank_between(textful, true_end, curr.start_sec) and is_repeat_utterance(
        curr.start_sec - true_end,
        float(getattr(config, "sample_step_sec", 0.0) or 0.0),
        true_end - prev.start_sec, curr.duration_sec,
        pixel_blank=confirmed_blank_between(confirmed_blanks, true_end, curr.start_sec),
        observed_blank_samples=observed_frames_between(true_end, curr.start_sec),
    )


#: [v3.23.720] Bản đọc dài hơn một chữ ở mép thắng khi có ít nhất chừng này khung và chừng này lần số khung bản ngắn.
_EDGE_VARIANT_MIN_READS = 3
_EDGE_VARIANT_MIN_SHARE = 0.6
_EDGE_VARIANT_MIN_CONFIDENCE = 0.8


def _is_edge_variant(text_a: str, text_b: str) -> bool:
    """[v3.23.720] Hai bản đọc của cùng một dòng: giống hệt, hoặc bản dài hơn đúng MỘT chữ Hán ở đầu / cuối.

    Dùng cho câu ngắn bị kẹp GIỮA hai lần đọc y hệt nhau (nhánh "nhảy qua câu ngắn"): khác :func:`is_superset_within_one_char`,
    cho phép bản ngắn chỉ 1 chữ ("瑶" / "瑶儿") vì bị kẹp giữa hai lần đọc giống nhau thì không thể là câu khác.
    """
    a, b = "".join(text_a.split()), "".join(text_b.split())
    if not a or not b:
        return False
    if a == b:
        return True
    longer, shorter = (a, b) if len(a) > len(b) else (b, a)
    if len(longer) - len(shorter) != 1:
        return False
    if longer.startswith(shorter):
        extra = longer[-1]
    elif longer.endswith(shorter):
        extra = longer[0]
    else:
        return False
    return "\u4e00" <= extra <= "\u9fff" or "\u3400" <= extra <= "\u4dbf"


def _is_short_fragment(event: SubtitleEvent, sample_step_sec: float) -> bool:
    """[v3.23.719] Câu chỉ là MẢNH (hiện ngắn hơn ``_VARIANT_MIN_SPAN_SEC``; không rõ bước lấy mẫu thì ≤ 4 khung)."""
    if sample_step_sec > 0:
        return event.frame_count * sample_step_sec < _VARIANT_MIN_SPAN_SEC
    return event.frame_count <= 4


def _strip_fragment_padding(event: SubtitleEvent, textful: Sequence[float] | None) -> SubtitleEvent:
    """[v3.23.715/719] Bỏ phần đệm thời lượng tối thiểu của một mảnh sắp được gộp vào câu cùng chữ.

    Mảnh ngắn đã bị :func:`absorb_flickers` kéo tới thời lượng tối thiểu; khi gộp vào câu kề, mốc cuối phải là khung
    CÓ chữ cuối cùng của mảnh chứ không phải mốc đệm (nếu không câu ghép kết thúc trễ đúng bằng phần đệm, lấn sang
    khung trống đã quan sát). +1 ms: hàm tìm trong nửa khoảng mở — mốc cuối chưa đệm chính là khung có chữ cuối cùng.
    """
    true_end = last_textful_at_or_before(textful, event.start_sec, event.end_sec + 1e-3)
    if true_end is None or true_end >= event.end_sec:
        return event
    return dataclasses.replace(
        event, interval=TimeInterval(event.start_sec, max(true_end, event.start_sec))
    )


def post_merge_duplicates(
    finalized_events: list[SubtitleEvent],
    config: SubtitleBuilderConfig,
    textful: Sequence[float] | None = None,
    confirmed_blanks: Sequence[float] | None = None,
) -> list[SubtitleEvent]:
    """Pass merge cuối: gộp events còn duplicate sau khi đã filter rác.

    Có 2 strategies:
        1. **Direct merge**: gộp current với previous nếu similarity >= ngưỡng.
        2. **Superset merge**: gộp khi 1 trong 2 là superset chênh 1 ký tự CJK
           VÀ 1 cluster nhỏ (<=4 frames). Tránh case 2 câu phụ đề thật.
        3. **Skip-1 merge**: nếu prev event quá ngắn (<=0.25s, có thể là rác
           đã miss), thử gộp current với prev-2 (loại prev khỏi result).

    Args:
        finalized_events: Events sau filter rác.
        config: Cấu hình builder.

    Returns:
        Events đã merge.
    """
    if len(finalized_events) <= 1:
        return finalized_events

    configured_similarity_threshold = config.similarity_threshold
    configured_maximum_gap = config.merge_gap_sec
    # ── v3.6+: Identical-text extended gap ──
    # Khi 2 event có text HOÀN TOÀN GIỐNG NHAU, cho phép gap rộng hơn
    # (2× merge_gap_sec) để xử lý animated subtitle "nhấp nháy" — phụ đề
    # xuất hiện, mờ 1 giây, rồi hiện lại → nên gộp thành 1 event duy nhất.
    _IDENTICAL_TEXT_GAP_MULTIPLIER: float = 2.0
    post_merged_result: list[SubtitleEvent] = [finalized_events[0]]

    for current_candidate_event in finalized_events[1:]:
        has_been_merged = False

        immediate_previous_event = post_merged_result[-1]
        time_gap_to_previous = (
            current_candidate_event.start_sec - immediate_previous_event.end_sec
        )

        # Xác định ngưỡng gap cho lần merge này: text giống nhau hoàn toàn
        # VÀ cả 2 cluster đều nhỏ (fc <= 8) → dùng gap rộng hơn để bắt
        # animated blink pattern (phụ đề nhấp nháy ngắn giữa 2 fragment).
        # Không mở rộng cho cluster lớn (fc > 8) vì đó là 2 câu thoại riêng biệt.
        effective_max_gap = configured_maximum_gap
        normalized_prev1_text_early = normalize_cjk_punctuation(
            immediate_previous_event.text
        )
        normalized_curr_text_early = normalize_cjk_punctuation(
            current_candidate_event.text
        )
        _BLINK_FC_THRESHOLD: int = 8
        both_small_clusters = (
            immediate_previous_event.frame_count <= _BLINK_FC_THRESHOLD
            and current_candidate_event.frame_count <= _BLINK_FC_THRESHOLD
        )
        # [v3.23.715] Một bên chỉ là MẢNH — hiện ngắn hơn ``_VARIANT_MIN_SPAN_SEC`` (0,30 s) nên theo chính luật câu lặp
        # (:func:`is_repeat_utterance`) không thể là một lần nói riêng — mang đúng chữ của câu kề bên: OCR hụt vài
        # khung giữa chừng chứ không phải câu mới. Lượt chạy thật 3.23.713 với khoảng gộp 0,1 s: "切" hụt 1 khung rồi
        # đọc lại 1 khung (cách 0,12 s) → câu kết thúc sớm 0,12 s; "你" 3269 s hụt 2 khung rồi đọc lại 3 khung → sớm
        # 0,28 s; ảnh thật cho thấy chữ hiện liên tục. Không rõ bước lấy mẫu thì mảnh = ≤ 4 khung.
        fragment_step = float(getattr(config, "sample_step_sec", 0.0) or 0.0)
        smaller_frame_count = min(immediate_previous_event.frame_count, current_candidate_event.frame_count)
        one_is_fragment = (
            smaller_frame_count * fragment_step < _VARIANT_MIN_SPAN_SEC if fragment_step > 0
            else smaller_frame_count <= 4
        )
        if one_is_fragment and fragment_step > 0:
            # Câu lặp thật cách nhau ≥ 3 khung ("散了吧" nói hai lần cách 4 khung, lần đầu chỉ 0,28 s): khoảng hở đủ số
            # khung đó thì KHÔNG coi là OCR hụt. Điểm đầu khoảng hở = mốc cuối câu trước, lùi về khung có chữ cuối cùng
            # nếu mốc cuối đã bị đệm; khung xen giữa đọc ra chữ KHÁC (vật trong cảnh) vẫn tính là hở.
            fragment_true_end = last_textful_at_or_before(
                textful, immediate_previous_event.start_sec, current_candidate_event.start_sec,
            )
            hole_start = immediate_previous_event.end_sec
            if fragment_true_end is not None:
                hole_start = min(hole_start, fragment_true_end)
            hole_samples = round((current_candidate_event.start_sec - hole_start) / fragment_step) - 1
            observed_hole = observed_frames_between(hole_start, current_candidate_event.start_sec)
            if observed_hole is not None:
                hole_samples = observed_hole       # [v3.23.718] đếm khung trống có thật (khung mất không tính)
            one_is_fragment = hole_samples < _REPEAT_MIN_BLANK_SAMPLES_DENSE
        if normalized_prev1_text_early == normalized_curr_text_early and (both_small_clusters or one_is_fragment):
            effective_max_gap = configured_maximum_gap * _IDENTICAL_TEXT_GAP_MULTIPLIER

        if time_gap_to_previous <= effective_max_gap:
            # Reuse texts already computed for gap decision (identical path).
            normalized_prev1_text = normalized_prev1_text_early
            normalized_curr_text = normalized_curr_text_early

            # [Repeat Utterance Guard] Text GIỐNG HỆT + cả 2 cluster ổn định
            # (không phải mảnh blink) + phụ đề đã biến mất >= 0.35s giữa chừng
            # → nhân vật nói LẶP câu, giữ 2 event riêng (khớp phụ đề gốc).
            if normalized_prev1_text == normalized_curr_text and (
                (
                    time_gap_to_previous >= 0.35
                    and immediate_previous_event.frame_count > _BLINK_FC_THRESHOLD
                    and current_candidate_event.frame_count > _BLINK_FC_THRESHOLD
                )
                or _is_repeat_event_pair(
                    immediate_previous_event, current_candidate_event, config, textful, confirmed_blanks)
            ):
                post_merged_result.append(current_candidate_event)
                continue

            calculated_sim_score = calculate_effective_similarity(
                normalized_prev1_text,
                normalized_curr_text,
                float(immediate_previous_event.confidence),
                float(current_candidate_event.confidence),
                time_gap_to_previous,
            )

            # Phân biệt 2 câu khác nhau (cả 2 cluster lớn) vs OCR drop transient.
            are_both_clusters_substantial = (
                immediate_previous_event.frame_count >= 5
                and current_candidate_event.frame_count >= 5
            )
            is_short_superset_pattern = is_superset_within_one_char(
                normalized_prev1_text, normalized_curr_text
            ) and (
                max(len(normalized_prev1_text), len(normalized_curr_text)) <= 4
            )
            should_treat_as_distinct_pair = (
                are_both_clusters_substantial and is_short_superset_pattern
            ) or is_stable_distinct_pair(
                normalized_prev1_text, normalized_curr_text,
                immediate_previous_event.duration_sec, current_candidate_event.duration_sec,
                immediate_previous_event.frame_count, current_candidate_event.frame_count,
            )

            if (
                calculated_sim_score >= configured_similarity_threshold
                and not should_treat_as_distinct_pair
            ):
                merged_fragment = current_candidate_event
                if one_is_fragment and normalized_prev1_text == normalized_curr_text:
                    # [v3.23.715] Mảnh đã bị đệm tới thời lượng tối thiểu: ghép theo khung CÓ chữ cuối cùng của nó.
                    merged_fragment = _strip_fragment_padding(current_candidate_event, textful)
                _execute_merge_onto_last(
                    post_merged_result,
                    immediate_previous_event,
                    merged_fragment,
                )
                has_been_merged = True
            elif (
                is_superset_within_one_char(
                    normalized_prev1_text, normalized_curr_text
                )
                and min(
                    immediate_previous_event.frame_count,
                    current_candidate_event.frame_count,
                )
                <= 4
            ):
                _execute_superset_merge(
                    post_merged_result,
                    immediate_previous_event,
                    current_candidate_event,
                )
                has_been_merged = True

        if not has_been_merged and len(post_merged_result) >= 2:
            second_previous_event = post_merged_result[-2]
            time_gap_to_second_prev = (
                current_candidate_event.start_sec - second_previous_event.end_sec
            )

            if (
                immediate_previous_event.duration_sec <= 0.25
                and time_gap_to_second_prev <= configured_maximum_gap * 1.5
            ):
                normalized_prev2_text = normalize_cjk_punctuation(
                    second_previous_event.text
                )
                normalized_curr_text = normalize_cjk_punctuation(
                    current_candidate_event.text
                )
                calculated_sim2_score = calculate_effective_similarity(
                    normalized_prev2_text,
                    normalized_curr_text,
                    float(second_previous_event.confidence),
                    float(current_candidate_event.confidence),
                    time_gap_to_second_prev,
                )

                if calculated_sim2_score >= configured_similarity_threshold:
                    post_merged_result.pop()
                    skipped_fragment = current_candidate_event
                    if normalized_prev2_text == normalized_curr_text and _is_short_fragment(
                        current_candidate_event, fragment_step
                    ):
                        # [v3.23.719] Cùng lỗi đệm như nhánh gộp trực tiếp (v3.23.715) nhưng qua câu rác bị nhảy qua:
                        # "啊" khung cuối (1 khung, đệm tới 0,15 s) gộp vào câu "啊" trước, vượt khung TRỐNG đã quan sát
                        # → câu kết thúc trễ đúng phần đệm (豪门Z外 936,48 s: 937,34 thay vì 937,19).
                        skipped_fragment = _strip_fragment_padding(current_candidate_event, textful)
                    anchor = second_previous_event
                    normalized_skipped_text = normalize_cjk_punctuation(immediate_previous_event.text)
                    if _is_edge_variant(normalized_skipped_text, normalized_curr_text):
                        # [v3.23.720] Câu ngắn bị nhảy qua là MỘT BẢN ĐỌC KHÁC của chính câu này (hơn / kém một chữ),
                        # không phải rác: gộp cả nó để số khung của nó được bầu. Trước đây nó bị vứt — test3 642,59 s
                        # hai lượt "瑶儿" (2 và 6 khung) bị bỏ để nối các mảnh "瑶" → câu ra "瑶" dù "瑶儿" 12 / 19 khung.
                        _execute_merge_onto_last(post_merged_result, second_previous_event, immediate_previous_event)
                        anchor = post_merged_result[-1]
                    _execute_merge_onto_last(
                        post_merged_result,
                        anchor,
                        skipped_fragment,
                    )
                    has_been_merged = True

        if not has_been_merged:
            post_merged_result.append(current_candidate_event)

    return post_merged_result


# ---------------------------------------------------------------------------
# Pass 7: Echo trail filter ở cấp event
# ---------------------------------------------------------------------------


def filter_echo_trail_events(
    source_events: list[SubtitleEvent],
    config: SubtitleBuilderConfig,
) -> list[SubtitleEvent]:
    """Filter echo trail ở cấp event — lớp filter cuối bắt rác còn sót.

    Có 2 loại candidate:
        1. **Echo dư âm**: duration < 0.20s, conf < 0.65, similar với
           neighbor conf >= 0.80, raw fuzz >= 0.20.
        2. **Single-frame outlier**: frame_count=1, duration < 0.12s,
           conf 0.65-0.90, similar với neighbor conf >= 0.90, raw fuzz >= 0.65.

    Args:
        source_events: Events sau toàn bộ pipeline.
        config: Cấu hình builder.

    Returns:
        Events đã loại bỏ echo còn sót.
    """
    if len(source_events) <= 1:
        return source_events

    max_neighbor_gap_seconds = config.merge_gap_sec * 1.5
    minimum_echo_ratio = 0.20
    single_frame_outlier_min_fuzz = 0.65
    kept_events: list[SubtitleEvent] = []
    dropped_count = 0

    for event_index, current_event in enumerate(source_events):
        current_confidence_value = float(current_event.confidence)

        is_echo_candidate = (
            current_event.duration_sec < 0.20 and current_confidence_value < 0.65
        )
        is_single_frame_outlier_candidate = (
            current_event.frame_count == 1
            and current_event.duration_sec < 0.12
            and 0.65 <= current_confidence_value < 0.90
        )

        if not is_echo_candidate and not is_single_frame_outlier_candidate:
            kept_events.append(current_event)
            continue

        current_normalized_text = normalize_cjk_punctuation(current_event.text)
        if not current_normalized_text:
            kept_events.append(current_event)
            continue

        required_minimum_ratio = (
            single_frame_outlier_min_fuzz
            if is_single_frame_outlier_candidate and not is_echo_candidate
            else minimum_echo_ratio
        )
        required_neighbor_confidence = (
            0.90
            if is_single_frame_outlier_candidate and not is_echo_candidate
            else 0.80
        )

        absorbed_by_neighbor = False

        if kept_events:
            prev_event = kept_events[-1]
            gap_to_prev = current_event.start_sec - prev_event.end_sec
            prev_confidence_value = float(prev_event.confidence)
            if (
                0.0 <= gap_to_prev <= max_neighbor_gap_seconds
                and prev_confidence_value >= required_neighbor_confidence
            ):
                prev_normalized_text = normalize_cjk_punctuation(prev_event.text)
                if prev_normalized_text:
                    echo_ratio = (
                        fuzz.ratio(prev_normalized_text, current_normalized_text)
                        / 100.0
                    )
                    if echo_ratio >= required_minimum_ratio:
                        absorbed_by_neighbor = True

        if not absorbed_by_neighbor and event_index + 1 < len(source_events):
            next_event = source_events[event_index + 1]
            gap_to_next = next_event.start_sec - current_event.end_sec
            next_confidence_value = float(next_event.confidence)
            if (
                0.0 <= gap_to_next <= max_neighbor_gap_seconds
                and next_confidence_value >= required_neighbor_confidence
            ):
                next_normalized_text = normalize_cjk_punctuation(next_event.text)
                if next_normalized_text:
                    echo_ratio = (
                        fuzz.ratio(next_normalized_text, current_normalized_text)
                        / 100.0
                    )
                    if echo_ratio >= required_minimum_ratio:
                        absorbed_by_neighbor = True

        if absorbed_by_neighbor:
            dropped_count += 1
            continue

        kept_events.append(current_event)

    if dropped_count:
        logger.debug("Event-level echo filter đã loại {} event.", dropped_count)

    return kept_events


#: [v3.23.615] Khoảng lấy mẫu tối đa (giây) vẫn coi là lấy TỪNG KHUNG (≥ 20fps): khi đó
#: một khung giữ trên màn hình đúng một khoảng lấy mẫu.
FRAME_LEVEL_MAX_INTERVAL_SEC = 0.05


def frame_display_hold(timestamps: list[float]) -> float:
    """[v3.23.615] Thời gian một khung lấy mẫu CÒN ĐỨNG trên màn hình sau mốc của nó.

    Hàm thuần.

    Khung ở mốc ``t`` hiển thị trong ``[t, t + khoảng khung)``. Câu phụ đề biến mất lúc
    khung KẾ TIẾP (không còn chữ) hiện ra, không phải lúc khung cuối cùng có chữ bắt đầu.
    Lấy mẫu từng khung thì độ giữ đúng bằng khoảng lấy mẫu; lấy thưa hơn thì lúc chữ biến
    mất nằm đâu đó trong khoảng, lấy nửa khoảng cho sai số trung bình bằng 0.

    Args:
        timestamps: Mốc các khung đã lấy mẫu (kể cả khung trùng).

    Returns:
        Số giây cộng vào mốc kết thúc mỗi câu; 0 nếu không đủ khung để biết.
    """
    ordered = sorted(set(timestamps))
    intervals = [b - a for a, b in pairwise(ordered) if b - a > 1e-6]
    if not intervals:
        return 0.0
    intervals.sort()
    interval = intervals[len(intervals) // 2]
    return interval if interval <= FRAME_LEVEL_MAX_INTERVAL_SEC else interval / 2.0


def extend_ends_by_display_hold(
    events: list[SubtitleEvent], hold_sec: float, min_gap_sec: float = 0.001
) -> list[SubtitleEvent]:
    """[v3.23.615] Mốc kết thúc = lúc chữ BIẾN MẤT, không phải mốc khung cuối có chữ.

    Đo bằng điểm ảnh vùng phụ đề trên video thật 14/10 (25fps, lấy mẫu mọi khung): 84/94
    câu kết thúc sớm đúng 40ms (một khung) so với lúc chữ biến mất, và giữa hai câu liền
    nhau luôn có một khe 40ms không có thật. Lồng tiếng mất 40ms khung đọc mỗi câu. Sau
    khi cộng độ giữ khung: 83/94 câu kết thúc đúng mốc (sai số dưới 1ms).

    Args:
        events: Câu phụ đề đã sắp theo thời gian (bị sửa tại chỗ).
        hold_sec: Độ giữ khung (:func:`frame_display_hold`).
        min_gap_sec: Khe tối thiểu với câu kế tiếp (không bao giờ chồng câu sau).

    Returns:
        Chính danh sách ``events``.
    """
    if hold_sec <= 0:
        return events
    for position, event in enumerate(events):
        end = event.end_sec + hold_sec
        if position + 1 < len(events):
            end = min(end, events[position + 1].start_sec - min_gap_sec)
        if end > event.end_sec:
            event.interval = TimeInterval(start_sec=event.start_sec, end_sec=end)
    return events


def reindex_events(unindexed_events_list: list[SubtitleEvent]) -> list[SubtitleEvent]:
    """Cập nhật trường ``index`` của events thành thứ tự 1-based.

    Args:
        unindexed_events_list: Events đã sort theo thời gian.

    Returns:
        Cùng list đã được mutate (index 1, 2, 3, ...).
    """
    for new_index, event_object in enumerate(unindexed_events_list, start=1):
        event_object.index = new_index
    return unindexed_events_list


# ---------------------------------------------------------------------------
# Pass 8 (optional): Strip persistent watermark lines từ multi-line events
# ---------------------------------------------------------------------------


def _is_watermark_latin_line(line: str) -> bool:
    """Kiểm tra line có phải dòng watermark Latin trong event CJK.

    Watermark thường là: 1 từ viết hoa đơn (vd 'ARBOR', 'LOGO', 'HOME'),
    hoặc code/số (vd 'HD1080', 'EP01'). Không có CJK, không có dấu câu CJK.

    Args:
        line: Dòng văn bản cần kiểm tra.

    Returns:
        ``True`` nếu là watermark Latin đơn thuần.
    """
    stripped = line.strip()
    if not stripped:
        return False
    # Phải là ASCII thuần — không lẫn CJK.
    if any(ord(c) > 0x3000 for c in stripped):
        return False
    # Ít nhất 2 ký tự.
    if len(stripped) < 2:
        return False
    # Viết hoa hoàn toàn hoặc mixed alphanumeric ngắn.
    upper_clean = stripped.upper().replace("-", "").replace("_", "")
    if not upper_clean.replace(" ", "").isalnum():
        return False
    words = stripped.split()
    # Tối đa 3 từ để tránh nhầm subtitle tiếng Anh thật.
    if len(words) > 3:
        return False
    # Ít nhất 50% là chữ in hoa (loại câu tiếng Anh bình thường viết thường).
    upper_count = sum(1 for c in stripped if c.isupper())
    alpha_count = sum(1 for c in stripped if c.isalpha())
    if alpha_count == 0:
        return False
    return upper_count / alpha_count >= 0.80


def strip_latin_watermark_prefix_from_cjk_events(
    subtitle_events: list[SubtitleEvent],
) -> list[SubtitleEvent]:
    """Strip dòng watermark Latin khỏi event CJK multi-line.

    Nhận biết và loại bỏ dòng prefix/suffix Latin thuần (vd 'ARBOR',
    'HOME', 'LOGO') trong event có phần còn lại là CJK. Pattern:

    * Event có >= 2 dòng.
    * Dòng đầu hoặc cuối là Latin watermark (``_is_watermark_latin_line``).
    * Các dòng còn lại có chứa CJK.

    Không drop event — chỉ strip dòng Latin. Nếu sau strip event chỉ còn
    1 dòng CJK vẫn giữ nguyên.

    Args:
        subtitle_events: Danh sách events.

    Returns:
        Events đã strip watermark Latin prefix/suffix.
    """
    stripped_count = 0
    result: list[SubtitleEvent] = []
    import dataclasses as _dc

    for event in subtitle_events:
        lines = [ln.strip() for ln in event.text.splitlines() if ln.strip()]
        if len(lines) < 2:
            result.append(event)
            continue

        # Kiểm tra dòng đầu / dòng cuối là watermark Latin.
        head_is_watermark = _is_watermark_latin_line(lines[0])
        tail_is_watermark = _is_watermark_latin_line(lines[-1])

        if not head_is_watermark and not tail_is_watermark:
            result.append(event)
            continue

        # Chỉ strip nếu phần còn lại có CJK để tránh mất subtitle tiếng Anh thật.
        remaining_after_strip = lines[1:] if head_is_watermark else lines
        if tail_is_watermark and remaining_after_strip:
            remaining_after_strip = remaining_after_strip[:-1] if tail_is_watermark else remaining_after_strip

        # Kiểm tra lại: strip cả đầu lẫn cuối nếu cần.
        clean_lines = list(lines)
        if head_is_watermark and contains_cjk("".join(clean_lines[1:])):
            clean_lines = clean_lines[1:]
            stripped_count += 1
        if (
            clean_lines
            and tail_is_watermark
            and contains_cjk("".join(clean_lines[:-1]))
        ):
            clean_lines = clean_lines[:-1]
            stripped_count += 1

        if not clean_lines:
            # Không còn gì sau strip → drop.
            continue

        if len(clean_lines) == len(lines):
            result.append(event)
        else:
            result.append(_dc.replace(event, text="\n".join(clean_lines)))

    if stripped_count:
        logger.debug(
            "Latin-watermark-prefix stripper: {} line stripped từ multi-line events.",
            stripped_count,
        )
    return result



def strip_persistent_watermark_lines(
    subtitle_events: list[SubtitleEvent],
    min_occurrence_ratio: float = 0.08,
    min_absolute_count: int = 5,
) -> list[SubtitleEvent]:
    """Loại bỏ dòng watermark cứng nhắc xuất hiện trong nhiều event.

    Watermark logo/thương hiệu (vd 'ARBOR', 'HOME', '天之') hay xuất hiện
    như line đầu hoặc cuối của event multi-line, lặp lại liên tục qua hàng
    chục đến hàng trăm event. Hàm này:

    1. Đếm tần suất xuất hiện của từng line trong TẤT CẢ multi-line events.
    2. Nếu một line có:
       * tần suất >= ``min_occurrence_ratio`` × số lượng multi-line events
       * VÀ xuất hiện >= ``min_absolute_count`` lần tuyệt đối
    → đánh dấu là watermark line cần loại.
    3. Với mỗi multi-line event có chứa watermark line → strip, giữ lại
       các dòng còn lại.
    4. Nếu sau khi strip, event chỉ còn 1 dòng → giữ lại (không drop).
    5. Nếu event chỉ có watermark, bỏ hoàn toàn.

    Chỉ áp dụng cho **Latin-dominant** line để tránh strip nhầm CJK dialogue
    ngắn (vd '哦', '嗯'). CJK watermark thường dài hơn và bị bắt bởi các
    filter khác.

    Args:
        subtitle_events: Danh sách events sau toàn bộ pipeline chính.
        min_occurrence_ratio: Tỉ lệ tối thiểu (0.08 = 8% multi-line events).
        min_absolute_count: Số lần xuất hiện tuyệt đối tối thiểu.

    Returns:
        Danh sách events đã strip watermark lines.
    """
    if not subtitle_events:
        return subtitle_events

    # Bước 1: Thu thập tất cả multi-line events và đếm tần suất từng line.
    multiline_events_indices: list[int] = []
    line_occurrence_counter: dict[str, int] = {}

    for event_idx, event in enumerate(subtitle_events):
        raw_lines = [ln.strip() for ln in event.text.splitlines() if ln.strip()]
        if len(raw_lines) < 2:
            continue
        multiline_events_indices.append(event_idx)
        for line_text in raw_lines:
            line_occurrence_counter[line_text] = (
                line_occurrence_counter.get(line_text, 0) + 1
            )

    if not multiline_events_indices:
        return subtitle_events

    total_multiline_count = len(multiline_events_indices)
    absolute_threshold = max(min_absolute_count, 1)
    ratio_threshold = total_multiline_count * min_occurrence_ratio

    # Bước 2: Xác định watermark lines — chỉ Latin-dominant để tránh
    # false-positive với CJK dialogue ngắn.
    def _is_latin_dominant_line(line: str) -> bool:
        """Kiểm tra line có phải Latin-dominant (>50% Latin char)."""
        stripped = line.replace(" ", "")
        if not stripped:
            return False
        latin_chars = sum(1 for c in stripped if c.isascii() and c.isalpha())
        return latin_chars / len(stripped) > 0.50

    watermark_lines: frozenset[str] = frozenset(
        line
        for line, count in line_occurrence_counter.items()
        if (
            count >= absolute_threshold
            and count >= ratio_threshold
            and _is_latin_dominant_line(line)
        )
    )

    if not watermark_lines:
        return subtitle_events

    stripped_count = 0
    dropped_count = 0
    result_events: list[SubtitleEvent] = []

    for event in subtitle_events:
        raw_lines = [ln.strip() for ln in event.text.splitlines() if ln.strip()]

        if len(raw_lines) < 2:
            # [v3.23.150] Event MỘT DÒNG mà nội dung CHÍNH LÀ watermark đã xác định
            # (bằng chứng thống kê từ multi-line: >= min_absolute_count lần và >=
            # min_occurrence_ratio) -> đây là lúc logo hiện MỘT MÌNH trên màn hình
            # (không có thoại) -> drop. Trước đây nhánh này giữ nguyên mọi event
            # một dòng nên phụ đề đầu ra lẫn hàng loạt event watermark đơn độc.
            if len(raw_lines) == 1 and raw_lines[0] in watermark_lines:
                dropped_count += 1
                continue
            result_events.append(event)
            continue

        clean_lines = [ln for ln in raw_lines if ln not in watermark_lines]

        if not clean_lines:
            # Event chỉ toàn watermark → drop hoàn toàn.
            dropped_count += 1
            continue

        if len(clean_lines) == len(raw_lines):
            # Không có gì bị strip → giữ nguyên.
            result_events.append(event)
            continue

        # Tạo event mới với text đã strip watermark.
        stripped_count += 1
        new_text = "\n".join(clean_lines)
        import dataclasses as _dc

        result_events.append(
            _dc.replace(event, text=new_text)
        )

    if stripped_count or dropped_count:
        logger.debug(
            "Watermark-line stripper: {} event stripped, {} event dropped. "
            "Watermark lines: {}",
            stripped_count,
            dropped_count,
            list(watermark_lines)[:5],
        )

    return result_events


BoundingBox = tuple[int, int, int, int]


#: [v3.23.717] Câu "đọc sai một chữ": bên tin cậy thấp phải dưới mức này (câu thật của phim mẫu: ≥ 0,995)...
MISREAD_MAX_CONFIDENCE: float = 0.97
#: ...và thấp hơn bên kia ít nhất chừng này.
MISREAD_CONFIDENCE_MARGIN: float = 0.02
#: Câu từ bấy nhiêu chữ trở lên mới xét (hai chữ khác một chữ là hai câu khác hẳn: "师兄" / "师妹").
MISREAD_MIN_CHARS: int = 3


def merge_misread_substitution_events(
    events: Sequence[SubtitleEvent], sample_step_sec: float,
) -> tuple[list[SubtitleEvent], int]:
    """[v3.23.717] Hai câu LIỀN NHAU, cùng chỗ, chỉ khác MỘT chữ, một bên tin cậy thấp rõ rệt → một câu.

    Thử sức trên OCR thật (3 % lần đọc bị đổi một chữ, tin cậy 0,70–0,95): câu ngắn như "知道了" có một lần đọc thành
    "知她了" — độ giống 0,67 dưới ngưỡng gộp nên ra hai câu, mốc đầu trễ 0,5 s. Hai câu thật liền nhau khác một chữ
    ("我去哪" → "你去哪", "我不闭关了" → "好不闭关了") đều được đọc với tin cậy ≥ 0,995 ở cả hai bên nên không bị đụng.

    Trả ``(câu, số cặp đã gộp)``.
    """
    if len(events) < 2 or sample_step_sec <= 0:
        return list(events), 0
    merged: list[SubtitleEvent] = [events[0]]
    count = 0
    for event in events[1:]:
        previous = merged[-1]
        left, right = previous.text or "", event.text or ""
        low, high = sorted((previous, event), key=lambda item: float(item.confidence))
        if (
            len(left) == len(right) >= MISREAD_MIN_CHARS
            and sum(1 for a, b in zip(left, right, strict=True) if a != b) == 1
            and event.start_sec - previous.end_sec <= 1.5 * sample_step_sec
            and float(low.confidence) < MISREAD_MAX_CONFIDENCE
            and float(high.confidence) - float(low.confidence) >= MISREAD_CONFIDENCE_MARGIN
            and (
                previous.bounding_box is None or event.bounding_box is None
                or _boxes_overlap(previous.bounding_box, event.bounding_box)
            )
        ):
            frames = previous.frame_count + event.frame_count
            # Chữ của câu gộp: :func:`_pick_reading` — không theo tin cậy một mình (mô hình có thể đọc sai rất tự tin).
            chosen = _pick_reading(
                left, (previous.frame_count, float(previous.confidence)),
                right, (event.frame_count, float(event.confidence)),
            )
            winner = previous if chosen == left else event
            merged[-1] = dataclasses.replace(
                winner,
                interval=TimeInterval(previous.start_sec, max(previous.end_sec, event.end_sec)),
                frame_count=frames,
                confidence=Confidence(
                    (float(previous.confidence) * previous.frame_count + float(event.confidence) * event.frame_count)
                    / max(1, frames)
                ),
            )
            count += 1
            continue
        merged.append(event)
    return merged, count


def _boxes_overlap(left: BoundingBox, right: BoundingBox) -> bool:
    """Hai bbox ``(xmin, ymin, xmax, ymax)`` có phần giao dương không (hàm thuần)."""
    return (
        min(left[2], right[2]) > max(left[0], right[0])
        and min(left[3], right[3]) > max(left[1], right[1])
    )


def _drop_belongs_to_group(
    group: FrameGroup,
    drop_timestamp: float,
    soft_dropped_boxes: dict[float, list[BoundingBox]] | None,
    soft_dropped_texts: dict[float, str] | None = None,
) -> bool:
    """[v3.23.616] Khung soft-drop có phải khung của CHÍNH câu này không (hàm thuần).

    Đo trên OCR thật 14/10 (phim "天骄女友退婚" tập 1): cả 2/2 lần nới biên bằng khung
    soft-drop mà chữ của khung đó nằm ở chỗ KHÁC câu (``山`` ở góc phải trên, ``公`` ở
    mép trái) đều sai so với điểm ảnh: câu "就是" bắt đầu sớm 40ms, câu "瑶瑶" kết
    thúc trễ 40ms. Khung mờ dần/hiện dần của chính câu (lý do có bước nới biên) luôn
    nằm cùng chỗ.

    Không biết vị trí (thiếu bbox) thì giữ cách cũ: chấp nhận.
    """
    if soft_dropped_texts is not None:
        # [v3.23.715] Khung bị lọc đọc ra chữ KHÔNG chung ký tự nào với câu là vật trong cảnh ("N", "V", "11!" với hộp
        # cao hết vùng chữ nên luôn "chồng" lên câu), không phải khung hiện dần / mờ dần của câu.
        dropped_text = soft_dropped_texts.get(drop_timestamp)
        if dropped_text is not None:
            own = {char for char in group.reconstructed_text if char.isalnum()}
            if own and not own & {char for char in dropped_text if char.isalnum()}:
                return False
    if soft_dropped_boxes is None or group.aggregated_bounding_box is None:
        return True
    boxes = soft_dropped_boxes.get(drop_timestamp)
    if not boxes:
        return True
    return any(_boxes_overlap(group.aggregated_bounding_box, box) for box in boxes)


def extend_group_timing_from_soft_drops(
    groups: list[FrameGroup],
    soft_dropped_sorted: list[float],
    sample_step_sec: float,
    soft_dropped_boxes: dict[float, list[BoundingBox]] | None = None,
    soft_dropped_texts: dict[float, str] | None = None,
) -> list[FrameGroup]:
    """Phục hồi timing biên của group bằng timestamp frame bị lọc sát biên.

    **Vấn đề cần giải quyết**: Stage 2 (spatial cleanup) và Stage 3
    (confidence filter) loại bỏ các frame có nội dung nhưng không đạt
    ngưỡng chất lượng. Các frame này thường là **frame đầu/cuối của một
    phụ đề** — khi text mới xuất hiện (animation fade-in, trượt vào) hoặc
    bắt đầu mờ dần (fade-out). Kết quả:

    * ``start_timestamp_sec`` của group **trễ hơn** thời điểm thực tế ≈ 1
      sample step (40 ms ở 25 fps lấy mẫu).
    * ``end_timestamp_sec`` **sớm hơn** thời điểm thực tế ≈ 1 sample step.

    **Cách sửa**: Thu thập timestamp của các frame "bị lọc nhưng có nội
    dung" (soft-drop) trong stages 2+3. Với mỗi group, kiểm tra xem có
    soft-drop nào nằm sát biên (trong cửa sổ 0.5–1.5 × sample_step) không.
    Nếu có, mở rộng timing biên tới soft-drop đó — nhưng không vượt qua
    group liền kề để tránh overlap.

    Args:
        groups:              FrameGroup đã merge (sort theo timestamp).
        soft_dropped_sorted: Timestamp sort tăng dần của frame bị lọc có nội dung.
        sample_step_sec:     Bước lấy mẫu (config.sample_step_sec).
        soft_dropped_texts:  [v3.23.715] ``timestamp → chữ`` của từng khung soft-drop; có thì khung đọc ra chữ không
                             chung ký tự nào với câu không được dùng để nới biên.
        soft_dropped_boxes:  [v3.23.616] ``timestamp → bbox`` chữ của từng khung
                             soft-drop; có thì chỉ nới biên bằng khung có chữ chồng
                             lên vùng của group (xem :func:`_drop_belongs_to_group`).

    Returns:
        Danh sách group đã mở rộng timing (mutate in-place và return).
    """
    if not groups or not soft_dropped_sorted:
        return groups

    # Cửa sổ tìm kiếm: 0.5× – 1.5× step.
    # < 0.5×: quá gần, có thể là PTS float noise, không phải frame bị lọc.
    # > 1.5×: quá xa, không còn chắc là frame của cùng phụ đề.
    _WINDOW_LOW = sample_step_sec * 0.5
    _WINDOW_HIGH = sample_step_sec * 1.5

    n = len(groups)
    for i, group in enumerate(groups):
        # --- Mở rộng START về phía trước ---
        target_start = group.start_timestamp_sec
        min_allowed = groups[i - 1].end_timestamp_sec if i > 0 else -1e9

        lo = bisect.bisect_left(soft_dropped_sorted, target_start - _WINDOW_HIGH)
        hi = bisect.bisect_left(soft_dropped_sorted, target_start - _WINDOW_LOW)
        candidates_start = [
            t for t in soft_dropped_sorted[lo:hi]
            if t > min_allowed
            and _drop_belongs_to_group(group, t, soft_dropped_boxes, soft_dropped_texts)
        ]
        if candidates_start:
            new_start = candidates_start[0]  # frame bị lọc SỚM NHẤT liền kề
            group.start_timestamp_sec = max(new_start, min_allowed + 0.001)

        # --- Mở rộng END về phía sau ---
        target_end = group.end_timestamp_sec
        max_allowed = groups[i + 1].start_timestamp_sec if i + 1 < n else 1e9

        lo2 = bisect.bisect_right(soft_dropped_sorted, target_end + _WINDOW_LOW)
        hi2 = bisect.bisect_right(soft_dropped_sorted, target_end + _WINDOW_HIGH)
        candidates_end = [
            t for t in soft_dropped_sorted[lo2:hi2]
            if t < max_allowed
            and _drop_belongs_to_group(group, t, soft_dropped_boxes, soft_dropped_texts)
        ]
        if candidates_end:
            new_end = candidates_end[-1]  # frame bị lọc MUỘN NHẤT liền kề
            group.end_timestamp_sec = min(new_end, max_allowed - 0.001)
        elif soft_dropped_texts is not None:
            # [v3.23.716] OCR hụt 1–2 khung rồi đọc lại ĐÚNG NGUYÊN VĂN câu ở một khung yếu (bị lọc): "切" 1729,36 s có
            # 17 khung, 2 khung rỗng, rồi "切" tin cậy 0,57 hộp cao hết vùng chữ — ảnh thật: chữ còn tới khung đó. Khoảng
            # hở < 3 mẫu là ngưỡng "OCR hụt" của luật câu lặp (từ 3 mẫu trống trở lên mới coi là nói lại).
            own_text = "".join(char for char in group.reconstructed_text if char.isalnum())
            reach = target_end + sample_step_sec * (_REPEAT_MIN_BLANK_SAMPLES_DENSE + 0.5)
            hi3 = bisect.bisect_right(soft_dropped_sorted, reach)
            same_text_reads = [
                t for t in soft_dropped_sorted[hi2:hi3]
                if t < max_allowed
                and own_text
                and "".join(char for char in soft_dropped_texts.get(t, "") if char.isalnum()) == own_text
                and _drop_belongs_to_group(group, t, soft_dropped_boxes, soft_dropped_texts)
            ]
            if same_text_reads:
                group.end_timestamp_sec = min(same_text_reads[-1], max_allowed - 0.001)

    return groups


# [v3.23.699] Rác OCR từ khung có hoa văn/vật thể (mảnh chữ Latin, số, ký hiệu) trong phim phụ đề chữ Hán.
_CJK_CHAR_RE = re.compile(r"[\u3400-\u9fff]")
_NOISE_PUNCT_RE = re.compile(r"[\s，。！？、…“”‘’\"'!?,.~～—\-:：;；()（）《》<>]")
#: Phim coi là phụ đề chữ Hán khi ít nhất chừng này phần câu có chữ Hán.
CJK_FILM_MIN_SHARE = 0.8
#: Câu rác: dưới 2 chữ Hán, có ký tự khác (chữ Latin / số / ký hiệu) và độ tin cậy dưới mức này.
NOISE_FRAGMENT_MAX_CONFIDENCE = 0.7


def drop_mixed_noise_fragments(events: Sequence[SubtitleEvent]) -> tuple[list[SubtitleEvent], int]:
    """[v3.23.699] Bỏ câu "mảnh rác" (vd "色6%") trong phim phụ đề chữ Hán.

    Lượt trích xuất thật cả phim 86 tập (GPU, bước 0,03 s): một khung hoa văn cho ~20 hộp chữ vụn (độ tin cậy 0,35–0,8),
    ghép thành câu "色6%" (tin cậy 0,60). Đối chiếu cả hai bộ dữ liệu phim mẫu (3663 + 3660 câu): luật chỉ bắt đúng câu
    này. Không áp cho phim có nhiều câu không chữ Hán (phụ đề Latin) — tránh xoá câu tiếng Anh tin cậy thấp.
    """
    if not events:
        return list(events), 0
    cjk_events = sum(1 for e in events if _CJK_CHAR_RE.search(e.text or ""))
    if cjk_events < CJK_FILM_MIN_SHARE * len(events):
        return list(events), 0
    kept: list[SubtitleEvent] = []
    for event in events:
        text = event.text or ""
        cjk = len(_CJK_CHAR_RE.findall(text))
        other = _NOISE_PUNCT_RE.sub("", _CJK_CHAR_RE.sub("", text))
        if cjk < 2 and other and float(event.confidence) < NOISE_FRAGMENT_MAX_CONFIDENCE:
            continue
        kept.append(event)
    return kept, len(events) - len(kept)


__all__ = [
    "drop_mixed_noise_fragments",
    "convert_groups_to_events",
    "extend_group_timing_from_soft_drops",
    "filter_echo_trail_events",
    "filter_echo_trail_groups",
    "filter_short_duration_events",
    "filter_short_text_events",
    "is_superset_within_one_char",
    "merge_adjacent_duplicates",
    "post_merge_duplicates",
    "reindex_events",
    "strip_latin_watermark_prefix_from_cjk_events",
    "strip_persistent_watermark_lines",
]
