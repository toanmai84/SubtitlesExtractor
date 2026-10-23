"""[v3.23.630] Chuẩn bị / khép lượt TTS tăng dần (nối ``tts_line_cache`` với đĩa + engine).

Luồng của một lần bấm "Tổng hợp":

1. :func:`prepare_run` đọc cache cạnh các segment, so chữ ký từng dòng → kế hoạch. Dòng dùng lại
   được dời về đúng tên ``seg_<vị trí mới>.wav`` và cache được ghi ngay (huỷ giữa chừng vẫn
   nhất quán).
2. Người gọi chạy cả phim hoặc chỉ các dòng đổi.
3. :func:`record_run` ghi cache mới từ kết quả (dòng lỗi không vào cache → lần sau thử lại).

Engine không có ``supports_line_cache`` (Edge/Gemini) bỏ qua cả module: ``prepare_run`` trả
``None``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from subtitles_extractor.application.services.tts_line_cache import (
    CacheEntry,
    IncrementalPlan,
    LineCache,
    PlanMode,
    assemble_prior_results,
    build_cache,
    global_key,
    line_signatures,
    plan_incremental,
    remap_cache,
)
from subtitles_extractor.domain.ports.subtitle_tts_port import (
    TTSRequest,
    TTSSegmentResult,
)
from subtitles_extractor.infrastructure.tts.line_cache_store import (
    audio_ok,
    cache_path,
    discard_cache,
    load_cache,
    save_cache,
)
from subtitles_extractor.infrastructure.tts.segment_store import (
    renumber_results,
    segments_dir_for,
)
from subtitles_extractor.infrastructure.tts.timing_math import master_length_samples

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PreparedRun:
    """Kết quả :func:`prepare_run` — mọi thứ người gọi cần để chạy rồi khép lượt.

    Attributes:
        plan: Kế hoạch (chạy cả phim / chỉ dòng đổi / không gì đổi).
        request: Yêu cầu của lượt chạy (bản lúc bắt đầu).
        engine_key: Dấu engine của cache.
        cache_file: File cache.
        sample_rate: Tần số lấy mẫu của segment cũ (0 khi chạy cả phim, chưa biết).
        prior_results: Kết quả "lượt trước" đủ mọi vị trí (rỗng khi chạy cả phim).
        total_length_samples: Độ dài master (0 khi chạy cả phim).
    """

    plan: IncrementalPlan
    request: TTSRequest
    engine_key: str
    cache_file: Path
    sample_rate: int = 0
    prior_results: tuple[TTSSegmentResult, ...] = ()
    total_length_samples: int = 0


def cache_file_for(request: TTSRequest, output_path: Path) -> Path:
    """File cache của lượt TTS ghi ra ``output_path``."""
    return cache_path(segments_dir_for(output_path, getattr(request, "segments_dir", "")))


def _master_samples(request: TTSRequest, sample_rate: int) -> int:
    """Độ dài master như lượt đầy đủ (cùng ``master_length_samples``); 0 nếu không có dòng."""
    spoken = [event.end_sec for event in request.events if event.text.strip()]
    if not spoken:
        return 0
    return int(master_length_samples(max(spoken), sample_rate, request.media_duration_s))


def prepare_run(
    request: TTSRequest, adapter: object, output_path: Path, incremental: bool = True
) -> PreparedRun | None:
    """Lập kế hoạch cho lần Tổng hợp này và chuẩn bị đĩa.

    Args:
        request: Yêu cầu TTS đầy đủ (mọi dòng, giọng từng dòng).
        adapter: Adapter đã tạo; cần ``supports_line_cache`` + ``line_cache_key()``.
        output_path: File output (đã đồng bộ đuôi theo định dạng).
        incremental: ``False`` = người dùng muốn chạy lại cả phim.

    Returns:
        ``None`` nếu engine không hỗ trợ cache; ngược lại :class:`PreparedRun`.
    """
    if not getattr(adapter, "supports_line_cache", False):
        return None
    engine_key = adapter.line_cache_key()  # type: ignore[attr-defined]
    cache_file = cache_file_for(request, output_path)
    cache = load_cache(cache_file) if incremental else None
    plan = plan_incremental(cache, request, engine_key, audio_ok)
    if not incremental:
        plan = IncrementalPlan(
            plan.mode, "disabled", plan.reuse, plan.stale, plan.total, plan.signatures
        )
    if plan.mode is PlanMode.FULL or cache is None:
        # Audio dòng sắp bị ghi đè → cache cũ hết đúng; xoá để huỷ giữa chừng không để sót.
        discard_cache(cache_file)
        return PreparedRun(plan, request, engine_key, cache_file)

    total_samples = _master_samples(request, cache.sample_rate)
    if total_samples <= 0:
        discard_cache(cache_file)
        return PreparedRun(plan, request, engine_key, cache_file)

    old_results = [cache.entries[old].result for old in plan.reuse]
    moved = renumber_results(old_results, plan.reuse, delete_removed=False)
    # Ghi cache NGAY với phần dùng lại (đã đúng tên mới): huỷ/hỏng giữa lượt vẫn nhất quán.
    save_cache(cache_file, build_cache(
        request, engine_key, cache.sample_rate, moved, plan.signatures,
        complete=cache.complete,        # lượt tiếp tục mà lại bị ngắt thì vẫn là "chưa xong"
    ))
    return PreparedRun(
        plan, request, engine_key, cache_file, cache.sample_rate,
        tuple(assemble_prior_results(request, moved)), total_samples,
    )


def _detect_sample_rate(results: Iterable[TTSSegmentResult]) -> int:
    """Tần số lấy mẫu của segment (đọc đầu một file audio dòng); 0 nếu không đọc được."""
    import soundfile as sf

    for result in results:
        if result.segment_audio_path:
            try:
                return int(sf.info(result.segment_audio_path).samplerate)
            except (OSError, RuntimeError):
                continue
    return 0


def record_run(run: PreparedRun, results: list[TTSSegmentResult]) -> None:
    """Ghi cache sau lượt chạy (cả phim hoặc tăng dần).

    Args:
        run: Kết quả :func:`prepare_run` của lượt này.
        results: Kết quả đủ mọi dòng.
    """
    sample_rate = run.sample_rate or _detect_sample_rate(results)
    if sample_rate <= 0:
        return
    fresh = build_cache(run.request, run.engine_key, sample_rate, results, run.plan.signatures)
    save_cache(run.cache_file, fresh)


def record_checkpoint(run: PreparedRun, results: list[TTSSegmentResult], sample_rate: int) -> bool:
    """[v3.23.647] Ghi điểm kiểm tra GIỮA lượt cả phim: các dòng đã xong tới lúc này.

    Gọi từ luồng TTS; chỉ đụng tới ``run`` (bất biến) và đĩa nên an toàn. Cache ghi ở đây mang cờ ``complete=False`` —
    nếu lượt bị huỷ/sập, lần Tổng hợp sau dùng lại các dòng này (xem :func:`plan_incremental`).

    Args:
        run: Kết quả :func:`prepare_run` của lượt đang chạy.
        results: Kết quả các dòng đã xong, theo thứ tự.
        sample_rate: Tần số lấy mẫu segment của engine.

    Returns:
        ``True`` nếu đã ghi.
    """
    if sample_rate <= 0 or not results:
        return False
    partial = build_cache(
        run.request, run.engine_key, sample_rate, results, run.plan.signatures, complete=False,
    )
    return save_cache(run.cache_file, partial)


def record_selected(
    request: TTSRequest,
    adapter: object,
    output_path: Path,
    results: list[TTSSegmentResult],
    positions: set[int],
) -> None:
    """Ghi cache sau khi TTS lại vài dòng chọn (nút "TTS lại dòng chọn" / dòng vừa thêm).

    Giữ nguyên các dòng khác của cache; dòng ``positions`` nhận chữ ký của yêu cầu vừa dùng.
    Không có cache hợp lệ trước đó thì cache mới chỉ gồm các dòng này.
    """
    if not getattr(adapter, "supports_line_cache", False):
        return
    engine_key = adapter.line_cache_key()  # type: ignore[attr-defined]
    cache_file = cache_file_for(request, output_path)
    key = global_key(request, engine_key)
    signatures = line_signatures(request)
    existing = load_cache(cache_file)
    if existing is not None and existing.global_key != key:
        existing = None
    sample_rate = (existing.sample_rate if existing else 0) or _detect_sample_rate(results)
    if sample_rate <= 0:
        return
    entries: dict[int, CacheEntry] = dict(existing.entries) if existing else {}
    updated = build_cache(request, engine_key, sample_rate, results, signatures)
    for position in positions:
        if position in updated.entries:
            entries[position] = updated.entries[position]
        else:
            entries.pop(position, None)
    save_cache(cache_file, LineCache(key, sample_rate, entries, existing.complete if existing else True))


def remap_saved_cache(
    request: TTSRequest, adapter: object, output_path: Path, mapping: dict[int, int]
) -> None:
    """Cache theo đĩa sau khi thêm/xoá/tách/gộp dòng (audio dòng đã được đánh số lại)."""
    if not getattr(adapter, "supports_line_cache", False):
        return
    cache_file = cache_file_for(request, output_path)
    cache = load_cache(cache_file)
    if cache is not None:
        save_cache(cache_file, remap_cache(cache, mapping))


__all__ = [
    "PreparedRun", "cache_file_for", "prepare_run", "record_checkpoint", "record_run",
    "record_selected", "remap_saved_cache",
]
