"""[v3.23.630] Bộ nhớ đệm TTS theo TỪNG DÒNG — chỉ tổng hợp lại dòng đã đổi.

Log thật của phim 86 tập (3649 dòng): 21 lượt "Tổng hợp" cả phim, mỗi lượt ~18 phút = 6h14m, dù
mỗi lần người dùng chỉ sửa vài chục dòng (đổi người nói, sửa lời). Module này cho mỗi dòng một
CHỮ KÝ (lời + mốc + giọng + mốc câu kế) và một KHOÁ TOÀN CỤC (mọi cài đặt ảnh hưởng audio từng
dòng). Lần Tổng hợp sau so chữ ký: dòng không đổi dùng lại file audio đã có, chỉ dòng đổi mới
đưa vào engine, rồi ghép lại master.

Chữ ký dựa trên NỘI DUNG chứ không theo vị trí → thêm/xoá dòng (vị trí dịch) hay mở lại ứng dụng
vẫn nhận lại đúng audio; audio dòng đổi vị trí được đổi tên theo vị trí mới.

Toàn bộ hàm trong module là hàm thuần (không I/O) — lưu/đọc file ở
``infrastructure.tts.line_cache_store``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from enum import Enum

from subtitles_extractor.domain.ports.subtitle_tts_port import (
    TTSRequest,
    TTSSegmentResult,
)

#: Phiên bản định dạng cache (đổi khi đổi cách tính chữ ký → cache cũ tự bị bỏ).
CACHE_VERSION = 1

#: Dùng lại cache chỉ khi có ít nhất ngần này dòng được tái dùng — dưới mức đó chạy lại cả
#: phim cho gọn (lập lịch thời gian toàn cục chính xác hơn, tiết kiệm không đáng kể).
MIN_REUSE_RATIO = 0.30

#: Trường ``TTSRequest`` KHÔNG ảnh hưởng audio từng dòng (chỉ ảnh hưởng master/ghi file, hoặc
#: được tính riêng theo từng dòng). Đổi các trường này KHÔNG làm mất cache.
_NEUTRAL_FIELDS = frozenset({
    "events", "line_voices", "neighbors", "segments_dir", "keep_line_audio",
    "output_format", "output_bitrate_kbps", "wav_subtype",
    "normalize", "target_lufs", "voice_clarity", "high_quality",
    "edge_concurrency", "retry_count", "retry_delay_s", "media_duration_s",
})


class PlanMode(str, Enum):
    """Cách chạy lần Tổng hợp này."""

    FULL = "full"            # chạy cả phim
    INCREMENTAL = "incremental"  # chỉ dòng đổi
    UNCHANGED = "unchanged"  # không dòng nào đổi — chỉ ghép lại master


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """Một dòng đã tổng hợp: chữ ký lúc đó + kết quả (kèm đường dẫn audio dòng)."""

    signature: str
    result: TTSSegmentResult


@dataclass(frozen=True, slots=True)
class LineCache:
    """Cache của MỘT file output: khoá toàn cục + các dòng đã tổng hợp theo vị trí."""

    global_key: str
    sample_rate: int
    entries: Mapping[int, CacheEntry] = field(default_factory=dict)
    #: ``False`` = cache ghi GIỮA một lượt cả phim (điểm kiểm tra): lượt đó chưa xong, các dòng còn lại chưa có.
    #: Cache cũ không có trường này = đã xong.
    complete: bool = True


@dataclass(frozen=True, slots=True)
class IncrementalPlan:
    """Kế hoạch cho lần Tổng hợp: chạy gì, dùng lại gì.

    Attributes:
        mode: Cách chạy.
        reason: Lý do (mã ngắn, để ghi log): ``no_cache`` / ``settings_changed`` /
            ``too_many_changed`` / ``ok``.
        reuse: ``{vị trí cũ trong cache: vị trí mới}`` của dòng dùng lại được.
        stale: Vị trí (mới) các dòng phải tổng hợp lại.
        total: Tổng số dòng.
        signatures: Chữ ký hiện tại của từng dòng theo vị trí (để ghi cache sau lượt chạy).
    """

    mode: PlanMode
    reason: str
    reuse: Mapping[int, int]
    stale: frozenset[int]
    total: int
    signatures: tuple[str, ...]

    @property
    def reused_count(self) -> int:
        """Số dòng dùng lại audio cũ."""
        return len(self.reuse)


def global_key(request: TTSRequest, engine_key: str) -> str:
    """Khoá toàn cục: engine + mọi cài đặt ảnh hưởng audio/lập lịch từng dòng.

    Args:
        request: Yêu cầu TTS (bỏ qua trường trung tính — xem ``_NEUTRAL_FIELDS``).
        engine_key: Dấu của engine + chế độ (``adapter.line_cache_key()``).

    Returns:
        Chuỗi băm ổn định.
    """
    parts = {
        f.name: getattr(request, f.name)
        for f in fields(request) if f.name not in _NEUTRAL_FIELDS
    }
    blob = json.dumps(
        {"v": CACHE_VERSION, "engine": engine_key, "request": parts},
        sort_keys=True, default=str, ensure_ascii=False,
    )
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def line_signatures(
    request: TTSRequest, media_duration_s: float | None = None
) -> tuple[str, ...]:
    """Chữ ký từng dòng theo vị trí trong ``request.events``.

    Gồm: lời, mốc đầu/cuối, giọng của dòng, mốc đầu câu kế (khung thời gian của dòng phụ
    thuộc câu kế — xoá/thêm/dời câu kế phải tổng hợp lại dòng này).

    Args:
        request: Yêu cầu TTS.
        media_duration_s: Biên cho dòng cuối (thời lượng video); ``None`` = lấy từ request.

    Returns:
        Chữ ký theo vị trí.
    """
    events = request.events
    boundary = media_duration_s if media_duration_s is not None else request.media_duration_s
    signatures: list[str] = []
    for position, event in enumerate(events):
        following = (
            round(float(events[position + 1].start_sec), 3)
            if position + 1 < len(events)
            else (None if boundary is None else round(float(boundary), 3))
        )
        blob = json.dumps(
            [
                str(getattr(event, "text", "") or ""),
                round(float(event.start_sec), 3), round(float(event.end_sec), 3),
                request.voice_at(position, request.speaker), following,
            ],
            ensure_ascii=False,
        )
        signatures.append(hashlib.sha1(blob.encode("utf-8")).hexdigest()[:20])
    return tuple(signatures)


def _reusable(entry: CacheEntry, audio_ok: Callable[[str], bool]) -> bool:
    """Kết quả cache dùng lại được: không lỗi và (bỏ qua có chủ đích, hoặc còn file audio)."""
    result = entry.result
    if result.error_msg:
        return False
    if result.was_skipped:
        return True
    return bool(result.segment_audio_path) and audio_ok(result.segment_audio_path)


def plan_incremental(
    cache: LineCache | None,
    request: TTSRequest,
    engine_key: str,
    audio_ok: Callable[[str], bool],
    media_duration_s: float | None = None,
) -> IncrementalPlan:
    """Quyết định chạy cả phim hay chỉ các dòng đã đổi.

    Args:
        cache: Cache đọc từ đĩa (``None`` nếu chưa có).
        request: Yêu cầu TTS hiện tại.
        engine_key: Dấu engine (``adapter.line_cache_key()``).
        audio_ok: ``đường dẫn -> còn file audio dòng dùng được``.
        media_duration_s: Biên dòng cuối.

    Returns:
        :class:`IncrementalPlan`.
    """
    signatures = line_signatures(request, media_duration_s)
    total = len(signatures)
    everything = frozenset(range(total))
    if cache is None or not cache.entries:
        return IncrementalPlan(PlanMode.FULL, "no_cache", {}, everything, total, signatures)
    if cache.global_key != global_key(request, engine_key) or cache.sample_rate <= 0:
        return IncrementalPlan(
            PlanMode.FULL, "settings_changed", {}, everything, total, signatures
        )

    by_signature: dict[str, int] = {}
    for old_position, entry in cache.entries.items():
        if _reusable(entry, audio_ok):
            by_signature[entry.signature] = old_position
    reuse: dict[int, int] = {}
    claimed: set[int] = set()
    for position, signature in enumerate(signatures):
        old_position = by_signature.get(signature)
        if old_position is None or old_position in claimed:
            continue
        claimed.add(old_position)
        reuse[old_position] = position
    stale = everything - frozenset(reuse.values())
    if not stale:
        return IncrementalPlan(PlanMode.UNCHANGED, "ok", reuse, stale, total, signatures)
    # [v3.23.647] Lượt cả phim bị huỷ/sập giữa chừng (cache chưa "complete"): các dòng còn thiếu chỉ là chưa chạy tới,
    # không phải "đổi quá nhiều" — dùng lại MỌI dòng đã xong dù ít, thay vì bắt chạy lại từ đầu.
    if not cache.complete:
        if not reuse:
            return IncrementalPlan(PlanMode.FULL, "no_cache", {}, everything, total, signatures)
        return IncrementalPlan(PlanMode.INCREMENTAL, "resumed", reuse, stale, total, signatures)
    if total == 0 or len(reuse) < MIN_REUSE_RATIO * total:
        return IncrementalPlan(
            PlanMode.FULL, "too_many_changed", {}, everything, total, signatures
        )
    return IncrementalPlan(PlanMode.INCREMENTAL, "ok", reuse, stale, total, signatures)


def assemble_prior_results(
    request: TTSRequest,
    reused: Sequence[TTSSegmentResult],
) -> list[TTSSegmentResult]:
    """Kết quả "lượt trước" đủ MỌI vị trí cho use case TTS lại.

    Dòng dùng lại giữ kết quả cũ (đã đánh số theo vị trí mới); dòng phải tổng hợp lại nhận
    kết quả tạm "bỏ qua" (sẽ bị thay bằng kết quả mới).

    Args:
        request: Yêu cầu TTS hiện tại.
        reused: Kết quả dòng dùng lại, ``event_index`` đã là vị trí mới.

    Returns:
        Danh sách theo vị trí ``0..len(events)-1``.
    """
    by_position = {int(r.event_index): r for r in reused}
    prior: list[TTSSegmentResult] = []
    for position, event in enumerate(request.events):
        existing = by_position.get(position)
        if existing is None:
            text = str(getattr(event, "text", "") or "")
            existing = TTSSegmentResult(
                event_index=position, start_sec=float(event.start_sec),
                end_sec=float(event.end_sec), text=text, subtitle_text=text,
                was_skipped=True,
            )
        prior.append(existing)
    return prior


def build_cache(
    request: TTSRequest,
    engine_key: str,
    sample_rate: int,
    results: Sequence[TTSSegmentResult],
    signatures: Sequence[str],
    complete: bool = True,
) -> LineCache:
    """Cache sau một lượt chạy: mọi dòng có kết quả tốt, theo chữ ký lúc bắt đầu lượt.

    Dòng lỗi (``error_msg``) hoặc đáng ra có tiếng mà không có file audio KHÔNG vào cache →
    lần sau tự được thử lại.

    Args:
        request: Yêu cầu của lượt chạy (đúng bản lúc bắt đầu).
        engine_key: Dấu engine.
        sample_rate: Tần số lấy mẫu của segment.
        results: Kết quả đủ mọi dòng.
        signatures: Chữ ký từng dòng tính LÚC BẮT ĐẦU lượt (không tính lại sau khi người dùng
            sửa dở dang).
        complete: ``False`` khi ghi điểm kiểm tra giữa lượt (xem :class:`LineCache`).

    Returns:
        :class:`LineCache`.
    """
    entries: dict[int, CacheEntry] = {}
    for result in results:
        position = int(result.event_index)
        if not 0 <= position < len(signatures) or result.error_msg:
            continue
        if not result.was_skipped and not result.segment_audio_path:
            continue
        entries[position] = CacheEntry(signatures[position], result)
    return LineCache(global_key(request, engine_key), int(sample_rate), entries, complete)


def remap_cache(cache: LineCache, mapping: Mapping[int, int]) -> LineCache:
    """Cache sau khi thêm/xoá dòng: đánh số lại theo ``{vị trí cũ: vị trí mới}``.

    Đường dẫn audio đổi theo tên ``seg_<vị trí>.wav`` mới (cùng quy ước ``segment_store``).
    """
    from pathlib import Path

    from subtitles_extractor.infrastructure.tts.segment_store import segment_path

    entries: dict[int, CacheEntry] = {}
    for old_position, entry in cache.entries.items():
        new_position = mapping.get(old_position)
        if new_position is None:
            continue
        path = entry.result.segment_audio_path
        if path:
            path = str(segment_path(Path(path).parent, new_position))
        entries[new_position] = CacheEntry(
            entry.signature,
            replace(entry.result, event_index=new_position, segment_audio_path=path),
        )
    return replace(cache, entries=entries)


__all__ = [
    "CACHE_VERSION", "MIN_REUSE_RATIO", "CacheEntry", "IncrementalPlan", "LineCache",
    "PlanMode", "assemble_prior_results", "build_cache", "global_key", "line_signatures",
    "plan_incremental", "remap_cache",
]
