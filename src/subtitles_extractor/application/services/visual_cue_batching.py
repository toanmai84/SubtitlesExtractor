"""[v3.23.560] Chia lô dòng phụ đề cho bộ quét Visual Cues THEO ĐOẠN VIDEO (hàm thuần).

Bộ quét gửi kèm mỗi lô các đoạn video phủ khoảng thời gian của lô. Chia lô chỉ theo số
dòng thì lô hay vắt qua ranh giới đoạn và phải kèm 2-4 đoạn (~138k token mỗi đoạn). Đo
trên phim thật khi quét bổ sung 2856 dòng thiếu cues: 17/20 lô kèm từ 2 đoạn trở lên, một
lô kèm 4 đoạn (~550k token) — model nhẹ trả 503/504 DEADLINE_EXCEEDED liên tục. Cắt lô
tại ranh giới đoạn thì mỗi lô chỉ kèm đúng một đoạn.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from typing import Any

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationLine,
    VisualCue,
)

#: [v3.23.581] Số dòng ĐÃ có nhãn tối đa đi kèm mỗi lô quét làm mạch hội thoại.
MAX_KNOWN_CONTEXT_LINES = 300
#: Mở rộng vùng ngữ cảnh ra ngoài hai đầu lô ngần này dòng.
KNOWN_CONTEXT_PADDING = 3


def chunk_index_for_time(
    start_sec: float, chunk_bounds: Sequence[tuple[float, float]]
) -> int:
    """Chỉ số đoạn video chứa thời điểm ``start_sec`` (hàm thuần).

    Args:
        start_sec: Thời điểm cần tra (giây).
        chunk_bounds: ``(bắt_đầu, kết_thúc)`` của từng đoạn, theo thứ tự thời gian.

    Returns:
        Chỉ số đoạn chứa thời điểm; thời điểm nằm ngoài mọi đoạn (khe hở, sau đoạn cuối)
        được gán cho đoạn gần nhất bắt đầu trước nó, hoặc đoạn 0 nếu trước đoạn đầu.
    """
    chosen = 0
    for position, (chunk_start, chunk_end) in enumerate(chunk_bounds):
        if chunk_start <= start_sec < chunk_end:
            return position
        if chunk_start <= start_sec:
            chosen = position
    return chosen


def batch_lines_by_video_chunk(
    lines: Sequence[TranslationLine],
    chunk_bounds: Sequence[tuple[float, float]],
    batch_size: int,
) -> list[list[TranslationLine]]:
    """Chia ``lines`` thành lô ≤ ``batch_size`` dòng, KHÔNG lô nào vắt qua hai đoạn video.

    Dòng được gán cho đoạn chứa thời điểm BẮT ĐẦU của nó. Thứ tự dòng được giữ nguyên.

    Args:
        lines: Các dòng cần quét (có thể là tập con rời rạc của phim).
        chunk_bounds: ``(bắt_đầu, kết_thúc)`` (giây) của từng đoạn video; rỗng nghĩa là
            không có video — chỉ chia theo số dòng.
        batch_size: Số dòng tối đa mỗi lô.

    Returns:
        Danh sách lô, mỗi lô là danh sách dòng liên tiếp theo thứ tự đầu vào.

    Raises:
        ValueError: Nếu ``batch_size`` không dương.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size phải dương, nhận {batch_size}.")
    batches: list[list[TranslationLine]] = []
    current: list[TranslationLine] = []
    current_chunk: int | None = None
    for line in lines:
        chunk = (
            chunk_index_for_time(line.start_ms / 1000.0, chunk_bounds)
            if chunk_bounds
            else 0
        )
        if current and (chunk != current_chunk or len(current) >= batch_size):
            batches.append(current)
            current = []
        current.append(line)
        current_chunk = chunk
    if current:
        batches.append(current)
    return batches


def clip_timestamp(start_ms: int, clip_start_sec: float) -> str:
    """[v3.23.581] Thời điểm của dòng TÍNH TỪ ĐẦU đoạn video kèm theo (hàm thuần).

    Bộ quét trước đây chỉ gửi ``id`` + chữ: model phải tự dò trong đoạn video 6 phút xem
    dòng nào hiện lúc nào. Ca thật: khi quét RIÊNG các dòng còn thiếu nhãn, cuộc gọi của
    Lâm Dật với chị dâu (dòng 1304-1307) bị gán cho một cảnh khác của đoạn phim. Gemini
    nhận mốc thời gian dạng ``MM:SS`` trong prompt để trỏ đúng khung hình.

    Args:
        start_ms: Mốc bắt đầu của dòng (mili-giây, tính từ đầu phim).
        clip_start_sec: Mốc bắt đầu của đoạn video (giây, tính từ đầu phim).

    Returns:
        ``"MM:SS"`` (hoặc ``"H:MM:SS"`` khi từ một giờ trở lên); âm thì về ``"00:00"``.
    """
    seconds = max(0, int(start_ms / 1000.0 - clip_start_sec))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def lines_in_clip(
    lines: Sequence[TranslationLine], start_sec: float, end_sec: float
) -> list[TranslationLine]:
    """[v3.23.596] Các dòng BẮT ĐẦU trong đoạn video ``[start_sec, end_sec)`` (hàm thuần).

    Cùng quy tắc chia lô với :func:`batch_lines_by_video_chunk`: câu kéo dài qua mép đoạn
    thuộc về đoạn chứa mốc bắt đầu của nó.

    Args:
        lines: Các dòng phụ đề (mốc mili-giây).
        start_sec: Mốc đầu đoạn (giây, tính từ đầu phim).
        end_sec: Mốc cuối đoạn (giây).

    Returns:
        Các dòng thuộc đoạn, giữ thứ tự.
    """
    start_ms, end_ms = start_sec * 1000.0, end_sec * 1000.0
    return [line for line in lines if start_ms <= line.start_ms < end_ms]


def scan_line_items(
    batch: Sequence[TranslationLine], clip_start_sec: float | None
) -> list[dict[str, Any]]:
    """Các dòng cần gắn nhãn của một lô quét, kèm mốc thời gian nếu biết đoạn video.

    Args:
        batch: Các dòng của lô.
        clip_start_sec: Mốc bắt đầu của ĐÚNG MỘT đoạn video kèm lô; ``None`` khi không có
            video hoặc lô kèm nhiều đoạn (mốc tương đối khi đó không rõ nghĩa).

    Returns:
        ``[{"id", "text"(, "t")}]``.
    """
    items: list[dict[str, Any]] = []
    for line in batch:
        item: dict[str, Any] = {"id": line.index, "text": line.text}
        if clip_start_sec is not None:
            item["t"] = clip_timestamp(line.start_ms, clip_start_sec)
        items.append(item)
    return items


def known_context_items(
    batch: Sequence[TranslationLine],
    context_lines: Sequence[TranslationLine],
    known_cues: Sequence[VisualCue],
    clip_start_sec: float | None,
) -> list[dict[str, Any]]:
    """[v3.23.581] Dòng xen kẽ ĐÃ biết ai nói với ai, trong vùng của lô (hàm thuần).

    Quét bổ sung chỉ gửi các dòng còn thiếu nhãn — rời rạc, mất mạch hội thoại. Các dòng
    đã có nhãn nằm giữa chúng được gửi kèm (chỉ để tham khảo) để model biết cuộc nói
    chuyện nào đang diễn ra.

    Args:
        batch: Các dòng cần gắn nhãn.
        context_lines: Toàn bộ dòng phụ đề.
        known_cues: Nhãn đã có (tin cậy).
        clip_start_sec: Như :func:`scan_line_items`.

    Returns:
        ``[{"id", "text", "spk", "to"(, "t")}]`` tăng dần theo ``id``, tối đa
        :data:`MAX_KNOWN_CONTEXT_LINES` dòng gần lô nhất.
    """
    if not batch:
        return []
    wanted = {line.index for line in batch}
    low = min(wanted) - KNOWN_CONTEXT_PADDING
    high = max(wanted) + KNOWN_CONTEXT_PADDING
    labels = {
        cue.line_no: cue for cue in known_cues if cue.speaker or cue.addressee
    }
    ordered = sorted(wanted)
    candidates = [
        line
        for line in context_lines
        if low <= line.index <= high
        and line.index not in wanted
        and line.index in labels
    ]
    if len(candidates) > MAX_KNOWN_CONTEXT_LINES:
        candidates.sort(key=lambda line: _distance(line.index, ordered))
        candidates = candidates[:MAX_KNOWN_CONTEXT_LINES]
    items: list[dict[str, Any]] = []
    for line in sorted(candidates, key=lambda item: item.index):
        cue = labels[line.index]
        item: dict[str, Any] = {
            "id": line.index, "text": line.text, "spk": cue.speaker, "to": cue.addressee,
        }
        if clip_start_sec is not None:
            item["t"] = clip_timestamp(line.start_ms, clip_start_sec)
        items.append(item)
    return items


def _distance(index: int, ordered: list[int]) -> int:
    """Khoảng cách (số dòng) từ ``index`` tới dòng gần nhất của lô."""
    position = bisect.bisect_left(ordered, index)
    nearest = [
        abs(ordered[pos] - index) for pos in (position - 1, position)
        if 0 <= pos < len(ordered)
    ]
    return min(nearest) if nearest else 0
