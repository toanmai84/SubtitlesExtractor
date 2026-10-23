"""[v3.23.566] Ghép tóm tắt phim = tóm tắt TOÀN CẢNH + diễn biến CHI TIẾT theo từng đoạn.

Chuẩn đoán thật v3.23.565 (phân tích bằng gemini-3.8-flash): mỗi đoạn video, model làm
đúng chỉ thị "hợp nhất thành bản cập nhật đầy đủ tới thời điểm này" — tức trả lại một bản
tóm tắt TOÀN PHIM từ đầu. Bộ gộp cũ lại NỐI các bản đó với nhau, nên tóm tắt 20 049 ký tự
có 7/13 đoạn văn mở đầu y hệt "Thế giới quan đan xen giữa cuộc sống nông thôn…" (kể lại
phần thiết lập), chạm trần độ dài và bị cắt MẤT phần giữa phim — đúng phần các lô dịch ở
giữa phim cần nhất.

Cách mới tách hai việc: ``overview`` là bản toàn cảnh NGẮN, luôn được THAY bằng bản mới
nhất; ``segment_summary`` là diễn biến chi tiết CHỈ của đoạn đang xem, lưu riêng theo mốc
thời gian và ghép theo thứ tự thời gian — không trùng lặp, không phụ thuộc thứ tự chạy
(chạy tiếp các đoạn thiếu hôm sau vẫn đúng thứ tự).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Trần độ dài bản ghép (≈ trần cũ của tóm tắt tích luỹ).
MAX_COMPOSED_OVERVIEW_CHARS = 20_000

#: Phần toàn cảnh chiếm tối đa chừng này — còn lại dành cho diễn biến theo đoạn.
MAX_GLOBAL_OVERVIEW_CHARS = 5_000

TIMELINE_HEADER = "DIỄN BIẾN THEO THỜI GIAN:"

_SENTENCE_END_RE = re.compile(r"[.!?…](?=\s|$)")

#: Gạch nối khoảng thời gian (en dash).
_RANGE_DASH = "\u2013"


@dataclass(frozen=True)
class SegmentSummary:
    """Diễn biến chi tiết của MỘT đoạn video.

    Attributes:
        start_sec: Mốc bắt đầu đoạn (giây).
        end_sec: Mốc kết thúc đoạn (giây).
        text: Tóm tắt diễn biến trong đoạn.
    """

    start_sec: float
    end_sec: float
    text: str


def _clock(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def trim_at_sentence(text: str, max_chars: int) -> str:
    """Cắt ``text`` về ≤ ``max_chars`` ký tự, ưu tiên dừng ở cuối một câu (hàm thuần).

    Args:
        text: Văn bản.
        max_chars: Độ dài tối đa.

    Returns:
        Văn bản đã cắt; thêm "…" khi phải cắt giữa câu.
    """
    clean = " ".join((text or "").split())
    if len(clean) <= max_chars:
        return clean
    window = clean[:max_chars]
    ends = [match.end() for match in _SENTENCE_END_RE.finditer(window)]
    if ends and ends[-1] >= max_chars // 3:
        return window[: ends[-1]].rstrip()
    return window[: max(0, max_chars - 1)].rstrip() + "…"


def compose_overview(
    global_overview: str,
    segments: list[SegmentSummary],
    max_chars: int = MAX_COMPOSED_OVERVIEW_CHARS,
) -> str:
    """Ghép toàn cảnh + diễn biến theo thời gian, không vượt ``max_chars`` (hàm thuần).

    Args:
        global_overview: Bản toàn cảnh mới nhất.
        segments: Diễn biến từng đoạn (thứ tự bất kỳ; đoạn rỗng bị bỏ).
        max_chars: Trần độ dài kết quả.

    Returns:
        Tóm tắt đã ghép. Không có đoạn nào thì trả nguyên bản toàn cảnh (đã cắt trần).
    """
    ordered = sorted(
        (seg for seg in segments if seg.text.strip()), key=lambda seg: seg.start_sec
    )
    head = trim_at_sentence(global_overview, min(MAX_GLOBAL_OVERVIEW_CHARS, max_chars))
    if not ordered:
        return head
    labels = [
        f"[{_clock(seg.start_sec)}{_RANGE_DASH}{_clock(seg.end_sec)}] " for seg in ordered
    ]
    fixed = len(head) + len(TIMELINE_HEADER) + 4 + sum(len(label) + 1 for label in labels)
    per_segment = max(80, (max_chars - fixed) // len(ordered))
    body = "\n".join(
        label + trim_at_sentence(seg.text, per_segment)
        for label, seg in zip(labels, ordered, strict=True)
    )
    timeline = f"{TIMELINE_HEADER}\n{body}"
    return f"{head}\n\n{timeline}" if head else timeline


__all__ = [
    "MAX_COMPOSED_OVERVIEW_CHARS",
    "MAX_GLOBAL_OVERVIEW_CHARS",
    "TIMELINE_HEADER",
    "SegmentSummary",
    "compose_overview",
    "trim_at_sentence",
]
