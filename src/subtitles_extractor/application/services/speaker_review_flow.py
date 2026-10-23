"""[v3.23.630] Duyệt người nói NHANH: hàng đợi dòng nghi sai, gợi ý người nói, đoạn liền kề.

Log thật của phim 86 tập (3649 dòng): người dùng sửa tay người nói ở 861 dòng (23,6%), mỗi dòng
một lần chuột phải → menu 24 vai → chọn. Phần lõi (hàm thuần) cho hộp thoại duyệt nhanh:

* :func:`review_queue` — dòng cần soát xếp theo độ chắc chắn của bằng chứng (gọi tên / giới
  tính > chỉ biết giọng > chưa rõ), cùng nhóm thì theo thứ tự phim.
* :func:`candidate_speakers` — vài người nói khả dĩ nhất cho một dòng (người đang hội thoại
  quanh đó, người vừa chọn gần đây, người được nói tới), để bấm phím số thay vì lục menu.
* :func:`same_speaker_run` — đoạn liền kề cùng người nói (một lời thoại dài tách nhiều dòng
  phụ đề hay bị gán sai CẢ đoạn) để đổi một lần.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from subtitles_extractor.application.services.speaker_review import (
    REVIEW_GENDER,
    REVIEW_SELF_CALL,
    REVIEW_UNKNOWN,
    REVIEW_VOICE_ONLY,
    SpeakerIssue,
)

#: Khoảng lặng tối đa (giây) giữa hai dòng liền nhau vẫn coi là CÙNG một đoạn lời thoại.
RUN_MAX_GAP_S = 2.5
#: Số dòng mỗi phía xét khi gợi ý người nói.
CANDIDATE_WINDOW = 10
#: Số gợi ý tối đa (bấm phím 1–9).
MAX_CANDIDATES = 9

#: Bằng chứng càng chắc càng soát trước (số nhỏ = trước).
_PRIORITY = {
    REVIEW_SELF_CALL: 0,
    REVIEW_GENDER: 1,
    REVIEW_VOICE_ONLY: 2,
    REVIEW_UNKNOWN: 3,
}
_RECENT_WEIGHT = 0.6
_ADDRESSEE_WEIGHT = 1.5


@dataclass(frozen=True, slots=True)
class ReviewLine:
    """Một dòng TTS với đủ ngữ cảnh để quyết định người nói (hiện trong hộp thoại duyệt nhanh).

    Attributes:
        position: Vị trí dòng (0-based).
        start_sec: Mốc bắt đầu.
        end_sec: Mốc kết thúc.
        text: Lời đang đọc (bản dịch).
        source: Câu gốc tương ứng (rỗng nếu không ghép được).
        speaker: Người nói hiện tại (đã áp sửa tay; rỗng = chưa rõ).
        addressee: Người được nói tới (rỗng nếu không biết).
    """

    position: int
    start_sec: float
    end_sec: float
    text: str
    source: str = ""
    speaker: str = ""
    addressee: str = ""


@dataclass(frozen=True, slots=True)
class ReviewItem:
    """Một dòng trong hàng đợi duyệt: vị trí + vấn đề nghi ngờ."""

    position: int
    issue: SpeakerIssue


def review_queue(issues: Mapping[int, SpeakerIssue]) -> list[ReviewItem]:
    """Hàng đợi duyệt: bằng chứng chắc nhất trước, cùng nhóm thì theo thứ tự phim.

    Args:
        issues: ``{vị trí dòng: vấn đề}`` (từ ``speaker_review.review_speakers``).

    Returns:
        Danh sách :class:`ReviewItem` theo thứ tự nên duyệt.
    """
    items = [ReviewItem(position, issue) for position, issue in issues.items()]
    items.sort(key=lambda item: (_PRIORITY.get(item.issue.reason, 9), item.position))
    return items


def same_speaker_run(
    speakers: Sequence[str],
    starts: Sequence[float],
    ends: Sequence[float],
    position: int,
    max_gap_s: float = RUN_MAX_GAP_S,
) -> list[int]:
    """Đoạn liền kề CÙNG người nói chứa ``position`` (gồm cả ``position``).

    Mở rộng sang hai phía chừng nào dòng kề có cùng người nói (khác rỗng) và khoảng lặng giữa
    hai dòng không quá ``max_gap_s`` — qua khoảng lặng dài thường đã sang cảnh khác.

    Args:
        speakers: Người nói từng dòng (đã áp sửa tay).
        starts: Mốc bắt đầu từng dòng.
        ends: Mốc kết thúc từng dòng.
        position: Dòng xuất phát.
        max_gap_s: Khoảng lặng tối đa trong cùng đoạn.

    Returns:
        Vị trí các dòng của đoạn, tăng dần; ``[position]`` khi dòng không có người nói.
    """
    if not 0 <= position < len(speakers) or not speakers[position]:
        return [position] if 0 <= position < len(speakers) else []
    who = speakers[position]
    low = high = position
    while low > 0 and speakers[low - 1] == who and starts[low] - ends[low - 1] <= max_gap_s:
        low -= 1
    while (
        high + 1 < len(speakers) and speakers[high + 1] == who
        and starts[high + 1] - ends[high] <= max_gap_s
    ):
        high += 1
    return list(range(low, high + 1))


def candidate_speakers(
    speakers: Sequence[str],
    position: int,
    *,
    recent: Sequence[str] = (),
    addressee: str = "",
    window: int = CANDIDATE_WINDOW,
    limit: int = MAX_CANDIDATES,
) -> list[str]:
    """Người nói khả dĩ nhất cho dòng ``position``, khả dĩ nhất trước.

    Điểm: mỗi lần một người xuất hiện quanh dòng cộng ``1/khoảng cách``; người vừa được chọn
    gần đây (``recent``, mới nhất trước) và người được nói tới (``addressee`` — người nghe
    thường là người đáp lời tiếp theo) được cộng thêm. Dòng đang xét tính theo chính người
    nói hiện tại của nó nên luôn có mặt nếu có người nói.

    Args:
        speakers: Người nói từng dòng (đã áp sửa tay; rỗng = chưa rõ).
        position: Dòng cần gợi ý.
        recent: Người nói người dùng chọn gần đây, mới nhất trước.
        addressee: Người được nói tới ở dòng này (rỗng nếu không biết).
        window: Số dòng mỗi phía xét.
        limit: Số gợi ý tối đa.

    Returns:
        Danh sách tên vai, không trùng, không rỗng.
    """
    scores: dict[str, float] = {}
    order: dict[str, int] = {}

    def add(name: str, weight: float) -> None:
        if not name:
            return
        scores[name] = scores.get(name, 0.0) + weight
        order.setdefault(name, len(order))

    if 0 <= position < len(speakers):
        add(speakers[position], 1.0)
    for distance in range(1, window + 1):
        for index in (position - distance, position + distance):
            if 0 <= index < len(speakers):
                add(speakers[index], 1.0 / distance)
    for rank, name in enumerate(recent):
        add(name, _RECENT_WEIGHT / (1 + rank))
    add(addressee, _ADDRESSEE_WEIGHT)
    ranked = sorted(scores, key=lambda name: (-scores[name], order[name]))
    return ranked[:limit]


__all__ = [
    "CANDIDATE_WINDOW", "MAX_CANDIDATES", "RUN_MAX_GAP_S", "ReviewItem", "ReviewLine",
    "candidate_speakers", "review_queue", "same_speaker_run",
]
