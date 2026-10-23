"""[v3.23.630] Phiên duyệt người nói nhanh (hàm thuần, không Qt): con trỏ, lựa chọn, hoàn tác.

Hộp thoại ``SpeakerReviewDialog`` chỉ vẽ; mọi quyết định nằm ở đây để kiểm thử được:

* đi qua hàng đợi dòng nghi sai (hoặc mọi dòng từ một vị trí);
* chọn người nói cho dòng — hoặc cả đoạn liền kề cùng người nói — rồi tự sang dòng kế;
* nhớ người vừa chọn (gợi ý lần sau), hoàn tác từng lựa chọn;
* trả về ``{vị trí: người nói}`` CHỈ cho dòng thật sự đổi, để áp một lượt khi đóng hộp thoại.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from subtitles_extractor.application.services.speaker_review_flow import (
    RUN_MAX_GAP_S,
    ReviewLine,
    candidate_speakers,
    same_speaker_run,
)

#: Số người vừa chọn nhớ để gợi ý.
RECENT_LIMIT = 6


@dataclass(frozen=True, slots=True)
class _Choice:
    """Một lựa chọn đã làm: để hoàn tác."""

    cursor: int
    previous: tuple[tuple[int, str], ...]
    recent: tuple[str, ...]


class SpeakerReviewSession:
    """Trạng thái duyệt người nói; lựa chọn chỉ nằm trong phiên cho tới khi áp dụng."""

    def __init__(self, lines: Sequence[ReviewLine], queue: Sequence[int]) -> None:
        """Dựng phiên.

        Args:
            lines: MỌI dòng, đúng thứ tự vị trí (``lines[i].position == i``).
            queue: Vị trí các dòng cần duyệt, theo thứ tự nên duyệt.
        """
        self._lines = list(lines)
        self._original = [line.speaker for line in self._lines]
        self._speakers = list(self._original)
        self._flagged = [p for p in queue if 0 <= p < len(self._lines)]
        self._queue = list(self._flagged)
        self._cursor = 0
        self._all_lines = False
        self._recent: list[str] = []
        self._last: int | None = None      # dòng cuối cùng đã đi qua
        self._history: list[_Choice] = []
        self._reviewed: set[int] = set()

    # ── Trạng thái ────────────────────────────────────────────────────────────

    @property
    def total(self) -> int:
        """Số dòng trong hàng đợi đang duyệt."""
        return len(self._queue)

    @property
    def finished(self) -> bool:
        """Đã đi hết hàng đợi (hoặc hàng đợi rỗng)."""
        return self._cursor >= len(self._queue)

    @property
    def position(self) -> int | None:
        """Vị trí dòng đang duyệt; ``None`` khi đã hết."""
        return None if self.finished else self._queue[self._cursor]

    @property
    def progress(self) -> tuple[int, int]:
        """``(số thứ tự dòng đang duyệt 1-based, tổng)``."""
        return min(self._cursor + 1, len(self._queue)), len(self._queue)

    @property
    def all_lines(self) -> bool:
        """Đang duyệt MỌI dòng (không chỉ dòng nghi sai)."""
        return self._all_lines

    @property
    def speakers(self) -> list[str]:
        """Người nói từng dòng sau các lựa chọn trong phiên."""
        return list(self._speakers)

    @property
    def reviewed_count(self) -> int:
        """Số dòng đã xem qua (chọn hoặc giữ nguyên)."""
        return len(self._reviewed)

    @property
    def can_undo(self) -> bool:
        return bool(self._history)

    def changes(self) -> dict[int, str]:
        """``{vị trí: người nói mới}`` của các dòng khác với lúc mở hộp thoại."""
        return {
            position: speaker
            for position, (speaker, before) in enumerate(
                zip(self._speakers, self._original, strict=True)
            )
            if speaker != before
        }

    def line(self, position: int) -> ReviewLine:
        """Dòng ``position`` với người nói HIỆN TẠI của phiên."""
        return replace(self._lines[position], speaker=self._speakers[position])

    def context(self, radius: int = 3) -> list[ReviewLine]:
        """Các dòng quanh dòng đang duyệt (gồm nó), người nói theo phiên."""
        current = self.position
        if current is None:
            return []
        low, high = max(0, current - radius), min(len(self._lines), current + radius + 1)
        return [self.line(position) for position in range(low, high)]

    # ── Gợi ý ─────────────────────────────────────────────────────────────────

    def candidates(self) -> list[str]:
        """Người nói khả dĩ nhất cho dòng đang duyệt (bấm phím 1–9)."""
        current = self.position
        if current is None:
            return []
        return candidate_speakers(
            self._speakers, current, recent=self._recent,
            addressee=self._lines[current].addressee,
        )

    def run(self) -> list[int]:
        """Đoạn liền kề cùng người nói chứa dòng đang duyệt."""
        current = self.position
        if current is None:
            return []
        return same_speaker_run(
            self._speakers, [line.start_sec for line in self._lines],
            [line.end_sec for line in self._lines], current, RUN_MAX_GAP_S,
        )

    # ── Điều khiển ────────────────────────────────────────────────────────────

    def set_scope(self, all_lines: bool) -> None:
        """Chuyển giữa "chỉ dòng nghi sai" và "mọi dòng" (giữ vị trí đang duyệt)."""
        if all_lines == self._all_lines:
            return
        anchor = self.position
        if anchor is None:   # đã hết (hoặc hàng đợi rỗng): tiếp từ dòng sau dòng cuối đã xem
            anchor = 0 if self._last is None else self._last + 1
        self._all_lines = all_lines
        self._queue = list(range(len(self._lines))) if all_lines else list(self._flagged)
        if anchor in self._queue:
            self._cursor = self._queue.index(anchor)
        else:  # thoát chế độ "mọi dòng" ở một dòng không nghi sai: sang dòng nghi sai kế
            later = [i for i, p in enumerate(self._queue) if p > anchor]
            self._cursor = later[0] if later else len(self._queue)

    def start_at(self, position: int) -> None:
        """Đặt con trỏ vào dòng ``position`` (chế độ "mọi dòng" nếu dòng không trong hàng đợi)."""
        if position in self._queue:
            self._cursor = self._queue.index(position)
            return
        if 0 <= position < len(self._lines):
            self.set_scope(True)
            self._cursor = position

    def next(self) -> bool:
        """Sang dòng kế (đánh dấu dòng này đã xem). ``False`` khi đã hết."""
        current = self.position
        if current is None:
            return False
        self._reviewed.add(current)
        self._last = current
        self._cursor += 1
        return not self.finished

    def previous(self) -> bool:
        """Về dòng trước trong hàng đợi. ``False`` khi đang ở dòng đầu."""
        if self._cursor <= 0:
            return False
        self._cursor -= 1
        return True

    def choose(self, speaker: str, whole_run: bool = False, advance: bool = True) -> list[int]:
        """Chọn người nói cho dòng đang duyệt (hoặc cả đoạn liền kề cùng người nói).

        Args:
            speaker: Tên vai; rỗng = giọng chung.
            whole_run: Áp cho cả đoạn liền kề cùng người nói hiện tại.
            advance: Tự sang dòng kế sau khi chọn.

        Returns:
            Vị trí các dòng thật sự đổi người nói.
        """
        current = self.position
        if current is None:
            return []
        targets = self.run() if whole_run else [current]
        changed = [p for p in targets if self._speakers[p] != speaker]
        self._history.append(_Choice(
            self._cursor, tuple((p, self._speakers[p]) for p in changed), tuple(self._recent),
        ))
        for position in changed:
            self._speakers[position] = speaker
        self._reviewed.update(targets)
        if speaker:
            self._recent = [speaker, *(r for r in self._recent if r != speaker)][:RECENT_LIMIT]
        if advance:
            self.next()
        return changed

    def undo(self) -> bool:
        """Hoàn tác lựa chọn gần nhất và quay lại dòng đó. ``False`` nếu chưa chọn gì."""
        if not self._history:
            return False
        choice = self._history.pop()
        for position, speaker in choice.previous:
            self._speakers[position] = speaker
        self._recent = list(choice.recent)
        self._cursor = min(choice.cursor, max(0, len(self._queue) - 1))
        if self.position is not None:
            self._reviewed.discard(self.position)
        return True


__all__ = ["RECENT_LIMIT", "SpeakerReviewSession"]
