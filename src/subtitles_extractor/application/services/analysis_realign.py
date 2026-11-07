"""[v3.23.668] Giữ phân tích ngữ cảnh khi phụ đề nguồn đổi một ít (ReOCR thêm/bớt/sửa dòng).

Trước bản này phân tích lưu theo mã băm của TOÀN BỘ chữ phụ đề: sửa một ký tự là mất cả bảng
nhân vật, tóm tắt, thuật ngữ lẫn người nói → phân tích lại từ đầu (tải lại video, tốn lượt của
model hạn mức thấp). Bảng nhân vật / tóm tắt / thuật ngữ không phụ thuộc từng dòng; chỉ người nói
(Visual Cues) gắn theo SỐ DÒNG.

Module (hàm thuần) này khớp dòng mới ↔ dòng cũ theo chữ + thời gian, chuyển người nói của dòng
khớp sang số dòng mới, và trả các dòng mới/khác hẳn để chỉ hỏi lại người nói cho đúng những dòng
đó. Mô phỏng ReOCR (thêm/xoá/tách/gộp/sửa chữ, 10 lần ngẫu nhiên mỗi mức) trên tập 1-2 phim mẫu:
giữ theo số dòng cũ còn 59-60% người nói đúng sau 8 chỗ sửa; căn dòng giữ nguyên 73% / 78% như
trước khi sửa, chỉ còn 1-3 dòng phải hỏi lại.
"""

from __future__ import annotations

import bisect
import difflib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from subtitles_extractor.application.services.visual_cue_serializer import (
    deserialize_visual_cues,
    serialize_visual_cues,
)

#: Điểm khớp tối thiểu (½ độ giống chữ + ½ độ chồng thời gian) để coi là cùng một dòng.
MIN_MATCH_SCORE = 0.5
#: Độ giống CHỮ tối thiểu: dòng chèn mới ("啊？") trùng thời gian với dòng cũ nhưng là câu khác,
#: có thể của người khác → phải hỏi lại người nói, không mượn nhãn của dòng cũ.
MIN_TEXT_SIMILARITY = 0.25
#: Tỉ lệ dòng mới khớp được tối thiểu để còn dùng lại phân tích (dưới mức này coi là phụ đề khác).
MIN_REUSE_RATIO = 0.6
#: Dòng giống hệt chữ nhưng lệch giờ quá mức này (ms) không coi là cùng dòng: câu ngắn lặp lại
#: ("嗯", "好", "走吧") ở chỗ khác trong phim làm neo sai và kéo lệch các dòng quanh nó.
MAX_EQUAL_SHIFT_MS = 3000
#: Độ dài dòng tối đa giả định khi tìm dòng cũ chồng thời gian (ms).
_MAX_LINE_MS = 30_000


@dataclass(frozen=True, slots=True)
class SourceLine:
    """Dòng phụ đề nguồn tối giản lưu kèm phân tích (để so khi phụ đề đổi)."""

    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class Realignment:
    """Kết quả khớp dòng mới (1-based) ↔ dòng cũ (1-based).

    Attributes:
        mapping: ``số dòng mới → số dòng cũ`` cho các dòng khớp được.
        unmatched: Số dòng mới KHÔNG khớp dòng cũ nào (dòng mới thật / sửa khác hẳn).
        changed: Số dòng mới khớp nhưng chữ đã đổi.
        total: Số dòng mới.
    """

    mapping: dict[int, int] = field(default_factory=dict)
    unmatched: tuple[int, ...] = ()
    changed: tuple[int, ...] = ()
    total: int = 0
    old_total: int = 0

    @property
    def matched_ratio(self) -> float:
        """Tỉ lệ dòng mới khớp được."""
        return len(self.mapping) / self.total if self.total else 0.0

    @property
    def old_coverage(self) -> float:
        """Tỉ lệ dòng CŨ còn được dòng mới nào đó khớp tới."""
        return len(set(self.mapping.values())) / self.old_total if self.old_total else 0.0

    @property
    def removed(self) -> int:
        """Số dòng cũ không còn dòng mới nào khớp (đã xoá / gộp mất)."""
        return self.old_total - len(set(self.mapping.values()))

    @property
    def reusable(self) -> bool:
        """Đủ giống để dùng lại phân tích cũ — xét CẢ HAI phía.

        Chỉ xét phía mới thì một đoạn trích 60 dòng của phim 1000 dòng vẫn "khớp 100%" và bị lưu đè
        lên phân tích cả phim (mất 940 dòng người nói).
        """
        return (
            self.total > 0 and self.matched_ratio >= MIN_REUSE_RATIO
            and self.old_coverage >= MIN_REUSE_RATIO
        )

    @property
    def identical(self) -> bool:
        """Không có dòng nào đổi."""
        return (
            not self.unmatched and not self.changed
            and all(new == old for new, old in self.mapping.items())
        )


def encode_lines(lines: Sequence[SourceLine]) -> str:
    """Chuỗi JSON gọn ``[[start_ms, end_ms, text], …]`` để lưu trong phiên dịch."""
    return json.dumps([[ln.start_ms, ln.end_ms, ln.text] for ln in lines], ensure_ascii=False,
                      separators=(",", ":"))


def decode_lines(raw: str | None) -> list[SourceLine]:
    """Đọc lại chuỗi của :func:`encode_lines`; hỏng/rỗng → danh sách rỗng."""
    try:
        data: Any = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    out: list[SourceLine] = []
    if not isinstance(data, list):
        return []
    for item in data:
        if isinstance(item, list) and len(item) == 3:
            try:
                out.append(SourceLine(int(item[0]), int(item[1]), str(item[2])))
            except (TypeError, ValueError):
                return []
    return out


def _score(old: SourceLine, new: SourceLine) -> float:
    overlap = min(old.end_ms, new.end_ms) - max(old.start_ms, new.start_ms)
    if overlap <= 0:
        return 0.0
    cover = overlap / max(1, new.end_ms - new.start_ms)
    similar = difflib.SequenceMatcher(None, old.text, new.text).ratio()
    if similar < MIN_TEXT_SIMILARITY:
        return 0.0
    return 0.5 * similar + 0.5 * min(1.0, cover)


def realign(old: Sequence[SourceLine], new: Sequence[SourceLine]) -> Realignment:
    """Khớp dòng mới với dòng cũ (hàm thuần).

    Bước 1: các khối chữ GIỐNG HỆT (difflib, giữ thứ tự). Bước 2: dòng trong vùng đổi khớp với
    dòng cũ cùng vùng (± một dòng) có điểm ``½·độ giống chữ + ½·độ chồng thời gian`` cao nhất,
    đạt :data:`MIN_MATCH_SCORE` — bắt được tách/gộp dòng, sửa vài chữ, dịch mốc thời gian.

    Args:
        old: Dòng lúc phân tích.
        new: Dòng hiện tại.

    Returns:
        :class:`Realignment` (số dòng 1-based).
    """
    matcher = difflib.SequenceMatcher(
        None, [ln.text for ln in old], [ln.text for ln in new], autojunk=False
    )
    mapping: dict[int, int] = {}
    pending: list[int] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                o, n = old[i1 + k], new[j1 + k]
                if abs(o.start_ms - n.start_ms) <= MAX_EQUAL_SHIFT_MS:
                    mapping[j1 + k + 1] = i1 + k + 1
                else:
                    pending.append(j1 + k)     # giống chữ nhưng ở chỗ khác → tìm theo thời gian
            continue
        pending.extend(range(j1, j2))
    # Khớp lỏng: chỉ xét các dòng cũ CHỒNG thời gian với dòng mới (tìm nhị phân theo mốc bắt đầu)
    # — vùng đổi lớn (vd chuẩn hoá dấu câu cả phim) không còn là O(n²) trên luồng giao diện.
    order = sorted(range(len(old)), key=lambda i: old[i].start_ms)
    starts = [old[i].start_ms for i in order]
    unmatched: list[int] = []
    changed: list[int] = []
    for j in pending:
        line = new[j]
        lo = bisect.bisect_left(starts, line.start_ms - _MAX_LINE_MS)
        hi = bisect.bisect_left(starts, line.end_ms)
        best, best_score = -1, 0.0
        for pos in range(lo, hi):
            i = order[pos]
            score = _score(old[i], line)
            if score > best_score:
                best, best_score = i, score
        if best >= 0 and best_score >= MIN_MATCH_SCORE:
            mapping[j + 1] = best + 1
            changed.append(j + 1)
        else:
            unmatched.append(j + 1)
    return Realignment(mapping=dict(sorted(mapping.items())), unmatched=tuple(unmatched),
                       changed=tuple(sorted(changed)), total=len(new), old_total=len(old))


def remap_cues(raw: str | None, alignment: Realignment) -> str:
    """Chuyển Visual Cues sang số dòng mới: dòng khớp giữ nhãn của dòng cũ, dòng không khớp bỏ
    trống (để bước "Bổ sung người nói" / khâu dịch hỏi lại đúng các dòng đó)."""
    by_old = {cue.line_no: cue for cue in deserialize_visual_cues(raw)}
    out = []
    for new_no in sorted(alignment.mapping):
        cue = by_old.get(alignment.mapping[new_no])
        if cue is not None:
            out.append(replace(cue, line_no=new_no))
    return serialize_visual_cues(out) if out else ""


__all__ = [
    "MIN_MATCH_SCORE",
    "MIN_REUSE_RATIO",
    "MIN_TEXT_SIMILARITY",
    "Realignment",
    "SourceLine",
    "decode_lines",
    "encode_lines",
    "realign",
    "remap_cues",
]
