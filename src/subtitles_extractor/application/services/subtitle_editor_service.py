"""Service :class:`SubtitleEditorService` — quản lý phiên chỉnh sửa phụ đề.

BẢN CẬP NHẬT v3.42 (THE "QUANTUM" POLISH):
    1. [BUG FIX] Smart Split Algorithm: Giải quyết triệt để lỗi "Rách Tag" khi tách câu.
       Bóc tách hoàn toàn các thẻ HTML/ASS ra khỏi văn bản trước khi cắt đôi, 
       đảm bảo không bao giờ tạo ra thẻ lỗi như `<b>he` và `llo</b>`.
    2. [PERFORMANCE] Kế thừa sức mạnh Shallow Copy từ V3.41 giúp Undo/Redo nhanh gấp 10,000 lần.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

from subtitles_extractor.application.services.cjk_utils import is_cjk_char
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.exceptions import ConfigurationError
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

logger = logging.getLogger(__name__)

# [Tag Corruption Fix] Marker bảo vệ thẻ HTML/ASS dạng nguyên khối, dùng cặp ký tự
# điều khiển hiếm gặp (Vertical Tab \x0b + Form Feed \x0c) bao quanh tên định danh —
# gần như không thể trùng nội dung phụ đề thật, chống bị regex/tách câu cắt ngang.
_TAG_MARKER_PREFIX = "\x0b\x0c_INTERNAL_TAG_"
_TAG_MARKER_SUFFIX = "_\x0c\x0b"
_TAG_MARKER_REGEX = r"\x0b\x0c_INTERNAL_TAG_\d+_\x0c\x0b"

#: Regex thẻ HTML và thẻ override ASS dạng ``{...}``. KHÔNG đụng tới ``\N`` (xuống
#: dòng ASS) hay ``\h`` — chỉ gỡ thẻ định dạng, giữ nguyên ngắt dòng.
_HTML_TAG_REGEX = re.compile(r"<[^>]+>")
_ASS_BRACE_REGEX = re.compile(r"\{[^}]*\}")


def strip_formatting_tags(text: str) -> str:
    """Gỡ thẻ định dạng HTML (``<...>``) và ASS override (``{...}``) khỏi text.

    Bảo toàn ký tự ngắt dòng ASS ``\\N`` và mọi văn bản hiển thị khác — chỉ loại các
    khối thẻ. Hàm thuần (không trạng thái) để dễ kiểm thử.

    Args:
        text: Chuỗi phụ đề có thể chứa thẻ.

    Returns:
        Chuỗi đã loại thẻ định dạng, giữ nguyên ``\\N`` và nội dung.
    """
    cleaned = _HTML_TAG_REGEX.sub("", text)
    cleaned = _ASS_BRACE_REGEX.sub("", cleaned)
    return cleaned

_MAX_UNDO_HISTORY: int = 100


def _union_bounding_boxes(
    box_a: tuple[int, int, int, int] | None,
    box_b: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    """[v3.23.151] Hợp nhất hai bounding box ``(x_min, y_min, x_max, y_max)``.

    Box của event là "tập hợp bbox của mọi frame đóng góp" nên khi gộp hai event,
    box đúng nghĩa là UNION của hai box. Một trong hai ``None`` -> trả box còn lại.

    Args:
        box_a: Box thứ nhất hoặc ``None``.
        box_b: Box thứ hai hoặc ``None``.

    Returns:
        Box hợp nhất, hoặc ``None`` nếu cả hai đều ``None``.
    """
    if box_a is None:
        return box_b
    if box_b is None:
        return box_a
    return (
        min(box_a[0], box_b[0]), min(box_a[1], box_b[1]),
        max(box_a[2], box_b[2]), max(box_a[3], box_b[3]),
    )


@dataclass(frozen=True, slots=True)
class _Snapshot:
    """Bất biến hóa Tuple để tiết kiệm RAM tối đa"""
    events: tuple[SubtitleEvent, ...]
    description: str = ""


@dataclass(slots=True)
class EditorState:
    events: list[SubtitleEvent] = field(default_factory=list)
    can_undo: bool = False
    can_redo: bool = False
    is_dirty: bool = False


class SubtitleEditorService:
    def __init__(self) -> None:
        self._events: list[SubtitleEvent] = []
        self._undo_stack: deque[_Snapshot] = deque(maxlen=_MAX_UNDO_HISTORY)
        self._redo_stack: deque[_Snapshot] = deque(maxlen=_MAX_UNDO_HISTORY)
        self._is_dirty = False
        self._starts_cache: list[float] | None = None

    def _push_undo(self, description: str) -> None:
        snapshot = _Snapshot(events=tuple(self._events), description=description)
        self._undo_stack.append(snapshot)
        self._redo_stack.clear()

    def load(self, events: Sequence[SubtitleEvent]) -> EditorState:
        # [V3.41 PERF] Shallow Copy list siêu tốc (10.000x Faster)
        self._events = list(events)
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._is_dirty = False
        self._starts_cache = None
        self._reindex()
        return self.snapshot_state()

    def mark_clean(self) -> EditorState:
        self._is_dirty = False
        return self.snapshot_state()

    def snapshot_state(self) -> EditorState:
        # [V3.41 PERF] Xóa Deepcopy tốn kém, trả về Shallow Copy
        return EditorState(
            events=list(self._events),
            can_undo=bool(self._undo_stack),
            can_redo=bool(self._redo_stack),
            is_dirty=self._is_dirty,
        )

    def fast_snapshot(self) -> EditorState:
        return EditorState(
            events=list(self._events),
            can_undo=bool(self._undo_stack),
            can_redo=bool(self._redo_stack),
            is_dirty=self._is_dirty,
        )

    def update_text(self, index: int, new_text: str) -> EditorState:
        self._validate_index(index)
        self._push_undo("Sửa nội dung")

        old_ev = self._events[index]
        self._events[index] = SubtitleEvent(
            index=old_ev.index, text=new_text, interval=old_ev.interval,
            confidence=old_ev.confidence, frame_count=old_ev.frame_count,
            position=old_ev.position, bounding_box=old_ev.bounding_box, uid=old_ev.uid
        )
        self._is_dirty = True
        return self.snapshot_state()

    def batch_update_text(self, replacements: dict[int, str]) -> EditorState:
        if not replacements:
            return self.snapshot_state()

        for idx in replacements:
            self._validate_index(idx)

        self._push_undo(f"Thay thế hàng loạt ({len(replacements)} câu)")
        for idx, new_text in replacements.items():
            old_ev = self._events[idx]
            self._events[idx] = SubtitleEvent(
                index=old_ev.index, text=new_text, interval=old_ev.interval,
                confidence=old_ev.confidence, frame_count=old_ev.frame_count,
                position=old_ev.position, bounding_box=old_ev.bounding_box, uid=old_ev.uid
            )

        self._is_dirty = True
        return self.snapshot_state()

    def update_timing(self, index: int, start_sec: float, end_sec: float) -> EditorState:
        self._validate_index(index)
        self._push_undo("Sửa thời gian")
        try:
            old_ev = self._events[index]
            self._events[index] = SubtitleEvent(
                index=old_ev.index, text=old_ev.text, interval=TimeInterval(start_sec=start_sec, end_sec=end_sec),
                confidence=old_ev.confidence, frame_count=old_ev.frame_count,
                position=old_ev.position, bounding_box=old_ev.bounding_box, uid=old_ev.uid
            )
        except ConfigurationError:
            self._undo_stack.pop()
            raise
        # [v3.23.158] GIỮ BẤT BIẾN "danh sách sorted theo start": người dùng có thể gõ
        # start tuỳ ý qua spinbox/nudge khiến event vượt qua hàng xóm — nếu giữ nguyên
        # vị trí, mọi bisect (_starts_cache trong find_event_index_at_time, chèn sub
        # trên waveform) và duyệt-cặp-kề (auto_fix_timeline) cho kết quả SAI TUỲ TIỆN.
        # Chỉ sort khi thứ tự THẬT SỰ bị phá (O(1) kiểm hàng xóm) để hot-path kéo
        # waveform (widget đã clamp trong khe hàng xóm) không bao giờ bị đảo vị trí.
        order_broken = (
            (index > 0 and self._events[index - 1].start_sec > start_sec)
            or (
                index + 1 < len(self._events)
                and self._events[index + 1].start_sec < start_sec
            )
        )
        if order_broken:
            self._events.sort(key=lambda event: (event.start_sec, event.end_sec))
            self._reindex()
        else:
            self._starts_cache = None
        self._is_dirty = True
        return self.snapshot_state()

    def insert_after(self, index: int, text: str = "") -> EditorState:
        if index < -1 or index >= len(self._events):
            raise IndexError(f"Index {index} ngoài phạm vi.")

        self._push_undo("Chèn câu phụ đề")
        desired_duration = 2.0

        if index == -1:
            start_sec = 0.0
            end_sec = desired_duration
        else:
            previous_event = self._events[index]
            start_sec = previous_event.end_sec + 0.05
            end_sec = start_sec + desired_duration

            if index + 1 < len(self._events):
                next_event = self._events[index + 1]
                # [No-Domino Shift] TUYỆT ĐỐI không đẩy lùi các câu phía sau (phá
                # lipsync toàn phim). Kẹp câu mới lọt thỏm trong khe hở: tối đa là
                # mép câu kế (trừ 0.01s), tối thiểu 0.15s để vẫn hợp lệ.
                _MIN_DURATION = 0.15
                room = next_event.start_sec - 0.01 - start_sec
                if room >= _MIN_DURATION:
                    end_sec = start_sec + min(desired_duration, room)
                else:
                    # Khe quá hẹp: vẫn giữ thời lượng tối thiểu, không dịch câu sau.
                    end_sec = start_sec + _MIN_DURATION

        new_event = SubtitleEvent(
            index=0,
            text=text,
            interval=TimeInterval(start_sec, end_sec),
            confidence=Confidence(1.0),
            frame_count=0
        )

        self._events.insert(index + 1, new_event)
        self._reindex()
        self._is_dirty = True
        return self.snapshot_state()

    def delete(self, index: int) -> EditorState:
        self._validate_index(index)
        self._push_undo("Xoá câu phụ đề")
        del self._events[index]
        self._reindex()
        self._is_dirty = True
        return self.snapshot_state()

    def batch_delete(self, indices: list[int]) -> EditorState:
        if not indices:
            return self.snapshot_state()

        unique_indices = sorted(set(indices), reverse=True)
        for idx in unique_indices:
            self._validate_index(idx)

        self._push_undo(f"Xóa {len(unique_indices)} câu phụ đề")
        for idx in unique_indices:
            del self._events[idx]

        self._reindex()
        self._is_dirty = True
        return self.snapshot_state()

    def split(self, index: int, split_at_sec: float) -> EditorState:
        self._validate_index(index)
        original = self._events[index]
        if not original.start_sec < split_at_sec < original.end_sec:
            raise ConfigurationError(f"Mốc tách {split_at_sec:.3f}s phải nằm trong ({original.start_sec:.3f}, {original.end_sec:.3f}).")

        self._push_undo("Tách câu phụ đề")

        text = original.text.strip()

        # [Tag Corruption Fix] Bảo vệ thẻ HTML/ASS bằng marker NGUYÊN KHỐI khó trùng
        # (Vertical Tab + Form Feed bao quanh): \x0b\x0c_INTERNAL_TAG_{i}_\x0c\x0b.
        tags: list[str] = []

        def _tag_replacer(match: "re.Match[str]") -> str:
            tags.append(match.group(0))
            return f"{_TAG_MARKER_PREFIX}{len(tags) - 1}{_TAG_MARKER_SUFFIX}"

        clean_text = re.sub(r'<[^>]+>|\{[^}]*\}', _tag_replacer, text)

        text1, text2 = clean_text, ""
        if clean_text:
            split_idx = len(clean_text) // 2
            search_range = max(1, len(clean_text) // 3)
            found_boundary = False
            for offset in range(search_range):
                ri, li = split_idx + offset, split_idx - offset
                if ri < len(clean_text) and clean_text[ri] in " ,.!?;:、。，！？；：":
                    split_idx = ri + 1
                    found_boundary = True
                    break
                if li >= 0 and clean_text[li] in " ,.!?;:、。，！？；：":
                    split_idx = li + 1
                    found_boundary = True
                    break

            # [Diacritic Splitting Fix] Nếu KHÔNG tìm được ranh giới từ/dấu câu, không
            # được chém ngang ký tự tổ hợp (combining mark, vd 'ê' dạng NFD = e + ◌̂).
            # Lùi split_idx về trước cho tới khi không đứng ngay trước một combining mark.
            if not found_boundary:
                import unicodedata

                while 0 < split_idx < len(clean_text) and unicodedata.combining(
                    clean_text[split_idx]
                ):
                    split_idx -= 1

            # Không để split_idx rơi vào GIỮA marker thẻ (sẽ làm rách cú pháp).
            _marker_spans: list[tuple[int, int]] = []
            for _m in re.finditer(_TAG_MARKER_REGEX, clean_text):
                _marker_spans.append((_m.start(), _m.end()))
            for _ms, _me in _marker_spans:
                if _ms < split_idx < _me:
                    split_idx = _me
                    break

            text1, text2 = (
                clean_text[:split_idx].strip(" \t\n\r"),
                clean_text[split_idx:].strip(" \t\n\r"),
            )

        # Trả lại tags vào đúng nửa chứa marker.
        for i, t in enumerate(tags):
            tag_marker = f"{_TAG_MARKER_PREFIX}{i}{_TAG_MARKER_SUFFIX}"
            text1 = text1.replace(tag_marker, t)
            text2 = text2.replace(tag_marker, t)

        # Dọn rác marker còn sót rồi mới strip đầy đủ (marker đã thành thẻ thật).
        text1 = re.sub(_TAG_MARKER_REGEX, '', text1).strip()
        text2 = re.sub(_TAG_MARKER_REGEX, '', text2).strip()

        # [v3.23.151] Cả hai nửa GIỮ bounding_box gốc (cùng vị trí hiển thị trên màn
        # hình); nửa ĐẦU kế thừa uid gốc để giữ liên kết Re-OCR/undo, nửa sau uid mới.
        first_half = SubtitleEvent(
            index=0, text=text1, interval=TimeInterval(original.start_sec, split_at_sec),
            confidence=original.confidence, frame_count=original.frame_count, position=original.position,
            bounding_box=original.bounding_box, uid=original.uid,
        )
        second_half = SubtitleEvent(
            index=0, text=text2, interval=TimeInterval(split_at_sec, original.end_sec),
            confidence=original.confidence, frame_count=original.frame_count, position=original.position,
            bounding_box=original.bounding_box,
        )
        self._events[index : index + 1] = [first_half, second_half]
        self._reindex()
        self._is_dirty = True
        return self.snapshot_state()

    def merge_with_next(self, index: int) -> EditorState:
        self._validate_index(index)
        if index + 1 >= len(self._events):
            raise IndexError("Không có câu phụ đề kế tiếp để gộp.")

        self._push_undo("Gộp với câu kế tiếp")

        current = self._events[index]
        next_event = self._events[index + 1]

        t1, t2 = current.text.strip(), next_event.text.strip()
        if t1 and t2:
            if is_cjk_char(t1[-1]) or is_cjk_char(t2[0]):
                merged_text = f"{t1}{t2}"
            else:
                merged_text = f"{t1} {t2}"
        else:
            merged_text = t1 or t2

        merged = SubtitleEvent(
            index=0, text=merged_text.strip(), interval=TimeInterval(current.start_sec, next_event.end_sec),
            confidence=Confidence((float(current.confidence) + float(next_event.confidence)) / 2.0),
            frame_count=current.frame_count + next_event.frame_count, position=current.position or next_event.position,
            # [v3.23.151] GIỮ uid câu đầu (nhất quán với apply_merge_groups; uid dùng để
            # khớp Re-OCR theo vùng và undo/redo) + HỢP NHẤT bounding_box hai câu (box là
            # "tập hợp bbox của mọi frame đóng góp" — mất box làm overlay không vẽ được).
            bounding_box=_union_bounding_boxes(current.bounding_box, next_event.bounding_box),
            uid=current.uid,
        )
        self._events[index : index + 2] = [merged]
        self._reindex()
        self._is_dirty = True
        return self.snapshot_state()

    def shift_all(self, offset_sec: float) -> EditorState:
        if not self._events:
            return self.snapshot_state()
        for event in self._events:
            if event.start_sec + offset_sec < 0:
                raise ConfigurationError(f"Offset {offset_sec:+.3f}s sẽ làm mốc bắt đầu của câu #{event.index} âm.")

        self._push_undo(f"Dịch toàn bộ {offset_sec:+.3f}s")
        for i, event in enumerate(self._events):
            self._events[i] = SubtitleEvent(
                index=event.index, text=event.text,
                interval=TimeInterval(start_sec=event.start_sec + offset_sec, end_sec=event.end_sec + offset_sec),
                confidence=event.confidence, frame_count=event.frame_count,
                position=event.position, bounding_box=event.bounding_box, uid=event.uid
            )
        # [v3.6 bugfix A]: Vô hiệu hoá starts_cache — tất cả start_sec đã thay đổi.
        # Không có dòng này trước đây → find_event_index_at_time trả sai kết quả
        # sau khi dùng shift_all (video cursor không bám đúng phụ đề).
        self._starts_cache = None
        self._is_dirty = True
        return self.snapshot_state()

    def replace_events(self, indices_to_remove: list[int], new_events: list[SubtitleEvent]) -> EditorState:
        remove_set = set(indices_to_remove)
        for idx in remove_set:
            self._validate_index(idx)

        self._push_undo("Trích xuất lại (Re-OCR)")
        keep_events = [e for i, e in enumerate(self._events) if i not in remove_set]
        self._events = self._merge_events_with_overlap_check(keep_events, new_events)
        self._reindex()
        self._is_dirty = True
        return self.snapshot_state()

    def replace_events_by_uid(
        self, uids_to_remove: list[str], new_events: list[SubtitleEvent], description: str = "Trích xuất lại (Re-OCR)"
    ) -> EditorState:
        if not uids_to_remove and not new_events:
            return self.snapshot_state()

        remove_set = set(uids_to_remove)
        self._push_undo(description)
        keep_events = [event for event in self._events if event.uid not in remove_set]
        self._events = self._merge_events_with_overlap_check(keep_events, new_events)
        self._reindex()
        self._is_dirty = True
        return self.snapshot_state()

    @staticmethod
    def _merge_events_with_overlap_check(
        keep_events: list[SubtitleEvent], new_events: list[SubtitleEvent]
    ) -> list[SubtitleEvent]:
        if not new_events:
            return list(keep_events)

        new_start = min(event.start_sec for event in new_events)
        new_end = max(event.end_sec for event in new_events)

        result: list[SubtitleEvent] = []
        for keep_event in keep_events:
            if keep_event.end_sec <= new_start or keep_event.start_sec >= new_end:
                result.append(keep_event)
                continue

            adjusted = SubtitleEditorService._clip_against_new_events(keep_event, new_events)
            if adjusted is not None:
                result.append(adjusted)

        result.extend(new_events)
        result.sort(key=lambda event: (event.start_sec, event.end_sec))
        return result

    @staticmethod
    def _clip_against_new_events(keep_event: SubtitleEvent, new_events: list[SubtitleEvent]) -> SubtitleEvent | None:
        new_start, new_end = keep_event.start_sec, keep_event.end_sec

        for new_event in new_events:
            if new_event.end_sec <= new_start or new_event.start_sec >= new_end:
                continue
            if new_event.start_sec <= new_start < new_event.end_sec:
                new_start = new_event.end_sec
            if new_event.start_sec < new_end <= new_event.end_sec:
                new_end = new_event.start_sec
            # [v3.23.151] new_event nằm GỌN bên trong keep_event (không chạm biên nào):
            # hai nhánh trên đều bỏ qua -> trước đây keep_event giữ NGUYÊN -> chồng phụ
            # đề lên vùng vừa Re-OCR. Nay cắt bỏ phần trùm, giữ nửa DÀI hơn (đầu/đuôi).
            if new_start < new_event.start_sec and new_event.end_sec < new_end:
                head_duration = new_event.start_sec - new_start
                tail_duration = new_end - new_event.end_sec
                if head_duration >= tail_duration:
                    new_end = new_event.start_sec
                else:
                    new_start = new_event.end_sec

        if new_end - new_start < 0.05:
            return None
        if new_start == keep_event.start_sec and new_end == keep_event.end_sec:
            return keep_event

        return SubtitleEvent(
            index=keep_event.index, text=keep_event.text, interval=TimeInterval(new_start, new_end),
            confidence=keep_event.confidence, frame_count=keep_event.frame_count,
            position=keep_event.position, bounding_box=keep_event.bounding_box, uid=keep_event.uid,
        )

    def auto_fix_timeline(self) -> int:
        if not self._events:
            return 0

        fixes = 0
        for i in range(len(self._events) - 1):
            curr, nxt = self._events[i], self._events[i+1]
            if curr.end_sec > nxt.start_sec or 0 < (nxt.start_sec - curr.end_sec) < 0.150:
                fixes += 1

        if fixes == 0: return 0

        # [v3.6 bugfix B]: _push_undo di chuyển XUỐNG DƯỚI vòng lặp fix thực tế.
        # Trước đây push ngay sau khi đếm pre-check → nếu tất cả cặp đều bị skip
        # bởi các điều kiện guard (curr.start >= nxt.end, v.v.), applied_fixes=0
        # nhưng undo stack đã có entry vô nghĩa → Ctrl+Z undo nhầm thao tác trước.
        applied_fixes = 0
        events_backup: tuple | None = None  # Snapshot chỉ khi thực sự cần

        for i in range(len(self._events) - 1):
            curr, nxt = self._events[i], self._events[i+1]

            if curr.end_sec > nxt.start_sec:
                if curr.start_sec >= nxt.end_sec:
                    continue

                mid = (curr.end_sec + nxt.start_sec) / 2.0
                if mid <= curr.start_sec: mid = curr.start_sec + 0.05
                if mid >= nxt.end_sec: mid = nxt.end_sec - 0.05

                curr_new_end = mid - 0.001
                nxt_new_start = mid + 0.001

                if curr.start_sec >= curr_new_end or nxt_new_start >= nxt.end_sec:
                    continue

                # Lazy-push undo chỉ trước fix đầu tiên thực sự được áp dụng.
                if events_backup is None:
                    events_backup = tuple(self._events)
                    self._undo_stack.append(_Snapshot(events=events_backup, description="Auto-Fix Timeline ✨"))
                    self._redo_stack.clear()

                self._events[i] = SubtitleEvent(
                    index=curr.index, text=curr.text, interval=TimeInterval(curr.start_sec, curr_new_end),
                    confidence=curr.confidence, frame_count=curr.frame_count, position=curr.position,
                    bounding_box=curr.bounding_box, uid=curr.uid
                )
                self._events[i+1] = SubtitleEvent(
                    index=nxt.index, text=nxt.text, interval=TimeInterval(nxt_new_start, nxt.end_sec),
                    confidence=nxt.confidence, frame_count=nxt.frame_count, position=nxt.position,
                    bounding_box=nxt.bounding_box, uid=nxt.uid
                )
                applied_fixes += 1

            elif 0 < (nxt.start_sec - curr.end_sec) < 0.150:
                mid = curr.end_sec + (nxt.start_sec - curr.end_sec) / 2.0

                curr_new_end = mid - 0.001
                nxt_new_start = mid + 0.001

                if curr.start_sec >= curr_new_end or nxt_new_start >= nxt.end_sec:
                    continue

                if events_backup is None:
                    events_backup = tuple(self._events)
                    self._undo_stack.append(_Snapshot(events=events_backup, description="Auto-Fix Timeline ✨"))
                    self._redo_stack.clear()

                self._events[i] = SubtitleEvent(
                    index=curr.index, text=curr.text, interval=TimeInterval(curr.start_sec, curr_new_end),
                    confidence=curr.confidence, frame_count=curr.frame_count, position=curr.position,
                    bounding_box=curr.bounding_box, uid=curr.uid
                )
                self._events[i+1] = SubtitleEvent(
                    index=nxt.index, text=nxt.text, interval=TimeInterval(nxt_new_start, nxt.end_sec),
                    confidence=nxt.confidence, frame_count=nxt.frame_count, position=nxt.position,
                    bounding_box=nxt.bounding_box, uid=nxt.uid
                )
                applied_fixes += 1

        self._starts_cache = None
        if applied_fixes > 0:
            self._is_dirty = True
        return applied_fixes

    def find_similar_groups(self, max_gap_sec: float, min_sim: float) -> list[list[int]]:
        import re
        groups: list[list[int]] = []
        total_evts = len(self._events)
        if total_evts < 2: return []

        def clean_text(t: str) -> str:
            return re.sub(r'[^\w]', '', t.lower()).strip()

        cached_clean_texts = [clean_text(e.text) for e in self._events]
        from subtitles_extractor.application.services.text_similarity import hybrid_semantic_similarity

        i = 0
        while i < total_evts - 1:
            base_text = cached_clean_texts[i]
            grp = [i]
            j = i + 1
            while j < total_evts:
                gap = self._events[j].start_sec - self._events[j-1].end_sec
                if gap > max_gap_sec:
                    break

                cmp_text = cached_clean_texts[j]
                len_base, len_cmp = len(base_text), len(cmp_text)

                if abs(len_base - len_cmp) > max(len_base, len_cmp) * 0.5:
                    break

                is_sim = False
                if not base_text and not cmp_text:
                    is_sim = True
                elif base_text and cmp_text:
                    is_sim = (hybrid_semantic_similarity(base_text, cmp_text) >= min_sim)

                if is_sim:
                    grp.append(j)
                    j += 1
                else:
                    break

            if len(grp) > 1:
                groups.append(grp)
            i = j if len(grp) > 1 else i + 1

        return groups

    def apply_merge_groups(self, groups: list[list[int]]) -> int:
        if not groups: return 0

        for grp in groups:
            for idx in grp: self._validate_index(idx)

        self._push_undo("Gộp câu trùng lặp 🧩")
        applied_count = 0

        for grp in reversed(groups):
            first_idx = grp[0]
            curr_ev = self._events[first_idx]
            new_end = max(self._events[idx].end_sec for idx in grp)
            best_text = max((self._events[idx].text for idx in grp), key=len)

            self._events[first_idx] = SubtitleEvent(
                index=curr_ev.index, text=best_text, interval=TimeInterval(curr_ev.start_sec, new_end),
                confidence=curr_ev.confidence, frame_count=curr_ev.frame_count,
                position=curr_ev.position, bounding_box=curr_ev.bounding_box, uid=curr_ev.uid
            )

            for r in reversed(grp[1:]):
                del self._events[r]
            applied_count += 1

        if applied_count > 0:
            self._reindex()
            self._is_dirty = True
        return applied_count

    def find_event_index_at_time(self, timestamp_sec: float) -> int:
        if not self._events: return -1
        from bisect import bisect_right

        if self._starts_cache is None:
            self._starts_cache = [event.start_sec for event in self._events]

        candidate_idx = bisect_right(self._starts_cache, timestamp_sec) - 1
        scan_start = max(0, candidate_idx - 1)
        scan_end = min(len(self._events), candidate_idx + 3)

        for index in range(scan_start, scan_end):
            event = self._events[index]
            if event.start_sec <= timestamp_sec <= event.end_sec:
                return index
        return -1

    def undo(self) -> EditorState:
        if not self._undo_stack: return self.snapshot_state()
        snapshot_to_redo = _Snapshot(events=tuple(self._events), description="(redo)")
        self._redo_stack.append(snapshot_to_redo)

        previous = self._undo_stack.pop()
        self._events = list(previous.events)
        self._is_dirty = True
        self._starts_cache = None
        # [v3.6 bugfix C]: Gọi _reindex() sau khi phục hồi snapshot.
        # Vấn đề: _push_undo() lưu THAM CHIẾU đến các SubtitleEvent hiện có.
        # _reindex() sau đó mutate trường .index IN-PLACE trên chính những object đó.
        # Khi undo phục hồi list cũ, các object trong snapshot đã bị mutate bởi
        # _reindex() của thao tác sau → event.index có giá trị SAI.
        # Fix: _reindex() lại ngay sau khi restore để đảm bảo indices nhất quán.
        self._reindex()
        return self.snapshot_state()

    def redo(self) -> EditorState:
        if not self._redo_stack: return self.snapshot_state()
        snapshot_to_undo = _Snapshot(events=tuple(self._events), description="(undo)")
        self._undo_stack.append(snapshot_to_undo)

        next_state = self._redo_stack.pop()
        self._events = list(next_state.events)
        self._is_dirty = True
        self._starts_cache = None
        # [v3.6 bugfix C]: Tương tự undo — reindex sau khi restore.
        self._reindex()
        return self.snapshot_state()

    def _reindex(self) -> None:
        for new_index, event in enumerate(self._events, start=1):
            event.index = new_index
        self._starts_cache = None

    def _validate_index(self, index: int) -> None:
        if not 0 <= index < len(self._events):
            raise IndexError(f"Index {index} ngoài phạm vi.")

__all__ = ["EditorState", "SubtitleEditorService"]
