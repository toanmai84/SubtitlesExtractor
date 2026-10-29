"""ViewModel cho trang TTS phụ đề."""

from __future__ import annotations

import functools
import logging
import time
import os
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal

from subtitles_extractor.application.services.dubbing_cast import (
    CastMember,
    DubbingSource,
    align_to_source,
    auto_cast,
    build_cast,
    conversation_weights,
    line_speakers,
    voices_for_lines,
)
from subtitles_extractor.application.services.dubbing_line_edits import (
    LineEdits,
    LineSnapshot,
    TextEdit,
    apply_speaker_overrides,
    restorable_text_edits,
    subtitle_fingerprint,
)
from subtitles_extractor.application.services.tts_line_cache import PlanMode
from subtitles_extractor.application.services.dubbing_line_structure import (
    build_snapshot,
    default_line_timing,
    delete_mapping,
    insert_mapping,
    insertion_position,
    remap_keys,
    remap_set,
    snapshot_is_structural,
    valid_snapshot,
)
from subtitles_extractor.application.services import tts_incremental_run as incremental_run
from subtitles_extractor.application.services.speaker_review_flow import (
    ReviewItem,
    ReviewLine,
    review_queue,
)
from subtitles_extractor.application.services.speaker_review import (
    SpeakerIssue,
    review_speakers,
)
from subtitles_extractor.domain.ports.subtitle_tts_port import (
    TTSRequest,
    TTSSegmentResult,
)
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.domain.value_objects.voice_profile import VoiceProfile
from subtitles_extractor.infrastructure.tts.segment_store import renumber_results
from subtitles_extractor.infrastructure.tts.tts_fit import (
    LineFit,
    measured_misfits,
    predict_misfits,
)

if TYPE_CHECKING:
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent

logger = logging.getLogger(__name__)


def _rewritten_since(path: str, started: float) -> bool:
    """[v3.23.659] Tệp audio từng dòng đã bị ghi lại kể từ ``started`` (giây epoch)."""
    if not path:
        return False
    try:
        return os.path.getmtime(path) >= started - 1.0
    except OSError:
        return False


#: [v3.23.627] Số bước hoàn tác tối đa giữ lại.
MAX_UNDO_STEPS = 100


def _addressee_map(visual_cues: str) -> dict[int, str]:
    """``{số câu gốc (1-based): người được nói tới}`` từ JSON Visual Cues (rỗng nếu hỏng)."""
    if not visual_cues.strip():
        return {}
    try:
        import json

        result: dict[int, str] = {}
        for cue in json.loads(visual_cues):
            target = str(cue.get("to") or "").strip()
            if target:
                result[int(cue.get("id", -1))] = target
        return result
    except (ValueError, TypeError, AttributeError):
        return {}


@dataclass(frozen=True, slots=True)
class _EditState:
    """Ảnh chụp mọi chỉnh sửa tay trên danh sách dòng (để hoàn tác / làm lại)."""

    events: tuple
    origins: tuple[int, ...]
    speakers: tuple[tuple[int, str], ...]
    texts: tuple[tuple[int, TextEdit], ...]
    edited: frozenset[int]
    addressees: tuple[tuple[int, str], ...] = ()


def _undoable(method):
    """Ghi MỘT bước hoàn tác cho cả thao tác (thao tác lồng nhau chỉ tính bước ngoài cùng)."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        outermost = self._tx_depth == 0
        before = self._capture_state() if outermost else None
        self._tx_depth += 1
        try:
            return method(self, *args, **kwargs)
        finally:
            self._tx_depth -= 1
            if outermost and before is not None:
                self._record_history(before)

    return wrapper


class TTSPageViewModel(QObject):
    """ViewModel trang TTS."""

    source_changed = Signal(int)
    busy_changed = Signal(bool)
    progress_changed = Signal(float, str)
    result_ready = Signal(object)
    error_occurred = Signal(str)
    status_message = Signal(str)
    tts_cancelled = Signal()
    engine_available = Signal(str, bool)
    #: [v3.23.598] Người nói / lời thoại sửa tay, hoặc tập "dòng đã sửa chưa TTS lại" đổi.
    line_edits_changed = Signal()
    #: [v3.23.625] Danh sách dòng vừa đổi (thêm/xoá dòng) — trang vẽ lại bảng dòng.
    structure_changed = Signal()

    def __init__(self, container: "ApplicationContainer") -> None:
        super().__init__()
        self._container = container
        self._source_events: list["SubtitleEvent"] = []
        self._last_results: list[TTSSegmentResult] = []
        self._last_output_path: Path | None = None
        self._last_request: "TTSRequest | None" = None
        self._last_engine_name: str = ""
        self._run_rewrites_line_audio: bool = False  # [v3.23.659]
        self._worker = None
        self._is_busy = False
        self._last_adapter = None  # [v3.23.450] adapter lần TTS gần nhất (để TTS lại dòng chọn)
        #: [v3.23.630] Lượt TTS đang chạy dùng cache từng dòng (ghi lại cache khi xong).
        self._run: incremental_run.PreparedRun | None = None
        #: [v3.23.630] ``(dùng lại, tổng, tổng hợp lại)`` của lượt tăng dần vừa xong; ``None``
        #: khi lượt gần nhất chạy cả phim.
        self._reuse_summary: tuple[int, int, int] | None = None
        #: [v3.23.630] TTS lại dòng chọn đang chạy: (yêu cầu, vị trí dòng) để cập nhật cache.
        self._manual_run: tuple[TTSRequest, frozenset[int]] | None = None
        # [v3.23.593] Lồng tiếng: ngữ cảnh phân vai (từ trang Dịch) + vai đã tìm thấy.
        self._dubbing_source: DubbingSource | None = None
        self._dubbing_cast: list[CastMember] = []
        self._dubbing_weights: dict[frozenset[str], int] = {}
        #: [v3.23.594] Visual Cues có ghép được vào phụ đề TTS đang nạp không (cùng phim).
        self._dubbing_cues_matched: bool = False
        # [v3.23.598] Sửa tay từng dòng (theo VỊ TRÍ dòng nguồn) — xem dubbing_line_edits.
        self._dubbing_prepared = False
        self._auto_speakers: list[str] = []
        self._speaker_overrides: dict[int, str] = {}
        # [v3.23.654] "Nói với ai" chọn tay theo vị trí dòng ("" = không nói với ai / độc thoại).
        self._addressee_overrides: dict[int, str] = {}
        self._text_edits: dict[int, TextEdit] = {}
        #: Dòng sửa tay SAU lần tổng hợp gần nhất của dòng đó (cần TTS lại).
        self._edited_since_synthesis: set[int] = set()
        #: Dòng sửa tay mà lượt TTS đang chạy đã đọc bản mới — xong thì hết "cần TTS lại".
        self._synthesizing: set[int] = set()
        self._fingerprint = ""
        #: [v3.23.625] Phụ đề lúc nạp + vị trí gốc của từng dòng hiện tại (-1 = dòng thêm).
        self._original_events: list["SubtitleEvent"] = []
        self._origins: list[int] = []
        # [v3.23.652] (số dòng gốc bớt, số dòng thêm) của ảnh chụp vừa áp khi nạp phụ đề; ``None`` = không áp.
        self._restored_structure: tuple[int, int] | None = None
        #: [v3.23.627] Hoàn tác / làm lại chỉnh sửa dòng — CHỈ khi chưa có kết quả TTS (audio
        #: từng dòng không khôi phục được nguyên vẹn sau khi tổng hợp lại).
        self._undo_stack: list[_EditState] = []
        self._redo_stack: list[_EditState] = []
        self._tx_depth = 0
        # [v3.23.599] Soát người nói: vị trí câu gốc từng dòng + kết quả soát gần nhất.
        self._source_positions: list[int | None] = []
        self._issues_cache: (
            tuple[tuple[object, ...], dict[int, SpeakerIssue]] | None
        ) = None
        #: [v3.23.630] ``{số câu gốc (1-based): người được nói tới}`` của Visual Cues.
        self._addressees: dict[int, str] = {}

    @property
    def source_events(self) -> list["SubtitleEvent"]:
        return self._source_events

    @property
    def has_source(self) -> bool:
        return len(self._source_events) > 0

    @property
    def has_result(self) -> bool:
        return bool(self._last_results)

    @property
    def is_busy(self) -> bool:
        return self._is_busy

    @property
    def last_output_path(self) -> Path | None:
        return self._last_output_path

    @property
    def last_results(self) -> list[TTSSegmentResult]:
        return self._last_results

    @property
    def last_request(self) -> "TTSRequest | None":
        return self._last_request

    @property
    def last_engine_name(self) -> str:
        return self._last_engine_name

    @property
    def dubbing_cast(self) -> list[CastMember]:
        """Vai tìm thấy ở lần nạp vai gần nhất."""
        return self._dubbing_cast

    @property
    def dubbing_cues_matched(self) -> bool:
        """[v3.23.594] Lần nạp vai gần nhất có dùng được Visual Cues của trang Dịch không.

        ``False`` khi trang Dịch không có cues, hoặc phụ đề TTS không cùng phim với phụ đề
        trang Dịch (khi đó chỉ tag ``[Tên:]`` đầu dòng được dùng).
        """
        return self._dubbing_cues_matched

    @property
    def dubbing_prepared(self) -> bool:
        """[v3.23.598] Đã tìm người nói cho phụ đề ĐANG nạp chưa (nạp vai sau khi nạp)."""
        return self._dubbing_prepared

    def auto_line_addressee(self, position: int) -> str:
        """[v3.23.654] "Nói với ai" của dòng theo Visual Cues (chưa áp sửa tay); rỗng nếu không biết."""
        if not self._dubbing_prepared or not (0 <= position < len(self._source_positions)):
            return ""
        source_position = self._source_positions[position]
        return self._addressees.get(source_position + 1, "") if source_position is not None else ""

    def line_addressee(self, position: int) -> str:
        """[v3.23.654] "Nói với ai" đang dùng của dòng (sửa tay thắng Visual Cues)."""
        if position in self._addressee_overrides:
            return self._addressee_overrides[position]
        return self.auto_line_addressee(position)

    @property
    def addressee_overrides(self) -> dict[int, str]:
        """[v3.23.654] "Nói với ai" chọn tay ``{vị trí dòng: người nghe}``."""
        return dict(self._addressee_overrides)

    @_undoable
    def set_addressee_override(self, positions: Iterable[int], addressee: str | None) -> int:
        """[v3.23.654] Chọn tay "nói với ai" cho các dòng.

        Không đổi giọng đọc (giọng theo người nói) nên KHÔNG đánh dấu cần TTS lại; dùng cho xưng hô ở lần dịch sau
        (đồng bộ về Visual Cues của trang Dịch).

        Args:
            positions: Vị trí dòng.
            addressee: Tên người nghe; ``""`` = không nói với ai (độc thoại / nói chung); ``None`` = bỏ chọn tay.

        Returns:
            Số dòng thay đổi.
        """
        changed = 0
        for position in self._valid_positions(positions):
            if addressee is None:
                if self._addressee_overrides.pop(position, None) is not None:
                    changed += 1
            elif self._addressee_overrides.get(position) != addressee:
                self._addressee_overrides[position] = addressee
                changed += 1
        if changed:
            self.line_edits_changed.emit()
        return changed

    def speaker_corrections(self) -> dict[int, str]:
        """[v3.23.653] Người nói chọn tay quy về câu GỐC của trang Dịch: ``{id cue (1-based): tên vai}``.

        Chỉ khi đã nạp vai cho phụ đề này và phụ đề TTS ghép được với câu gốc (cùng phim — xem ``align_to_source``).
        Bỏ "giọng chung" (không phải một người nói) và câu gốc nhận hai người nói khác nhau (dòng đã tách).
        """
        if not self._dubbing_prepared or not self._source_positions:
            return {}
        corrections: dict[int, str] = {}
        conflicts: set[int] = set()
        for position, speaker in self._speaker_overrides.items():
            speaker = (speaker or "").strip()
            if not speaker or not (0 <= position < len(self._source_positions)):
                continue
            source_position = self._source_positions[position]
            if source_position is None:
                continue
            line_no = source_position + 1
            if corrections.get(line_no, speaker) != speaker:
                conflicts.add(line_no)
            corrections[line_no] = speaker
        for line_no in conflicts:
            corrections.pop(line_no, None)
        return corrections

    def cue_corrections(self) -> dict[int, tuple[str | None, str | None]]:
        """[v3.23.654] Sửa tay quy về câu gốc: ``{id cue: (người nói | None, người nghe | None)}`` (``None`` = không sửa).

        Câu gốc nhận hai giá trị khác nhau từ các dòng TTS (dòng đã tách) bị bỏ phần mâu thuẫn.
        """
        if not self._dubbing_prepared or not self._source_positions:
            return {}
        result: dict[int, list[str | None]] = {
            line_no: [speaker, None] for line_no, speaker in self.speaker_corrections().items()
        }
        conflicts: set[int] = set()
        seen: dict[int, str] = {}
        for position, addressee in self._addressee_overrides.items():
            if not (0 <= position < len(self._source_positions)):
                continue
            source_position = self._source_positions[position]
            if source_position is None:
                continue
            line_no = source_position + 1
            addressee = (addressee or "").strip()
            if seen.get(line_no, addressee) != addressee:
                conflicts.add(line_no)
            seen[line_no] = addressee
        for line_no, addressee in seen.items():
            if line_no in conflicts:
                continue
            result.setdefault(line_no, [None, None])[1] = addressee
        return {line_no: (pair[0], pair[1]) for line_no, pair in result.items()}

    @property
    def dubbing_source_spans(self) -> tuple[tuple[float, float], ...]:
        """[v3.23.653] Mốc câu gốc của ngữ cảnh phân vai đang dùng (để biết còn khớp trang Dịch không)."""
        source = self._dubbing_source
        return tuple(source.source_spans) if self._dubbing_prepared and source is not None else ()

    @property
    def speaker_overrides(self) -> dict[int, str]:
        """[v3.23.598] Người nói chọn tay ``{vị trí dòng: vai}`` (rỗng: giọng chung)."""
        return dict(self._speaker_overrides)

    @property
    def edited_positions(self) -> frozenset[int]:
        """[v3.23.598] Dòng sửa tay (người nói/lời) sau lần tổng hợp gần nhất của dòng."""
        return frozenset(self._edited_since_synthesis)

    @property
    def subtitle_fingerprint(self) -> str:
        """[v3.23.598] Dấu của phụ đề đang nạp (để nhớ chỉnh sửa tay theo phụ đề)."""
        return self._fingerprint

    @property
    def can_resynthesize(self) -> bool:
        """[v3.23.598] Lần TTS gần nhất có giữ audio từng dòng (TTS lại được dòng chọn).

        [v3.23.655] Edge/Gemini nay cũng ghi audio từng dòng; kết quả cũ (trước 655) thì không. Bản trước vẫn cho bấm "TTS lại dòng chọn" sau khi
        chạy Edge/Gemini: tổng hợp xong các dòng chọn rồi mới báo lỗi "Không có segment
        nào để ghép lại master".
        """
        return self._last_adapter is not None and any(
            getattr(result, "segment_audio_path", "") for result in self._last_results
        )

    def _line_audio_dirs(self) -> list[Path]:
        """[v3.23.657] Thư mục audio từng dòng của lần TTS gần nhất.

        Lấy từ đường dẫn segment THỰC trong kết quả (đúng cả khi ``segments_dir`` bị chỉ định
        riêng); thêm thư mục mặc định cạnh output để dọn cả phần còn sót của lượt trước.
        """
        dirs: list[Path] = []
        if not self._last_results:
            return dirs  # chưa có lượt TTS cho phụ đề đang nạp (nạp phụ đề mới xoá kết quả cũ)
        for result in self._last_results:
            path = getattr(result, "segment_audio_path", "")
            if path and Path(path).parent not in dirs:
                dirs.append(Path(path).parent)
        if self._last_output_path is not None:
            from subtitles_extractor.infrastructure.tts.segment_store import segments_dir_for

            override = getattr(self._last_request, "segments_dir", "") if self._last_request else ""
            default = segments_dir_for(self._last_output_path, override or "")
            if default not in dirs:
                dirs.append(default)
        return dirs

    def line_audio_usage(self) -> int:
        """[v3.23.657] Số byte audio từng dòng đang chiếm đĩa (0 nếu không có)."""
        from subtitles_extractor.infrastructure.tts.segment_store import line_audio_usage

        return sum(line_audio_usage(d)[0] for d in self._line_audio_dirs())

    def clean_line_audio(self) -> int:
        """[v3.23.657] Người dùng bấm "Dọn": xoá audio từng dòng, giữ master + SRT.

        Sau khi dọn, "Nghe/TTS lại dòng chọn" cần chạy lại cả phim (``can_resynthesize`` = False).

        Returns:
            Số byte đã giải phóng; ``-1`` nếu đang bận (không dọn).
        """
        if self._is_busy:
            return -1
        from dataclasses import replace as _replace

        from subtitles_extractor.infrastructure.tts.segment_store import clean_line_audio

        freed = sum(clean_line_audio(d) for d in self._line_audio_dirs())
        self._last_results = [
            _replace(r, segment_audio_path="") if getattr(r, "segment_audio_path", "") else r
            for r in self._last_results
        ]
        return freed

    def line_edits(self) -> LineEdits:
        """[v3.23.598] Chỉnh sửa tay hiện có (để lưu theo phụ đề)."""
        return LineEdits(
            speakers=dict(self._speaker_overrides), texts=dict(self._text_edits),
            lines=self._line_snapshot(), addressees=dict(self._addressee_overrides),
        )

    def _line_snapshot(self) -> tuple[LineSnapshot, ...] | None:
        """[v3.23.625] Ảnh chụp danh sách dòng khi đã thêm/xoá dòng; ``None`` nếu chưa."""
        if not (
            snapshot_is_structural(self._origins, len(self._original_events))
            or self._timing_changed()
        ):
            return None
        events = self._source_events
        return build_snapshot(
            self._origins,
            [float(e.start_sec) for e in events],
            [float(e.end_sec) for e in events],
            [str(getattr(e, "text", "") or "") for e in events],
        )

    def _timing_changed(self) -> bool:
        """[v3.23.626] Có dòng nào đã đổi mốc thời gian so với lúc nạp không."""
        for event, origin in zip(self._source_events, self._origins, strict=False):
            if 0 <= origin < len(self._original_events):
                base = self._original_events[origin]
                if (event.start_sec, event.end_sec) != (base.start_sec, base.end_sec):
                    return True
        return False

    @property
    def reuse_summary(self) -> tuple[int, int, int] | None:
        """[v3.23.630] ``(dòng dùng lại, tổng, dòng tổng hợp lại)`` của lượt tăng dần gần nhất."""
        return self._reuse_summary

    @property
    def line_structure_changed(self) -> bool:
        """[v3.23.625] Danh sách dòng đã khác lúc nạp (có thêm/xoá dòng)."""
        return self._line_snapshot() is not None

    def has_line_edits(self, positions: Iterable[int] | None = None) -> bool:
        """[v3.23.598] Có chỉnh sửa tay nào (trong ``positions`` nếu có) không."""
        edited = set(self._speaker_overrides) | set(self._text_edits) | set(self._addressee_overrides)
        if positions is None:
            return bool(edited)
        return any(position in edited for position in positions)

    def auto_line_speakers(self) -> list[str]:
        """[v3.23.598] Người nói ứng dụng tự tìm cho từng dòng nguồn (chưa áp sửa tay).

        Returns:
            Theo vị trí dòng; toàn rỗng khi chưa nạp vai cho phụ đề này.
        """
        count = len(self._source_events)
        if not self._dubbing_prepared or self._dubbing_source is None:
            return [""] * count
        if len(self._auto_speakers) != count:
            self._auto_speakers = line_speakers(self._source_events, self._dubbing_source)
        return list(self._auto_speakers)

    def effective_line_speakers(self) -> list[str]:
        """[v3.23.598] Người nói từng dòng sau khi áp lựa chọn tay của người dùng."""
        return apply_speaker_overrides(self.auto_line_speakers(), self._speaker_overrides)

    def speaker_issues(self) -> dict[int, SpeakerIssue]:
        """[v3.23.599] Dòng cần soát người nói (xem ``speaker_review``).

        Dòng người dùng đã chọn tay người nói coi như đã soát. Chưa nạp vai cho phụ đề
        này thì rỗng.

        Returns:
            ``{vị trí dòng: vấn đề}``.
        """
        source = self._dubbing_source
        if not self._dubbing_prepared or source is None:
            return {}
        speakers = self.effective_line_speakers()
        signature = (tuple(speakers), tuple(sorted(self._speaker_overrides)))
        if self._issues_cache is not None and self._issues_cache[0] == signature:
            return dict(self._issues_cache[1])
        texts = [
            source.source_texts[position]
            if position is not None and position < len(source.source_texts) else ""
            for position in self._source_positions
        ]
        if len(texts) != len(speakers):
            texts = [""] * len(speakers)
        issues = review_speakers(
            speakers, texts, source.characters,
            {member.name: member.gender for member in self._dubbing_cast},
            frozenset(self._speaker_overrides),
        )
        self._issues_cache = (signature, issues)
        return dict(issues)

    @_undoable
    def set_speaker_override(
        self, positions: Iterable[int], speaker: str | None
    ) -> int:
        """[v3.23.598] Chọn tay người nói cho các dòng (sửa phân vai sai của khâu dịch).

        Args:
            positions: Vị trí dòng (0-based).
            speaker: Tên vai; :data:`SHARED_VOICE_SPEAKER` = giọng chung; ``None`` = bỏ
                lựa chọn tay, dùng lại người nói tự tìm.

        Returns:
            Số dòng thay đổi.
        """
        changed: list[int] = []
        for position in self._valid_positions(positions):
            if speaker is None:
                if self._speaker_overrides.pop(position, None) is not None:
                    changed.append(position)
            elif self._speaker_overrides.get(position) != speaker:
                self._speaker_overrides[position] = speaker
                changed.append(position)
        if changed:
            self._mark_edited(changed)
            self.line_edits_changed.emit()
        return len(changed)

    @_undoable
    def revert_line_edits(self, positions: Iterable[int]) -> list[int]:
        """[v3.23.598] Bỏ mọi sửa tay của các dòng: lời về chữ lúc nạp, người nói tự tìm.

        Args:
            positions: Vị trí dòng.

        Returns:
            Vị trí các dòng vừa được trả lại lời gốc (để trang cập nhật ô nội dung).
        """
        reverted_texts: list[int] = []
        changed: list[int] = []
        for position in self._valid_positions(positions):
            if self._speaker_overrides.pop(position, None) is not None:
                changed.append(position)
            if self._addressee_overrides.pop(position, None) is not None and position not in changed:
                changed.append(position)
            edit = self._text_edits.pop(position, None)
            if edit is not None:
                self._replace_event_text(position, edit.original)
                reverted_texts.append(position)
                changed.append(position)
        if changed:
            if reverted_texts:
                self._refresh_auto_speakers()
            self._mark_edited(changed)
            self.line_edits_changed.emit()
        return reverted_texts

    def restore_line_edits(self, edits: LineEdits) -> tuple[int, int]:
        """[v3.23.598] Áp lại chỉnh sửa tay đã lưu cho phụ đề vừa nạp.

        Người nói áp theo vị trí (cùng dấu phụ đề = cùng các dòng). Lời sửa chỉ áp khi
        dòng còn đúng chữ lúc sửa (:func:`restorable_text_edits`).

        Args:
            edits: Chỉnh sửa đã lưu của phụ đề này.

        Returns:
            ``(số dòng áp người nói, số dòng áp lời sửa)``.
        """
        before = len(self._source_events)
        structural = self._restore_structure(edits.lines, edits.texts)
        count = len(self._source_events)
        # [v3.23.652] Trang báo rõ việc áp lại danh sách dòng (trước đây lặng lẽ, log ghi "0 dòng").
        kept = {origin for origin in self._origins if origin >= 0}
        self._restored_structure = (
            (sum(1 for origin in range(before) if origin not in kept), self._origins.count(-1))
            if structural else None
        )
        speakers = {
            position: speaker
            for position, speaker in edits.speakers.items()
            if 0 <= position < count
        }
        if structural:
            # Lời trong ảnh chụp đã là lời sau khi sửa; chỉ nhớ lại chữ gốc để "Bỏ sửa tay".
            texts = {
                position: edit for position, edit in edits.texts.items()
                if 0 <= position < count
                and edit.edited != edit.original
                and str(getattr(self._source_events[position], "text", "") or "") == edit.edited
            }
        else:
            texts = restorable_text_edits(self._source_events, edits.texts)
        addressees = {
            position: addressee for position, addressee in edits.addressees.items() if 0 <= position < count
        }
        self._addressee_overrides.update(addressees)
        if not speakers and not texts:
            if structural or addressees:
                self.line_edits_changed.emit()
            return 0, 0
        self._speaker_overrides.update(speakers)
        for position, edit in texts.items():
            if not structural:
                self._replace_event_text(position, edit.edited)
            self._text_edits[position] = edit
        if texts:
            self._refresh_auto_speakers()
        self.line_edits_changed.emit()
        return len(speakers), len(texts)

    def prepare_dubbing(self, source: DubbingSource) -> list[str]:
        """[v3.23.593] Tìm người nói từng dòng TTS và lập danh sách vai.

        [v3.23.598] Áp người nói chọn tay trước khi lập danh sách vai: vai người dùng gán
        cho một dòng (kể cả nhân vật Visual Cues bỏ sót) có mặt trong bảng phân vai và
        nhận giọng như mọi vai khác.

        Args:
            source: Bảng nhân vật + Visual Cues + mốc câu gốc từ trang Dịch.

        Returns:
            Người nói của từng dòng nguồn TTS (rỗng = không rõ / giọng chung).
        """
        self._dubbing_source = source
        self._dubbing_prepared = True
        self._issues_cache = None
        self._addressees = _addressee_map(source.visual_cues)
        self._auto_speakers = line_speakers(self._source_events, source)
        speakers = self.effective_line_speakers()
        self._source_positions = align_to_source(self._source_events, source.source_spans)
        self._dubbing_cues_matched = bool(source.visual_cues.strip()) and any(
            position is not None for position in self._source_positions
        )
        self._dubbing_cast = build_cast(speakers, source.characters)
        self._dubbing_weights = conversation_weights(speakers)
        return speakers

    def voice_catalog(self, engine: str, language: str) -> list[VoiceProfile]:
        """Bộ giọng phân vai của Edge/Gemini (tức thì, không nạp model).

        VieNeu phải nạp model mới biết giọng nên trang lấy qua luồng nền riêng.

        Args:
            engine: ``"edge"`` hoặc ``"gemini"``.
            language: Locale (Edge) hoặc tên model (Gemini).

        Returns:
            Danh sách giọng; rỗng với engine khác.
        """
        if engine == "edge":
            return self._container.make_edge_tts_adapter().list_voice_catalog(language)
        if engine == "gemini":
            return self._container.make_gemini_tts_adapter().list_voice_catalog(language)
        return []

    def auto_cast(self, catalog: list[VoiceProfile]) -> dict[str, str]:
        """Giao giọng tự động cho các vai đã nạp (xem ``dubbing_cast.auto_cast``)."""
        return auto_cast(self._dubbing_cast, catalog, self._dubbing_weights)

    def dubbing_line_voices(self, casting: dict[str, str]) -> tuple[str, ...]:
        """Giọng từng dòng nguồn HIỆN TẠI theo bảng phân vai.

        Tính lại người nói trên phụ đề đang nạp (người dùng có thể đã nạp lại phụ đề sau
        khi nạp vai), nên giọng luôn khớp đúng dòng. [v3.23.598] Có áp người nói chọn tay.

        Args:
            casting: ``{tên vai: voice_id}``.

        Returns:
            Giọng theo vị trí dòng; rỗng khi chưa nạp vai hoặc không dòng nào có giọng
            riêng.
        """
        if self._dubbing_source is None:
            return ()
        speakers = apply_speaker_overrides(
            line_speakers(self._source_events, self._dubbing_source),
            self._speaker_overrides,
        )
        return voices_for_lines(speakers, casting)

    def set_source_events(self, events: list["SubtitleEvent"]) -> None:
        self._restored_structure = None
        self._source_events = list(events)
        self._original_events = list(events)
        self._origins = list(range(len(events)))
        self._undo_stack, self._redo_stack = [], []
        # [v3.23.598] Sửa tay gắn với phụ đề CŨ; trang khôi phục sửa tay đã lưu của
        # phụ đề mới (theo dấu phụ đề) khi nhận source_changed.
        self._fingerprint = subtitle_fingerprint(self._source_events)
        self._speaker_overrides = {}
        self._addressee_overrides = {}
        self._text_edits = {}
        self._edited_since_synthesis = set()
        self._synthesizing = set()
        self._auto_speakers = []
        self._dubbing_prepared = False
        self._source_positions = []
        self._issues_cache = None
        # [v3.23.598] Kết quả TTS cũ là của phụ đề CŨ. Bảng dòng nay hiện ngay các dòng
        # vừa nạp và cho chọn dòng: "TTS lại dòng chọn" trên kết quả cũ sẽ ghép lời phụ đề
        # mới vào audio phụ đề cũ.
        self._last_results = []
        self._last_request = None
        self._last_adapter = None
        self.source_changed.emit(len(self._source_events))

    @_undoable
    def update_event_text(self, position: int, new_text: str) -> None:
        """[v3.23.451] Cập nhật nội dung dòng nguồn theo VỊ TRÍ (0-based, = ``event_index`` của
        kết quả) — KHÔNG dùng ``event.index`` (1-based) để tránh lệch n→n-1.

        [v3.23.598] Ghi nhận lời sửa tay (nhớ chữ gốc để "Bỏ sửa tay" và để lưu theo phụ
        đề) và đánh dấu dòng cần TTS lại. Sửa chỉ khác khoảng trắng không tính là sửa.

        Args:
            position: Vị trí enumerate của dòng (bằng ``TTSSegmentResult.event_index``).
            new_text: Nội dung mới (đã sửa trên bảng).
        """
        if not (0 <= position < len(self._source_events)):
            return
        current = str(getattr(self._source_events[position], "text", "") or "")
        if new_text.strip() == current.strip():
            return
        edit = self._text_edits.get(position)
        original = edit.original if edit is not None else current
        self._replace_event_text(position, new_text)
        if new_text.strip() == original.strip():
            self._text_edits.pop(position, None)
        else:
            self._text_edits[position] = TextEdit(original=original, edited=new_text)
        # Lời mới có thể vừa thêm/bỏ tag [Tên:] đầu dòng — tag quyết định người nói.
        self._refresh_auto_speakers()
        self._mark_edited([position])
        self.line_edits_changed.emit()

    def _replace_event_text(self, position: int, text: str) -> None:
        event = self._source_events[position]
        try:
            self._source_events[position] = replace(event, text=text)
        except (TypeError, ValueError):
            event.text = text  # type: ignore[attr-defined]

    def _refresh_auto_speakers(self) -> None:
        if self._dubbing_prepared and self._dubbing_source is not None:
            self._auto_speakers = line_speakers(self._source_events, self._dubbing_source)

    def _mark_edited(self, positions: Iterable[int]) -> None:
        for position in positions:
            self._edited_since_synthesis.add(position)
            # Sửa trong lúc lượt TTS đang chạy: lượt đó đọc bản CŨ → vẫn cần TTS lại.
            self._synthesizing.discard(position)

    def _valid_positions(self, positions: Iterable[int]) -> list[int]:
        count = len(self._source_events)
        return sorted({
            int(position) for position in positions if 0 <= int(position) < count
        })

    def resynthesize_lines(
        self,
        selected_indices: set[int],
        line_voices: tuple[str, ...] | None = None,
        remerge_only: bool = False,
    ) -> bool:
        """[v3.23.450] TTS lại các dòng được chọn (kể cả nội dung vừa sửa) rồi ghép lại master.

        Args:
            selected_indices: Vị trí (0-based) các dòng cần TTS lại.
            line_voices: [v3.23.594] Giọng từng dòng theo bảng phân vai HIỆN TẠI
                (người dùng vừa đổi giọng một vai rồi TTS lại các dòng của vai đó).
                ``None`` = giữ giọng của lần TTS trước.

            remerge_only: [v3.23.625] Chỉ ghép lại master (không tổng hợp dòng nào).

        Returns:
            True nếu đã khởi chạy; False nếu tiền điều kiện không thoả (chưa TTS, thiếu file,
            engine không hỗ trợ giữ audio per-line, hoặc đang bận).
        """
        if self._is_busy or self._worker is not None and self._worker.isRunning():
            return False
        if (
            (not selected_indices and not remerge_only)
            or self._last_adapter is None
            or self._last_request is None
            or self._last_output_path is None
            or not self._last_results
        ):
            return False
        if not self._last_output_path.exists() or not self.can_resynthesize:
            return False

        # Đọc sample_rate + độ dài master từ file WAV output hiện có (giữ nguyên tổng thời lượng).
        try:
            import soundfile as sf

            info = sf.info(str(self._last_output_path))
            sample_rate = int(info.samplerate)
            total_length_samples = int(info.frames)
        except (OSError, RuntimeError) as exc:
            # [v3.23.659] libsndfile không đọc được M4A/AAC (và có bản không đọc MP3/Opus): trước đây "TTS lại dòng chọn"
            # / ghép lại sau xoá-thêm-tách-gộp dòng LUÔN lỗi với các định dạng đó. Lấy tần số từ audio từng dòng và độ
            # dài master theo đúng công thức của lượt đầy đủ.
            from subtitles_extractor.application.services.tts_incremental_run import (
                _detect_sample_rate,
                _master_samples,
            )

            sample_rate = _detect_sample_rate(self._last_results)
            total_length_samples = (
                _master_samples(replace(self._last_request, events=list(self._source_events)), sample_rate)
                if sample_rate > 0 and self._last_request is not None else 0
            )
            if sample_rate <= 0 or total_length_samples <= 0:
                self.error_occurred.emit(f"Không đọc được file audio để ghép lại: {exc}")
                return False

        # Yêu cầu ghép nội dung MỚI (source_events đã cập nhật qua update_event_text).
        # [v3.23.598] Bản SAO danh sách dòng: luồng nền không được thấy sửa tay giữa chừng.
        full_request = replace(
            self._last_request, events=list(self._source_events), keep_line_audio=True
        )
        if line_voices is not None:
            full_request = replace(full_request, line_voices=tuple(line_voices))

        use_case = self._container.make_resynthesize_use_case(self._last_adapter)
        from subtitles_extractor.presentation.workers.resynthesize_worker import (
            ResynthesizeWorker,
        )
        self._worker = ResynthesizeWorker(
            use_case,
            full_request=full_request,
            selected_indices=set(int(i) for i in selected_indices),
            prior_results=self._last_results,
            output_path=self._last_output_path,
            sample_rate=sample_rate,
            total_length_samples=total_length_samples,
            remerge_only=remerge_only,
        )
        self._worker.progress_changed.connect(self.progress_changed.emit)
        self._worker.finished_ok.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.cancelled.connect(self._on_worker_cancelled)
        self._worker.finished.connect(self._on_thread_finished)
        self._set_busy(True)
        self._synthesizing = self._edited_since_synthesis & {
            int(i) for i in selected_indices
        }
        self._run = None
        self._run_rewrites_line_audio = False  # [v3.23.659] TTS lại dòng chọn ghi vào thư mục tạm — huỷ không hỏng gì
        self._reuse_summary = None
        self._manual_run = (
            None if remerge_only
            else (full_request, frozenset(int(i) for i in selected_indices))
        )
        self._worker.start()
        self.status_message.emit(
            "Đang ghép lại audio sau khi đổi danh sách dòng..." if remerge_only
            else f"Đang TTS lại {len(selected_indices)} dòng rồi ghép lại audio..."
        )
        return True

    # ── [v3.23.625] Thêm / xoá dòng ───────────────────────────────────────────

    def _restore_structure(
        self, lines: tuple[LineSnapshot, ...] | None, texts: dict[int, TextEdit] | None = None,
    ) -> bool:
        """Dựng lại danh sách dòng đã thêm/xoá từ ảnh chụp đã lưu. ``True`` nếu đã áp.

        [v3.23.652] Dòng gốc lấy chữ của phụ đề VỪA NẠP; chữ trong ảnh chụp chỉ dùng cho dòng thêm tay và dòng người
        dùng đã sửa lời trên ĐÚNG chữ gốc đó. Trước đây chữ cũ trong ảnh chụp đè cả bản dịch mới (dịch lại không đổi mốc
        giờ → cùng dấu phụ đề).
        """
        original = self._original_events
        if lines is None or not original:
            return False
        if not valid_snapshot(lines, len(original)):
            logger.warning(
                "Danh sách dòng TTS đã lưu (%d dòng) không khớp phụ đề %d dòng — bỏ qua, dùng phụ đề gốc.",
                len(lines), len(original),
            )
            return False
        texts = texts or {}
        rebuilt: list["SubtitleEvent"] = []
        for position, line in enumerate(lines):
            base = original[line.origin] if line.origin >= 0 else None
            text = line.text
            if base is not None:
                base_text = str(getattr(base, "text", "") or "")
                edit = texts.get(position)
                if edit is None or edit.original.strip() != base_text.strip():
                    text = base_text
            rebuilt.append(self._make_event(
                base or original[0], len(rebuilt) + 1, line.start_sec, line.end_sec,
                text, keep_identity=base is not None,
            ))
        self._source_events = rebuilt
        self._origins = [line.origin for line in lines]
        return True

    @staticmethod
    def _make_event(
        template: "SubtitleEvent", index: int, start_sec: float, end_sec: float, text: str,
        *, keep_identity: bool = False,
    ) -> "SubtitleEvent":
        """Bản sao ``template`` với số thứ tự, mốc và lời mới (dòng mới: danh tính mới)."""
        changes: dict[str, object] = {
            "index": index, "text": text, "interval": TimeInterval(start_sec, end_sec),
        }
        if not keep_identity:
            changes.update(
                uid=str(uuid.uuid4()), confidence=Confidence.zero(), frame_count=0,
                position=None, bounding_box=None,
            )
        return replace(template, **changes)  # type: ignore[type-var]

    def _renumber_events(self) -> None:
        """Số thứ tự ``index`` của dòng theo vị trí hiện tại (1-based)."""
        for position, event in enumerate(self._source_events):
            if getattr(event, "index", position + 1) != position + 1:
                self._source_events[position] = replace(event, index=position + 1)

    def _after_structure_change(self, mapping: dict[int, int]) -> None:
        """Đưa mọi trạng thái đánh theo vị trí sang vị trí mới rồi báo trang."""
        self._speaker_overrides = remap_keys(self._speaker_overrides, mapping)
        self._addressee_overrides = remap_keys(self._addressee_overrides, mapping)
        self._text_edits = remap_keys(self._text_edits, mapping)
        self._edited_since_synthesis = remap_set(self._edited_since_synthesis, mapping)
        self._synthesizing = set()
        self._renumber_events()
        self._rederive_speakers()
        self.structure_changed.emit()
        self.line_edits_changed.emit()

    def _rederive_speakers(self) -> None:
        """Tính lại người nói tự tìm và vị trí câu gốc theo danh sách dòng hiện tại."""
        self._auto_speakers = []
        self._source_positions = []
        self._issues_cache = None
        if self._dubbing_prepared and self._dubbing_source is not None:
            self._auto_speakers = line_speakers(self._source_events, self._dubbing_source)
            self._source_positions = align_to_source(
                self._source_events, self._dubbing_source.source_spans
            )

    # ── [v3.23.627] Hoàn tác / làm lại ────────────────────────────────────────

    def _capture_state(self) -> _EditState:
        return _EditState(
            events=tuple(self._source_events),
            origins=tuple(self._origins),
            speakers=tuple(sorted(self._speaker_overrides.items())),
            texts=tuple(sorted(self._text_edits.items())),
            edited=frozenset(self._edited_since_synthesis),
            addressees=tuple(sorted(self._addressee_overrides.items())),
        )

    def _record_history(self, before: _EditState) -> None:
        """Đẩy trạng thái trước thao tác vào ngăn hoàn tác nếu thao tác có đổi gì."""
        if self._last_results or self._capture_state() == before:
            return
        self._undo_stack.append(before)
        del self._undo_stack[:-MAX_UNDO_STEPS]
        self._redo_stack.clear()

    def _apply_state(self, state: _EditState) -> None:
        self._source_events = list(state.events)
        self._origins = list(state.origins)
        self._speaker_overrides = dict(state.speakers)
        self._addressee_overrides = dict(state.addressees)
        self._text_edits = dict(state.texts)
        self._edited_since_synthesis = set(state.edited)
        self._synthesizing = set()
        self._rederive_speakers()
        self.structure_changed.emit()
        self.line_edits_changed.emit()

    @property
    def can_undo(self) -> bool:
        """Có bước để hoàn tác (chưa có kết quả TTS, không bận)."""
        return bool(self._undo_stack) and not self._last_results and not self._is_busy

    @property
    def can_redo(self) -> bool:
        """Có bước để làm lại."""
        return bool(self._redo_stack) and not self._last_results and not self._is_busy

    def undo(self) -> bool:
        """Hoàn tác thao tác chỉnh dòng gần nhất (lời, người nói, thêm/xoá/gộp/tách, mốc)."""
        if not self.can_undo:
            return False
        self._redo_stack.append(self._capture_state())
        self._apply_state(self._undo_stack.pop())
        return True

    def redo(self) -> bool:
        """Làm lại thao tác vừa hoàn tác."""
        if not self.can_redo:
            return False
        self._undo_stack.append(self._capture_state())
        self._apply_state(self._redo_stack.pop())
        return True

    @_undoable
    def delete_lines(self, positions: Iterable[int]) -> int:
        """[v3.23.625] Xoá các dòng khỏi danh sách lồng tiếng.

        Người nói chọn tay, lời sửa tay, dòng cần TTS lại, kết quả tổng hợp và tệp audio từng
        dòng đều được đánh số lại theo vị trí mới; audio của dòng bị xoá bị dọn. Master chưa bị
        đổi — trang gọi :meth:`remerge_master` để ghép lại.

        Args:
            positions: Vị trí (0-based) các dòng cần xoá.

        Returns:
            Số dòng đã xoá (0 nếu đang bận, hoặc sẽ xoá hết mọi dòng).
        """
        if self._is_busy:
            return 0
        removed = self._valid_positions(positions)
        count = len(self._source_events)
        if not removed or len(removed) >= count:
            return 0
        mapping = delete_mapping(count, removed)
        origins_before = dict(enumerate(self._origins))
        self._source_events = [
            event for position, event in enumerate(self._source_events)
            if position in mapping
        ]
        self._origins = [o for position, o in enumerate(self._origins) if position in mapping]
        self._remap_results(mapping, removed=len(removed), origins_before=origins_before)
        self._after_structure_change(mapping)
        return len(removed)

    @_undoable
    def insert_line(
        self, start_sec: float, end_sec: float, text: str
    ) -> int | None:
        """[v3.23.625] Thêm một dòng mới (đúng thứ tự thời gian) vào danh sách lồng tiếng.

        Args:
            start_sec: Mốc bắt đầu (giây).
            end_sec: Mốc kết thúc (giây), lớn hơn mốc bắt đầu.
            text: Lời thoại (không rỗng).

        Returns:
            Vị trí của dòng mới; ``None`` nếu đang bận hoặc dữ liệu không hợp lệ.
        """
        text = (text or "").strip()
        if self._is_busy or not self._source_events or not text:
            return None
        if start_sec < 0 or end_sec <= start_sec:
            return None
        starts = [float(e.start_sec) for e in self._source_events]
        position = insertion_position(starts, start_sec)
        count = len(self._source_events)
        mapping = insert_mapping(count, position)
        template = self._source_events[max(0, position - 1)]
        event = self._make_event(template, position + 1, start_sec, end_sec, text)
        self._source_events.insert(position, event)
        self._origins.insert(position, -1)
        self._remap_results(mapping, inserted=(position, start_sec, end_sec, text))
        self._after_structure_change(mapping)
        self._mark_edited([position])
        return position

    # ── [v3.23.652] Khôi phục dòng gốc ────────────────────────────────────────

    @property
    def restored_structure(self) -> tuple[int, int] | None:
        """``(dòng gốc bớt, dòng thêm)`` nếu lúc nạp phụ đề vừa áp lại danh sách dòng đã sửa lưu trước."""
        return self._restored_structure

    def missing_original_lines(self) -> list[tuple[int, float, float, str]]:
        """Các dòng của phụ đề gốc đang không có trong danh sách: ``(origin, bắt đầu, kết thúc, lời)``."""
        present = {origin for origin in self._origins if origin >= 0}
        return [
            (origin, float(event.start_sec), float(event.end_sec), str(getattr(event, "text", "") or ""))
            for origin, event in enumerate(self._original_events)
            if origin not in present
        ]

    @_undoable
    def restore_original_lines(self, origins: Iterable[int] | None = None) -> int:
        """Chèn lại các dòng gốc đã xoá (``None`` = mọi dòng đang thiếu), theo thứ tự thời gian; dòng chèn lại cần TTS.

        Dựng lại danh sách MỘT lượt (một lần đánh số lại kết quả/tệp audio, một lần báo trang): chèn từng dòng một
        từng làm treo giao diện ~25 s với 600 dòng (vẽ lại bảng + ghi QSettings mỗi dòng).

        Returns:
            Số dòng đã chèn lại (0 nếu đang bận).
        """
        if self._is_busy or not self._original_events:
            return 0
        present = {origin for origin in self._origins if origin >= 0}
        total = len(self._original_events)
        wanted = range(total) if origins is None else origins
        chosen = sorted({o for o in wanted if 0 <= o < total and o not in present})
        if not chosen:
            return 0
        # Dòng hiện có giữ nguyên thứ tự; dòng chèn lại đặt theo mốc bắt đầu (sau các dòng cùng mốc).
        current_starts = [float(event.start_sec) for event in self._source_events]
        order_ok = all(a <= b for a, b in zip(current_starts, current_starts[1:], strict=False))
        merged: list[tuple[str, int]] = []
        if order_ok:
            import bisect

            inserts = sorted(
                (float(self._original_events[o].start_sec), o) for o in chosen
            )
            cursor = 0
            for start, origin in inserts:
                stop = bisect.bisect_right(current_starts, start, lo=cursor)
                merged.extend(("old", p) for p in range(cursor, stop))
                merged.append(("new", origin))
                cursor = stop
            merged.extend(("old", p) for p in range(cursor, len(current_starts)))
        else:  # danh sách hiện tại đã lệch thứ tự (dữ liệu cũ): xếp theo dòng gốc như trước
            pending = list(chosen)
            for position, origin in enumerate(self._origins):
                while pending and 0 <= origin and pending[0] < origin:
                    merged.append(("new", pending.pop(0)))
                merged.append(("old", position))
            merged.extend(("new", o) for o in pending)
        events: list["SubtitleEvent"] = []
        new_origins: list[int] = []
        mapping: dict[int, int] = {}
        inserted: list[int] = []
        for kind, key in merged:
            if kind == "old":
                mapping[key] = len(events)
                events.append(self._source_events[key])
                new_origins.append(self._origins[key])
            else:
                base = self._original_events[key]
                inserted.append(len(events))
                events.append(self._make_event(
                    base, len(events) + 1, float(base.start_sec), float(base.end_sec),
                    str(getattr(base, "text", "") or ""), keep_identity=True,
                ))
                new_origins.append(key)
        self._source_events = events
        self._origins = new_origins
        self._remap_results(mapping)
        recovered: list[int] = []
        if self._last_results:
            for position in inserted:
                event = events[position]
                result = self._placeholder_result(
                    position, float(event.start_sec), float(event.end_sec), str(getattr(event, "text", "") or ""),
                )
                if not result.was_skipped:
                    recovered.append(position)
                self._last_results.append(result)
            self._last_results.sort(key=lambda r: r.event_index)
        self._after_structure_change(mapping)
        self._mark_edited([position for position in inserted if position not in recovered])
        return len(chosen)

    def _segments_dir(self) -> Path | None:
        for result in self._last_results:
            if result.segment_audio_path:
                return Path(result.segment_audio_path).parent
        return None

    def _placeholder_result(self, position: int, start: float, end: float, text: str) -> TTSSegmentResult:
        """[v3.23.654] Kết quả cho dòng chèn lại: lấy lại audio cũ từ thùng rác nếu có (khỏi TTS lại), không thì chờ TTS."""
        from subtitles_extractor.infrastructure.tts.segment_store import recover_from_trash

        folder = self._segments_dir()
        origin = self._origins[position] if 0 <= position < len(self._origins) else -1
        found = recover_from_trash(folder, text, position, origin) if folder is not None and origin >= 0 else None
        if found is not None:
            path, duration = found
            return TTSSegmentResult(
                event_index=position, start_sec=start, end_sec=end, text=text, subtitle_text=text,
                segment_audio_path=str(path), audio_duration_s=duration,
            )
        return TTSSegmentResult(
            event_index=position, start_sec=start, end_sec=end, text=text, subtitle_text=text, was_skipped=True,
        )

    @_undoable
    def apply_saved_edits(self, edits: LineEdits) -> bool:
        """[v3.23.654] Thay mọi sửa tay hiện tại bằng một phiên bản đã lưu (lịch sử); hoàn tác được.

        Chỉ khi chưa có kết quả TTS (audio từng dòng gắn với danh sách dòng hiện tại).
        """
        if self._is_busy or self._last_results or not self._original_events:
            return False
        self._source_events = list(self._original_events)
        self._origins = list(range(len(self._original_events)))
        self._speaker_overrides, self._addressee_overrides, self._text_edits = {}, {}, {}
        self._edited_since_synthesis, self._synthesizing = set(), set()
        self.restore_line_edits(edits)
        self._renumber_events()
        self._rederive_speakers()
        self.structure_changed.emit()
        self.line_edits_changed.emit()
        return True

    @_undoable
    def reset_to_original(self, *, keep_speakers: bool = True) -> bool:
        """Trả danh sách dòng về đúng phụ đề nguồn: đủ dòng gốc theo thứ tự gốc, bỏ dòng thêm tay, mốc và lời về gốc.

        Args:
            keep_speakers: Giữ người nói đã chọn tay cho các dòng gốc.

        Returns:
            ``True`` nếu có thay đổi.
        """
        if self._is_busy or not self._original_events:
            return False
        original = self._original_events
        if (
            self._origins == list(range(len(original))) and not self._text_edits
            and not self._timing_changed()
            and (keep_speakers or not (self._speaker_overrides or self._addressee_overrides))
        ):
            return False
        old_events = self._source_events
        mapping = {position: origin for position, origin in enumerate(self._origins) if origin >= 0}
        present = set(mapping.values())
        self._source_events = [
            self._make_event(base, index + 1, float(base.start_sec), float(base.end_sec),
                             str(getattr(base, "text", "") or ""), keep_identity=True)
            for index, base in enumerate(original)
        ]
        self._origins = list(range(len(original)))
        self._text_edits = {}
        if not keep_speakers:
            self._speaker_overrides = {}
            self._addressee_overrides = {}
        self._remap_results(mapping)
        changed_lines = [o for o in range(len(original)) if o not in present]
        for position, origin in mapping.items():
            before, base = old_events[position], original[origin]
            if (
                (before.start_sec, before.end_sec) != (base.start_sec, base.end_sec)
                or str(getattr(before, "text", "") or "") != str(getattr(base, "text", "") or "")
            ):
                changed_lines.append(origin)
        if self._last_results:
            by_index = {int(r.event_index): i for i, r in enumerate(self._last_results)}
            for origin in list(changed_lines):
                base = original[origin]
                text = str(getattr(base, "text", "") or "")
                fresh = (
                    self._placeholder_result(origin, float(base.start_sec), float(base.end_sec), text)
                    if origin not in by_index else TTSSegmentResult(
                        event_index=origin, start_sec=float(base.start_sec), end_sec=float(base.end_sec),
                        text=text, subtitle_text=text, was_skipped=True,
                    )
                )
                if not fresh.was_skipped:
                    changed_lines.remove(origin)
                if origin in by_index:
                    self._last_results[by_index[origin]] = replace(
                        self._last_results[by_index[origin]], text=text, subtitle_text=text,
                        start_sec=fresh.start_sec, end_sec=fresh.end_sec,
                        adjusted_start_sec=-1.0, adjusted_end_sec=-1.0,
                    )
                else:
                    self._last_results.append(fresh)
            self._last_results.sort(key=lambda r: r.event_index)
        self._after_structure_change(mapping)
        self._mark_edited(changed_lines)
        return True

    def suggested_line_timing(self, after_position: int | None) -> tuple[float, float]:
        """[v3.23.625] Mốc gợi ý cho dòng mới đặt ngay sau dòng ``after_position``."""
        duration = getattr(self._last_request, "media_duration_s", None)
        return default_line_timing(
            [float(e.start_sec) for e in self._source_events],
            [float(e.end_sec) for e in self._source_events],
            after_position, duration,
        )

    def _remap_results(
        self, mapping: dict[int, int], removed: int = 0,
        inserted: tuple[int, float, float, str] | None = None,
        origins_before: dict[int, int] | None = None,
    ) -> None:
        """Đánh số lại kết quả TTS (và giọng từng dòng của lần chạy gần nhất)."""
        if not self._last_results:
            return
        old_voices = (
            self._last_request.line_voices if self._last_request is not None else ()
        )
        # [v3.23.654] Audio dòng bị xoá vào thùng rác kèm dòng GỐC của nó (khôi phục đúng dòng, không lẫn hai câu "好").
        self._last_results = renumber_results(self._last_results, mapping, trash_keys=origins_before)
        if (
            self._last_adapter is not None and self._last_request is not None
            and self._last_output_path is not None
        ):
            # [v3.23.630] Cache theo đĩa phải đi theo audio vừa đổi tên.
            incremental_run.remap_saved_cache(
                self._last_request, self._last_adapter, self._last_output_path, mapping
            )
        if inserted is not None:
            position, start, end, text = inserted
            self._last_results.append(TTSSegmentResult(
                event_index=position, start_sec=start, end_sec=end, text=text,
                subtitle_text=text, was_skipped=True,
            ))
            self._last_results.sort(key=lambda r: r.event_index)
        if self._last_request is not None and old_voices:
            voices = [""] * len(self._source_events)
            for old, voice in enumerate(old_voices):
                if old in mapping and mapping[old] < len(voices):
                    voices[mapping[old]] = voice
            self._last_request = replace(self._last_request, line_voices=tuple(voices))

    @_undoable
    def set_line_timing(self, position: int, start_sec: float, end_sec: float) -> bool:
        """[v3.23.626] Đổi mốc thời gian của MỘT dòng (giữ nguyên thứ tự các dòng).

        Args:
            position: Vị trí dòng (0-based).
            start_sec: Mốc bắt đầu mới.
            end_sec: Mốc kết thúc mới, sau mốc bắt đầu.

        Returns:
            ``True`` nếu đã đổi; ``False`` nếu đang bận, mốc không hợp lệ, hoặc làm dòng đứng
            trước dòng kề (đổi thứ tự dòng: hãy xoá rồi thêm lại).
        """
        events = self._source_events
        if self._is_busy or not (0 <= position < len(events)):
            return False
        if start_sec < 0 or end_sec <= start_sec:
            return False
        if position > 0 and start_sec < events[position - 1].start_sec:
            return False
        if position + 1 < len(events) and start_sec > events[position + 1].start_sec:
            return False
        event = events[position]
        if (event.start_sec, event.end_sec) == (start_sec, end_sec):
            return True
        self._set_interval(position, start_sec, end_sec)
        self._mark_edited([position])
        self.structure_changed.emit()
        self.line_edits_changed.emit()
        return True

    def _set_interval(self, position: int, start_sec: float, end_sec: float) -> None:
        """Ghi mốc mới vào dòng nguồn và kết quả TTS cùng dòng (mốc đã xếp lại bị xoá)."""
        self._source_events[position] = replace(
            self._source_events[position], interval=TimeInterval(start_sec, end_sec)
        )
        for index, result in enumerate(self._last_results):
            if int(result.event_index) == position:
                self._last_results[index] = replace(
                    result, start_sec=start_sec, end_sec=end_sec,
                    adjusted_start_sec=-1.0, adjusted_end_sec=-1.0,
                )

    @_undoable
    def merge_lines(self, positions: Iterable[int]) -> int | None:
        """[v3.23.626] Gộp các dòng LIỀN NHAU thành một (lời nối bằng dấu cách, mốc phủ cả hai).

        Dòng đầu giữ người nói đã chọn; các dòng sau bị xoá. Dòng gộp cần TTS lại.

        Returns:
            Vị trí dòng gộp; ``None`` nếu ít hơn hai dòng, không liền nhau, hoặc đang bận.
        """
        chosen = self._valid_positions(positions)
        if self._is_busy or len(chosen) < 2:
            return None
        if chosen[-1] - chosen[0] != len(chosen) - 1:
            return None
        events = self._source_events
        first = chosen[0]
        merged_text = " ".join(
            str(getattr(events[p], "text", "") or "").strip() for p in chosen
        ).strip()
        start = float(events[first].start_sec)
        end = max(float(events[p].end_sec) for p in chosen)
        if not merged_text or end <= start:
            return None
        self.update_event_text(first, merged_text)
        self._set_interval(first, start, end)
        for index, result in enumerate(self._last_results):
            if int(result.event_index) == first:
                self._last_results[index] = replace(
                    result, text=merged_text, subtitle_text=merged_text
                )
        self.delete_lines(chosen[1:])
        self._mark_edited([first])
        return first

    @_undoable
    def split_line(
        self, position: int, text_a: str, text_b: str, split_sec: float
    ) -> int | None:
        """[v3.23.626] Tách một dòng thành hai tại ``split_sec``; dòng sau giữ người nói.

        Returns:
            Vị trí dòng mới (dòng sau); ``None`` nếu dữ liệu không hợp lệ hoặc đang bận.
        """
        events = self._source_events
        text_a, text_b = (text_a or "").strip(), (text_b or "").strip()
        if self._is_busy or not (0 <= position < len(events)) or not text_a or not text_b:
            return None
        start, end = float(events[position].start_sec), float(events[position].end_sec)
        if not (start < split_sec < end):
            return None
        speaker = self._speaker_overrides.get(position)
        self.update_event_text(position, text_a)
        self._set_interval(position, start, split_sec)
        for index, result in enumerate(self._last_results):
            if int(result.event_index) == position:
                self._last_results[index] = replace(
                    result, text=text_a, subtitle_text=text_a
                )
        new_position = self.insert_line(split_sec, end, text_b)
        if new_position is not None and speaker is not None:
            self._speaker_overrides[new_position] = speaker
            self.line_edits_changed.emit()
        addressee = self._addressee_overrides.get(position)
        if new_position is not None and addressee is not None:
            self._addressee_overrides[new_position] = addressee
            self.line_edits_changed.emit()
        self._mark_edited([position])
        return new_position

    # ── [v3.23.629] Dịch ↔ TTS: dòng không vừa khung ─────────────────────────

    def speech_misfits(self) -> list[LineFit]:
        """Dòng bản dịch KHÔNG vừa khung giọng đọc.

        Đã tổng hợp thì dùng số đo THẬT (nén ≥ 1,5×, bị cắt, chồng tiếng); chưa tổng hợp thì dự đoán
        theo mô hình thời lượng VieNeu (ngưỡng cao — bộ lập lịch còn dời câu / mượn khoảng lặng).
        """
        if self._last_results:
            return measured_misfits(self._source_events, self._last_results)
        return predict_misfits(self._source_events)

    def line_context(self, position: int) -> tuple[str, str, str]:
        """``(câu gốc, người nói, người nghe)`` của một dòng — ngữ cảnh cho bước rút gọn bằng AI.

        Rỗng ở phần không biết (chưa nạp ngữ cảnh phân vai của trang Dịch).
        """
        source = self._dubbing_source
        if not self._dubbing_prepared or source is None:
            return "", "", ""
        speakers = self.effective_line_speakers()
        speaker = speakers[position] if 0 <= position < len(speakers) else ""
        source_pos = (
            self._source_positions[position]
            if 0 <= position < len(self._source_positions) else None
        )
        text = (
            source.source_texts[source_pos]
            if source_pos is not None and 0 <= source_pos < len(source.source_texts) else ""
        )
        return text, speaker, self.line_addressee(position)

    def speaker_review_lines(self) -> list[ReviewLine]:
        """[v3.23.630] Mọi dòng kèm ngữ cảnh người nói (cho hộp thoại duyệt nhanh).

        Chưa nạp vai cho phụ đề này thì người nói/câu gốc/người nghe để rỗng.
        """
        speakers = self.effective_line_speakers()
        source = self._dubbing_source if self._dubbing_prepared else None
        lines: list[ReviewLine] = []
        for position, event in enumerate(self._source_events):
            source_pos = (
                self._source_positions[position]
                if position < len(self._source_positions) else None
            )
            origin = (
                source.source_texts[source_pos]
                if source is not None and source_pos is not None
                and 0 <= source_pos < len(source.source_texts) else ""
            )
            lines.append(ReviewLine(
                position=position, start_sec=float(event.start_sec),
                end_sec=float(event.end_sec), text=str(getattr(event, "text", "") or ""),
                source=origin, speaker=speakers[position] if position < len(speakers) else "",
                addressee=self.line_addressee(position),
            ))
        return lines

    def speaker_review_queue(self) -> list[ReviewItem]:
        """[v3.23.630] Dòng cần soát người nói, bằng chứng chắc nhất trước."""
        return review_queue(self.speaker_issues())

    def remerge_master(self) -> bool:
        """[v3.23.625] Ghép lại master từ các segment còn lại (sau khi xoá dòng), không TTS.

        Returns:
            ``True`` nếu đã khởi chạy; ``False`` khi chưa có kết quả, đang bận, hoặc lần TTS
            gần nhất không giữ audio từng dòng (Edge/Gemini: phải TTS lại cả phim).
        """
        return self.resynthesize_lines(set(), remerge_only=True)

    def load_source_from_file(self, source_path: Path) -> None:
        try:
            use_case = self._container.make_import_subtitles_use_case()
            events = use_case.execute(source_path)
            self.set_source_events(events)
            self.status_message.emit(f"Đã nạp {len(events)} dòng từ {source_path.name}")
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(f"Không đọc được tệp: {exc}")

    def check_engines(self) -> None:
        edge   = self._container.make_edge_tts_adapter()
        gemini = self._container.make_gemini_tts_adapter()
        vieneu = self._container.make_vieneu_tts_adapter()
        self.engine_available.emit(edge.get_engine_name(),   edge.is_available())
        self.engine_available.emit(gemini.get_engine_name(), gemini.is_available())
        self.engine_available.emit(vieneu.get_engine_name(), vieneu.is_available())

    def start_tts(
        self,
        engine: str,
        language: str = "",
        speaker: str = "",
        base_speed: float = 1.0,
        max_speed: float = 3.0,
        device: str = "auto",
        normalize: bool = True,
        output_path: Path | None = None,
        retry_count: int = 3,
        retry_delay_s: float = 1.0,
        # Timing controls
        clean_tags: bool = True,
        dialog_pause_ms: int = 300,
        max_overlap_ms: int = 500,
        skip_overlap_ms: int = 0,
        double_pass: bool = True,
        elastic_timing: bool = True,
        max_drift_s: float = 2.5,
        lead_in_s: float = 0.3,
        anchor_gap_s: float = 0.7,
        max_segment_s: float = 10.0,
        comfort_speed_ratio: float = 1.25,
        min_pause_ratio: float = 0.35,
        max_intra_gap_s: float = 0.5,
        timing_strategy: str = "lipsync",
        allow_audio_overlap: bool = True,
        min_stretch_ratio: float = 0.75,
        edge_concurrency: int = 16,
        last_line_max_extend_s: float = 0.0,
        skip_paren: bool = True,
        skip_square: bool = True,
        skip_curly: bool = True,
        skip_music_pair: bool = True,
        skip_music_line: bool = True,
        target_lufs: float = -16.0,
        voice_clarity: bool = True,
        high_quality: bool = True,
        output_format: str = "wav",
        output_bitrate_kbps: int = 320,
        wav_subtype: str = "PCM_16",
        # Gemini / VieNeu (voice cloning)
        api_key: str = "",
        ref_audio_path: str = "",
        ref_text: str = "",
        style_prompt: str = "",
        affective_dialog: bool = True,
        gemini_temperature: float | None = None,
        # VieNeu
        vieneu_mode: str = "v3turbo",
        vieneu_emotion: str = "natural",
        vieneu_force_cpu: bool = True,
        vieneu_batch_size: int = 0,
        vieneu_emotion_cues: bool = False,
        vieneu_precision: str = "",
        media_duration_s: float | None = None,
        line_voices: tuple[str, ...] = (),
        incremental: bool = True,
    ) -> bool:
        """Chạy TTS cả bộ phụ đề (hoặc chỉ các dòng đã đổi khi ``incremental``).

        [v3.23.630] ``incremental``: engine ghi được audio từng dòng (VieNeu) và đã có kết quả
        cũ cạnh file output → chỉ tổng hợp lại dòng có lời/mốc/giọng/cài đặt khác lần trước
        rồi ghép lại master. ``False`` luôn chạy lại cả phim.
        """
        if self._is_busy:
            self.status_message.emit("Đang bận, vui lòng chờ.")
            return False
        if not self._source_events:
            self.error_occurred.emit("Chưa có phụ đề nguồn để TTS.")
            return False
        if output_path is None:
            self.error_occurred.emit("Chưa chọn đường dẫn file đầu ra.")
            return False
        # Đồng bộ đuôi file với định dạng đã chọn (để hiển thị/mở đúng file thực).
        fmt = (output_format or "wav").lower()
        if output_path.suffix.lower() != f".{fmt}":
            output_path = output_path.with_suffix(f".{fmt}")
        # [v3.23.654] Lượt TTS mới (giọng/phân vai/tốc độ có thể đã đổi): audio trong thùng rác không còn đáng tin.
        from subtitles_extractor.infrastructure.tts.segment_store import clear_trash, segments_dir_for

        clear_trash(segments_dir_for(output_path))

        request = TTSRequest(
            # [v3.23.598] Bản SAO: người dùng sửa lời/người nói trong lúc đang chạy thì luồng
            # nền vẫn đọc trọn bộ dòng lúc bấm Tổng hợp (dòng sửa giữa chừng được đánh dấu
            # cần TTS lại).
            events=list(self._source_events),
            language=language, speaker=speaker,
            base_speed=base_speed, max_speed=max_speed,
            device=device, normalize=normalize,
            retry_count=retry_count, retry_delay_s=retry_delay_s,
            clean_tags=clean_tags, dialog_pause_ms=dialog_pause_ms,
            max_overlap_ms=max_overlap_ms, skip_overlap_ms=skip_overlap_ms,
            double_pass=double_pass,
            elastic_timing=elastic_timing, max_drift_s=max_drift_s, lead_in_s=lead_in_s,
            anchor_gap_s=anchor_gap_s, max_segment_s=max_segment_s,
            comfort_speed_ratio=comfort_speed_ratio, min_pause_ratio=min_pause_ratio,
            max_intra_gap_s=max_intra_gap_s,
            allow_audio_overlap=allow_audio_overlap, min_stretch_ratio=min_stretch_ratio,
            timing_strategy=timing_strategy,
            edge_concurrency=edge_concurrency,
            last_line_max_extend_s=last_line_max_extend_s,
            skip_paren=skip_paren, skip_square=skip_square, skip_curly=skip_curly,
            skip_music_pair=skip_music_pair, skip_music_line=skip_music_line,
            target_lufs=target_lufs, voice_clarity=voice_clarity,
            high_quality=high_quality,
            output_format=fmt, output_bitrate_kbps=output_bitrate_kbps,
            wav_subtype=wav_subtype,
            ref_audio_path=ref_audio_path, ref_text=ref_text,
            style_prompt=style_prompt, affective_dialog=affective_dialog,
            gemini_temperature=gemini_temperature,
            media_duration_s=media_duration_s,
            line_voices=tuple(line_voices),
            # [v3.23.450] Luôn giữ audio từng dòng ở trang TTS → cho phép nghe lại + TTS lại
            # dòng chọn. Chi phí: mỗi dòng một WAV nhỏ trong thư mục _segments cạnh output.
            keep_line_audio=True,
        )

        if engine == "gemini":
            adapter = self._container.make_gemini_tts_adapter(api_key=api_key)
        elif engine == "vieneu":
            # [v3.23.416/422] Giải quyết batch size (số CÂU gom mỗi lần gọi GPU):
            #   0 = tắt · -1 = Tự động (theo VRAM) · -2 = MỘT LÔ cho TẤT CẢ câu · ≥2 = cố định.
            resolved_batch = vieneu_batch_size
            if vieneu_batch_size == -2:
                # Số lớn hơn mọi phụ đề thực tế → look-ahead gom HẾT câu vào một lần gọi.
                resolved_batch = 10_000_000
                logger.info("VieNeu batch: MỘT LÔ cho toàn bộ câu (gom hết một lần).")
            elif vieneu_batch_size < 0:
                from subtitles_extractor.infrastructure.tts.vieneu_batch_tuning import (
                    suggest_auto_batch_size,
                )

                resolved_batch = suggest_auto_batch_size()
                logger.info("VieNeu batch: Tự động → %d câu/lô (theo VRAM).", resolved_batch)
            adapter = self._container.make_vieneu_tts_adapter(
                mode=vieneu_mode, emotion=vieneu_emotion, force_cpu=vieneu_force_cpu,
                batch_size=resolved_batch, emotion_cues=vieneu_emotion_cues,
                precision=vieneu_precision or None,
            )
        else:
            adapter = self._container.make_edge_tts_adapter()

        if not adapter.is_available():
            install_map = {
                "edge": "edge-tts soundfile",
                "gemini": "google-genai soundfile",
                "vieneu": "vieneu soundfile",
            }
            self.error_occurred.emit(
                f"{adapter.get_engine_name()} chưa sẵn sàng.\n"
                f"Cài đặt: pip install {install_map.get(engine, 'soundfile')}"
            )
            return False

        prepared = incremental_run.prepare_run(request, adapter, output_path, incremental)
        if prepared is not None and prepared.plan.mode is not PlanMode.FULL:
            return self._start_incremental(prepared, adapter, output_path)
        self._run = prepared
        self._reuse_summary = None
        if prepared is not None:
            logger.info(
                "TTS cả phim (%s): %d dòng.", prepared.plan.reason, prepared.plan.total
            )
            # [v3.23.647] Điểm kiểm tra giữa lượt: huỷ/sập/mất điện sau 15 phút thì lần sau chạy tiếp từ dòng đã xong.
            setter = getattr(adapter, "set_checkpoint_callback", None)
            if setter is not None:
                setter(lambda results, rate, run=prepared: incremental_run.record_checkpoint(run, results, rate))

        use_case = self._container.make_generate_tts_use_case(adapter)
        from subtitles_extractor.presentation.workers.tts_worker import TTSWorker
        self._worker = TTSWorker(use_case, request, output_path)
        self._run_rewrites_line_audio = True  # [v3.23.659] xem _forget_stale_line_audio
        self._run_started_at = time.time()
        self._last_request = request  # snapshot cấu hình để xuất debug
        self._last_engine_name = adapter.get_engine_name()
        self._last_adapter = adapter  # [v3.23.450] để TTS lại dòng chọn không phải tạo lại
        self._worker.progress_changed.connect(self.progress_changed.emit)
        self._worker.finished_ok.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.cancelled.connect(self._on_worker_cancelled)
        self._worker.finished.connect(self._on_thread_finished)
        self._set_busy(True)
        self._last_output_path = output_path
        self._synthesizing = set(self._edited_since_synthesis)
        self._worker.start()
        dubbing_note = (
            f" — lồng tiếng {len({v for v in request.line_voices if v})} giọng"
            if request.line_voices else ""
        )
        self.status_message.emit(
            f"Đang TTS {len(self._source_events)} dòng bằng "
            f"{adapter.get_engine_name()}{dubbing_note}..."
        )
        return True

    def _start_incremental(
        self, run: incremental_run.PreparedRun, adapter: object, output_path: Path
    ) -> bool:
        """[v3.23.630] Chỉ tổng hợp lại dòng đã đổi (hoặc chỉ ghép lại master) rồi trả kết quả.

        Dòng không đổi dùng lại audio đã có (``prepare_run`` đã dời đúng tên); lượt chạy đi
        qua đúng đường "TTS lại dòng chọn" nên ghép master/phụ đề y như các lượt đó.
        """
        from subtitles_extractor.presentation.workers.resynthesize_worker import (
            ResynthesizeWorker,
        )

        plan = run.plan
        unchanged = plan.mode is PlanMode.UNCHANGED
        setter = getattr(adapter, "set_checkpoint_callback", None)
        if setter is not None:
            setter(None)
        use_case = self._container.make_resynthesize_use_case(adapter)
        self._worker = ResynthesizeWorker(
            use_case,
            full_request=run.request,
            selected_indices=set(plan.stale),
            prior_results=list(run.prior_results),
            output_path=output_path,
            sample_rate=run.sample_rate,
            total_length_samples=run.total_length_samples,
            remerge_only=unchanged,
            # [v3.23.647] Lượt tiếp tục sau khi bị ngắt có thể còn hàng nghìn dòng: cũng ghi điểm kiểm tra.
            checkpoint_cb=None if unchanged else (
                lambda results, rate, prepared_run=run: incremental_run.record_checkpoint(
                    prepared_run, results, rate
                )
            ),
        )
        self._run = run
        self._run_rewrites_line_audio = not unchanged  # [v3.23.659] xem _forget_stale_line_audio
        self._run_started_at = time.time()
        self._reuse_summary = (plan.reused_count, plan.total, len(plan.stale))
        self._last_request = run.request
        self._last_engine_name = adapter.get_engine_name()  # type: ignore[attr-defined]
        self._last_adapter = adapter
        self._last_output_path = output_path
        self._worker.progress_changed.connect(self.progress_changed.emit)
        self._worker.finished_ok.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.cancelled.connect(self._on_worker_cancelled)
        self._worker.finished.connect(self._on_thread_finished)
        self._set_busy(True)
        # Dòng đã sửa mà chữ ký khớp cache = audio đã đúng bản sửa → cũng hết "cần TTS lại".
        self._synthesizing = set(self._edited_since_synthesis)
        self._worker.start()
        reused = plan.reused_count
        logger.info(
            "TTS tăng dần: dùng lại %d/%d dòng, tổng hợp lại %d dòng.",
            reused, plan.total, len(plan.stale),
        )
        self.status_message.emit(
            f"Không dòng nào đổi ({plan.total} dòng) — chỉ ghép lại audio..." if unchanged
            else f"Dùng lại audio {reused}/{plan.total} dòng, tổng hợp lại "
                 f"{len(plan.stale)} dòng đã đổi..."
        )
        return True

    def cancel_tts(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.request_cancel()
            self.status_message.emit("Đang huỷ TTS...")

    def _clear_checkpoint_callback(self) -> None:
        """[v3.23.647] Gỡ hàm điểm kiểm tra khỏi adapter (nó chỉ có nghĩa cho đúng một lượt cả phim)."""
        setter = getattr(getattr(self, "_last_adapter", None), "set_checkpoint_callback", None)
        if setter is not None:
            setter(None)

    def _record_line_cache(self, results: list[TTSSegmentResult]) -> None:
        """[v3.23.630] Ghi cache từng dòng sau lượt chạy (lỗi ghi cache không chặn kết quả)."""
        run, self._run = self._run, None
        manual, self._manual_run = self._manual_run, None
        self._clear_checkpoint_callback()
        try:
            if run is not None:
                incremental_run.record_run(run, results)
            elif manual is not None and self._last_adapter is not None and (
                self._last_output_path is not None
            ):
                incremental_run.record_selected(
                    manual[0], self._last_adapter, self._last_output_path,
                    results, set(manual[1]),
                )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Không cập nhật được cache TTS từng dòng: %s", exc)

    def _on_worker_finished(self, results: object) -> None:
        self._run_rewrites_line_audio = False
        self._last_results = list(results)  # type: ignore[arg-type]
        self._record_line_cache(self._last_results)
        self._undo_stack, self._redo_stack = [], []
        # [v3.23.598] Dòng sửa tay mà lượt này đã đọc bản mới thì hết "cần TTS lại".
        finished_edits = self._synthesizing & self._edited_since_synthesis
        self._edited_since_synthesis -= self._synthesizing
        self._synthesizing = set()
        self.result_ready.emit(self._last_results)
        if finished_edits:
            self.line_edits_changed.emit()

    def _forget_stale_line_audio(self) -> None:
        """[v3.23.659] Lượt TTS cả phim bị huỷ / lỗi giữa chừng: audio từng dòng trên đĩa ĐÃ là của lượt mới.

        Kết quả cũ (mốc thời gian, thời lượng của lượt trước) mà ghép với tệp mới thì "TTS lại dòng chọn" / xoá dòng
        / khôi phục dòng đặt audio mới vào mốc cũ (giọng có thể đổi giữa phim). Bỏ liên kết audio từng dòng — muốn
        sửa từng dòng thì chạy TTS lại (lượt sau tự dùng lại các dòng đã xong qua điểm kiểm tra).
        """
        if not getattr(self, "_run_rewrites_line_audio", False):
            return
        self._run_rewrites_line_audio = False
        started = float(getattr(self, "_run_started_at", 0.0))
        # Chỉ bỏ liên kết khi tệp từng dòng THẬT SỰ bị lượt này ghi lại (huỷ sớm / ghi ra thư mục khác → giữ nguyên).
        if any(_rewritten_since(getattr(r, "segment_audio_path", ""), started) for r in self._last_results):
            self._last_results = [
                replace(r, segment_audio_path="") if r.segment_audio_path else r for r in self._last_results
            ]
            logger.info("TTS bị dừng giữa chừng — bỏ liên kết audio từng dòng của lượt trước (không còn khớp).")

    def _on_worker_failed(self, message: str) -> None:
        self._run, self._manual_run = None, None
        self._clear_checkpoint_callback()
        self._forget_stale_line_audio()
        self.error_occurred.emit(message)

    def _on_worker_cancelled(self) -> None:
        self._run, self._manual_run = None, None
        self._clear_checkpoint_callback()
        self._forget_stale_line_audio()
        self.tts_cancelled.emit()

    def _on_thread_finished(self) -> None:
        # [1.2] Chống race: chỉ luồng HIỆN HÀNH (đang phát tín hiệu) mới được đổi
        # trạng thái UI. Nếu người dùng Huỷ rồi Bắt đầu ngay, tín hiệu 'finished' trễ
        # của luồng cũ không được phép gỡ busy của luồng mới.
        sender = self.sender()
        if sender is not None and sender is not self._worker and self._worker is not None:
            # Luồng cũ kết thúc muộn: chỉ dọn chính nó, không đụng trạng thái hiện tại.
            sender.deleteLater()
            return
        self._set_busy(False)
        worker = self._worker
        self._worker = None
        if worker is not None:
            # [1.1] Tín hiệu 'finished' chỉ phát SAU khi run() kết thúc, nên luồng đã
            # dừng hẳn — KHÔNG cần worker.wait() (vốn chặn luồng giao diện tới 5s).
            # Dùng deleteLater() chuẩn của Qt để dọn an toàn ngoài vòng sự kiện.
            worker.deleteLater()
            logger.debug("Đã dọn luồng TTS.")

    def _set_busy(self, busy: bool) -> None:
        if busy != self._is_busy:
            self._is_busy = busy
            self.busy_changed.emit(busy)

    def cleanup(self) -> None:
        """Dừng worker an toàn khi đóng/chuyển trang.

        Ngắt mọi kết nối signal TRƯỚC để slot không chạy sau khi VM/page đã bị
        huỷ (tránh crash do truy cập widget đã xoá). Terminate nếu không dừng kịp
        (tránh treo ứng dụng vô hạn khi API chậm).
        """
        worker = self._worker
        self._worker = None
        if worker is None:
            return
        # Ngắt tất cả signal — chặn slot chạy sau khi đối tượng đã bị huỷ.
        for signal in (
            worker.progress_changed, worker.finished_ok,
            worker.failed, worker.cancelled, worker.finished,
        ):
            try:
                signal.disconnect()
            except (TypeError, RuntimeError):
                pass  # chưa kết nối hoặc đã ngắt — bỏ qua
        if worker.isRunning():
            worker.request_cancel()
            if not worker.wait(8000):
                logger.warning("TTSWorker không dừng kịp trong 8s — buộc dừng.")
                worker.terminate()
                worker.wait(2000)
        worker.deleteLater()


__all__ = ["TTSPageViewModel"]
