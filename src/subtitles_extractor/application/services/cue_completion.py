"""[v3.23.600] Đủ NGƯỜI NÓI cho mọi dòng có lời khi phân tích/dịch có kèm video.

Lồng tiếng đọc mỗi dòng bằng giọng của người nói dòng đó; dòng không có người nói chỉ đọc
được bằng giọng chung. Trước bản này:

* Khâu dịch chỉ quét bổ sung Visual Cues khi nhãn dùng được phủ DƯỚI 60% số dòng. Phủ
  97% là thôi — 3% còn lại không bao giờ được hỏi: lượt dịch mới nhất phim "傻子落水…"
  còn 103/3902 dòng, phim "he thong sung the" 125/5540 dòng không có người nói (nhật ký:
  "Dùng 5415 Visual Cues từ phân tích ngữ cảnh (đã có sẵn, không quét lại video)").
* Dòng chỉ có người NGHE (người nói trống) vẫn được tính là "đã có nhãn".
* Khâu phân tích không tự hỏi lại dòng nó bỏ trống.

Mô-đun (hàm thuần; việc gọi model được tiêm vào qua ``scan``) tìm các dòng thiếu người
nói, hỏi lại đúng các dòng đó (tối đa :data:`SPEAKER_COMPLETION_PASSES` lượt — lượt sau
chỉ hỏi dòng model còn bỏ sót), và chỉ ĐIỀN chỗ trống: nhãn đã có (có thể người dùng đã
sửa tay) luôn được giữ.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from subtitles_extractor.application.services.cue_contradiction_guard import (
    contradicted_cue_lines,
)
from subtitles_extractor.application.services.cue_label_canonicalizer import (
    canonicalize_cue_labels,
)
from subtitles_extractor.application.services.cue_run_filler import LineSpan
from subtitles_extractor.application.services.fill_support_guard import (
    demote_unsupported_fills,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationLine,
    VisualCue,
)
from subtitles_extractor.domain.value_objects.speaker_label import VOICE_ONLY_SPEAKERS

#: Số lượt hỏi lại tối đa: lượt 2 chỉ hỏi các dòng lượt 1 còn bỏ sót.
SPEAKER_COMPLETION_PASSES = 2

#: ``scan(dòng cần hỏi, nhãn đã biết làm mạch hội thoại) -> nhãn trả về``.
CueScanner = Callable[[list[TranslationLine], list[VisualCue]], list[VisualCue]]


@dataclass(frozen=True, slots=True)
class SpeakerCoverage:
    """Độ phủ người nói trên các dòng có lời.

    Attributes:
        spoken_lines: Số dòng có chữ (cần người nói).
        with_speaker: Số dòng đã có người nói.
        voice_only: Trong số đó, số dòng chỉ biết GIỌNG (nhãn "Giọng nam chưa rõ"…).
        missing: Số thứ tự các dòng chưa có người nói.
    """

    spoken_lines: int
    with_speaker: int
    voice_only: int
    missing: tuple[int, ...]

    @property
    def ratio(self) -> float:
        """Tỉ lệ dòng có người nói (1.0 khi không có dòng nào cần)."""
        return self.with_speaker / self.spoken_lines if self.spoken_lines else 1.0


def lines_missing_speaker(
    cues: Sequence[VisualCue], lines: Sequence[TranslationLine]
) -> list[TranslationLine]:
    """Dòng có chữ mà chưa có người nói (dòng trống không cần lồng tiếng).

    Args:
        cues: Visual Cues hiện có.
        lines: Toàn bộ dòng phụ đề.

    Returns:
        Các dòng thiếu người nói, theo thứ tự.
    """
    with_speaker = {cue.line_no for cue in cues if cue.speaker.strip()}
    return [
        line for line in lines
        if (line.text or "").strip() and line.index not in with_speaker
    ]


def fill_missing_speakers(
    existing: Sequence[VisualCue], scanned: Sequence[VisualCue]
) -> tuple[list[VisualCue], list[int]]:
    """Điền người nói từ ``scanned`` vào các dòng CHƯA có người nói.

    Dòng đã có người nói giữ nguyên. Dòng chỉ có người nghe giữ người nghe cũ (nhận người
    nghe/mô tả mới chỉ khi ô cũ trống).

    Args:
        existing: Visual Cues hiện có.
        scanned: Nhãn vừa hỏi được.

    Returns:
        ``(cues sau khi điền, các dòng vừa được điền)``.
    """
    by_line = {cue.line_no: cue for cue in existing}
    filled: list[int] = []
    for cue in scanned:
        if not cue.speaker.strip():
            continue
        current = by_line.get(cue.line_no)
        if current is not None and current.speaker.strip():
            continue
        by_line[cue.line_no] = cue if current is None else replace(
            current,
            speaker=cue.speaker,
            addressee=current.addressee or cue.addressee,
            scene=current.scene or cue.scene,
        )
        filled.append(cue.line_no)
    merged = sorted(by_line.values(), key=lambda cue: cue.line_no)
    return merged, sorted(set(filled))


def complete_speakers(
    cues: Sequence[VisualCue],
    lines: Sequence[TranslationLine],
    scan: CueScanner,
    passes: int = SPEAKER_COMPLETION_PASSES,
) -> tuple[list[VisualCue], list[int]]:
    """Hỏi lại các dòng thiếu người nói cho tới khi đủ hoặc hết lượt.

    Args:
        cues: Visual Cues hiện có.
        lines: Toàn bộ dòng phụ đề.
        scan: Hàm hỏi model (lỗi API thì trả danh sách rỗng — không làm hỏng khâu chính).
        passes: Số lượt hỏi tối đa.

    Returns:
        ``(cues sau khi điền, mọi dòng đã được điền)``.
    """
    current = list(cues)
    filled_all: list[int] = []
    for _pass in range(max(0, passes)):
        missing = lines_missing_speaker(current, lines)
        if not missing:
            break
        known = [cue for cue in current if cue.speaker or cue.addressee]
        current, filled = fill_missing_speakers(current, scan(missing, known))
        filled_all += filled
        if not filled:
            break
    return current, sorted(set(filled_all))


def drop_contradicted_fills(
    original: Sequence[VisualCue],
    completed: Sequence[VisualCue],
    filled: Sequence[int],
    lines: Sequence[TranslationLine],
    characters: str,
) -> tuple[list[VisualCue], list[int]]:
    """Bỏ nhãn vừa điền mà "lạc cảnh" (lưới v3.23.581), trả lại nhãn cũ của dòng đó.

    Nhãn hỏi lại đến từ bộ quét — loại nhãn từng sai kiểu "lạc cảnh" (gán cho cặp nhân vật
    của một cảnh khác). Dòng bị bỏ nhãn giữ lại cue cũ (vd chỉ có người nghe) hoặc trống.

    Args:
        original: Cues trước khi điền.
        completed: Cues sau khi điền.
        filled: Các dòng vừa điền.
        lines: Toàn bộ dòng phụ đề (mốc thời gian + câu gốc).
        characters: Bảng nhân vật.

    Returns:
        ``(cues, các dòng bị bỏ nhãn vừa điền)``.
    """
    if not filled:
        return list(completed), []
    spans = {
        line.index: LineSpan(line.index, line.start_ms, line.end_ms) for line in lines
    }
    sources = {line.index: (line.original_text or line.text or "") for line in lines}
    dropped = contradicted_cue_lines(
        canonicalize_cue_labels(list(completed), characters),
        set(filled), spans, sources, characters,
    )
    if not dropped:
        return list(completed), []
    previous = {cue.line_no: cue for cue in original}
    removed = set(dropped)
    kept = [cue for cue in completed if cue.line_no not in removed]
    kept += [previous[line] for line in dropped if line in previous]
    return sorted(kept, key=lambda cue: cue.line_no), sorted(dropped)


def speaker_coverage(
    cues: Sequence[VisualCue], lines: Sequence[TranslationLine]
) -> SpeakerCoverage:
    """Đo độ phủ người nói (để ghi nhật ký, chuẩn đoán và báo người dùng)."""
    spoken = [line for line in lines if (line.text or "").strip()]
    speakers = {cue.line_no: cue.speaker.strip() for cue in cues if cue.speaker.strip()}
    missing = tuple(line.index for line in spoken if line.index not in speakers)
    voice_only = sum(
        1 for line in spoken if speakers.get(line.index, "") in VOICE_ONLY_SPEAKERS
    )
    return SpeakerCoverage(
        spoken_lines=len(spoken),
        with_speaker=len(spoken) - len(missing),
        voice_only=voice_only,
        missing=missing,
    )


@dataclass(frozen=True, slots=True)
class SpeakerCompletionOutcome:
    """[v3.23.601] Kết quả một lượt hỏi lại người nói (đã qua lưới "lạc cảnh").

    Attributes:
        cues: Visual Cues sau khi điền.
        before: Độ phủ trước khi hỏi.
        after: Độ phủ sau khi hỏi.
        dropped: Các dòng có nhãn vừa điền bị bỏ vì mâu thuẫn hội thoại quanh nó.
        demoted: Các dòng có nhãn tên vừa điền bị hạ xuống nhãn giọng.
    """

    cues: list[VisualCue]
    before: SpeakerCoverage
    after: SpeakerCoverage
    dropped: tuple[int, ...] = ()
    #: [v3.23.602] Dòng có nhãn TÊN vừa hỏi lại nhưng không có chỗ dựa trong cảnh → đã hạ
    #: xuống nhãn giọng (xem ``fill_support_guard``).
    demoted: tuple[int, ...] = ()

    @property
    def changed(self) -> bool:
        """Có thêm dòng nào được điền người nói không."""
        return self.after.with_speaker != self.before.with_speaker


def complete_and_check_speakers(
    cues: Sequence[VisualCue],
    lines: Sequence[TranslationLine],
    characters: str,
    scan: CueScanner,
    passes: int = SPEAKER_COMPLETION_PASSES,
) -> SpeakerCompletionOutcome:
    """[v3.23.601] Hỏi lại các dòng thiếu người nói rồi bỏ nhãn vừa điền mà "lạc cảnh".

    Gộp chung cho khâu phân tích, khâu dịch và nút "Bổ sung người nói" (phân tích khôi
    phục từ phiên cũ chưa từng được hỏi lại).

    Args:
        cues: Visual Cues hiện có (đã quy nhãn về bảng nhân vật).
        lines: Toàn bộ dòng phụ đề.
        characters: Bảng nhân vật.
        scan: Hàm hỏi model.
        passes: Số lượt hỏi tối đa.

    Returns:
        :class:`SpeakerCompletionOutcome`; không gọi ``scan`` khi đã đủ người nói.
    """
    original = list(cues)
    before = speaker_coverage(original, lines)
    if not before.missing:
        return SpeakerCompletionOutcome(original, before, before)
    completed, filled = complete_speakers(original, lines, scan, passes)
    completed, dropped = drop_contradicted_fills(
        original, completed, filled, lines, characters
    )
    # [v3.23.602] Tên người nói vừa hỏi lại phải có mặt trong cảnh (fill_support_guard).
    completed, demoted = demote_unsupported_fills(
        original, completed, [line for line in filled if line not in set(dropped)],
        characters,
    )
    return SpeakerCompletionOutcome(
        completed, before, speaker_coverage(completed, lines), tuple(dropped),
        tuple(finding.line_no for finding in demoted),
    )


__all__ = [
    "SPEAKER_COMPLETION_PASSES",
    "CueScanner",
    "SpeakerCompletionOutcome",
    "SpeakerCoverage",
    "complete_and_check_speakers",
    "complete_speakers",
    "drop_contradicted_fills",
    "fill_missing_speakers",
    "lines_missing_speaker",
    "speaker_coverage",
]
