"""[v3.23.609] Dòng người dùng sửa trên BẢNG KẾT QUẢ (bản dịch cuối) — giữ khi dịch lại.

Từ 3.23.608 bấm Dịch lại chỉ mất vài giây (các giai đoạn lấy từ bộ nhớ đệm) nên dịch lại
sau khi sửa một giai đoạn là thao tác thường gặp. Bản dịch cuối được dựng lại từ các giai
đoạn, nên trước bản này mọi dòng người dùng đã sửa (hay đã "Dịch lại các dòng đã chọn")
trên bảng kết quả bị ghi đè mà không báo gì.

Mỗi dòng sửa ghi:

* câu người dùng chốt;
* câu máy dịch gần nhất (để "Bỏ sửa tay" trả về đúng bản máy MỚI NHẤT);
* vân tay câu gốc lúc sửa — câu gốc dòng đó đổi (sửa OCR, thêm/bớt dòng) thì bỏ câu sửa;
* ngôn ngữ đích — câu sửa tiếng Việt không được áp vào bản dịch tiếng Anh.

Các hàm ở đây là hàm thuần (không I/O), dễ kiểm thử.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from subtitles_extractor.application.services.stage_checkpoint import (
    line_input_digest,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent

__all__ = [
    "FINAL_EDITS_STAGE_ID",
    "FinalEditsOutcome",
    "FinalLineEdit",
    "apply_final_edits",
    "discard_final_edits",
    "final_edits_from_json",
    "final_edits_to_json",
    "record_final_edit",
]

#: Mã "giai đoạn" lưu các dòng sửa trong phiên dịch (cạnh 'final' và các giai đoạn).
FINAL_EDITS_STAGE_ID = "final_edits"


@dataclass(frozen=True)
class FinalLineEdit:
    """Một dòng người dùng chốt trên bảng kết quả.

    Attributes:
        index: Chỉ số câu gốc của dòng.
        text: Câu người dùng chốt.
        auto_text: Câu máy dịch gần nhất cho dòng này.
        source_digest: Vân tay câu gốc lúc sửa.
        target_lang: Ngôn ngữ đích của bản dịch được sửa (rỗng = không rõ).
    """

    index: int
    text: str
    auto_text: str
    source_digest: str
    target_lang: str = ""

    def to_json(self) -> dict[str, Any]:
        """Dạng lưu phiên."""
        return {
            "index": self.index,
            "text": self.text,
            "auto": self.auto_text,
            "h": self.source_digest,
            "lang": self.target_lang,
        }

    @classmethod
    def from_json(cls, raw: Any) -> FinalLineEdit | None:
        """Đọc từ phiên; ``None`` nếu dữ liệu hỏng."""
        if not isinstance(raw, dict):
            return None
        try:
            index = int(raw.get("index"))
        except (TypeError, ValueError):
            return None
        return cls(
            index=index,
            text=str(raw.get("text") or ""),
            auto_text=str(raw.get("auto") or ""),
            source_digest=str(raw.get("h") or ""),
            target_lang=str(raw.get("lang") or ""),
        )


@dataclass(frozen=True)
class FinalEditsOutcome:
    """Kết quả áp các dòng sửa lên một bản dịch mới.

    Attributes:
        events: Bản dịch sau khi áp.
        edits: Dòng sửa còn giữ (câu máy đã cập nhật theo bản dịch mới).
        applied: Chỉ số câu gốc của các dòng vừa được áp.
        dropped: Chỉ số các dòng sửa bị bỏ vì câu gốc đã đổi.
    """

    events: list[SubtitleEvent]
    edits: dict[int, FinalLineEdit]
    applied: tuple[int, ...]
    dropped: tuple[int, ...]


def record_final_edit(
    edits: Mapping[int, FinalLineEdit],
    *,
    index: int,
    new_text: str,
    current_text: str,
    source_text: str,
    target_lang: str,
) -> dict[int, FinalLineEdit]:
    """Ghi (hoặc bỏ) dòng sửa sau khi người dùng đổi câu của dòng ``index``.

    Args:
        edits: Dòng sửa hiện có.
        index: Chỉ số câu gốc của dòng.
        new_text: Câu mới.
        current_text: Câu đang hiện trước khi sửa.
        source_text: Câu gốc của dòng.
        target_lang: Ngôn ngữ đích của bản dịch.

    Returns:
        Bản sao ``edits`` đã cập nhật (sửa về đúng câu máy = bỏ dòng sửa).
    """
    updated = dict(edits)
    existing = updated.get(index)
    auto_text = existing.auto_text if existing is not None else current_text
    if new_text == auto_text:
        updated.pop(index, None)
        return updated
    updated[index] = FinalLineEdit(
        index=index,
        text=new_text,
        auto_text=auto_text,
        source_digest=line_input_digest(source_text),
        target_lang=target_lang or (existing.target_lang if existing else ""),
    )
    return updated


def apply_final_edits(
    events: Sequence[SubtitleEvent],
    source_events: Sequence[SubtitleEvent],
    edits: Mapping[int, FinalLineEdit],
    target_lang: str,
) -> FinalEditsOutcome:
    """Áp các dòng sửa lên bản dịch vừa dựng lại (sau một lượt dịch).

    Dòng sửa của ngôn ngữ khác được giữ nguyên trong danh sách nhưng không áp; dòng sửa
    có câu gốc đã đổi bị bỏ.

    Args:
        events: Bản dịch mới (theo thứ tự câu gốc).
        source_events: Câu gốc.
        edits: Dòng sửa đã lưu.
        target_lang: Ngôn ngữ đích của bản dịch mới.

    Returns:
        :class:`FinalEditsOutcome`.
    """
    source_by_index = {event.index: event for event in source_events}
    kept = dict(edits)
    applied: list[int] = []
    dropped: list[int] = []
    for index, edit in edits.items():
        source = source_by_index.get(index)
        if source is None or line_input_digest(source.text) != edit.source_digest:
            kept.pop(index, None)
            dropped.append(index)
    result: list[SubtitleEvent] = []
    for event in events:
        edit = kept.get(event.index)
        if edit is None or (
            edit.target_lang and target_lang and edit.target_lang != target_lang
        ):
            result.append(event)
            continue
        kept[event.index] = replace(edit, auto_text=event.text)
        if event.text != edit.text:
            event = replace(event, text=edit.text)
        applied.append(edit.index)
        result.append(event)
    return FinalEditsOutcome(result, kept, tuple(applied), tuple(sorted(dropped)))


def discard_final_edits(
    events: Sequence[SubtitleEvent],
    edits: Mapping[int, FinalLineEdit],
    indices: Iterable[int],
) -> tuple[list[SubtitleEvent], dict[int, FinalLineEdit], tuple[int, ...]]:
    """Trả các dòng sửa về câu máy dịch gần nhất.

    Args:
        events: Bản dịch hiện tại.
        edits: Dòng sửa hiện có.
        indices: Chỉ số câu gốc của các dòng cần trả lại.

    Returns:
        ``(bản dịch, dòng sửa còn lại, các chỉ số đã trả lại)``.
    """
    wanted = {index for index in indices if index in edits}
    if not wanted:
        return list(events), dict(edits), ()
    result = [
        replace(event, text=edits[event.index].auto_text)
        if event.index in wanted else event
        for event in events
    ]
    remaining = {index: edit for index, edit in edits.items() if index not in wanted}
    return result, remaining, tuple(sorted(wanted))


def final_edits_to_json(edits: Mapping[int, FinalLineEdit]) -> str:
    """Dòng sửa → JSON lưu phiên."""
    return json.dumps(
        [edits[index].to_json() for index in sorted(edits)], ensure_ascii=False
    )


def final_edits_from_json(text: str | None) -> dict[int, FinalLineEdit]:
    """JSON lưu phiên → dòng sửa (bỏ qua mục hỏng)."""
    if not text:
        return {}
    try:
        rows = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(rows, list):
        return {}
    edits: dict[int, FinalLineEdit] = {}
    for row in rows:
        edit = FinalLineEdit.from_json(row)
        if edit is not None:
            edits[edit.index] = edit
    return edits
