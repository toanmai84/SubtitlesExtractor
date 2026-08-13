"""Tuần tự hoá Visual Cues để lưu/khôi phục qua QSettings (Auto-Save).

Tách riêng phần logic *thuần* (không phụ thuộc Qt) khỏi widget để dễ kiểm thử:
widget chỉ cần gọi :func:`serialize_visual_cues` trước khi ghi ``QSettings`` và
:func:`deserialize_visual_cues` khi nạp lại. Nhờ vậy chỉnh sửa của người dùng trên
bảng Visual Cues không mất khi tắt/mở lại ứng dụng.
"""

from __future__ import annotations

import json

from subtitles_extractor.domain.ports.subtitle_translator_port import VisualCue


def serialize_visual_cues(cues: list[VisualCue]) -> str:
    """Chuyển danh sách :class:`VisualCue` thành chuỗi JSON gọn (khoá rút gọn).

    Args:
        cues: Danh sách gợi ý hình ảnh (có thể đã được người dùng chỉnh sửa).

    Returns:
        Chuỗi JSON UTF-8 an toàn để lưu vào ``QSettings``.
    """
    payload = [
        {"id": cue.line_no, "spk": cue.speaker, "to": cue.addressee, "cue": cue.scene}
        for cue in cues
    ]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def deserialize_visual_cues(raw: str | None) -> list[VisualCue]:
    """Khôi phục danh sách :class:`VisualCue` từ chuỗi JSON đã lưu.

    Bỏ qua phần tử hỏng/sai định dạng thay vì ném lỗi, để một bản ghi lỗi không
    làm mất toàn bộ dữ liệu đã lưu của người dùng.

    Args:
        raw: Chuỗi JSON đã lưu (hoặc ``None``/rỗng).

    Returns:
        Danh sách :class:`VisualCue` khôi phục được (theo thứ tự ``line_no``).
    """
    if not raw:
        return []
    try:
        items = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(items, list):
        return []

    cues: list[VisualCue] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            line_no = int(item.get("id", item.get("line_no", 0)))
        except (TypeError, ValueError):
            continue
        if line_no <= 0:
            continue
        cues.append(
            VisualCue(
                line_no=line_no,
                speaker=str(item.get("spk", item.get("speaker", "")) or ""),
                addressee=str(item.get("to", item.get("addressee", "")) or ""),
                scene=str(item.get("cue", item.get("scene", "")) or ""),
            )
        )
    cues.sort(key=lambda cue: cue.line_no)
    return cues
