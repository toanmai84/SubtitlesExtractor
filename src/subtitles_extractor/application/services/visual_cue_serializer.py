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


def remap_visual_cues_to_subset(raw: str | None, positions: list[int]) -> str:
    """[v3.23.609] Visual Cues cho một TẬP CON dòng, đánh số lại theo tập con (hàm thuần).

    Cues gắn theo số thứ tự dòng của cả phim. "Dịch lại các dòng đã chọn" gửi tập con mà
    khâu dịch đánh số lại 1..k, nên nếu chuyển nguyên cues thì dòng thứ nhất của tập con
    (vd dòng 723 của phim) nhận người nói/người nghe của dòng 1 của phim → sai xưng hô.

    Args:
        raw: JSON Visual Cues của cả phim.
        positions: Vị trí (từ 0) trong cả phim của từng dòng tập con, theo thứ tự gửi.

    Returns:
        JSON Visual Cues chỉ gồm các dòng của tập con, ``line_no`` theo tập con; rỗng
        nếu không còn cue nào.
    """
    by_line = {cue.line_no: cue for cue in deserialize_visual_cues(raw)}
    remapped = [
        VisualCue(
            line_no=subset_no,
            speaker=by_line[position + 1].speaker,
            addressee=by_line[position + 1].addressee,
            scene=by_line[position + 1].scene,
        )
        for subset_no, position in enumerate(positions, start=1)
        if position + 1 in by_line
    ]
    return serialize_visual_cues(remapped) if remapped else ""
