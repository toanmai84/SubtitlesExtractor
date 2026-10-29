"""Tuần tự hoá Visual Cues để lưu/khôi phục qua QSettings (Auto-Save).

Tách riêng phần logic *thuần* (không phụ thuộc Qt) khỏi widget để dễ kiểm thử:
widget chỉ cần gọi :func:`serialize_visual_cues` trước khi ghi ``QSettings`` và
:func:`deserialize_visual_cues` khi nạp lại. Nhờ vậy chỉnh sửa của người dùng trên
bảng Visual Cues không mất khi tắt/mở lại ứng dụng.
"""

from __future__ import annotations

import json

from dataclasses import replace

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    CUE_ORIGIN_USER,
    TranslationLine,
    VisualCue,
)


#: [v3.23.655] Giá trị sửa "chọn lại Tự động" cho người nói / người nghe trong :func:`apply_user_cues`.
CUE_AUTO = "\x00auto"
_SEP = "\x1f"


def _auto_label_from(value: object) -> str:
    if isinstance(value, list | tuple) and len(value) == 2:
        return f"{str(value[0] or '')}{_SEP}{str(value[1] or '')}"
    return ""


def serialize_visual_cues(cues: list[VisualCue]) -> str:
    """Chuyển danh sách :class:`VisualCue` thành chuỗi JSON gọn (khoá rút gọn).

    Args:
        cues: Danh sách gợi ý hình ảnh (có thể đã được người dùng chỉnh sửa).

    Returns:
        Chuỗi JSON UTF-8 an toàn để lưu vào ``QSettings``.
    """
    payload = []
    for cue in cues:
        item = {"id": cue.line_no, "spk": cue.speaker, "to": cue.addressee, "cue": cue.scene}
        if cue.origin:
            item["src"] = cue.origin  # [v3.23.653] chỉ ghi khi có: JSON của cue AI giữ nguyên như cũ
            if cue.auto_label:
                item["ai"] = cue.auto_label.split(_SEP, 1)  # [v3.23.655] nhãn AI trước khi người dùng sửa
        payload.append(item)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def visual_cues_syntax_error(raw: str | None) -> str | None:
    """[v3.23.652] Lỗi cú pháp của ô Visual Cues (sửa tay), hoặc ``None`` nếu rỗng/hợp lệ.

    ``deserialize_visual_cues`` cố ý im lặng trả ``[]`` khi JSON hỏng. Với ô người dùng sửa tay, im lặng = thiếu một
    dấu phẩy là cả 3648 cues biến mất: lượt dịch quét lại toàn video (tốn hạn mức) rồi ghi đè lên ô. Phải báo trước.
    """
    if not raw or not raw.strip():
        return None
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as exc:
        return f"JSON lỗi ở dòng {exc.lineno}, cột {exc.colno}: {exc.msg}"
    if not isinstance(items, list):
        return "JSON phải là một danh sách [ {...}, {...} ]"
    return None


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
                origin=str(item.get("src", "") or ""),
                auto_label=_auto_label_from(item.get("ai")),
            )
        )
    cues.sort(key=lambda cue: cue.line_no)
    return cues


def keep_user_cues(new_cues: list[VisualCue], old_cues: list[VisualCue]) -> list[VisualCue]:
    """[v3.23.653] Gộp kết quả tự động ``new_cues`` mà GIỮ nguyên mọi cue người dùng chọn trong ``old_cues``.

    Người nói người dùng sửa (vd ở trang TTS) thắng AI: vá phân tích, lưới tiếng gọi hay phân tích lại không được đè.
    """
    user = {cue.line_no: cue for cue in old_cues if cue.origin == CUE_ORIGIN_USER}
    if not user:
        return list(new_cues)
    merged = {cue.line_no: cue for cue in new_cues}
    merged.update(user)
    return [merged[line_no] for line_no in sorted(merged)]


def keep_user_cues_json(new_raw: str | None, old_raw: str | None) -> str:
    """[v3.23.653] :func:`keep_user_cues` trên chuỗi JSON (trả ``new_raw`` nguyên vẹn nếu cũ không có cue người dùng)."""
    old = deserialize_visual_cues(old_raw)
    if not any(cue.origin == CUE_ORIGIN_USER for cue in old):
        return new_raw or ""
    return serialize_visual_cues(keep_user_cues(deserialize_visual_cues(new_raw), old))


def merge_cues_three_way(base_raw: str | None, mine_raw: str | None, theirs_raw: str | None) -> str:
    """[v3.23.655] Gộp cues do lượt chạy nền trả về (``theirs``) với ô người dùng sửa TRONG LÚC chạy (``mine``).

    ``base`` = nội dung ô lúc bắt đầu lượt chạy (lượt dịch/quét bổ sung dựng ``theirs`` từ bản này). Trước đây kết quả
    ghi đè thẳng lên ô: sửa tay trong lúc dịch mất sạch (bug #9).

    * Dòng người dùng đã sửa/thêm/xoá so với ``base`` → giữ của người dùng.
    * Dòng người dùng không đụng tới → lấy của lượt chạy (gồm cả dòng mới quét bổ sung).
    * Ô không đổi so với ``base`` → trả nguyên ``theirs``; ô đang hỏng JSON → giữ nguyên ô (không phá sửa dở).
    """
    if (mine_raw or "").strip() == (base_raw or "").strip():
        return theirs_raw or ""
    if visual_cues_syntax_error(mine_raw) is not None:
        return mine_raw or ""
    base = {cue.line_no: cue for cue in deserialize_visual_cues(base_raw)}
    mine = {cue.line_no: cue for cue in deserialize_visual_cues(mine_raw)}
    theirs = {cue.line_no: cue for cue in deserialize_visual_cues(theirs_raw)}
    merged: dict[int, VisualCue] = {}
    for line_no in set(base) | set(mine) | set(theirs):
        if mine.get(line_no) != base.get(line_no):          # người dùng đổi dòng này
            if line_no in mine:
                merged[line_no] = mine[line_no]
            continue                                         # người dùng xoá → giữ xoá
        chosen = theirs.get(line_no, mine.get(line_no))
        if chosen is not None:
            merged[line_no] = chosen
    result = keep_user_cues([merged[line_no] for line_no in sorted(merged)], list(mine.values()))
    return serialize_visual_cues(result)


def user_cue_key(cue: VisualCue) -> str:
    """[v3.23.653] Khoá người nói/người nghe của một cue người dùng (đưa vào vân tay bộ nhớ đệm dịch)."""
    return f"{cue.speaker}\x1f{cue.addressee}"


def tag_user_cues(lines: list[TranslationLine], cues_raw: str | None) -> list[TranslationLine]:
    """[v3.23.653] Gắn :attr:`TranslationLine.user_cue` cho dòng có cue người dùng (theo ``index`` = ``id`` cue).

    Gắn lại ở MỖI giai đoạn (dòng lấy từ bộ nhớ đệm hay do model trả không mang trường này) để vân tay ổn định.
    """
    keys = {cue.line_no: user_cue_key(cue) for cue in deserialize_visual_cues(cues_raw) if cue.origin == CUE_ORIGIN_USER}
    if not keys and not any(line.user_cue for line in lines):
        return lines
    return [
        line if line.user_cue == keys.get(line.index, "") else replace(line, user_cue=keys.get(line.index, ""))
        for line in lines
    ]


def apply_user_speakers(raw: str | None, speakers: dict[int, str]) -> tuple[str, list[int]]:
    """[v3.23.653] Ghi người nói người dùng chọn (``id`` câu gốc → tên vai) vào JSON cues, đánh dấu nguồn người dùng.

    Returns:
        ``(JSON mới, các id đã đổi)``.
    """
    return apply_user_cues(raw, {line_no: (speaker, None) for line_no, speaker in speakers.items()})


def apply_user_cues(
    raw: str | None, corrections: dict[int, tuple[str | None, str | None]],
) -> tuple[str, list[int]]:
    """[v3.23.654] Ghi người nói / "nói với ai" người dùng chọn vào JSON cues (nguồn người dùng).

    Args:
        raw: JSON Visual Cues hiện tại.
        corrections: ``{id câu gốc: (người nói | None, người nghe | None)}`` — ``None`` = giữ nguyên phần đó;
            người nghe ``""`` = không nói với ai.

    Returns:
        ``(JSON mới, các id đã đổi)``.
    """
    cues = {cue.line_no: cue for cue in deserialize_visual_cues(raw)}
    changed: list[int] = []
    for line_no, (speaker, addressee) in sorted(corrections.items()):
        speaker = None if speaker is None else (speaker if speaker == CUE_AUTO else speaker.strip() or None)
        if line_no <= 0 or (speaker is None and addressee is None):
            continue
        if CUE_AUTO in (speaker, addressee):
            # [v3.23.655] Trang TTS gửi TRẠNG THÁI ĐỦ của dòng: phần ``None`` là phần không còn sửa tay → cũng về nhãn AI
            # (vd người nghe bị xoá kèm khi chọn người nói, chọn lại "Tự động" thì người nghe cũng phải về như cũ).
            speaker = CUE_AUTO if speaker is None else speaker
            addressee = CUE_AUTO if addressee is None else addressee
        old = cues.get(line_no) or VisualCue(line_no=line_no)
        # [v3.23.655] Nhãn AI gốc: cất lần đầu cue thành cue người dùng, để "chọn lại Tự động" trả về đúng nhãn này.
        if old.origin == CUE_ORIGIN_USER:
            ai_speaker, _, ai_addressee = old.auto_label.partition(_SEP) if old.auto_label else (old.speaker, "", old.addressee)
        else:
            ai_speaker, ai_addressee = old.speaker, old.addressee
        new = replace(old, auto_label=f"{ai_speaker}{_SEP}{ai_addressee}")
        if speaker == CUE_AUTO:
            new = replace(new, speaker=ai_speaker)
        elif speaker is not None:
            new = replace(new, speaker=speaker)
            if addressee is None and old.addressee == speaker:
                new = replace(new, addressee="")  # người nghe trùng người nói mới = nhãn cũ đã đảo
        if addressee == CUE_AUTO:
            new = replace(new, addressee=ai_addressee)
        elif addressee is not None:
            new = replace(new, addressee=addressee.strip())
        if (new.speaker, new.addressee) == (ai_speaker, ai_addressee) and CUE_AUTO in (speaker, addressee):
            new = replace(new, origin="", auto_label="")   # không còn gì người dùng chọn → lại là cue tự động
        else:
            new = replace(new, origin=CUE_ORIGIN_USER)
        if new == old:
            continue
        cues[line_no] = new
        changed.append(line_no)
    if not changed:
        return raw or "", []
    return serialize_visual_cues([cues[k] for k in sorted(cues)]), changed

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
            origin=by_line[position + 1].origin,
        )
        for subset_no, position in enumerate(positions, start=1)
        if position + 1 in by_line
    ]
    return serialize_visual_cues(remapped) if remapped else ""
