"""[v3.23.671] Gộp kết quả bước NGHE GIỌNG vào bảng nhân vật đã chốt bằng chữ (hàm thuần).

Bảng nhân vật lập từ chữ sót người không tên có giọng riêng: tập 1 phim mẫu có một nam đệ tử,
một nữ đệ tử và giọng đám đông phụ hoạ, mà bảng chỉ có một mục "Đệ tử" (hoặc không có). Bước
nghe giọng (``listen_roster_voices`` ở adapter) đưa ra người còn thiếu + giới tính; mô-đun này
chỉ nhận phần AN TOÀN:

* Chỉ THÊM người mới — không đổi tên, giới, vai trò của người đã có.
* Người mới phải mang nhãn MÔ TẢ (đuôi 弟子/修士/旁白…, phần trước là phe có trong phụ đề + giới/số)
  hoặc tên riêng có thật trong phụ đề và không chung chữ với tên người đã có (biệt danh): đưa âm
  thanh vào model nhỏ thì nó từng bịa tên riêng ("林风") không hề xuất hiện trong phim.
* Phần "vai trò" (kể cả mệnh đề xưng hô) của người MỚI lấy như model ghi — chỉ ảnh hưởng chính
  người không tên đó.
* Không thêm trùng người đã có (theo mọi bí danh và phiên âm).
* KHÔNG đổi giới tính người đã có: model nhỏ nghe giới không ổn định (3 lượt cho "师父" — không
  hề nói — lúc 'nam' lúc 'nữ'; Nam Cung Mộng bị nghe thành 'nam').
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from subtitles_extractor.application.services.roster_format import (
    RosterEntry,
    parse_roster,
    render_roster,
)

#: Số người mới tối đa nhận từ một lượt nghe giọng.
MAX_NEW_VOICES = 8
#: Đuôi cho thấy tên là nhãn MÔ TẢ người không tên (không phải tên riêng có thể bịa). Chỉ
#: từ 2 chữ trở lên và phải ĐỨNG CUỐI tên: chữ lẻ ("女", "甲", "群") khớp cả "林女", "王甲".
_ROLE_SUFFIXES = (
    "弟子", "修士", "长老", "护法", "士兵", "侍卫", "守卫", "村民", "路人", "百姓", "旁白",
    "掌柜", "小二", "伙计", "丫鬟", "仆人", "管家", "魔修", "门人", "老者", "少年", "少女",
    "师弟", "师妹", "师兄", "师姐", "众人", "广播", "主持人", "记者", "医生", "护士", "警察",
    "保安", "司机", "服务员", "侍女", "家丁", "随从", "手下", "喽啰", "围观者", "群众",
)
#: Chữ bổ nghĩa được phép đứng trước/sau đuôi mô tả (giới, số đếm, "đám").
_MODIFIERS = "男女众老小大甲乙丙丁一二三四五六七八九十0123456789 "
#: Ký tự phá cú pháp dòng bảng nhân vật ("名 (x)", "旁白, 男").
_SYNTAX_CHARS = re.compile(r"[()（）\[\]【】—–,，、:：;；/|]")
#: Đại từ / từ không phải tên dù có trong phụ đề.
_NOT_NAMES = frozenset({"我", "你", "他", "她", "我们", "你们", "他们", "她们", "大家", "自己"})
_GENDER_WORDS = {
    "nam": "nam", "male": "nam", "nữ": "nữ", "nu": "nữ", "female": "nữ",
    "nhiều người": "nhiều người", "nhóm": "nhiều người", "mixed": "nhiều người",
}


def _norm_gender(value: Any) -> str:
    text = " ".join(str(value or "").strip().lower().split())
    if text in _GENDER_WORDS:
        return _GENDER_WORDS[text]
    if text.startswith(("nữ", "nu ", "female", "女")):
        return "nữ"
    if text.startswith(("nam", "male", "男")):
        return "nam"
    if text.startswith(("nhiều", "nhóm", "đám", "mixed", "group")):
        return "nhiều người"
    return ""


def _keys(entry: RosterEntry) -> set[str]:
    parts = re.split(r"\s*[/|]\s*", f"{entry.name}/{entry.aka}")
    return {p.strip().lower() for p in parts if p.strip()}


def _descriptive(name: str, source_text: str) -> bool:
    """Nhãn mô tả: <phe/nơi chốn có trong phụ đề hoặc bảng><giới/số><đuôi vai> (灵霄宗女弟子, 旁白)."""
    core = name.rstrip(_MODIFIERS)
    for suffix in _ROLE_SUFFIXES:
        if core.endswith(suffix):
            prefix = core[: -len(suffix)].strip(_MODIFIERS)
            return not prefix or prefix in source_text
    return False


def _cjk(text: str) -> set[str]:
    return {ch for ch in text if "\u4e00" <= ch <= "\u9fff"}


def merge_voice_roster(
    characters: str, patch: Mapping[str, Any] | None, source_text: str = ""
) -> tuple[str, list[str]]:
    """Gộp bản bổ sung của bước nghe giọng vào bảng nhân vật.

    Args:
        characters: Bảng nhân vật hiện tại (đã chốt bằng chữ).
        patch: ``{new_characters: [...]}`` do model trả.
        source_text: Toàn văn phụ đề (tên riêng người mới phải xuất hiện trong đó).

    Returns:
        ``(bảng mới, tên người đã thêm)``. Không thêm ai thì bảng trả về y nguyên chuỗi cũ.
    """
    if not isinstance(patch, Mapping):
        return characters, []
    entries = parse_roster(characters or "")
    taken: set[str] = set().union(*(_keys(e) for e in entries)) if entries else set()
    #: Chữ Hán trong TÊN RIÊNG đã có — tên mới "瑶儿" chung chữ với "楚瑶 / 瑶瑶" là biệt danh.
    named_chars = set().union(*(
        _cjk(e.name) for e in entries if not _descriptive(e.name.split("/")[0].strip(), "")
    )) if entries else set()
    added: list[str] = []
    for item in patch.get("new_characters") or []:
        if not isinstance(item, Mapping) or len(added) >= MAX_NEW_VOICES:
            continue
        name = str(item.get("name") or "").strip()
        aka = str(item.get("aka") or "").strip()
        if not name or len(name) > 24 or _SYNTAX_CHARS.search(name) or name in _NOT_NAMES:
            continue
        if _SYNTAX_CHARS.search(aka.replace("/", "")) or "/" in aka:
            continue
        if not _descriptive(name, f"{source_text}\n{characters or ''}"):
            # Tên riêng: phải có trong phụ đề (model nhỏ từng bịa "林风") và không phải biệt
            # danh của người đã có (chung chữ Hán với tên của họ).
            if len(name) < 2 or name not in source_text or _cjk(name) & named_chars:
                continue
        entry = RosterEntry(
            name=name, aka=aka, gender=_norm_gender(item.get("gender")),
            age=str(item.get("age") or "").strip(), role=str(item.get("role") or "").strip(),
        )
        keys = _keys(entry)
        if keys & taken:
            continue
        entries.append(entry)
        taken |= keys
        added.append(name)
    if not added:
        return characters, []
    return render_roster(entries), added


__all__ = ["MAX_NEW_VOICES", "merge_voice_roster"]
