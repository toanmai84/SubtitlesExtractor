"""[v3.23.488] QUY CÁCH CỐ ĐỊNH cho bảng nhân vật & bảng thuật ngữ.

Vấn đề trước đây: schema bắt AI trả ``characters``/``glossary`` là CHUỖI TỰ DO, nên mỗi lần chạy
model lại viết một kiểu (khi ngăn bằng dấu phẩy, khi bằng dấu chấm, mô tả lại chứa thêm dấu hai
chấm...). Mọi bộ phân tích cú pháp phía sau đều thành đoán mò và sai.

Cách làm ở module này:

1. AI trả về DỮ LIỆU CÓ CẤU TRÚC (mảng đối tượng) — xem schema trong adapter.
2. Ngay khi nhận, chuẩn hoá về VĂN BẢN QUY CÁCH CỐ ĐỊNH: **mỗi mục đúng MỘT DÒNG**.

   * Nhân vật: ``Tên (Phiên âm) [giới, tuổi] — vai trò: mô tả``
   * Thuật ngữ: ``Thuật ngữ gốc = bản dịch``

   Chỉ phần ``Tên`` là bắt buộc; các phần sau có thì thêm, không có thì bỏ.
3. Đọc lại: tách theo DÒNG rồi cắt tại dấu phân giới đầu tiên → tất định, không cần đoán.

Dữ liệu cũ hoặc do người dùng gõ tay không theo quy cách vẫn đọc được nhờ nhánh dự phòng
``_parse_freeform_names`` (heuristic, chấp nhận sai sót) — nhưng dữ liệu MỚI luôn đúng quy cách.

Toàn bộ hàm ở đây là HÀM THUẦN: không I/O, không phụ thuộc Qt/AI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from subtitles_extractor.application.services.roster_kinship import (
    KinPerson,
    KinshipGraph,
)

NAME_SEPARATORS = "(["
"""Ký tự mở đầu phần chú thích đứng ngay sau tên (phiên âm, giới/tuổi)."""

ROLE_SEPARATORS = ("—", "–", " - ", ":")
"""Dấu phân giới giữa phần định danh và phần vai trò/mô tả."""

MAX_NAME_CHARS = 40
"""Tên dài hơn ngần này gần như chắc chắn là câu mô tả bị cắt nhầm."""

MAX_NAME_WORDS = 5
"""Tên nhân vật hiếm khi quá 5 từ."""

_FULLWIDTH_MAP = str.maketrans({
    "（": "(", "）": ")", "［": "[", "］": "]", "【": "[", "】": "]",
    "：": ":", "；": ";", "，": ",", "、": ",",
})
"""[v3.23.492] Văn bản CJK thường dùng dấu câu FULL-WIDTH. Không quy đổi thì ``林逸（Lâm Dật）``
bị đọc thành một tên dính liền — sai ngay ở ca phổ biến nhất."""


def normalize_punctuation(text: str) -> str:
    """Quy đổi dấu câu full-width về ASCII để phân tích cú pháp nhất quán."""
    return text.translate(_FULLWIDTH_MAP)


def _split_glossary_line(line: str) -> tuple[str, str]:
    """[v3.23.492] Tách một dòng thuật ngữ thành (gốc, bản dịch) theo THỨ TỰ ƯU TIÊN.

    Vì sao cần ưu tiên: cả thuật ngữ lẫn bản dịch đều có thể chứa dấu ngăn. Ví dụ
    ``"tỉ lệ gạo:nước = cách gọi"`` phải tách ở ``=`` (không phải ``:`` nằm trong thuật ngữ), còn
    ``"C=O = liên kết đôi"`` phải tách ở ``" = "`` có khoảng trắng (không phải ``=`` trong công
    thức). Thứ tự: dấu có KHOẢNG TRẮNG hai bên → dấu mạnh (=>, =, ->, →, |) → dấu hai chấm.
    """
    for pattern in (
        r"\s+(?:=>|=|->|→|\|)\s+",   # ưu tiên 1: dấu mạnh có khoảng trắng bao quanh
        r"\s+:\s+",                   # ưu tiên 2: dấu hai chấm có khoảng trắng
        r"(?:=>|=|->|→|\|)",         # ưu tiên 3: dấu mạnh không khoảng trắng
        r":",                         # cuối cùng: dấu hai chấm dính liền
    ):
        parts = re.split(pattern, line, maxsplit=1)
        if len(parts) > 1 and parts[0].strip():
            return parts[0].strip(), parts[1].strip()
    return line.strip(), ""


@dataclass(frozen=True, slots=True)
class RosterEntry:
    """Một nhân vật trong bảng nhân vật.

    Attributes:
        name: Tên gốc (bắt buộc), có thể chứa bí danh ngăn bằng ``/``.
        aka: Phiên âm / tên gọi trong bản dịch.
        gender: Giới tính (tuỳ chọn).
        age: Mô tả tuổi (tuỳ chọn), ví dụ ``"~24 tuổi"``.
        role: Vai trò + mô tả ngắn.
    """

    name: str
    aka: str = ""
    gender: str = ""
    age: str = ""
    role: str = ""

    def render(self) -> str:
        """Kết xuất một DÒNG đúng quy cách ``Tên (Phiên âm) [giới, tuổi] — vai trò``."""
        parts = [self.name.strip()]
        if self.aka.strip():
            parts.append(f"({self.aka.strip()})")
        traits = ", ".join(p for p in (self.gender.strip(), self.age.strip()) if p)
        if traits:
            parts.append(f"[{traits}]")
        line = " ".join(parts)
        if self.role.strip():
            # Dùng "—" (gạch dài) làm phân giới CỐ ĐỊNH: không xuất hiện trong tên người,
            # và khác dấu hai chấm vốn hay có sẵn trong phần mô tả.
            line = f"{line} — {self.role.strip()}"
        return line


@dataclass(frozen=True, slots=True)
class GlossaryEntry:
    """Một mục thuật ngữ: ``term`` (gốc) → ``translation`` (bản dịch chuẩn)."""

    term: str
    translation: str = ""

    def render(self) -> str:
        """Kết xuất một DÒNG ``Thuật ngữ gốc = bản dịch`` (bỏ vế phải nếu rỗng)."""
        term = self.term.strip()
        translation = self.translation.strip()
        return f"{term} = {translation}" if translation else term


#: [v3.23.566] Nhãn trường/tiêu đề hay bị model trả nhầm thành một "nhân vật".
_HEADING_LABELS = frozenset({
    "xưng hô", "cách xưng hô", "nhân vật", "danh sách nhân vật", "bảng nhân vật",
    "vai trò", "ghi chú", "mô tả", "quan hệ", "characters", "roster", "notes",
    "address", "relationships",
})


def _is_heading_entry(entry: RosterEntry) -> bool:
    """Mục chỉ có TÊN (không phiên âm/giới tính/tuổi/vai trò) trùng một nhãn tiêu đề."""
    details = (entry.aka, entry.gender, entry.age, entry.role)
    if any(part.strip() for part in details):
        return False
    return entry.name.strip().rstrip(":").strip().casefold() in _HEADING_LABELS


def _looks_like_name(candidate: str) -> bool:
    """Heuristic loại các mảnh MÔ TẢ bị cắt nhầm thành tên (chỉ dùng ở nhánh dự phòng)."""
    name = candidate.strip()
    if not name or len(name) > MAX_NAME_CHARS:
        return False
    if name.endswith((".", "!", "?")):
        return False
    if len(name.split()) > MAX_NAME_WORDS:
        return False
    if any(ch in name for ch in "]})") and not any(ch in name for ch in "[({"):
        return False
    has_cjk = any("\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff" for ch in name)
    if has_cjk:
        return True
    first_alpha = next((ch for ch in name if ch.isalpha()), "")
    return bool(first_alpha) and first_alpha.isupper()


def name_of_line(line: str) -> str:
    """Cắt lấy TÊN từ một dòng đúng quy cách (tất định, không đoán).

    Args:
        line: Dòng dạng ``"Tên (Phiên âm) [giới, tuổi] — vai trò: mô tả"``.

    Returns:
        Phần tên đã bỏ chú thích/vai trò; chuỗi rỗng nếu dòng trống.
    """
    text = normalize_punctuation(line).strip().lstrip("-–—*• \t")
    if not text:
        return ""
    for separator in ROLE_SEPARATORS:
        index = text.find(separator)
        if index > 0:
            text = text[:index]
            break
    for char in NAME_SEPARATORS:
        index = text.find(char)
        if index > 0:
            text = text[:index]
            break
    return text.strip()


def _parse_freeform_names(text: str) -> list[str]:
    """Nhánh DỰ PHÒNG cho dữ liệu cũ/gõ tay không theo quy cách (heuristic, có thể sai sót)."""
    entry_pattern = re.compile(
        r"(?:^|[\n;•]|(?<=[.!?])\s|,\s)"
        r"\s*[-–—*]?\s*([^,;\n:()\[\].!?]{1,40}?)\s*"
        r"(?:\([^)]*\)\s*)?(?:\[[^\]]*\]\s*)?:",
    )
    names: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        name = raw.strip().strip("-–—•*").strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            names.append(name)

    structured_found = False
    for line in re.split(r"[;\n]+", text):
        line = line.strip()
        if not line:
            continue
        matches = [m.group(1) for m in entry_pattern.finditer(line) if _looks_like_name(m.group(1))]
        if matches:
            structured_found = True
            for name in matches:
                add(name)
            continue
        head = name_of_line(line)
        if head and head != line and _looks_like_name(head):
            structured_found = True
            add(head)
            continue
        if not structured_found:
            for chunk in line.split(","):
                candidate = name_of_line(chunk)
                if candidate:
                    add(candidate)
    return names


_CANONICAL_LINE_RE = re.compile(
    r"^\s*[-–—*•]?\s*"
    # TÊN không được chứa dấu phân giới vai trò (— – :) hay ngoặc chú thích.
    r"(?P<name>[^(\[\n:—–]+?)\s*"
    r"(?:\((?P<aka>[^)]*)\)\s*)?"
    r"(?:\[(?P<traits>[^\]]*)\]\s*)?"
    # Phân giới vai trò: "—" (quy cách mới) hoặc ":" (dữ liệu cũ / gõ tay).
    r"(?:[—–:]\s*(?P<role>.*))?$",
)


#: Đầu dòng quy cách mới: tên + chú thích ``(…)``/``[…]`` (nếu có) + phân giới gạch dài
#: (em dash hoặc en dash; en dash viết dạng ``\u2013`` để khỏi nhầm với gạch nối).
_STRUCTURED_HEAD_RE = re.compile(
    r"^\s*[-\u2013—*•]?\s*[^(\[\n:—\u2013]+?\s*"
    r"(?:\([^)]*\)\s*)?(?:\[[^\]]*\]\s*)?[—\u2013]"
)


def _is_single_entry_line(line: str, entry: RosterEntry | None) -> bool:
    """Dòng chứa đúng MỘT nhân vật.

    [v3.23.614] Dòng đúng quy cách mới ``Tên (Phiên âm) [giới, tuổi] — mô tả``: mọi thứ
    sau ``—`` là mô tả tự do, kể cả một câu mới "Xưng hô: với …" hay "Quan hệ: …". Ca thật
    gemini-3.8-flash: câu "… Luyện Khí. Xưng hô: với Sở Dao …" bị đếm là nhân vật thứ hai
    trên dòng, cả bảng 7 người rơi xuống nhánh dự phòng và còn 4 tên trơn (mất nữ chính,
    giới tính, phiên âm, mọi quy tắc xưng hô). Dòng dữ liệu cũ (không có chú thích trước
    dấu phân giới) vẫn được đếm như trước để tách kiểu "A: … . B: …".
    """
    if entry is None:
        return False
    if entry.role and _STRUCTURED_HEAD_RE.match(normalize_punctuation(line).lstrip()):
        return True
    return len(_parse_freeform_names(line)) <= 1


#: Cách ghi giới tính trong ngoặc vuông của bảng nhân vật.
_GENDER_TRAIT_PREFIXES = ("nam", "nữ", "nu", "male", "female", "không rõ", "chưa rõ")


def _is_gender_trait(trait: str) -> bool:
    return trait.casefold().startswith(_GENDER_TRAIT_PREFIXES)


def _split_traits(traits: list[str]) -> tuple[str, str]:
    """``[giới, tuổi]`` → (giới, tuổi), theo NỘI DUNG chứ không theo vị trí.

    [v3.23.614] Nhân vật chưa rõ giới mà có tuổi được kết xuất ``[~70]``; đọc theo vị trí
    biến "~70" thành giới tính và mất tuổi (giọng lồng tiếng theo tuổi, "先生" → ông/anh
    theo tuổi đều dựa vào trường này). Phát hiện bằng kiểm thử kết xuất ↔ đọc lại.
    """
    if not traits:
        return "", ""
    if len(traits) == 1:
        return (traits[0], "") if _is_gender_trait(traits[0]) else ("", traits[0])
    first, second = traits[0], traits[1]
    if not _is_gender_trait(first) and _is_gender_trait(second):
        return second, first
    return first, second


def parse_canonical_line(line: str) -> RosterEntry | None:
    """Đọc MỘT DÒNG đúng quy cách ``Tên (Phiên âm) [giới, tuổi] — vai trò``.

    Args:
        line: Dòng cần đọc.

    Returns:
        ``RosterEntry`` đầy đủ trường, hoặc None nếu dòng không đúng quy cách (thiếu dấu phân
        giới ``—`` lẫn phần chú thích ``(...)``/``[...]`` thì coi như không phải quy cách chuẩn,
        để nhánh dự phòng xử lý).
    """
    text = normalize_punctuation(line).strip()
    if not text:
        return None
    # Bỏ ký tự gạch đầu dòng, rồi quy dấu gạch ngang ASCII có khoảng trắng (" - ") về phân giới
    # chuẩn "—" để dữ liệu gõ tay kiểu "Tên - vai trò" cũng đọc được tất định.
    text = text.lstrip("-–—*• \t")
    text = re.sub(r"\s+-\s+", " — ", text, count=1)
    has_marker = "—" in text or "–" in text or "(" in text or "[" in text
    match = _CANONICAL_LINE_RE.match(text)
    if match is None:
        return None
    name = (match.group("name") or "").strip()
    if not name or not _looks_like_name(name):
        return None
    if not has_marker:
        # Dòng chỉ có mỗi tên: chấp nhận, nhưng phải KHÔNG chứa dấu phẩy (tránh nuốt nhầm
        # danh sách kiểu "A, B, C" thành một tên).
        if "," in text:
            return None
        return RosterEntry(name=name)
    traits = [t.strip() for t in (match.group("traits") or "").split(",") if t.strip()]
    gender, age = _split_traits(traits)
    return RosterEntry(
        name=name,
        aka=(match.group("aka") or "").strip(),
        gender=gender,
        age=age,
        role=(match.group("role") or "").strip(),
    )


def _aliases_of(name: str) -> list[str]:
    """Tách tên thành danh sách bí danh đã chuẩn hoá (``"胖子 / 三胖子"`` → ``["胖子", "三胖子"]``)."""
    return [a.strip().lower() for a in re.split(r"\s*[/|]\s*", name) if a.strip()]


def _union_aliases(first: str, second: str) -> str:
    """[v3.23.567] Hợp hai chuỗi bí danh ``a / b`` (giữ thứ tự, bỏ trùng hoa/thường).

    Ca thật: ``柳杰 / 柳河 (Liễu Kiệt)`` gộp với ``柳河 / 柳杰 (Liễu Hà / Liễu Kiệt)`` — giữ
    phiên âm của bản đầu thì 柳河 mất tên tiếng Việt.
    """
    parts: dict[str, str] = {}
    for part in re.split(r"\s*[/|]\s*", f"{first}/{second}"):
        if part.strip():
            parts.setdefault(part.strip().lower(), part.strip())
    return " / ".join(parts.values())


def coalesce_aliases(entries: list[RosterEntry]) -> list[RosterEntry]:
    """[v3.23.496] Gộp các mục CÙNG MỘT NGƯỜI dựa trên bí danh chung.

    Dữ liệu thật có ``胖子 / 三胖子`` và ``胖子 / 三胖子 / 王大福`` là hai mục cho cùng một người
    (model tách nhầm dù prompt đã yêu cầu gộp). Hai mục chia sẻ ít nhất một bí danh thì chắc chắn
    là một người → gộp thành mục có TẬP bí danh đầy đủ, giữ mô tả dài hơn.

    Args:
        entries: Danh sách nhân vật đã đọc.

    Returns:
        Danh sách đã gộp, giữ thứ tự xuất hiện đầu tiên.
    """
    merged: list[RosterEntry] = []
    alias_to_index: dict[str, int] = {}
    for entry in entries:
        aliases = _aliases_of(entry.name)
        hit = next((alias_to_index[a] for a in aliases if a in alias_to_index), None)
        if hit is None:
            merged.append(entry)
            for alias in aliases:
                alias_to_index[alias] = len(merged) - 1
            continue
        current = merged[hit]
        # Hợp nhất tập bí danh, giữ thứ tự: bí danh cũ trước, bí danh mới bổ sung sau.
        names = list(dict.fromkeys(
            [n.strip() for n in re.split(r"\s*[/|]\s*", current.name) if n.strip()]
            + [n.strip() for n in re.split(r"\s*[/|]\s*", entry.name) if n.strip()]
        ))
        merged[hit] = RosterEntry(
            name=" / ".join(names),
            aka=_union_aliases(current.aka, entry.aka),
            gender=current.gender or entry.gender,
            age=current.age or entry.age,
            role=max((current.role, entry.role), key=len),
        )
        for alias in _aliases_of(merged[hit].name):
            alias_to_index[alias] = hit
    return merged


def parse_roster(value: Any) -> list[RosterEntry]:
    """Chuẩn hoá bảng nhân vật từ MỌI nguồn về danh sách ``RosterEntry``.

    Args:
        value: Mảng đối tượng do AI trả (ưu tiên), hoặc chuỗi (quy cách mới hoặc dữ liệu cũ).

    Returns:
        Danh sách nhân vật đã khử trùng tên, giữ nguyên thứ tự xuất hiện.
    """
    entries: list[RosterEntry] = []
    seen: set[str] = set()

    def add(entry: RosterEntry) -> None:
        name = entry.name.strip()
        if not name or name.lower() in seen:
            return
        # [v3.23.566] Dòng TIÊU ĐỀ lạc vào bảng (ca thật gemini-3.8-flash: một mục chỉ có
        # tên "Xưng hô", không phiên âm/giới tính/vai trò) không phải nhân vật.
        if _is_heading_entry(entry):
            return
        seen.add(name.lower())
        entries.append(entry)

    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                add(RosterEntry(
                    name=str(item.get("name") or "").strip(),
                    aka=str(item.get("aka") or "").strip(),
                    gender=str(item.get("gender") or "").strip(),
                    age=str(item.get("age") or "").strip(),
                    role=str(item.get("role") or "").strip(),
                ))
            elif isinstance(item, str):
                add(RosterEntry(name=name_of_line(item), role=""))
        return coalesce_aliases(entries)

    text = str(value or "").strip()
    if not text:
        return coalesce_aliases(entries)

    lines = [ln for ln in text.splitlines() if ln.strip()]
    # Quy cách mới: MỖI MỤC MỘT DÒNG. Đọc TẤT ĐỊNH khi MỌI dòng đều hợp quy cách VÀ mỗi dòng chỉ
    # chứa ĐÚNG MỘT mục — nếu một dòng gói nhiều mục (dữ liệu cũ kiểu "A: ... . B: ...") thì phải
    # rơi xuống nhánh dự phòng, nếu không sẽ nuốt cả cụm thành một tên.
    canonical = [parse_canonical_line(ln) for ln in lines]
    one_entry_per_line = all(
        _is_single_entry_line(line, entry)
        for line, entry in zip(lines, canonical, strict=True)
    )
    if lines and one_entry_per_line and all(entry is not None for entry in canonical):
        for entry in canonical:
            add(entry)  # type: ignore[arg-type]
        return coalesce_aliases(entries)

    for name in _parse_freeform_names(text):
        add(RosterEntry(name=name))
    return coalesce_aliases(entries)


def render_roster(entries: list[RosterEntry]) -> str:
    """Kết xuất bảng nhân vật theo quy cách cố định — mỗi nhân vật đúng MỘT DÒNG."""
    return "\n".join(entry.render() for entry in entries if entry.name.strip())


def parse_glossary(value: Any) -> list[GlossaryEntry]:
    """Chuẩn hoá bảng thuật ngữ từ mảng đối tượng hoặc văn bản (mỗi dòng ``gốc = dịch``)."""
    entries: list[GlossaryEntry] = []
    seen: set[str] = set()

    def add(term: str, translation: str) -> None:
        key = term.strip()
        if not key or key.lower() in seen:
            return
        seen.add(key.lower())
        entries.append(GlossaryEntry(term=key, translation=translation.strip()))

    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                add(str(item.get("term") or ""), str(item.get("translation") or ""))
            elif isinstance(item, str):
                left, _, right = item.partition("=")
                add(left, right)
        return entries

    for line in str(value or "").splitlines():
        if not line.strip():
            continue
        term, translation = _split_glossary_line(normalize_punctuation(line))
        add(term, translation)
    return entries


def render_glossary(entries: list[GlossaryEntry]) -> str:
    """Kết xuất bảng thuật ngữ theo quy cách cố định — mỗi thuật ngữ đúng MỘT DÒNG."""
    return "\n".join(entry.render() for entry in entries if entry.term.strip())


def unread_roster_lines(characters: str, entries: list[RosterEntry]) -> list[str]:
    """[v3.23.614] Dòng của bảng nhân vật mà kết quả đọc ``entries`` không có (hàm thuần).

    Lưới an toàn cho chẩn đoán: bộ đọc bỏ sót dòng là mọi lưới phía sau (quy tắc xưng hô,
    giới tính khi lồng tiếng, xưng hô thuần Việt) mù theo — ca thật 14/10: bảng 7 người
    đọc ra 4 tên trơn mà chẩn đoán không báo gì.

    Args:
        characters: Bảng nhân vật dạng văn bản.
        entries: Kết quả :func:`parse_roster` của chính văn bản đó.

    Returns:
        Các dòng (đã bỏ khoảng trắng hai đầu) không ứng với nhân vật nào; dòng tiêu đề
        ("Xưng hô", "Nhân vật"…) không tính.
    """
    known = {alias for entry in entries for alias in _aliases_of(entry.name)}
    unread: list[str] = []
    for line in (characters or "").splitlines():
        text = line.strip()
        head = name_of_line(text)
        if not head or head.rstrip(":").strip().casefold() in _HEADING_LABELS:
            continue
        if not any(alias in known for alias in _aliases_of(head)):
            unread.append(text)
    return unread


def roster_names(value: Any) -> list[str]:
    """Tiện ích: lấy danh sách TÊN nhân vật từ mọi dạng dữ liệu đầu vào."""
    return [entry.name for entry in parse_roster(value)]


MAX_OVERVIEW_CHARS = 20_000
"""Trần độ dài tóm tắt tích luỹ (~5k token) — đủ chi tiết mà không phình prompt các đoạn sau."""


def merge_roster(previous: Any, current: Any) -> str:
    """[v3.23.489] HỢP NHẤT hai bảng nhân vật theo TÊN (giữ mục cũ, thêm mục mới).

    Khi cùng một nhân vật xuất hiện ở cả hai bên, giữ bản ghi có thông tin ĐẦY ĐỦ hơn (mô tả dài
    hơn thường là bản đã bổ sung), nhưng luôn giữ phiên âm/giới/tuổi nếu bên kia có mà bên này thiếu.

    Args:
        previous: Bảng nhân vật tích luỹ (mảng đối tượng hoặc văn bản).
        current: Bảng nhân vật của đoạn mới.

    Returns:
        Văn bản quy cách cố định, mỗi nhân vật một dòng.
    """
    merged: dict[str, RosterEntry] = {}
    order: list[str] = []
    for entry in [*parse_roster(previous), *parse_roster(current)]:
        key = entry.name.strip().lower()
        if not key:
            continue
        existing = merged.get(key)
        if existing is None:
            merged[key] = entry
            order.append(key)
            continue
        merged[key] = RosterEntry(
            name=existing.name,
            aka=existing.aka or entry.aka,
            gender=existing.gender or entry.gender,
            age=existing.age or entry.age,
            role=max((existing.role, entry.role), key=len),
        )
    # [v3.23.567] Khoá theo TÊN nguyên văn nên ``柳杰 / 柳河`` và ``柳河 / 柳杰`` (một
    # người, model đảo thứ tự ở đoạn sau) thành hai dòng. Dữ liệu thật v3.23.566: bảng 30
    # dòng có 3 nhân vật bị lặp — prompt dịch nhận hai mô tả cho cùng một người.
    return render_roster(coalesce_aliases([merged[key] for key in order]))


def merge_glossary(previous: Any, current: Any) -> str:
    """[v3.23.489] HỢP NHẤT hai bảng thuật ngữ theo THUẬT NGỮ GỐC (giữ cũ, thêm mới)."""
    merged: dict[str, GlossaryEntry] = {}
    order: list[str] = []
    for entry in [*parse_glossary(previous), *parse_glossary(current)]:
        key = entry.term.strip().lower()
        if not key:
            continue
        if key not in merged:
            merged[key] = entry
            order.append(key)
        elif not merged[key].translation and entry.translation:
            merged[key] = entry
    return render_glossary([merged[key] for key in order])


MIN_DEDUP_SENTENCE_CHARS = 30
"""[v3.23.493] Chỉ khử trùng những câu ĐỦ DÀI: câu ngắn ("Anh ta bỏ đi.") có thể lặp lại hợp lệ
ở nhiều đoạn khác nhau, xoá đi sẽ mất thông tin."""


def _dedupe_against(previous: str, current: str) -> str:
    """Bỏ khỏi ``current`` những câu ĐÃ có nguyên văn trong ``previous``.

    Vì sao cần: khi phân tích tuần tự, mỗi đoạn thường tóm tắt lại phần trước rồi mới kể phần
    mới. Nối thẳng qua 23 đoạn sẽ tích tụ nội dung TRÙNG, vừa phình prompt vừa làm model rối.

    Args:
        previous: Tóm tắt đã tích luỹ.
        current: Tóm tắt đoạn mới.

    Returns:
        Phần nội dung MỚI của ``current`` (chuỗi rỗng nếu không có gì mới).
    """
    seen = previous
    kept: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", current):
        stripped = sentence.strip()
        if not stripped:
            continue
        if len(stripped) >= MIN_DEDUP_SENTENCE_CHARS and stripped in seen:
            continue
        kept.append(stripped)
    return " ".join(kept)


def merge_overview(previous: str, current: str, max_chars: int = MAX_OVERVIEW_CHARS) -> str:
    """[v3.23.489] NỐI TIẾP tóm tắt theo thứ tự thời gian thay vì để đoạn sau ghi đè đoạn trước.

    Nếu bản mới đã BAO HÀM bản cũ (model tự mở rộng đúng) thì dùng thẳng bản mới, tránh lặp.
    Khi vượt trần, cắt bớt phần GIỮA — giữ đầu phim (thiết lập bối cảnh) và cuối phim (diễn biến
    mới nhất), là hai phần quan trọng nhất cho việc dịch.

    Args:
        previous: Tóm tắt đã tích luỹ.
        current: Tóm tắt của đoạn mới.
        max_chars: Trần độ dài kết quả.

    Returns:
        Tóm tắt tích luỹ.
    """
    old = (previous or "").strip()
    new = (current or "").strip()
    if not old:
        return new
    if not new or new in old:
        return old
    if old in new:
        return new
    # [v3.23.493] Bỏ phần TRÙNG trước khi nối — tránh tích tụ nội dung lặp qua nhiều đoạn.
    fresh = _dedupe_against(old, new)
    if not fresh:
        return old
    combined = f"{old}\n\n{fresh}"
    if len(combined) <= max_chars:
        return combined
    head = combined[: max_chars // 2].rstrip()
    tail = combined[-(max_chars // 2) :].lstrip()
    return f"{head}\n\n[… lược bớt phần giữa để giữ độ dài hợp lý …]\n\n{tail}"


# [v3.23.501] Kính ngữ/hậu tố CJK hay bị model bỏ sót khi dịch, tạo chữ lai kiểu "Lâm先生".
# Chỉ gồm các từ có bản dịch ỔN ĐỊNH, không phụ thuộc ngữ cảnh, nên thay tất định là an toàn.
COMMON_CJK_SUFFIXES: dict[str, str] = {
    "先生": " tiên sinh",
    "小姐": " tiểu thư",
    "女士": " quý cô",
    "太太": " phu nhân",
    "老师": " thầy",
    "医生": " bác sĩ",
    "护士": " y tá",
    "经理": " giám đốc",
    "总裁": " tổng tài",
    "董事长": " chủ tịch",
    "哥": " ca",
    "姐": " tỷ",
    "叔": " chú",
    "婶": " thím",
    "爷": " ông",
    "奶奶": " bà",
}


def substitute_known_cjk(text: str, mappings: dict[str, str]) -> str:
    """[v3.23.501] Thay các mảnh chữ Hán còn sót bằng bản dịch ĐÃ DUYỆT (roster/glossary).

    Vì sao cần: model đôi khi dịch NỬA VỜI, để lại chữ lai như ``"Lâm先生"``, ``"Cường哥"`` hay bỏ
    nguyên tên ``"秦雅"``. Dịch lại cả dòng thường cho ra đúng kết quả cũ, nên vòng dọn hiện có
    không cứu được. Thay tất định bằng chính bảng tên/thuật ngữ người dùng đã duyệt thì không thể
    bịa nội dung, và luôn nhất quán với phần còn lại của bản dịch.

    Args:
        text: Câu dịch còn lẫn chữ Hán.
        mappings: Bản đồ ``chữ gốc -> bản dịch`` (tên nhân vật, thuật ngữ, kính ngữ).

    Returns:
        Câu đã thay thế; phần chữ Hán không có trong bản đồ được giữ lại (ở dạng giản thể
        nếu là chữ phồn có trong bảng quy đổi).
    """
    if not text:
        return text
    # [v3.23.615] Chữ Hán còn sót được đưa về giản thể trước khi tra: bảng tên ghi "楚瑶"
    # còn bản dịch có thể giữ nguyên dạng OCR "楚瑤". Chỉ chữ Hán bị đổi — mà chữ Hán còn
    # sót trong bản dịch tiếng Việt đằng nào cũng phải thay hoặc bị báo.
    from subtitles_extractor.application.services.han_variants import fold_han_variants

    text = fold_han_variants(text)
    # Thay chuỗi DÀI trước để "董事长" không bị "长" cắt mất.
    for source in sorted(mappings, key=len, reverse=True):
        target = mappings[source]
        folded = fold_han_variants(source)
        if folded and target and folded in text:
            text = text.replace(folded, target)
    return re.sub(r"\s{2,}", " ", text).strip()


def cjk_substitution_map(characters: str, glossary: str) -> dict[str, str]:
    """Dựng bản đồ thay thế từ bảng nhân vật + bảng thuật ngữ + kính ngữ thông dụng.

    Với nhân vật, mỗi bí danh đều được ánh xạ sang tên tiếng đích; mục thuật ngữ ghi nhiều phương
    án (``"A / B"``) thì lấy phương án ĐẦU để giữ nhất quán.

    [v3.23.587] Hai vế của mục thuật ngữ tách bằng ``glossary_options`` — cùng một cách
    hiểu với khối nhắc thuật ngữ theo lô. Trước đây tách vế dịch theo ``/`` TRƯỚC khi bỏ
    ngoặc nên ``卸货 = dỡ hàng (sinh con / lâm bồn)`` cho đích thay thế "dỡ hàng (sinh
    con" (29/3457 mục của 17 bảng thật), còn mục nhiều cách viết ``老山 / 魯山`` dùng
    nguyên chuỗi làm khoá nên không bao giờ khớp (98 mục).
    """
    # Nạp trễ: ``glossary_options`` phụ thuộc ``parse_glossary`` của chính module này.
    from subtitles_extractor.application.services.glossary_options import term_renderings

    mapping: dict[str, str] = {}
    for entry in parse_roster(characters):
        target = entry.aka.strip() or entry.name.strip()
        if not target:
            continue
        for alias in re.split(r"\s*[/|]\s*", entry.name):
            alias = alias.strip()
            if alias and alias != target:
                mapping[alias] = target
    for rendering in term_renderings(glossary):
        # Vế dịch chỉ gồm chú thích trong ngoặc không phải thứ chèn được vào lời thoại.
        if not rendering.options[0].startswith(("(", "\uff08")):
            mapping[rendering.term] = rendering.options[0]
    for suffix, translation in COMMON_CJK_SUFFIXES.items():
        mapping.setdefault(suffix, translation)
    return mapping


def drop_variant_glossary_entries(entries: list[GlossaryEntry]) -> list[GlossaryEntry]:
    """[v3.23.506] Bỏ các mục thuật ngữ là BIẾN THỂ của mục đã có (cùng nghĩa, khác một ký tự).

    Dữ liệu thật có 9 cặp kiểu ``骨髓配对``/``骨髓配型``, ``满月宴``/``满月酒`` — cùng một thực thể
    ghi hai cách. Khi cả hai cùng nằm trong bảng, khâu dịch nhận hai chỉ thị cho cùng một khái niệm
    và không thể tuân thủ trọn vẹn cả hai. Prompt đã nhắc nhưng model vẫn sinh trùng, nên xử lý
    TẤT ĐỊNH ở đây.

    Chỉ bỏ khi bản dịch TRÙNG hoặc bao hàm nhau — hai từ khác một ký tự nhưng khác nghĩa
    (``装货`` chất hàng / ``卸货`` dỡ hàng) phải được GIỮ NGUYÊN cả hai.

    Args:
        entries: Danh sách mục thuật ngữ theo thứ tự xuất hiện.

    Returns:
        Danh sách đã bỏ biến thể, giữ mục xuất hiện TRƯỚC.
    """
    kept: list[GlossaryEntry] = []
    for entry in entries:
        duplicate_of_existing = False
        for existing in kept:
            if len(existing.term) != len(entry.term):
                continue
            differences = sum(
                1 for a, b in zip(existing.term, entry.term, strict=True) if a != b
            )
            if differences != 1:
                continue
            left = existing.translation.strip().lower()
            right = entry.translation.strip().lower()
            if left and right and (left == right or left in right or right in left):
                duplicate_of_existing = True
                break
        if not duplicate_of_existing:
            kept.append(entry)
    return kept


def drop_subsumed_glossary_entries(entries: list[GlossaryEntry]) -> list[GlossaryEntry]:
    """[v3.23.508] Bỏ mục thuật ngữ có tập bí danh NẰM GỌN trong một mục khác.

    Model vừa ghi mục gộp bí danh (``"满月宴 / 满月酒"``) vừa ghi mục lẻ (``"满月宴"``) cho cùng
    một khái niệm — dữ liệu thật có 21/287 mục thừa kiểu này. Hệ quả: khâu dịch nhận hai chỉ thị
    cho cùng một từ, và bảng dài ra vô ích. Mục THỪA bị loại, giữ mục có tập bí danh ĐẦY ĐỦ hơn.

    Args:
        entries: Danh sách mục thuật ngữ.

    Returns:
        Danh sách đã loại mục bị bao hàm, giữ thứ tự xuất hiện.
    """
    alias_sets: list[set[str]] = [
        {alias.strip().lower() for alias in re.split(r"\s*[/|]\s*", entry.term) if alias.strip()}
        for entry in entries
    ]
    kept: list[GlossaryEntry] = []
    for index, entry in enumerate(entries):
        current = alias_sets[index]
        if not current:
            continue
        subsumed = any(
            other_index != index and current < alias_sets[other_index]
            for other_index in range(len(entries))
        )
        if not subsumed:
            kept.append(entry)
    return kept


#: [v3.23.549] Ký tự Hán/Nhật/Hàn — dùng để phát hiện mục thuật ngữ DỊCH DỞ.
_CJK_CHAR_RE = re.compile(r"[一-鿿㐀-䶿぀-ヿ가-힯]")
#: Chữ cái Latinh/Việt — có mặt cùng CJK trong một vế dịch là dấu hiệu dịch dở dang.
_LATIN_CHAR_RE = re.compile(r"[a-zà-ỹđ]", re.IGNORECASE)


def is_half_translated(translation: str) -> bool:
    """[v3.23.549] True nếu vế dịch LẪN chữ Hán với chữ Latinh — mục hỏng, phải loại.

    Ca thật (phim 3902 dòng): mục ``生死道消`` có vế dịch ``"thân tử đạo消"`` — model dịch được
    ba chữ rồi bỏ dở chữ cuối. Mục hỏng này KHÔNG bị cờ nào bắt (``suspect_glossary_entries``
    bằng 0) mà lại đi thẳng vào prompt dịch, dạy model giữ nguyên chữ Hán trong bản tiếng Việt.

    Chỉ coi là hỏng khi vế dịch có CẢ HAI hệ chữ. Vế dịch TOÀN chữ Hán là hợp lệ khi ngôn ngữ
    đích chính là tiếng Trung, nên không được loại oan.

    Args:
        translation: Vế dịch của một mục thuật ngữ.

    Returns:
        True nếu mục dịch dở dang.
    """
    text = (translation or "").strip()
    if not text:
        return False
    return bool(_CJK_CHAR_RE.search(text)) and bool(_LATIN_CHAR_RE.search(text))


def split_half_translated_entries(
    entries: list[GlossaryEntry],
) -> tuple[list[GlossaryEntry], list[GlossaryEntry]]:
    """Tách mục thuật ngữ DỊCH DỞ khỏi các mục lành.

    Args:
        entries: Danh sách mục thuật ngữ.

    Returns:
        Cặp ``(muc_lanh, muc_hong)``, giữ nguyên thứ tự xuất hiện.
    """
    healthy: list[GlossaryEntry] = []
    broken: list[GlossaryEntry] = []
    for entry in entries:
        (broken if is_half_translated(entry.translation) else healthy).append(entry)
    return healthy, broken


#: [v3.23.550] Một quy tắc xưng hô trong roster: "với <Tên> (tự) xưng '<A>' gọi '<B>'".
_ADDRESS_RULE_RE = re.compile(
    r"với\s+(?P<other>[^,;:']+?)\s+(?:tự\s+)?xưng\s+(?:là\s+)?'(?P<self>[^']*)'"
    r"(?:\s+gọi\s+(?:là\s+)?'(?P<addr>[^']*)')?",
    re.IGNORECASE,
)
#: [v3.23.574] "xưng là 'x'", "gọi là 'y'" — cùng nghĩa với "xưng 'x'", "gọi 'y'". Quy tắc
#: được đưa về dạng không "là" để mọi bộ đọc phía sau (lưới xưng hô, vai vế) đọc được.
_RULE_LA_RE = re.compile(r"\b(xưng|gọi)\s+là\s+'", re.IGNORECASE)
#: [v3.23.574] Cách viết KHÁC của cùng thông tin: một cách tự xưng chung rồi liệt kê cách
#: gọi từng người — "tự xưng 'em', gọi Chân Kiếm Thanh 'Kiếm ca/anh', gọi Chu Chính Quân
#: 'A Quân'". Ca thật gemini-3.8-flash (phim "灭门惨案", 17 nhân vật): cả bảng viết kiểu
#: này, bộ đọc cũ đọc được 0 quy tắc nên khâu dịch và lưới xưng hô mù hoàn toàn.
_GLOBAL_SELF_RE = re.compile(
    r"(?:tự\s+)?xưng\s+(?:là\s+)?'(?P<self>[^']*)'", re.IGNORECASE
)
_CALL_LISTENER_RE = re.compile(
    r"(?:với\s+(?P<with>[^,;:'()]+?)\s+)?gọi\s+"
    r"(?:(?P<other>(?!là\b)[^,;:'.()]+?)\s+)?(?:là\s+)?'(?P<addr>[^']*)'",
    re.IGNORECASE,
)
#: Người nghe là một NHÓM — mỗi thành viên một cách gọi là đúng.
_GROUP_ADDRESSEE_RE = re.compile(r"\b(?:và|gia đình|mọi người|cả nhà|các|những|bọn|lũ)\b",
                                 re.IGNORECASE)
#: Từ báo hiệu quy tắc có ĐIỀU KIỆN theo giai đoạn — nhiều phương án là hợp lệ.
_ADDRESS_PHASE_RE = re.compile(r"\b(?:sau khi|trước khi|khi|lúc|từ khi|về sau|ban đầu)\b",
                               re.IGNORECASE)


#: Từ xưng hô thân tộc/cổ trang hay dùng THAY đại từ, bổ sung cho danh sách đại từ chuẩn.
_EXTRA_ADDRESS_HEADS = frozenset({"lão", "mình", "mẹ", "bố", "ba"})

#: [v3.23.591] DANH XƯNG mở đầu bằng một chữ trùng đại từ nhưng không phải đại từ đó:
#: "bác sĩ" là nghề, không phải "bác". Chuẩn đoán thật v3.23.590: ``龍傳人 → bác sĩ: xưng
#: 'tôi' gọi 'ông/bác sĩ'`` bị đếm là hai đại từ ("ông", "bác") nên báo MƠ HỒ kèm lời
#: khuyên phân tích lại — tốn 17-20 request của model hạn mức ít cho một bảng đang đúng.
#: Trong 20 bảng nhân vật của kho, 'bác sĩ' có mặt ở 20 phương án xưng hô; 'bà con' (6),
#: 'ông chủ' (2) cùng loại. Danh xưng thân tộc ('chị dâu', 'chú em') KHÔNG vào đây: chữ
#: đầu của chúng đúng là đại từ dùng để gọi.
_TITLE_COMPOUNDS = frozenset({
    "bác sĩ", "bà con", "ông chủ", "bà chủ", "cô chủ", "cậu chủ",
})


#: [v3.23.601] Đại từ cũng là LOẠI TỪ đứng đầu lời chửi ("con ranh", "con mụ").
_EPITHET_CLASSIFIERS = frozenset({"con"})


def is_classifier_epithet(words: list[str]) -> bool:
    """Phương án là lời chửi mở đầu bằng loại từ ("con ranh"), không phải đại từ "con".

    "con gái", "con trai" (gọi con cái) vẫn là đại từ "con": chữ thứ hai là từ thân tộc.

    Args:
        words: Các chữ (viết thường) của MỘT phương án xưng hô trong bảng nhân vật.

    Returns:
        True nếu chữ đầu là loại từ của lời chửi.
    """
    from subtitles_extractor.application.services.translation_diagnostics import (
        KIN_COMPOUND_TAIL,
    )

    if len(words) < 2 or words[0] not in _EPITHET_CLASSIFIERS:
        return False
    return words[1] not in KIN_COMPOUND_TAIL.split("|")


def _pronoun_count(forms: str) -> int:
    """Số ĐẠI TỪ KHÁC NHAU trong chuỗi phương án ``'a/b/c'`` (xét theo từ đầu của phương án).

    Chỉ đại từ mới buộc khâu dịch phải tự chọn. Danh xưng (``'Thiết Trụ ca'``, ``'Tống thần y'``)
    và lời chửi (``'thằng ngốc'``, ``'kẻ quê mùa'``) dùng song song với đại từ là bình thường nên
    không tính. Hai phương án cùng từ đầu (``'chị'`` / ``'chị dâu'``) tính là MỘT.
    """
    from subtitles_extractor.application.services.translation_diagnostics import (
        VIETNAMESE_PRONOUNS,
    )

    known = set(VIETNAMESE_PRONOUNS) | _EXTRA_ADDRESS_HEADS
    options = [part.strip().lower().split() for part in forms.split("/") if part.strip()]
    # [v3.23.591] Danh xưng nghề/chức ("bác sĩ") là danh xưng, không phải đại từ.
    options = [words for words in options if " ".join(words[:2]) not in _TITLE_COMPOUNDS]
    # [v3.23.601] "con ranh", "con đàn bà kia": chữ "con" là loại từ của lời chửi, không
    # phải đại từ "con" (gọi con cái). Chuẩn đoán thật v3.23.600: ``洪志權 → Tần Khả Tú:
    # xưng 'tao' gọi 'mày/con ranh'`` bị báo mơ hồ kèm lời khuyên phân tích lại cả phim.
    options = [words for words in options if not is_classifier_epithet(words)]
    heads = {words[0] for words in options} & known
    # [v3.23.586] Phương án chỉ THÊM từ đệm vào một phương án khác là cùng một cách gọi:
    # 'ba/lão ba', 'em/cô em'. Ca thật: ``洪志權 → Hồng Chấn: xưng 'con' gọi 'ba/lão ba'``
    # bị báo mơ hồ dù cả hai đều là "ba".
    decorated = {
        words[0]
        for words in options
        if any(other != words[0] and other in words[1:] for other in heads)
    }
    return len(heads - decorated)


def ambiguous_address_rules(characters: str) -> list[str]:
    """[v3.23.550] Các quy tắc xưng hô trong roster liệt kê NHIỀU đại từ mà không có điều kiện.

    Bảng xưng hô chỉ có tác dụng khi mỗi cặp nhân vật có ĐÚNG MỘT cách tự xưng và MỘT cách gọi.
    Ca thật: ``với Tống Tiền tự xưng 'tôi/Mộ Vân/chị' gọi 'cậu/Tống thần y/Tống Tiền'`` — khâu dịch
    được phép chọn bất kỳ nên mỗi lô chọn một kiểu, bản dịch trôi "tôi" → "em" chỉ trong 5 dòng.

    Quy tắc có ĐIỀU KIỆN giai đoạn ("sau khi thua trận xưng 'tôi'") là hợp lệ và không bị báo.

    Args:
        characters: Bảng nhân vật đã chuẩn hoá (mỗi nhân vật một dòng).

    Returns:
        Danh sách ``"<Người nói> → <Người nghe>: xưng '<A>' gọi '<B>'"`` bị mơ hồ.
    """
    findings: list[str] = []
    resolver = _ListenerResolver.build(parse_roster(characters))
    for line in (characters or "").splitlines():
        speaker = name_of_line(line) if line.strip() else ""
        for segment in re.split(r"[;,](?=\s*(?:với|sau khi|trước khi)\b)", line):
            match = _ADDRESS_RULE_RE.search(segment)
            if not match:
                continue
            self_form = match.group("self") or ""
            address_form = match.group("addr") or ""
            if _pronoun_count(self_form) < 2 and _pronoun_count(address_form) < 2:
                continue
            if _ADDRESS_PHASE_RE.search(segment):
                continue
            # Người nghe là NHÓM ("mẹ và em trai", "gia đình họ Liễu"): mỗi người một cách gọi
            # là đúng, không phải mơ hồ.
            if _GROUP_ADDRESSEE_RE.search(match.group("other")):
                continue
            # [v3.23.586] Người nghe không phải MỘT nhân vật của bảng — một loại người
            # ("cấp dưới", "khách hàng": gọi 'cậu/cô', 'anh/chị' theo giới là đúng) hoặc
            # nhiều nhân vật ("Kiếm Thanh/Khả Tú") — cũng không phải mơ hồ. Chuẩn đoán
            # thật v3.23.585: 4/6 cặp bị báo thuộc loại này, kèm lời khuyên phân tích lại.
            listeners, _, listener_count = resolver.resolve_all(
                match.group("other").strip(), speaker
            )
            if not listeners or listener_count > 1:
                continue
            findings.append(
                f"{speaker or '?'} → {match.group('other').strip()}: "
                f"xưng '{self_form}' gọi '{address_form}'"
            )
    return findings


def name_aliases(value: str) -> list[str]:
    """Tách bí danh ``"吴二狗 / 土狗"`` → ``["吴二狗", "土狗"]`` (bỏ khoảng trắng thừa)."""
    return [alias.strip() for alias in re.split(r"\s*[/|]\s*", value or "") if alias.strip()]


_name_aliases = name_aliases


#: [v3.23.568] Tách danh sách người nghe "A và B", "A / B", "A、B", "A和B".
_LISTENER_SPLIT_RE = re.compile(r"\s*(?:\bvà\b|/|&|和|、)\s*", re.IGNORECASE)
#: [v3.23.568] Cách gọi SỐ NHIỀU ("các người", "chúng mày") chỉ đúng khi nói với cả nhóm.
_PLURAL_FORM_RE = re.compile(
    r"\b(?:các|chúng|bọn|lũ|mấy|mọi người|hai người|hai vị|các vị)\b", re.IGNORECASE
)


@dataclass(frozen=True, slots=True)
class AddressClause:
    """[v3.23.568] Một mệnh đề xưng hô "với <người nghe> (tự) xưng … gọi …" đã tra tên.

    Attributes:
        speakers: Tên gốc + bí danh của người nói.
        listeners: Tên gốc + bí danh của MỌI người nghe tra được.
        rule: Quy tắc (giữ nguyên điều kiện giai đoạn), vd ``"tự xưng 'em' gọi 'anh'"``.
        raw_listener: Phần người nghe như roster ghi.
        group: Mệnh đề nói với NHIỀU người cùng lúc.
    """

    speakers: tuple[str, ...]
    listeners: tuple[str, ...]
    rule: str
    raw_listener: str
    group: bool = False


def _entry_names(entry: RosterEntry) -> tuple[str, ...]:
    """Tên gốc + bí danh của một nhân vật (tên gốc đứng đầu)."""
    return tuple(dict.fromkeys([entry.name.strip(), *_name_aliases(entry.name)]))


#: [v3.23.575] Phiên âm ba chữ viết hoa từng chữ ("Tần Khả Tú"): hai chữ sau là TÊN GỌI.
_GIVEN_NAME_WORDS = 3


def _given_name_keys(entries: list[RosterEntry]) -> dict[str, str]:
    """[v3.23.575] Khoá tên gọi (bỏ họ) → tên gốc; bỏ tên gọi trùng giữa các nhân vật.

    Ca thật gemini-3.8-flash: ``洪大嫂`` "với Khả Tú khi đe dọa tự xưng 'tao' gọi
    'mày'" — bảng ghi phiên âm "Tần Khả Tú" nên bộ tra theo tên đầy đủ bỏ sót, 32 dòng
    洪大嫂 → 秦可秀 dịch không có quy tắc. Chỉ nhận phiên âm đúng ba chữ đều viết hoa
    (tên riêng); "Mạc đại tẩu", "Bác sĩ phòng khám" là danh xưng, không phải họ + tên.
    """
    owners: dict[str, set[str]] = {}
    for entry in entries:
        for alias in _name_aliases(entry.aka):
            words = alias.split()
            if len(words) == _GIVEN_NAME_WORDS and all(w[:1].isupper() for w in words):
                given_name = " ".join(words[1:]).lower()
                owners.setdefault(given_name, set()).add(entry.name.strip())
    return {key: next(iter(names)) for key, names in owners.items() if len(names) == 1}


def _listener_index(entries: list[RosterEntry]) -> dict[str, list[str]]:
    """Khoá tra người nghe (phiên âm VÀ tên gốc, chữ thường) → tên + bí danh nhân vật."""
    index: dict[str, list[str]] = {}
    names_of = {entry.name.strip(): _entry_names(entry) for entry in entries}
    for entry in entries:
        names = _entry_names(entry)
        # [v3.23.568] Quy tắc có thể gọi người nghe bằng TÊN GỐC ("với 柳慕云 tự xưng…") —
        # ca thật gemini-3.6-flash: 140/160 quy tắc viết như vậy và bị bỏ sót toàn bộ.
        for key in [*_name_aliases(entry.aka), *_name_aliases(entry.name)]:
            bucket = index.setdefault(key.lower(), [])
            bucket.extend(name for name in names if name not in bucket)
    for key, owner in _given_name_keys(entries).items():
        # Tên đầy đủ/bí danh luôn thắng tên gọi suy ra.
        index.setdefault(key, list(names_of[owner]))
    return index


def _resolve_listener(part: str, index: dict[str, list[str]]) -> tuple[str, str] | None:
    """Khoá khớp dài nhất ở ĐẦU ``part`` và phần điều kiện còn lại ("trước khi thua")."""
    lowered = part.strip().lower()
    for key in sorted(index, key=len, reverse=True):
        if lowered == key or lowered.startswith(key + " "):
            return key, part.strip()[len(key):].strip()
    return None


_PARENTHESIS_RE = re.compile(r"\((?P<inner>[^)]*)\)")

#: [v3.23.623] Người nghe viết lệch tên (ca thật: nhân vật là "Dao Dao" nhưng mệnh đề ghi
#: "với Diêu Dao") chỉ được nhận khi MỘT ứng viên nổi hẳn lên: điểm tương đồng tối thiểu,
#: hơn ứng viên thứ hai một khoảng, và chung ít nhất một âm tiết nguyên vẹn. Gán nhầm sang
#: người khác tệ hơn không có quy tắc (Tống Tiền ≠ Tống Dao), nên ngưỡng khắt khe.
_FUZZY_MIN_SCORE = 60.0
_FUZZY_MIN_GAP = 25.0
_FUZZY_MAX_WORDS = 4


def _fuzzy_listener_key(part: str, index: dict[str, list[str]], speaker: str) -> str | None:
    """Khoá của ``index`` gần nhất với ``part`` nếu ĐỦ chắc chắn, ngược lại ``None``."""
    query = part.strip().lower()
    words = query.split()
    if not query or len(words) > _FUZZY_MAX_WORDS or re.search(r"[(),;:'\"]", query):
        return None
    try:
        from rapidfuzz import fuzz
    except ImportError:  # thiếu thư viện → không đoán
        return None
    scored: list[tuple[float, str]] = []
    for key, owners in index.items():
        # Cùng số âm tiết (lệch chính tả, không phải rút gọn/danh xưng: "đại tẩu" ≠ "Mạc đại tẩu").
        if speaker in owners or len(key.split()) != len(words) or not set(words) & set(key.split()):
            continue
        scored.append((fuzz.ratio(query, key), key))
    scored.sort(reverse=True)
    if not scored or scored[0][0] < _FUZZY_MIN_SCORE:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < _FUZZY_MIN_GAP:
        # Hai khoá có thể cùng chủ (tên gốc + phiên âm): chỉ mơ hồ khi khác chủ.
        if set(index[scored[0][1]]) != set(index[scored[1][1]]):
            return None
    return scored[0][1]


@dataclass(frozen=True, slots=True)
class _ListenerResolver:
    """[v3.23.575] Tra phần người nghe của một mệnh đề xưng hô ra tên nhân vật.

    Thứ tự: tên/bí danh (kể cả tên gọi) → tên trong ngoặc ("mẹ ruột (Bà Chu)") → quan
    hệ gia đình với người nói ("với vợ", "với cha") — xem ``roster_kinship``.
    """

    index: dict[str, list[str]]
    names_of: dict[str, tuple[str, ...]]
    kinship: KinshipGraph

    @classmethod
    def build(cls, entries: list[RosterEntry]) -> _ListenerResolver:
        """Dựng bộ tra cho một bảng nhân vật."""
        index = _listener_index(entries)
        names_of = {entry.name.strip(): _entry_names(entry) for entry in entries}

        def find_person(text: str) -> str | None:
            names = index.get(text.strip().lower(), [])
            owners = [name for name in names if name in names_of]
            return owners[0] if len(owners) == 1 else None

        people = [
            KinPerson(entry.name.strip(), entry.gender, entry.role) for entry in entries
        ]
        return cls(index, names_of, KinshipGraph.build(people, find_person))

    def resolve(self, part: str, speaker: str) -> tuple[tuple[str, ...], str] | None:
        """(tên + bí danh người nghe, điều kiện giai đoạn); không tra được → None."""
        hit = _resolve_listener(part, self.index)
        if hit is not None:
            return tuple(self.index[hit[0]]), hit[1]
        bracket = _PARENTHESIS_RE.search(part)
        if bracket is not None:
            inner = _resolve_listener(bracket.group("inner"), self.index)
            if inner is not None and not inner[1]:
                return tuple(self.index[inner[0]]), part[bracket.end():].strip()
            part = f"{part[:bracket.start()]} {part[bracket.end():]}".strip()
        kin = self.kinship.resolve(part, speaker)
        if kin is None:
            fuzzy = _fuzzy_listener_key(part, self.index, speaker)
            if fuzzy is None:
                return None
            return tuple(self.index[fuzzy]), ""
        person, condition = kin
        # Phần sau quan hệ chỉ nhận khi là điều kiện giai đoạn ("khi say"); còn lại là
        # mô tả khác ("vợ hai", "cha của A") — không đoán.
        if condition and not _ADDRESS_PHASE_RE.match(condition):
            return None
        return self.names_of[person], condition

    def resolve_all(
        self, raw_listener: str, speaker: str
    ) -> tuple[tuple[str, ...], str, int]:
        """Tra danh sách người nghe "A và B": (tên tra được, điều kiện, số người nghe)."""
        parts = [p for p in _LISTENER_SPLIT_RE.split(raw_listener) if p.strip()]
        hits = [hit for hit in (self.resolve(p, speaker) for p in parts) if hit]
        listeners = tuple(dict.fromkeys(name for names, _ in hits for name in names))
        condition = hits[-1][1] if hits else ""
        return listeners, condition, len(parts)


def address_clauses(characters: str) -> list[AddressClause]:
    """[v3.23.568] Đọc MỌI mệnh đề xưng hô của bảng nhân vật, tra người nghe (hàm thuần).

    Args:
        characters: Bảng nhân vật đã chuẩn hoá.

    Returns:
        Các mệnh đề theo thứ tự trong bảng. Mệnh đề không tra được người nghe nào vẫn được
        trả (``listeners`` rỗng) để file chuẩn đoán đếm được.
    """
    entries = parse_roster(characters)
    resolver = _ListenerResolver.build(entries)
    clauses: list[AddressClause] = []
    for entry in entries:
        speakers = _entry_names(entry)
        if not speakers[0] or not re.search(r"xưng|gọi", entry.role, re.IGNORECASE):
            continue
        for segment in re.split(r";|,(?=\s*với\b)", entry.role):
            match = _ADDRESS_RULE_RE.search(segment)
            if not match:
                clauses.extend(_call_list_clauses(segment, speakers, resolver))
                continue
            raw_listener = match.group("other").strip()
            tail = segment[match.end("other"):].strip().rstrip(".").strip()
            listeners, condition, count = resolver.resolve_all(raw_listener, speakers[0])
            clauses.append(
                AddressClause(
                    speakers=speakers,
                    listeners=listeners,
                    rule=_RULE_LA_RE.sub(r"\1 '", f"{condition} {tail}".strip()),
                    raw_listener=raw_listener,
                    group=count > 1,
                )
            )
    return clauses


def _call_list_clauses(
    segment: str, speakers: tuple[str, ...], resolver: _ListenerResolver
) -> list[AddressClause]:
    """[v3.23.574] Mệnh đề từ cách viết "tự xưng 'S', gọi <Tên> (là) 'A', gọi …".

    Cũng đọc "với <Tên> gọi là 'A'" (thiếu phần tự xưng nên cách đọc chuẩn bỏ qua).

    Cách tự xưng chung (nếu có) được gắn cho từng người nghe — đúng như bảng ghi. Người
    nghe không tra được ("gọi cấp dưới 'cô/cậu'") vẫn được trả với ``listeners`` rỗng để
    chuẩn đoán đếm được.
    """
    self_match = _GLOBAL_SELF_RE.search(segment)
    self_part = f"tự xưng '{self_match.group('self')}' " if self_match else ""
    clauses: list[AddressClause] = []
    for match in _CALL_LISTENER_RE.finditer(segment):
        raw_listener = (match.group("with") or match.group("other") or "").strip()
        if not raw_listener:
            continue
        listeners, _, count = resolver.resolve_all(raw_listener, speakers[0])
        clauses.append(
            AddressClause(
                speakers=speakers,
                listeners=listeners,
                rule=f"{self_part}gọi '{match.group('addr')}'",
                raw_listener=raw_listener,
                group=count > 1,
            )
        )
    return clauses


#: Từ chừng này nhân vật trở lên thì bảng phải có quy tắc xưng hô (một người thì không có
#: cặp nào để quy định).
MIN_CHARACTERS_FOR_ADDRESS_RULES = 2


def lacks_address_rules(characters: str) -> bool:
    """[v3.23.578] Bảng nhiều nhân vật mà KHÔNG có mệnh đề xưng hô nào (hàm thuần).

    Khâu phân tích bắt buộc chốt xưng hô từ v3.23.524, nhưng phân tích đã lưu được KHÔI
    PHỤC chỉ theo hash phụ đề + ngôn ngữ đích — bản phân tích cũ hơn thế được dùng lại
    mãi. Chuẩn đoán thật (phim 5540 dòng, bảng 48 nhân vật, 0 mệnh đề xưng hô): mọi lưới
    xưng hô (gắn quy tắc vào payload, sửa theo bảng, độc thoại theo cặp, chuẩn đoán
    ``address_rule_*``) đều im lặng, còn ``address_rule_violations = 0`` trông như tốt.

    Args:
        characters: Bảng nhân vật đã chuẩn hoá.

    Returns:
        True nếu bảng có ≥ :data:`MIN_CHARACTERS_FOR_ADDRESS_RULES` nhân vật mà không đọc
        được mệnh đề xưng hô nào.
    """
    return len(parse_roster(characters)) >= MIN_CHARACTERS_FOR_ADDRESS_RULES and not (
        address_clauses(characters)
    )


def address_rule_map(characters: str) -> dict[tuple[str, str], str]:
    """[v3.23.551] Tra sẵn quy tắc xưng hô cho từng cặp ``(người nói, người nghe)`` theo TÊN GỐC.

    Vì sao cần: khâu dịch nhận ``speaker``/``addressee`` ở từng dòng nhưng phải TỰ TRA quy tắc
    trong một bảng nhân vật dài (38 nhân vật, mỗi người một đoạn mô tả). Chuẩn đoán thật: roster
    quy định ``张依贝`` với Tống Tiền "tự xưng 'em' gọi 'anh Tống Tiền'", vậy mà dòng 601-655 vẫn
    ra "tôi"/"cậu" (tự xưng em 11 / tôi 5, gọi anh 13 / cậu 11 trong CÙNG MỘT CẢNH). Gắn thẳng quy
    tắc vào dòng cần dùng thay vì trông model tự tra.

    Người nghe được tra theo PHIÊN ÂM ("với Tống Tiền …") lẫn TÊN GỐC ("với 宋钱 …") [v3.23.568].
    Mệnh đề nói với NHIỀU người ("với Tống Tiền và Tống Dao") áp cho từng người, trừ khi
    cách xưng/gọi là số nhiều ("gọi 'các người'") — khi đó nó sai với từng người riêng lẻ.
    Quy tắc riêng từng cặp luôn thắng quy tắc nhóm.

    Args:
        characters: Bảng nhân vật đã chuẩn hoá.

    Returns:
        Bản đồ ``(bí_danh_người_nói, bí_danh_người_nghe) → "tự xưng '…' gọi '…'"``. Quy tắc có
        điều kiện giai đoạn được giữ NGUYÊN VĂN để model tự chọn đúng giai đoạn.
    """
    clauses = address_clauses(characters)
    rules: dict[tuple[str, str], str] = {}
    for clause in [c for c in clauses if not c.group] + [c for c in clauses if c.group]:
        if clause.group and _PLURAL_FORM_RE.search(clause.rule):
            continue
        for speaker in clause.speakers:
            for listener in clause.listeners:
                rules.setdefault((speaker, listener), clause.rule)
    return rules


def lookup_address_rule(
    rules: dict[tuple[str, str], str], speaker: str, addressee: str
) -> str:
    """[v3.23.551] Tra quy tắc xưng hô, chịu được khác biệt thứ tự/tập bí danh giữa cues và roster.

    Cues có thể ghi ``"柳河 / 柳杰"`` trong khi roster ghi ``"柳杰 / 柳河"``; thử mọi cặp bí danh.
    Người nói trùng người nghe (lỗi nhãn cues, thường là độc thoại) thì không trả quy tắc nào.

    Args:
        rules: Kết quả :func:`address_rule_map`.
        speaker: Nhãn người nói của dòng.
        addressee: Nhãn người nghe của dòng.

    Returns:
        Quy tắc xưng hô, hoặc chuỗi rỗng nếu không có.
    """
    if not rules or not speaker or not addressee:
        return ""
    direct = rules.get((speaker.strip(), addressee.strip()))
    if direct:
        return direct
    speakers = _name_aliases(speaker)
    listeners = _name_aliases(addressee)
    if set(speakers) & set(listeners):
        return ""
    for spk in speakers:
        for lis in listeners:
            rule = rules.get((spk, lis))
            if rule:
                return rule
    return ""
