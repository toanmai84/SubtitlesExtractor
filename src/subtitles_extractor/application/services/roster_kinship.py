"""[v3.23.575] Tra người nghe theo QUAN HỆ GIA ĐÌNH trong bảng nhân vật ("với vợ").

Chuẩn đoán thật (gemini-3.8-flash, phim "灭门惨案II借种", v3.23.574): bảng nhân vật
ghi quy tắc xưng hô của gia đình Hồng Chấn bằng QUAN HỆ thay vì bằng tên — ``洪震``:
"với con trai tự xưng 'ba' gọi 'con'; với vợ tự xưng 'tôi' gọi 'bà nó'", ``洪志權``:
"với cha tự xưng 'con' gọi 'ba'". Bộ tra chỉ biết tên nên 6 quy tắc này không gắn được
vào cặp nhân vật nào: 洪震 → 洪志權 (32 dòng), 洪震 → 洪大嫂 (31 dòng), 洪志權 → 洪震
(19 dòng) đều dịch KHÔNG có quy tắc, và bản dịch trộn "Tao bảo mày…" (dòng 826) với
"Ba không được thì con…" (dòng 829) trong cùng một cảnh.

Thông tin để tra ĐÃ CÓ trong chính bảng: phần mô tả của người kia mở đầu bằng quan hệ
— ``洪志權``: "Con trai ngỗ ngược của Hồng Chấn", ``洪大嫂``: "Vợ của Hồng Chấn". Module
này đọc các quan hệ đó thành đồ thị rồi tra người nghe:

* Quan hệ chỉ đọc ở ĐẦU một vế mô tả (sau dấu phẩy/chấm/hai chấm): "đưa cho mẹ Đào
  Hoa…" ở giữa câu nói về người KHÁC, không phải về nhân vật đang mô tả.
* Tên sau quan hệ phải khớp TRỌN một tên/bí danh trong bảng.
* "mẹ vợ", "chị dâu", "anh họ", "mẹ kế"… là quan hệ KHÁC nên bị loại.
* Mẹ của một người cũng là vợ của cha người đó (và ngược lại với con) — bảng thường
  chỉ ghi một chiều: "Con trai … của Hồng Chấn", "Vợ của Hồng Chấn".
* Chỉ nhận khi còn ĐÚNG MỘT ứng viên hợp giới tính; mơ hồ thì bỏ qua (không đoán).

Đo trên 50 bảng nhân vật thật: đọc được 214 quan hệ (61 quan hệ khác nhau), soát tay
không quan hệ nào sai. Cùng tên gọi và tên trong ngoặc (xem ``roster_format``), số
mệnh đề xưng hô không tra được người nghe giảm 57 → 43; 40 mệnh đề còn lại là nhóm
người ("cấp dưới", "quan khách", "đàn em"…), 3 mệnh đề gọi bằng danh xưng không có
trong bảng ("Liễu lão gia tử").

Toàn bộ là hàm/lớp thuần: không I/O, không phụ thuộc module đọc bảng nhân vật.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

#: Từ quan hệ → (lớp quan hệ, giới tính người mang quan hệ; rỗng = không ràng buộc).
KIN_RELATIONS: dict[str, tuple[str, str]] = {
    "vợ": ("spouse", "nữ"),
    "chồng": ("spouse", "nam"),
    "cha": ("parent", "nam"),
    "bố": ("parent", "nam"),
    "ba": ("parent", "nam"),
    "mẹ": ("parent", "nữ"),
    "con trai": ("child", "nam"),
    "con gái": ("child", "nữ"),
    "con": ("child", ""),
    "anh trai": ("elder", "nam"),
    "chị gái": ("elder", "nữ"),
    "em trai": ("younger", "nam"),
    "em gái": ("younger", "nữ"),
}
#: A là <lớp> của B ⇔ B là <lớp nghịch> của A.
_INVERSE_CLASS: dict[str, str] = {
    "spouse": "spouse",
    "parent": "child",
    "child": "parent",
    "elder": "younger",
    "younger": "elder",
}
#: Từ đứng NGAY sau từ quan hệ biến nó thành quan hệ khác: "mẹ vợ", "bố chồng",
#: "mẹ kế", "cha nuôi", "con dâu", "con rể", "anh họ", "cha đỡ đầu", "em cùng cha…".
EXCLUDED_NEXT_WORDS = frozenset({
    "vợ", "chồng", "kế", "ghẻ", "nuôi", "dượng", "rể", "dâu", "họ", "nội", "ngoại",
    "đỡ", "hờ", "cùng",
})
#: Từ bổ nghĩa đứng sau quan hệ ở phần người nghe mà không đổi quan hệ ("mẹ ruột").
ABSORBED_WORDS = frozenset({"ruột", "đẻ"})
#: Số từ bổ nghĩa tối đa giữa quan hệ và "của" ("Con trai ngỗ ngược của Hồng Chấn",
#: "em gái ngoan ngoãn hiếu thảo của Tống Tiền").
MAX_MODIFIER_WORDS = 4

_DESCRIPTION_END_RE = re.compile(r"xưng\s+hô\s*:", re.IGNORECASE)
_CLAUSE_SPLIT_RE = re.compile("[,;:.\u2014\u2013()]")
_LEADING_FILLER_RE = re.compile(r"^(?:người|là)\s+", re.IGNORECASE)
_OBJECT_SPLIT_RE = re.compile(r"\s*(?:\bvà\b|&|、|和)\s*", re.IGNORECASE)
_RELATION_WORDS = sorted(KIN_RELATIONS, key=len, reverse=True)

#: Tra một chuỗi (đúng TRỌN một tên/bí danh) ra tên gốc của nhân vật; không có → None.
PersonFinder = Callable[[str], str | None]


@dataclass(frozen=True, slots=True)
class KinPerson:
    """Một nhân vật, gọn cho việc đọc quan hệ.

    Attributes:
        key: Tên gốc của nhân vật trong bảng.
        gender: Giới tính như bảng ghi ("nam", "nữ", có thể rỗng).
        role: Phần vai trò + mô tả (kể cả phần "xưng hô:").
    """

    key: str
    gender: str
    role: str


@dataclass(frozen=True, slots=True)
class KinEdge:
    """``person`` là <``relation``> của ``of`` (vd 洪大嫂 là "spouse" của 洪震)."""

    person: str
    relation: str
    of: str


def normalize_gender(gender: str) -> str:
    """``"Nam"``/``"nam giới"`` → ``"nam"``, ``"nữ"`` → ``"nữ"``, còn lại → ``""``."""
    lowered = (gender or "").strip().lower()
    if lowered.startswith("nữ") or lowered.startswith("nu"):
        return "nữ"
    if lowered.startswith("nam"):
        return "nam"
    return ""


def split_relation(text: str) -> tuple[str, str] | None:
    """Tách từ quan hệ ở ĐẦU chuỗi: ``"Con trai ngỗ ngược…"`` → ``("con trai", …)``.

    Returns:
        (từ quan hệ viết thường, phần còn lại), hoặc None nếu chuỗi không mở đầu bằng
        quan hệ hay quan hệ bị từ đứng sau đổi nghĩa ("mẹ vợ", "chị dâu").
    """
    lowered = text.strip().lower()
    for relation in _RELATION_WORDS:
        if lowered != relation and not lowered.startswith(relation + " "):
            continue
        remainder = text.strip()[len(relation):].strip()
        first_word = remainder.split(maxsplit=1)[0].lower() if remainder else ""
        if first_word in EXCLUDED_NEXT_WORDS:
            return None
        return relation, remainder
    return None


def _objects_of(words: list[str], find_person: PersonFinder) -> list[str]:
    """Người được nói tới sau quan hệ: "của A và B" → [A, B] (dừng ở phần không phải tên).

    Trả rỗng khi phần ngay sau quan hệ (bỏ qua tối đa vài từ bổ nghĩa) không phải tên.
    """
    for skip in range(MAX_MODIFIER_WORDS + 1):
        tail = words[skip:]
        if tail and tail[0].lower() == "của":
            tail = tail[1:]
        elif skip:
            # Có từ bổ nghĩa thì bắt buộc có "của" — không thì dễ nuốt nhầm cả câu.
            continue
        found: list[str] = []
        for part in _OBJECT_SPLIT_RE.split(" ".join(tail)):
            person = find_person(part) if part.strip() else None
            if person is None:
                break
            found.append(person)
        if found:
            return found
    return []


def _description_clauses(role: str) -> list[str]:
    """Các vế mô tả (bỏ phần "xưng hô:" và từ đệm "người"/"là" ở đầu vế)."""
    description = _DESCRIPTION_END_RE.split(role or "", maxsplit=1)[0]
    return [
        _LEADING_FILLER_RE.sub("", clause.strip())
        for clause in _CLAUSE_SPLIT_RE.split(description)
        if clause.strip()
    ]


def kin_edges(people: Iterable[KinPerson], find_person: PersonFinder) -> list[KinEdge]:
    """Đọc mọi quan hệ gia đình ghi ở đầu các vế mô tả nhân vật (hàm thuần).

    Args:
        people: Các nhân vật của bảng.
        find_person: Tra TRỌN một tên/bí danh ra tên gốc.

    Returns:
        Các cạnh quan hệ, không trùng, theo thứ tự trong bảng.
    """
    edges: list[KinEdge] = []
    for person in people:
        for clause in _description_clauses(person.role):
            split = split_relation(clause)
            if split is None:
                continue
            relation_class = KIN_RELATIONS[split[0]][0]
            for other in _objects_of(split[1].split(), find_person):
                edge = KinEdge(person.key, relation_class, other)
                if other != person.key and edge not in edges:
                    edges.append(edge)
    return edges


@dataclass(frozen=True, slots=True)
class KinshipGraph:
    """Đồ thị quan hệ gia đình của một bảng nhân vật.

    Attributes:
        genders: Tên gốc → giới tính đã chuẩn hoá.
        edges: Các quan hệ đọc được.
    """

    genders: dict[str, str]
    edges: tuple[KinEdge, ...]

    @classmethod
    def build(cls, people: list[KinPerson], find_person: PersonFinder) -> KinshipGraph:
        """Dựng đồ thị từ bảng nhân vật."""
        return cls(
            genders={person.key: normalize_gender(person.gender) for person in people},
            edges=tuple(kin_edges(people, find_person)),
        )

    def relatives(self, key: str, relation_class: str) -> set[str]:
        """Những người là <``relation_class``> của ``key`` (đọc cả hai chiều cạnh)."""
        inverse = _INVERSE_CLASS[relation_class]
        found = {
            edge.person
            for edge in self.edges
            if edge.of == key and edge.relation == relation_class
        }
        found |= {e.of for e in self.edges if e.person == key and e.relation == inverse}
        return found

    def resolve(self, listener: str, speaker: str) -> tuple[str, str] | None:
        """Tra người nghe ghi bằng quan hệ ("vợ", "mẹ ruột khi say") của ``speaker``.

        Args:
            listener: Phần người nghe như bảng ghi.
            speaker: Tên gốc người nói.

        Returns:
            (tên gốc người nghe, phần còn lại sau quan hệ — thường là điều kiện giai
            đoạn), hoặc None nếu không phải quan hệ hay không có ĐÚNG MỘT ứng viên.
        """
        split = split_relation(listener)
        if split is None:
            return None
        relation, remainder = split
        words = remainder.split()
        while words and words[0].lower() in ABSORBED_WORDS:
            words.pop(0)
        relation_class, gender = KIN_RELATIONS[relation]
        candidates = self.relatives(speaker, relation_class)
        # Bảng thường chỉ ghi một chiều ("Con trai của Hồng Chấn", "Vợ của Hồng Chấn"):
        # mẹ của con là vợ của cha, con của vợ là con của chồng.
        if relation_class == "parent":
            for parent in set(candidates):
                candidates |= self.relatives(parent, "spouse")
        elif relation_class == "child":
            for spouse in self.relatives(speaker, "spouse"):
                candidates |= self.relatives(spouse, "child")
        matching = {
            candidate
            for candidate in candidates
            if candidate != speaker
            and (not gender or self.genders.get(candidate, "") in ("", gender))
        }
        if len(matching) != 1:
            return None
        return next(iter(matching)), " ".join(words)


__all__ = [
    "ABSORBED_WORDS",
    "EXCLUDED_NEXT_WORDS",
    "KIN_RELATIONS",
    "MAX_MODIFIER_WORDS",
    "KinEdge",
    "KinPerson",
    "KinshipGraph",
    "PersonFinder",
    "kin_edges",
    "normalize_gender",
    "split_relation",
]
