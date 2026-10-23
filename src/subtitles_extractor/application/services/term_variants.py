"""[v3.23.645] Nhận ra BIẾN THỂ chính tả của tên tổ chức trong phụ đề (灵霄宗 / 凌霄宗).

OCR và người viết kịch bản dùng lẫn hai chữ đồng âm/gần dạng cho cùng một tông môn. Bảng thuật
ngữ chỉ ghi MỘT dạng (``drop_variant_glossary_entries`` còn chủ động bỏ dạng thứ hai), nên khối
nhắc thuật ngữ theo lô — vốn khớp đúng chuỗi — không thấy dạng kia và model dịch nó một cách
khác ("Lăng Tiêu Tông" bên cạnh "Linh Tiêu Tông").

Chỉ xét tên kết thúc bằng hậu tố tổ chức ĐẶC TRƯNG (không dùng 会/帮/门/教/谷… vì là chữ thường gặp:
``天地会`` + "明天才会到" từng sinh ra biến thể giả ``天才会``), khác đúng MỘT chữ ở phần tên riêng, và dạng kia
KHÔNG phải một thuật ngữ khác trong bảng — hai tông môn có tên khác nhau một chữ mà đều được
ghi trong bảng vẫn là hai thực thể riêng.
"""

from __future__ import annotations

import re

ORGANISATION_SUFFIXES = "宗殿阁閣派"
"""Hậu tố chỉ tổ chức đủ đặc trưng; chỉ tên kết thúc bằng một trong các chữ này mới được xét biến thể."""

_MIN_TERM_LENGTH = 3
_MAX_TERM_LENGTH = 6
_HAN_RE = re.compile(r"[一-鿿㐀-䶿]+")


def is_organisation_term(term: str) -> bool:
    """Tên có dạng ``<tên riêng><hậu tố tổ chức>`` với tên riêng ít nhất hai chữ Hán."""
    if not (_MIN_TERM_LENGTH <= len(term) <= _MAX_TERM_LENGTH):
        return False
    return term[-1] in ORGANISATION_SUFFIXES and _HAN_RE.fullmatch(term) is not None


class VariantIndex:
    """Chỉ mục các cụm chữ Hán kết thúc bằng hậu tố tổ chức trong một văn bản.

    Dựng MỘT lần rồi tra cho nhiều thuật ngữ: chỉ cụm kết thúc bằng hậu tố (hiếm) được lưu, mỗi cụm
    kèm các "mặt nạ" thay một chữ — nên tra một thuật ngữ tốn vài lần băm, không quét văn bản và
    không dựng regex (bản đầu dùng regex: 400 tên tổ chức × 90 lô ≈ 104 giây CPU).
    """

    def __init__(self, text: str) -> None:
        self._masked: dict[tuple[str, int, str], dict[str, int]] = {}
        for run in _HAN_RE.finditer(text or ""):
            chunk, base = run.group(0), run.start()
            for end, char in enumerate(chunk):
                if char not in ORGANISATION_SUFFIXES:
                    continue
                for length in range(_MIN_TERM_LENGTH, _MAX_TERM_LENGTH + 1):
                    begin = end + 1 - length
                    if begin < 0:
                        break
                    gram = chunk[begin : end + 1]
                    for position in range(length - 1):
                        key = (gram[:position], position, gram[position + 1 :])
                        self._masked.setdefault(key, {}).setdefault(gram, base + begin)

    def variants_of(self, term: str, known_terms: frozenset[str] = frozenset()) -> list[str]:
        """Dạng khác (đúng một chữ ở phần tên riêng) của ``term``, theo thứ tự xuất hiện đầu tiên."""
        if not is_organisation_term(term):
            return []
        found: dict[str, int] = {}
        for position in range(len(term) - 1):
            bucket = self._masked.get((term[:position], position, term[position + 1 :]))
            for gram, where in (bucket or {}).items():
                if gram != term and gram not in known_terms:
                    found.setdefault(gram, where)
        return sorted(found, key=found.__getitem__)


def find_variants(term: str, text: str, known_terms: frozenset[str] = frozenset()) -> list[str]:
    """Các dạng viết khác của ``term`` trong ``text`` (tra một lần; nhiều thuật ngữ thì dùng :class:`VariantIndex`).

    Args:
        term: Tên tổ chức trong bảng thuật ngữ.
        text: Văn bản nguồn cần dò.
        known_terms: Mọi thuật ngữ của bảng — dạng trùng một mục khác không phải biến thể.

    Returns:
        Danh sách dạng khác, thứ tự xuất hiện đầu tiên trong ``text``, không trùng.
    """
    if not text or not is_organisation_term(term):
        return []
    return VariantIndex(text).variants_of(term, known_terms)
