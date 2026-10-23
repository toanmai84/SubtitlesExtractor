"""[v3.23.552] Lưới chặn: giai đoạn tinh chỉnh LÀM MẤT tên riêng mà giai đoạn trước đã dịch đúng.

Vì sao cần: chuẩn đoán thật (phim 3902 dòng) — dòng ``桃花嫂子的元神碎片``, giai đoạn DỊCH THÔ ra
"mảnh nguyên thần của **chị dâu** Đào Hoa" (đúng bảng thuật ngữ ``桃花嫂子 = chị dâu Đào Hoa``), giai
đoạn TINH CHỈNH đổi thành "của **vợ** Đào Hoa" — sai hẳn quan hệ (Hứa Đào Hoa là góa phụ). Ứng dụng
đã có lưới chặn cho hai dạng hồi quy cùng kiểu (chèn lại chữ Hán, làm rỗng dòng) nhưng chưa có cho
tên riêng.

Chỉ xét mục thuật ngữ loại TÊN RIÊNG — thành ngữ/khẩu ngữ được phép diễn đạt lại khi tinh chỉnh.
Đo trên 3 lần chạy thật cùng phim: đúng 1 lượt bắn, và đó chính là lỗi thật trên.

Toàn bộ module là HÀM THUẦN: không I/O, không phụ thuộc Qt/AI.
"""

from __future__ import annotations

import re

from subtitles_extractor.application.services.roster_format import parse_glossary
from subtitles_extractor.application.services.translation_diagnostics import (
    GLOSSARY_KIND_NAME,
    glossary_alternatives,
    glossary_entry_kind,
    matches_expected_term,
)

NEIGHBOUR_LINES = 1
"""Soi cả dòng kề khi kiểm bản MỚI: tinh chỉnh được phép dời tên sang dòng bên (mảnh câu)."""


def find_name_regressions(
    sources: dict[int, str],
    previous: dict[int, str],
    current: dict[int, str],
    glossary: str,
) -> list[tuple[int, str]]:
    """Tìm các dòng mà bản TRƯỚC thể hiện đúng tên riêng, còn bản MỚI (kể cả dòng kề) thì mất.

    Args:
        sources: Câu gốc theo chỉ số.
        previous: Bản dịch của giai đoạn trước theo chỉ số.
        current: Bản dịch của giai đoạn vừa chạy theo chỉ số.
        glossary: Bảng thuật ngữ ``gốc = dịch``.

    Returns:
        Danh sách ``(chỉ số dòng, thuật ngữ gốc)`` bị hồi quy, sắp theo chỉ số.
    """
    if not glossary.strip():
        return []
    names = [
        entry for entry in parse_glossary(glossary)
        if entry.translation.strip()
        and glossary_entry_kind(entry.translation) == GLOSSARY_KIND_NAME
    ]
    findings: dict[int, str] = {}
    for entry in names:
        alternatives = glossary_alternatives(entry.translation)
        aliases = [alias.strip() for alias in re.split(r"\s*[/|]\s*", entry.term) if alias.strip()]
        if not alternatives or not aliases:
            continue
        for index, source in sources.items():
            if index in findings or not any(alias in source for alias in aliases):
                continue
            if not matches_expected_term(alternatives, previous.get(index, "").lower()):
                continue
            window = " ".join(
                current.get(neighbour, "").lower()
                for neighbour in range(index - NEIGHBOUR_LINES, index + NEIGHBOUR_LINES + 1)
            )
            if not matches_expected_term(alternatives, window):
                findings[index] = entry.term
    return sorted(findings.items())


__all__ = ["find_name_regressions"]
