"""Kiểm tra tính nhất quán của bảng thuật ngữ trong bản dịch (QA hậu kỳ).

[v3.23.54] Theo nghiên cứu dịch máy nhận biết ngữ cảnh (Context-Aware NMT), tên riêng và
thuật ngữ thường bị dịch KHÔNG nhất quán giữa các câu — một lỗi khó phát hiện thủ công.
Module này đối chiếu bảng thuật ngữ (glossary) do người dùng cung cấp với cặp câu
nguồn → câu dịch, phát hiện các dòng mà thuật ngữ gốc xuất hiện nhưng bản dịch chuẩn lại
KHÔNG xuất hiện trong câu đích.

Thiết kế thuần (pure functions) — không phụ thuộc UI hay I/O — để dễ kiểm thử và tái dùng.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "GlossaryEntry",
    "GlossaryViolation",
    "parse_glossary",
    "check_glossary_consistency",
]


# Dấu phân tách giữa thuật ngữ gốc và bản dịch chuẩn trong một dòng glossary.
_SEPARATORS: tuple[str, ...] = ("=>", "->", "→", ":", "：", "=")


@dataclass(frozen=True, slots=True)
class GlossaryEntry:
    """Một mục thuật ngữ: ``source`` (gốc) → ``target`` (bản dịch chuẩn)."""

    source: str
    target: str


@dataclass(frozen=True, slots=True)
class GlossaryViolation:
    """Một vi phạm nhất quán thuật ngữ được phát hiện.

    Attributes:
        line_index: Chỉ số dòng phụ đề (1-based) nơi phát hiện vi phạm.
        source_term: Thuật ngữ gốc xuất hiện trong câu nguồn.
        expected_target: Bản dịch chuẩn lẽ ra phải xuất hiện trong câu đích.
        translated_text: Nội dung câu dịch thực tế (để người dùng đối chiếu).
    """

    line_index: int
    source_term: str
    expected_target: str
    translated_text: str


def _split_entry(line: str) -> GlossaryEntry | None:
    """Tách một dòng glossary thành (source, target). Trả None nếu không hợp lệ."""
    for sep in _SEPARATORS:
        if sep in line:
            left, _, right = line.partition(sep)
            source = left.strip()
            # Bản dịch chuẩn có thể kèm chú thích trong ngoặc — lấy phần chính trước ngoặc.
            target_full = right.strip()
            target_main = re.split(r"[(（]", target_full, maxsplit=1)[0].strip()
            if source and target_main:
                return GlossaryEntry(source=source, target=target_main)
            return None
    return None


def parse_glossary(glossary_text: str) -> list[GlossaryEntry]:
    """Phân tích văn bản glossary nhiều dòng thành danh sách :class:`GlossaryEntry`.

    Mỗi dòng có dạng ``gốc => bản dịch`` (chấp nhận nhiều loại dấu phân tách). Dòng trống
    hoặc không có dấu phân tách hợp lệ sẽ được bỏ qua.

    Args:
        glossary_text: Nội dung bảng thuật ngữ, mỗi mục một dòng.

    Returns:
        Danh sách các mục thuật ngữ đã phân tích (có thể rỗng).
    """
    entries: list[GlossaryEntry] = []
    seen: set[str] = set()
    for raw_line in (glossary_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        entry = _split_entry(line)
        if entry is not None and entry.source not in seen:
            entries.append(entry)
            seen.add(entry.source)
    return entries


def check_glossary_consistency(
    glossary_text: str,
    source_texts: list[str],
    translated_texts: list[str],
) -> list[GlossaryViolation]:
    """Phát hiện các dòng mà thuật ngữ gốc có mặt nhưng bản dịch chuẩn lại thiếu.

    Với mỗi cặp (câu nguồn, câu dịch) cùng chỉ số, nếu một thuật ngữ gốc trong glossary
    xuất hiện trong câu nguồn mà bản dịch chuẩn tương ứng KHÔNG xuất hiện trong câu dịch,
    thì ghi nhận một vi phạm. So khớp không phân biệt hoa/thường.

    Args:
        glossary_text: Bảng thuật ngữ (mỗi dòng ``gốc => chuẩn``).
        source_texts: Danh sách câu phụ đề gốc.
        translated_texts: Danh sách câu đã dịch (cùng thứ tự với ``source_texts``).

    Returns:
        Danh sách vi phạm theo thứ tự dòng. Rỗng nếu nhất quán hoàn toàn.
    """
    entries = parse_glossary(glossary_text)
    if not entries:
        return []

    violations: list[GlossaryViolation] = []
    pair_count = min(len(source_texts), len(translated_texts))
    for idx in range(pair_count):
        source_lower = source_texts[idx].lower()
        translated_lower = translated_texts[idx].lower()
        for entry in entries:
            if entry.source.lower() in source_lower:
                if entry.target.lower() not in translated_lower:
                    violations.append(
                        GlossaryViolation(
                            line_index=idx + 1,
                            source_term=entry.source,
                            expected_target=entry.target,
                            translated_text=translated_texts[idx],
                        )
                    )
    return violations
