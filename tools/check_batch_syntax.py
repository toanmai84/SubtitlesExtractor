"""Kiểm cú pháp tệp batch của dự án — bắt lỗi mà mắt thường dễ bỏ sót.

VÌ SAO cần
==========
v3.23.333 sửa ``build_windows.bat`` bằng phép thay chuỗi, nhưng mốc kết thúc dùng
``s.index(":skip_whisperx")`` — mà chuỗi đó **cũng nằm trong ``goto :skip_whisperx``**
ở đầu khối. Kết quả: chỉ phần đầu bị thay, khối cũ còn nguyên bên dưới, tạo ra:

* **Nhãn trùng** ``:skip_whisperx`` (hai lần).
* Một lệnh ``pip install whisperx`` vào môi trường CHÍNH chạy **vô điều kiện** — nó nằm
  SAU nhãn nên mọi ``goto`` đều rơi thẳng vào đó.

Hậu quả thật trên máy người dùng: ``huggingface-hub`` bị hạ từ 1.25.1 xuống 0.36.2 và
``gradio`` hỏng. Phép kiểm cũ của tôi CHỈ xét ngoặc cân bằng và ``goto`` có nhãn — nên
không thấy gì bất thường.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BatchIssue:
    """Một vấn đề phát hiện trong tệp batch."""

    line: int
    message: str


def _strip_comment(text: str) -> str:
    stripped = text.strip()
    if stripped.lower().startswith(("rem ", "rem")) or stripped.startswith("::"):
        return ""
    return stripped


def find_duplicate_labels(lines: list[str]) -> list[BatchIssue]:
    """Tìm nhãn bị định nghĩa nhiều lần.

    Batch chỉ nhảy tới nhãn ĐẦU TIÊN, nên nhãn trùng thường là dấu hiệu của một khối
    cũ chưa bị xoá — đúng lỗi đã xảy ra.
    """
    seen: dict[str, int] = {}
    issues: list[BatchIssue] = []
    for number, raw in enumerate(lines, 1):
        text = _strip_comment(raw)
        if not text.startswith(":") or text.startswith("::"):
            continue
        label = text[1:].split()[0].lower()
        if label in seen:
            issues.append(
                BatchIssue(
                    number,
                    f"Nhãn ':{label}' đã được định nghĩa ở dòng {seen[label]} — "
                    "nhiều khả năng còn sót khối cũ chưa xoá.",
                )
            )
        else:
            seen[label] = number
    return issues


def find_missing_labels(lines: list[str]) -> list[BatchIssue]:
    """Tìm ``goto`` trỏ tới nhãn không tồn tại."""
    labels = set()
    for raw in lines:
        text = _strip_comment(raw)
        if text.startswith(":") and not text.startswith("::"):
            labels.add(text[1:].split()[0].lower())

    issues: list[BatchIssue] = []
    for number, raw in enumerate(lines, 1):
        text = _strip_comment(raw)
        for match in re.finditer(r"goto\s+:?(\w+)", text, re.IGNORECASE):
            target = match.group(1).lower()
            if target not in labels and target != "eof":
                issues.append(BatchIssue(number, f"goto :{target} — không có nhãn này."))
    return issues


def find_unbalanced_parentheses(lines: list[str]) -> list[BatchIssue]:
    """Tìm ngoặc không cân bằng (làm hỏng khối ``if``/``for``)."""
    depth = 0
    for number, raw in enumerate(lines, 1):
        text = _strip_comment(raw)
        if text.lower().startswith("echo"):
            continue
        cleaned = re.sub(r'"[^"]*"', "", text).replace("^(", "").replace("^)", "")
        depth += cleaned.count("(") - cleaned.count(")")
        if depth < 0:
            return [BatchIssue(number, "Thừa dấu ')' — khối lệnh bị lệch.")]
    if depth != 0:
        return [BatchIssue(len(lines), f"Thiếu {depth} dấu ')' ở cuối tệp.")]
    return []


def find_unreachable_after_label(lines: list[str]) -> list[BatchIssue]:
    """Cảnh báo lệnh CÀI ĐẶT nằm ngay sau nhãn ``skip_*``.

    Nhãn tên ``skip_...`` hàm ý "bỏ qua phần trên". Nếu ngay sau nó lại có lệnh cài
    thì mọi nhánh bỏ qua đều rơi thẳng vào lệnh đó — chính là lỗi đã xảy ra với
    ``pip install whisperx``.
    """
    issues: list[BatchIssue] = []
    for index, raw in enumerate(lines):
        text = _strip_comment(raw)
        if not (text.lower().startswith(":skip") and not text.startswith("::")):
            continue
        for offset in range(1, 4):
            if index + offset >= len(lines):
                break
            following = _strip_comment(lines[index + offset]).lower()
            if "pip install" in following:
                issues.append(
                    BatchIssue(
                        index + offset + 1,
                        f"Lệnh cài nằm ngay sau nhãn '{text}' — sẽ chạy VÔ ĐIỀU KIỆN "
                        "kể cả khi tính năng bị tắt.",
                    )
                )
                break
    return issues


def find_statement_swallowed_by_comment(lines: list[str]) -> list[BatchIssue]:
    """Tìm lệnh điều kiện bị một ``REM`` nuốt mất phần thân.

    [v3.23.343] Bổ sung sau một lỗi THẬT: bản vá v3.23.341 chèn khối chú thích vào GIỮA
    dòng ``if exist build rmdir /s /q build``, biến nó thành::

        if exist build REM --- KIEM BAN DONG GOI ...

    Batch vẫn chạy (cú pháp hợp lệ!) nhưng lệnh ``rmdir`` biến mất — nên phép kiểm cũ
    (ngoặc cân bằng, nhãn trùng, goto thiếu nhãn) không thấy gì bất thường.

    Mẫu nguy hiểm: một lệnh điều kiện (``if``/``for``) mà phần thân chỉ là ``REM``.
    """
    issues: list[BatchIssue] = []
    pattern = re.compile(
        r"^\s*(if|for)\b.*?\s(rem|::)\s", re.IGNORECASE
    )
    for number, raw in enumerate(lines, 1):
        text = raw.strip()
        if text.lower().startswith(("rem", "::")):
            continue  # cả dòng là chú thích -> bình thường
        if pattern.match(text):
            issues.append(
                BatchIssue(
                    number,
                    "Lệnh điều kiện có phần thân là REM — nhiều khả năng một khối chú "
                    "thích đã bị chèn vào GIỮA dòng, làm mất lệnh thật.",
                )
            )
    return issues


def check_batch_file(path: Path) -> list[BatchIssue]:
    """Chạy mọi phép kiểm trên một tệp batch.

    Args:
        path: Tệp ``.bat``.

    Returns:
        Danh sách vấn đề, rỗng nghĩa là sạch.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    issues: list[BatchIssue] = []
    issues.extend(find_duplicate_labels(lines))
    issues.extend(find_missing_labels(lines))
    issues.extend(find_unbalanced_parentheses(lines))
    issues.extend(find_unreachable_after_label(lines))
    issues.extend(find_statement_swallowed_by_comment(lines))
    return sorted(issues, key=lambda issue: issue.line)


def main() -> int:
    """Kiểm mọi tệp .bat ở gốc dự án."""
    root = Path(__file__).resolve().parent.parent
    targets = sorted(root.glob("*.bat"))
    if not targets:
        print("Không tìm thấy tệp .bat nào.")
        return 0

    total = 0
    for path in targets:
        issues = check_batch_file(path)
        if not issues:
            print(f"  {path.name:28} ✓ sạch")
            continue
        print(f"  {path.name:28} ⛔ {len(issues)} vấn đề")
        for issue in issues:
            print(f"      dòng {issue.line}: {issue.message}")
        total += len(issues)

    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BatchIssue",
    "check_batch_file",
    "find_duplicate_labels",
    "find_missing_labels",
    "find_statement_swallowed_by_comment",
    "find_unbalanced_parentheses",
    "find_unreachable_after_label",
]
