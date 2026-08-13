"""[v3.23.74] Guard chống "việt hoá thiếu dấu" trong chuỗi văn bản (log/UI).

Ứng dụng yêu cầu việt hoá HOÀN TOÀN, tiếng Việt CÓ DẤU. Đã từng có khối "BÁO CÁO
AUTO-DUBBING" bị viết không dấu ("BAO CAO", "Toc do dinh", "Chong tieng"…). Guard này quét
mọi chuỗi literal trong ``src/`` và báo đỏ nếu xuất hiện các CỤM tiếng Việt không dấu đặc
trưng (chọn dạng cụm/bigram để tránh nhầm với tiếng Anh hay tên biến).

Phân tích AST nguồn → chạy headless, không cần dựng Qt.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[2] / "src" / "subtitles_extractor"

# Các CỤM tiếng Việt không dấu KHÔNG ĐƯỢC xuất hiện trong chuỗi văn bản.
# Dùng cụm nhiều từ để loại trừ trùng với tiếng Anh / định danh kỹ thuật.
_BANNED_UNACCENTED_PHRASES: tuple[str, ...] = (
    "bao cao",
    "toc do",
    "chong tieng",
    "vuot rao",
    "ep xung",
    "cat muot",
    "cau xu ly",
    "xu ly",
    "tieng noi",
    "phu de",
    "nhan vat",
    "thuc thi",
    "du kien",
    "lan dai nhat",
    "tong cong",
)


def _iter_string_literals(tree: ast.AST):
    """Sinh (giá trị chuỗi, dòng) cho mọi literal chuỗi trong cây AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value, node.lineno


def _python_files() -> list[Path]:
    return sorted(p for p in _SRC_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def test_no_unaccented_vietnamese_in_strings() -> None:
    violations: list[str] = []
    for path in _python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:  # pragma: no cover - nguồn luôn hợp lệ trong CI
            continue
        for text, lineno in _iter_string_literals(tree):
            lowered = text.lower()
            for phrase in _BANNED_UNACCENTED_PHRASES:
                if phrase in lowered:
                    rel = path.relative_to(_SRC_DIR.parent.parent)
                    snippet = text.strip().replace("\n", " ")[:60]
                    violations.append(f"{rel}:{lineno} chứa '{phrase}' → \"{snippet}\"")

    assert not violations, (
        "Phát hiện tiếng Việt KHÔNG DẤU trong chuỗi văn bản "
        "(vi phạm việt hoá hoàn toàn):\n" + "\n".join(sorted(violations))
    )


def test_banned_phrases_are_lowercase_and_unique() -> None:
    """Bảo trì: danh sách cụm cấm phải viết thường và không trùng lặp."""
    assert all(p == p.lower() for p in _BANNED_UNACCENTED_PHRASES)
    assert len(_BANNED_UNACCENTED_PHRASES) == len(set(_BANNED_UNACCENTED_PHRASES))


def test_detector_catches_known_pattern() -> None:
    """Tự kiểm: detector thực sự bắt được mẫu không dấu đã từng lỗi."""
    sample = 'logger.info("===== BAO CAO AUTO-DUBBING =====")'
    tree = ast.parse(sample)
    hits = [
        phrase
        for text, _ in _iter_string_literals(tree)
        for phrase in _BANNED_UNACCENTED_PHRASES
        if phrase in text.lower()
    ]
    assert "bao cao" in hits
