"""[v3.23.619] Tầng domain thuần Python: không phụ thuộc tầng ngoài hay thư viện nặng.

Trước đây ``domain/entities/ocr_frame_result.py`` import ``is_cjk_char`` từ tầng
application (bên trong hàm, mỗi lần nối box). README quy định
``presentation → application → domain ← infrastructure``.
"""
from __future__ import annotations

import ast
from pathlib import Path

DOMAIN = Path(__file__).resolve().parents[3] / "src" / "subtitles_extractor" / "domain"
FORBIDDEN_PREFIXES = (
    "subtitles_extractor.application",
    "subtitles_extractor.infrastructure",
    "subtitles_extractor.presentation",
    "subtitles_extractor.composition",
    # numpy được phép: các port dùng ndarray làm kiểu dữ liệu khung hình.
    "PySide6", "cv2", "paddle", "paddleocr", "av", "google",
)


def test_domain_imports_nothing_from_outer_layers() -> None:
    offenders: list[str] = []
    for path in DOMAIN.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for name in names:
                if name.startswith(FORBIDDEN_PREFIXES):
                    offenders.append(f"{path.relative_to(DOMAIN)}:{node.lineno} {name}")
    assert offenders == [], "\n".join(offenders)


def test_cjk_detection_shared_with_application() -> None:
    from subtitles_extractor.application.services import cjk_utils
    from subtitles_extractor.domain.value_objects.cjk_char import is_cjk_char

    assert cjk_utils.is_cjk_char is is_cjk_char
    assert is_cjk_char("瑶") and is_cjk_char("가") and not is_cjk_char("a")
