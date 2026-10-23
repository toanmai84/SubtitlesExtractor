"""[v3.23.619] Phụ thuộc khai báo phải khớp với mã thực sự import.

Trước đây: ``pgs_decoder.py`` import Pillow ngay khi nạp nhưng không khai báo ở đâu (chỉ có
nhờ paddleocr kéo theo); ``pyproject.toml`` thiếu 10 gói so với ``requirements.txt``.
"""
from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src" / "subtitles_extractor"

#: Tên module import → tên gói pip.
IMPORT_TO_DIST = {
    "PIL": "pillow",
    "cv2": "opencv-python",
    "mpv": "python-mpv",
    "google": "google-genai",
    "yaml": "pyyaml",
    "shiboken6": "pyside6",
    "paddle": "paddlepaddle",
    "paddlex": "paddleocr",  # paddleocr kéo theo
}


def _dist_name(spec: str) -> str:
    return re.split(r"[<>=!;\[ ]", spec.strip(), maxsplit=1)[0].lower().replace("_", "-")


def _requirements() -> set[str]:
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return {_dist_name(line.split("#")[0]) for line in lines if line.split("#")[0].strip()}


def test_pyproject_matches_requirements() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = {_dist_name(d) for d in project["dependencies"]}
    assert declared == _requirements()


def _unguarded_module_imports(tree: ast.Module) -> set[str]:
    """Import ở cấp module, không nằm trong ``try`` (thiếu gói là sập ngay khi nạp)."""
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def test_every_unguarded_third_party_import_is_declared() -> None:
    declared = _requirements()
    stdlib = set(sys.stdlib_module_names)
    missing: list[str] = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for module in _unguarded_module_imports(tree):
            if module in stdlib or module in {"subtitles_extractor", "__future__"}:
                continue
            dist = IMPORT_TO_DIST.get(module, module.lower().replace("_", "-"))
            if dist not in declared:
                missing.append(f"{path.relative_to(SRC)}: {module} ({dist})")
    assert missing == [], "Gói import ngay khi nạp nhưng chưa khai báo:\n" + "\n".join(missing)
