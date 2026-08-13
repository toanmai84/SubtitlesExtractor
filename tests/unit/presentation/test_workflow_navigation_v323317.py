"""Canh giữ hợp đồng điều hướng quy trình — v3.23.317.

Nút “Đi tới” trên thanh tiến độ chuyển trang bằng cách so ``objectName``. Nếu ai đó
đổi tên đối tượng của một trang mà quên sửa :meth:`WorkflowStage.next_page_key`, nút sẽ
**hỏng âm thầm** — không lỗi, không log, chỉ là bấm không có gì xảy ra.

Bộ test này đọc thẳng mã nguồn các trang (không cần Qt/màn hình) để bắt lỗi đó.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from subtitles_extractor.domain.entities.project_record import WorkflowStage

def _locate_pages_dir() -> Path:
    """Tìm thư mục ``presentation/pages`` mà KHÔNG import Qt.

    Suy ra từ vị trí của module domain (đã import được ở trên, không cần Qt) thay vì
    dò đường dẫn tương đối — nhờ vậy chạy đúng dù test được sao chép đi đâu.
    """
    import subtitles_extractor.domain.entities.project_record as anchor

    package_root = Path(anchor.__file__).resolve().parents[2]
    return package_root / "presentation" / "pages"


_PAGES_DIR = _locate_pages_dir()

# Trang tương ứng với từng khoá điều hướng.
_KEY_TO_MODULE = {
    "extractPage": "extract_page.py",
    "editorPage": "editor_page.py",
    "translatePage": "translate_page.py",
    "ttsPage": "tts_page.py",
    "publishPage": "publish_page.py",
}

_OBJECT_NAME_PATTERN = re.compile(r'setObjectName\(\s*"([^"]+)"\s*\)')


def _declared_object_name(module_file: str) -> str | None:
    """Đọc ``setObjectName("…")`` đầu tiên trong tệp trang."""
    source = (_PAGES_DIR / module_file).read_text(encoding="utf-8")
    match = _OBJECT_NAME_PATTERN.search(source)
    return match.group(1) if match else None


@pytest.mark.parametrize(("key", "module_file"), sorted(_KEY_TO_MODULE.items()))
def test_next_page_key_matches_real_object_name(key: str, module_file: str) -> None:
    """Mỗi khoá điều hướng phải khớp ``objectName`` thật của trang đó."""
    assert _declared_object_name(module_file) == key


def test_every_navigable_stage_points_to_an_existing_page() -> None:
    """Mọi khâu chưa hoàn thành phải trỏ tới một trang CÓ THẬT."""
    for stage in WorkflowStage:
        key = stage.next_page_key
        if key is None:
            assert stage.is_complete
            continue
        assert key in _KEY_TO_MODULE, f"{stage.name} trỏ tới trang không tồn tại: {key}"


def test_page_files_exist() -> None:
    for module_file in _KEY_TO_MODULE.values():
        assert (_PAGES_DIR / module_file).is_file()


def test_navigation_keys_are_unique() -> None:
    """Hai khâu khác nhau không được trỏ về cùng một trang."""
    keys = [s.next_page_key for s in WorkflowStage if s.next_page_key is not None]
    assert len(keys) == len(set(keys))
