"""[v3.23.397] Tìm trình thông dịch Python để CHẠY PIP cho các bước tải-lúc-chạy.

Bối cảnh
========
Bản đóng gói (.exe) không có Python bên trong dùng được cho pip (``sys.executable`` là chính
file .exe). Các bước TẢI-LÚC-CHẠY (paddlepaddle-gpu, CUDA runtime) chạy ``pip install --target``
nên cần một Python thật.

Trước đây phải dựa vào Python HỆ THỐNG của người dùng — cản trở mục tiêu "chia sẻ .exe cho bất
kỳ ai" (người nhận có thể KHÔNG cài Python, hoặc cài phiên bản khác → wheel paddle sai ABI).

Giải pháp: NHÚNG sẵn **Python embeddable** (bản chính thức, nhẹ ~15MB, ĐÚNG phiên bản Python đã
đóng gói app) vào ``vendor/python_embed/`` kèm pip. Runtime ưu tiên dùng nó → .exe tự lập,
không cần Python hệ thống, và luôn tải đúng wheel cho phiên bản Python bundled.

Lưu ý phạm vi: chỉ dùng cho ``pip install --target`` (paddle/CUDA — chỉ đặt file, không cần
tạo venv). WhisperX vẫn cần Python HỆ THỐNG vì nó tạo virtualenv riêng (Python embeddable không
kèm ``venv``).
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Thư mục con (trong ``vendor/``) chứa Python embeddable nhúng.
EMBEDDED_PYTHON_SUBDIR: str = "python_embed"

#: Tên file thực thi Python trong bản embeddable.
_EMBEDDED_PYTHON_EXE: str = "python.exe"


def find_bundled_python() -> str | None:
    """Trả về đường dẫn ``python.exe`` embeddable ĐÃ NHÚNG, hoặc ``None`` nếu không có.

    Chỉ có ý nghĩa khi chạy bản ĐÓNG GÓI (``sys.frozen``). Python embeddable được nhúng vào
    ``vendor/python_embed/`` và giải nén ra ``_MEIPASS`` khi chạy (one-file) hoặc nằm trong
    ``_internal`` (onedir).

    Returns:
        Đường dẫn tuyệt đối tới ``python.exe`` nhúng nếu tồn tại; ngược lại ``None``.
    """
    if not getattr(sys, "frozen", False):
        return None
    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return None
    candidate = (
        Path(meipass) / "vendor" / EMBEDDED_PYTHON_SUBDIR / _EMBEDDED_PYTHON_EXE
    )
    return str(candidate) if candidate.is_file() else None


def resolve_installer_python(system_python: str | None) -> str | None:
    """Chọn Python để chạy ``pip install --target`` (paddle/CUDA).

    Ưu tiên Python embeddable NHÚNG (tự lập, khớp phiên bản bundled); nếu không có thì lùi về
    Python HỆ THỐNG mà lời gọi truyền vào.

    Args:
        system_python: Python hệ thống (kết quả ``find_system_python()``), có thể ``None``.

    Returns:
        Đường dẫn Python nên dùng, hoặc ``None`` nếu không có lựa chọn nào.
    """
    return find_bundled_python() or system_python


__all__ = [
    "EMBEDDED_PYTHON_SUBDIR",
    "find_bundled_python",
    "resolve_installer_python",
]
