"""Shim thay ``chardet`` (LGPL) bằng ``charset-normalizer`` (MIT) — an toàn thương mại.

[v3.23.271] ``chardet`` (bản ổn định ≤6.x) dùng **LGPL 2.1** — copyleft. Khi đóng gói
PyInstaller, thư viện Python thuần bị nhúng TĨNH vào bundle → LGPL yêu cầu cho
phép thay thế, khó thoả mãn khi đóng gói. ``chardet`` bị ``paddlex`` kéo vào (app
KHÔNG dùng trực tiếp).

**Giải pháp:** cung cấp module ``chardet`` giả lập API tối thiểu mà paddlex dùng
(``chardet.detect``), nội bộ gọi ``charset-normalizer`` (MIT). Đăng ký vào ``sys.modules``
TRƯỚC khi paddlex import → paddlex nhận bản MIT, không cần chardet LGPL thật.

Xem docs/LICENSE_ANALYSIS.md.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any


def _detect(byte_str: bytes | bytearray) -> dict[str, Any]:
    """Phát hiện encoding — API tương thích ``chardet.detect``.

    Args:
        byte_str: Dữ liệu byte cần đoán encoding.

    Returns:
        Dict ``{"encoding": str|None, "confidence": float, "language": str}`` như chardet.
    """
    from charset_normalizer import from_bytes

    if not byte_str:
        return {"encoding": None, "confidence": 0.0, "language": ""}

    match = from_bytes(bytes(byte_str)).best()
    if match is None:
        return {"encoding": None, "confidence": 0.0, "language": ""}
    # charset-normalizer dùng '_' (utf_8); chardet dùng '-' (utf-8). Chuẩn hoá lại.
    encoding = match.encoding.replace("_", "-") if match.encoding else None
    languages = match.languages or []
    return {
        "encoding": encoding,
        "confidence": 1.0,  # charset-normalizer không cho điểm; báo cao để không bị lọc.
        "language": languages[0] if languages else "",
    }


def _detect_all(
    byte_str: bytes | bytearray, *_args: Any, **_kwargs: Any
) -> list[dict[str, Any]]:
    """API tương thích ``chardet.detect_all`` — trả danh sách (ở đây gói 1 kết quả)."""
    result = _detect(byte_str)
    return [result] if result["encoding"] else []


def install_chardet_shim() -> None:
    """Đăng ký module ``chardet`` giả (dựa trên charset-normalizer) vào ``sys.modules``.

    Gọi TRƯỚC khi import paddlex/paddleocr. Nếu ``chardet`` thật đã import rồi thì
    KHÔNG ghi đè (tránh xung đột); chỉ cài shim khi chardet chưa nạp — bảo đảm bản
    MIT được dùng trong bản đóng gói (nơi chardet LGPL đã bị loại khỏi bundle).
    """
    if "chardet" in sys.modules:
        return

    try:
        import charset_normalizer  # noqa: F401
    except ImportError:
        # Không có charset-normalizer → để paddlex tự xử lý (dùng chardet thật nếu có).
        return

    shim = ModuleType("chardet")
    shim.detect = _detect  # type: ignore[attr-defined]
    shim.detect_all = _detect_all  # type: ignore[attr-defined]
    # [v3.23.272] Báo version dạng SỐ hợp lệ (không phải chuỗi tự do) để các thư viện như
    # ``requests`` parse ``chardet.__version__.split(".")`` không cảnh báo. Dùng "5.9.9" —
    # nằm trong khoảng requests chấp nhận (>=3.0.2, <8.0.0) — bản tương thích.
    shim.__version__ = "5.9.9"  # type: ignore[attr-defined]
    # Đánh dấu là shim (để chẩn đoán) qua thuộc tính riêng, không lẫn vào __version__.
    shim.__subext_shim__ = "charset-normalizer"  # type: ignore[attr-defined]
    sys.modules["chardet"] = shim


__all__ = ["install_chardet_shim"]
