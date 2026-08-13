"""[v3.23.392] Tiền-kiểm tra môi trường build TRƯỚC bước PyInstaller (~20 phút).

VÌ SAO cần
==========
Bản one-file LOẠI lõi paddle khỏi bundle (tải lúc chạy). Điều đó làm lộ vài cấu hình môi
trường build phải đúng, nếu sai thì chỉ phát hiện SAU khi build xong + chạy thử — tốn cả một
chu kỳ build dài. Ví dụ đã gặp thật:

* ``setuptools>=80`` XÓA ``setuptools.command.easy_install`` mà ``paddlepaddle-gpu`` cần →
  ``import paddle`` lỗi lúc chạy, dù build "thành công".
* Thiếu ``paddleocr``/``paddlex`` trong môi trường build → ``collect_all`` trong spec không gom
  được → pipeline OCR hỏng lúc chạy.

Công cụ này kiểm THẲNG môi trường build và báo NGAY (fail-fast) trước khi tốn thời gian đóng
gói. Chạy tự động trong ``build_windows.bat``; cũng chạy tay được:

    python tools/preflight_build_check.py
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Kết quả một mục kiểm tra.

    Attributes:
        name: Tên ngắn của mục kiểm.
        passed: Đạt hay không.
        critical: ``True`` → không đạt thì DỪNG build; ``False`` → chỉ cảnh báo.
        detail: Mô tả để hiển thị (hướng dẫn khắc phục nếu lỗi).
    """

    name: str
    passed: bool
    critical: bool
    detail: str


def _check_setuptools_easy_install() -> CheckResult:
    """setuptools phải CÒN ``command.easy_install`` (bị xóa ở setuptools>=80).

    ``paddlepaddle-gpu 3.3.1`` import module này cho JIT/cpp_extension.
    """
    found = importlib.util.find_spec("setuptools.command.easy_install") is not None
    try:
        import setuptools  # noqa: PLC0415 — chỉ để lấy version cho thông điệp

        version = getattr(setuptools, "__version__", "?")
    except ImportError:
        version = "KHÔNG CÀI"
    if found:
        detail = f"setuptools {version} còn easy_install — OK."
    else:
        detail = (
            f"setuptools {version} THIẾU command.easy_install (bị xóa ở >=80). "
            'Chạy: pip install "setuptools<80"  (paddle cần module này).'
        )
    return CheckResult("setuptools.command.easy_install", found, True, detail)


def _check_importable(module: str, *, critical: bool, hint: str) -> CheckResult:
    """Kiểm một module có import được trong môi trường build không."""
    found = importlib.util.find_spec(module) is not None
    detail = f"{module}: OK." if found else f"{module}: THIẾU — {hint}"
    return CheckResult(module, found, critical, detail)


def run_all_checks() -> list[CheckResult]:
    """Chạy toàn bộ mục kiểm, trả về danh sách kết quả (thuần, dễ test)."""
    return [
        _check_setuptools_easy_install(),
        _check_importable(
            "PyInstaller", critical=True, hint="pip install 'pyinstaller>=6.0'"
        ),
        _check_importable(
            "paddle",
            critical=False,
            hint="cài paddlepaddle-gpu vào môi trường build để collect_all gom được.",
        ),
        _check_importable(
            "paddleocr",
            critical=False,
            hint="pip install paddleocr (pipeline OCR cần khi chạy).",
        ),
        _check_importable(
            "paddlex",
            critical=False,
            hint="paddleocr kéo paddlex — kiểm lại cài đặt.",
        ),
        _check_importable(
            "PySide6", critical=True, hint="pip install PySide6"
        ),
    ]


def main() -> int:
    """In báo cáo + trả mã thoát (0 nếu không có mục CRITICAL nào hỏng)."""
    results = run_all_checks()
    print("── Tiền-kiểm tra môi trường build ──")
    critical_failed = False
    for result in results:
        if result.passed:
            mark = "✓"
        elif result.critical:
            mark = "✗ [DỪNG]"
            critical_failed = True
        else:
            mark = "⚠ [cảnh báo]"
        print(f"  {mark} {result.detail}")

    if critical_failed:
        print(
            "\n[LỖI] Có mục BẮT BUỘC chưa đạt — sửa trước khi build để khỏi tốn "
            "công đóng gói rồi mới phát hiện."
        )
        return 1
    print("\n[OK] Môi trường build đạt yêu cầu (các cảnh báo nếu có: xem ở trên).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
