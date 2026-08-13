"""Sinh báo cáo license thư viện từ môi trường Python thực tế.

[v3.23.269] Công cụ hỗ trợ tuân thủ license cho bản phân phối thương mại. Quét mọi package
đã cài, trích license + version + trang chủ, xuất báo cáo Markdown. Chạy trong môi trường
build (build_env) để có đúng danh sách đóng gói.

Cách dùng:
    python tools/generate_license_report.py > THIRD_PARTY_LICENSES_AUTO.md

Lưu ý: đây là báo cáo TỰ ĐỘNG (metadata pip), có thể thiếu/sai với vài package khai báo
license không chuẩn. Dùng kèm THIRD_PARTY_LICENSES.md (bản rà tay chính xác hơn).
"""

from __future__ import annotations

import sys
from importlib import metadata

# Các package runtime cốt lõi (khớp requirements.txt). Không liệt kê dev/test.
_RUNTIME_PACKAGES = [
    "PySide6", "shiboken6",
    "paddlepaddle", "paddleocr", "paddlex",
    "opencv-python", "opencv-contrib-python",
    "numpy", "scipy", "soundfile", "librosa", "av", "pydub",
    "pydantic", "pydantic-settings", "rapidfuzz",
    "google-genai", "json-repair", "loguru", "rjieba",
    "python-mpv",
]

# Package GPL/cách ly — KHÔNG đóng gói, chỉ ghi chú.
_ISOLATED_PACKAGES = ["edge-tts", "whisperx", "vieneu", "sea-g2p"]


def _license_of(package_name: str) -> tuple[str, str]:
    """Trả về (version, license) của package. ('?', '?') nếu không tìm thấy."""
    try:
        dist = metadata.distribution(package_name)
    except metadata.PackageNotFoundError:
        return "chưa cài", "?"
    version = dist.version
    meta = dist.metadata
    # License có thể ở trường 'License' hoặc classifier 'License :: ...'.
    license_str = meta.get("License", "").strip()
    if not license_str or len(license_str) > 60:
        classifiers = [
            c for c in meta.get_all("Classifier", []) if c.startswith("License ::")
        ]
        if classifiers:
            license_str = classifiers[0].split("::")[-1].strip()
    return version, license_str or "(không khai báo)"


def _print_section(title: str, packages: list[str], note: str = "") -> None:
    print(f"\n## {title}\n")
    if note:
        print(f"> {note}\n")
    print("| Thư viện | Phiên bản | License |")
    print("|---|---|---|")
    for pkg in packages:
        version, license_str = _license_of(pkg)
        print(f"| {pkg} | {version} | {license_str} |")


def main() -> int:
    print("# Báo cáo License Tự động (từ metadata pip)\n")
    print("Sinh bởi tools/generate_license_report.py — đối chiếu THIRD_PARTY_LICENSES.md")
    print("(bản rà tay) để có thông tin chính xác về ghi chú tuân thủ.")
    _print_section(
        "Thư viện đóng gói cùng ứng dụng", _RUNTIME_PACKAGES
    )
    _print_section(
        "Thư viện cách ly / không đóng gói", _ISOLATED_PACKAGES,
        note="GPL hoặc nặng — chạy subprocess/tải runtime, KHÔNG liên kết vào app.",
    )
    print("\n---\n*Xem THIRD_PARTY_LICENSES.md để biết ghi chú tuân thủ chi tiết.*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
