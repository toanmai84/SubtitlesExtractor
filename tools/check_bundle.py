"""Kiểm bản đóng gói SAU khi build — bắt tệp bị thiếu trước khi người dùng gặp lỗi.

VÌ SAO cần
==========
v3.23.340 phát hiện ``whisperx_subprocess.py`` không có trong bundle. Hậu quả: Python
báo "can't open file" và thoát **mã 2** — trùng mã của lỗi sai đối số nên rất khó chẩn
đoán. Người dùng phải gửi log qua nhiều lượt mới tìm ra.

Điều đáng lo hơn: log build **không hề nói gì**. ``datas.append(...)`` trong tệp spec
không in ra dòng nào, nên không có cách nào biết tệp đã vào bundle hay chưa cho tới khi
tính năng đó hỏng lúc chạy.

Công cụ này kiểm thẳng thư mục ``dist/`` sau khi build và báo ngay nếu thiếu.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

#: Tệp/thư mục BẮT BUỘC phải có trong bundle, kèm lý do để thông điệp lỗi hữu ích.
REQUIRED_ENTRIES: Final[tuple[tuple[str, str], ...]] = (
    (
        "_internal/subtitles_extractor/infrastructure/stt/whisperx_subprocess.py",
        "worker phiên âm WhisperX — thiếu thì báo 'thoát mã 2' khó hiểu",
    ),
    (
        "_internal/subtitles_extractor/infrastructure/tts/edge_tts_subprocess.py",
        "worker Edge TTS — thiếu thì giọng Microsoft không dùng được",
    ),
    (
        "_internal/subtitles_extractor/infrastructure/tts/vieneu_gpu_subprocess.py",
        "worker VieNeu GPU — thiếu thì TTS âm thầm lùi về CPU dù đã cài đủ gói",
    ),
    (
        "_internal/subtitles_extractor/data",
        "chuỗi giao diện tiếng Việt (strings_vi.json)",
    ),
    (
        "_internal/vendor/ffmpeg/ffmpeg.exe",
        "ffmpeg — cần cho xuất bản video và nén ngữ cảnh Gemini",
    ),
    (
        "_internal/vendor/mpv/libmpv-2.dll",
        "libmpv — trình phát video chính",
    ),
    # [v3.23.343] Gộp từ `check_bundle_contents.py` (bản trùng, đã xoá).
    (
        "_internal/vendor/ffmpeg/ffprobe.exe",
        "ffprobe — cần để đọc metadata video và kiểm tệp đã xuất",
    ),
)

#: [v3.23.395] Thành phần CHỈ có khi build với ``SUBEXT_PREFETCH_MODELS=1`` (nhúng sẵn model).
#: MẶC ĐỊNH bản nhỏ KHÔNG nhúng — model tải-lúc-chạy vào ``models/`` cạnh exe. Vì vậy THIẾU
#: các mục này là BÌNH THƯỜNG (chỉ thông tin), KHÔNG phải lỗi build.
OPTIONAL_ENTRIES: Final[tuple[tuple[str, str], ...]] = (
    (
        "_internal/models/paddle/official_models",
        "mô hình OCR nhúng sẵn (chỉ khi SUBEXT_PREFETCH_MODELS=1) — thiếu thì tải lúc chạy",
    ),
    (
        "_internal/models/huggingface/hub",
        "mô hình VieNeu-TTS nhúng sẵn (chỉ khi SUBEXT_PREFETCH_MODELS=1) — thiếu thì tải lúc chạy",
    ),
)

#: Thư mục con của bundle mà LẼ RA không nên có (dấu hiệu đóng gói sai).
UNEXPECTED_ENTRIES: Final[tuple[tuple[str, str], ...]] = (
    (
        "_internal/torch",
        "torch KHÔNG nên nằm trong bundle — nó thuộc môi trường riêng whisperx_env",
    ),
    (
        "_internal/whisperx",
        "whisperx KHÔNG nên nằm trong bundle — thuộc môi trường riêng",
    ),
)


@dataclass(frozen=True, slots=True)
class BundleIssue:
    """Một vấn đề phát hiện trong bundle.

    Attributes:
        path: Đường dẫn tương đối bị thiếu hoặc không nên có.
        reason: Giải thích hệ quả.
        missing: ``True`` nếu thiếu; ``False`` nếu có mà không nên có.
        optional: ``True`` nếu chỉ là thành phần TÙY CHỌN (thiếu không phải lỗi).
    """

    path: str
    reason: str
    missing: bool
    optional: bool = False


def check_bundle(bundle_root: Path) -> list[BundleIssue]:
    """Kiểm bundle có đủ tệp bắt buộc và không chứa thứ không nên có.

    Args:
        bundle_root: Thư mục ``dist/SubtitlesExtractor``.

    Returns:
        Danh sách vấn đề; rỗng nghĩa là bundle hợp lệ.
    """
    issues: list[BundleIssue] = []

    for relative, reason in REQUIRED_ENTRIES:
        if not (bundle_root / relative).exists():
            issues.append(BundleIssue(relative, reason, missing=True))

    for relative, reason in OPTIONAL_ENTRIES:
        if not (bundle_root / relative).exists():
            issues.append(
                BundleIssue(relative, reason, missing=True, optional=True)
            )

    for relative, reason in UNEXPECTED_ENTRIES:
        if (bundle_root / relative).exists():
            issues.append(BundleIssue(relative, reason, missing=False))

    return issues


def find_bundle_root(project_root: Path) -> Path | None:
    """Tìm thư mục bundle trong ``dist/``.

    Args:
        project_root: Thư mục gốc dự án.

    Returns:
        Đường dẫn bundle, hoặc ``None`` nếu chưa build.
    """
    candidate = project_root / "dist" / "SubtitlesExtractor"
    return candidate if candidate.is_dir() else None


def main() -> int:
    """Kiểm bundle và in báo cáo."""
    project_root = Path(__file__).resolve().parent.parent
    bundle_root = find_bundle_root(project_root)

    print("=" * 68)
    print("  KIỂM BẢN ĐÓNG GÓI")
    print("=" * 68)

    if bundle_root is None:
        print("  ⓘ Chưa có thư mục dist/SubtitlesExtractor — bỏ qua.")
        return 0

    print(f"  Bundle: {bundle_root}")
    issues = check_bundle(bundle_root)

    if not issues:
        print(f"\n  ✓ Đủ {len(REQUIRED_ENTRIES)} thành phần bắt buộc, "
              "không có thứ lạ.")
        print("=" * 68)
        return 0

    missing = [i for i in issues if i.missing and not i.optional]
    optional_missing = [i for i in issues if i.missing and i.optional]
    unexpected = [issue for issue in issues if not issue.missing]

    if missing:
        print(f"\n  ⛔ THIẾU {len(missing)} thành phần:")
        for issue in missing:
            print(f"      • {issue.path}")
            print(f"        → {issue.reason}")
    if unexpected:
        print(f"\n  ⚠️ CÓ {len(unexpected)} thứ không nên có:")
        for issue in unexpected:
            print(f"      • {issue.path}")
            print(f"        → {issue.reason}")
    if optional_missing:
        print(f"\n  ⓘ {len(optional_missing)} thành phần TÙY CHỌN không nhúng "
              "(bình thường với bản nhỏ — tải lúc chạy):")
        for issue in optional_missing:
            print(f"      • {issue.path}")

    # Chỉ FAIL khi thiếu thành phần BẮT BUỘC hoặc có thứ lạ; thiếu tùy chọn thì không.
    if not missing and not unexpected:
        print("\n" + "=" * 68)
        print("  ✓ Bundle hợp lệ (chỉ thiếu thành phần tùy chọn — tải lúc chạy).")
        print("=" * 68)
        return 0

    print("\n" + "=" * 68)
    print("  Sửa tệp .spec rồi build lại.")
    print("=" * 68)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REQUIRED_ENTRIES",
    "OPTIONAL_ENTRIES",
    "UNEXPECTED_ENTRIES",
    "BundleIssue",
    "check_bundle",
    "find_bundle_root",
]
