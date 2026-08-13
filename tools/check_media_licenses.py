"""Kiểm license libmpv + FFmpeg (qua PyAV) — phát hiện thành phần GPL.

[v3.23.270] Công cụ xác minh binary media dùng bản LGPL (an toàn thương mại), KHÔNG phải
GPL. Chạy trên MÁY BUILD/ĐÍCH để xác nhận đúng binary được phân phối, vì license phụ thuộc
cách build từng bản.

Kiểm 2 thành phần:
1. **FFmpeg (qua PyAV):** đọc ``av._core.library_meta`` — trường ``license`` và
   ``configuration``. GPL nếu có ``--enable-gpl``/``--enable-nonfree``.
2. **libmpv:** chạy ``mpv --version`` (nếu có CLI) hoặc kiểm chuỗi trong DLL. Báo cáo cờ.

Cách dùng:
    python tools/check_media_licenses.py

Mã thoát: 0 nếu TẤT CẢ đều LGPL (an toàn); 1 nếu phát hiện GPL; 2 nếu không kiểm được.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
import sys


def _check_ffmpeg_via_pyav() -> tuple[str, bool]:
    """Kiểm license FFmpeg qua PyAV. Trả (mô tả, is_safe).

    is_safe = True nếu LGPL (không --enable-gpl/nonfree).
    """
    try:
        import av
    except ImportError:
        return "PyAV chưa cài — không kiểm được FFmpeg.", False

    try:
        meta = av._core.library_meta
    except AttributeError:
        return f"PyAV {av.__version__}: không đọc được library_meta.", False

    codec = meta.get("libavcodec", {})
    license_str = codec.get("license", "(không rõ)")
    config = codec.get("configuration", "")

    has_gpl = "--enable-gpl" in config
    has_nonfree = "--enable-nonfree" in config
    # Lưu ý: "LGPL" chứa chuỗi "GPL" -> KHÔNG dùng substring "GPL". Chỉ coi là GPL khi
    # license nói "GPL" mà KHÔNG phải "LGPL", hoặc có cờ --enable-gpl/nonfree.
    license_upper = license_str.upper()
    is_gpl_license = "GPL" in license_upper and "LGPL" not in license_upper
    is_safe = not has_gpl and not has_nonfree and not is_gpl_license

    lines = [
        f"FFmpeg (PyAV {av.__version__}):",
        f"  License khai báo : {license_str}",
        f"  --enable-gpl     : {'CÓ ⚠️' if has_gpl else 'KHÔNG ✓'}",
        f"  --enable-nonfree : {'CÓ ⚠️' if has_nonfree else 'KHÔNG ✓'}",
    ]
    # Ghi chú thành phần GPL tiềm ẩn (chỉ cảnh báo nếu ĐỒNG THỜI có --enable-gpl).
    gpl_components = ["libx264", "libx265", "libxvid", "libvidstab"]
    present = [c for c in gpl_components if f"--enable-{c}" in config]
    if present and has_gpl:
        lines.append(f"  ⚠️ Thành phần GPL BẬT: {', '.join(present)}")
    elif present:
        lines.append(
            f"  Ghi chú: có tham chiếu {', '.join(present)} nhưng KHÔNG --enable-gpl "
            "-> không biên dịch vào (an toàn)."
        )
    return "\n".join(lines), is_safe


# [v3.23.308] Dấu vết các thành phần GPL trong binary media.
# QUAN TRỌNG — các dấu vết dưới đây đã được KIỂM CHỨNG BẰNG SỐ LIỆU trên 2 file thật:
#   * ffmpeg LGPL (vendor/ffmpeg/ffmpeg.exe) -> 0 khớp  (không báo nhầm)
#   * libmpv GPL  (vendor/mpv/libmpv-2.dll)  -> >0 khớp (phát hiện đúng)
# Các dấu vết "ngây thơ" đã bị LOẠI vì gây báo nhầm:
#   - b"rubberband" / b"vidstab" / b"frei0r": khớp cả trong cờ "--disable-librubberband"…
#   - b"x264 - core": ffmpeg DECODER cũng chứa (dùng để đọc SEI nhận diện x264) dù
#     không hề build kèm encoder x264.
# Nguyên tắc: chỉ dùng chuỗi CHỈ tồn tại khi ENCODER GPL được biên dịch vào
# (tên dài AVCodec của encoder, hoặc chuỗi bản quyền nội bộ của chính thư viện).
_GPL_COMPONENT_MARKERS: dict[str, tuple[bytes, ...]] = {
    "x265 (GPL v2+)": (b"libx265 H.265 / HEVC", b"x265.org", b"Multicoreware"),
    "x264 (GPL v2+)": (b"libx264 H.264 / AVC",),
    "rubberband (GPL)": (b"RubberBandStretcher",),
}


def _scan_binary_for_gpl_components(binary: Path) -> dict[str, int]:
    """Quét binary tìm dấu vết thành phần GPL.

    Args:
        binary: File cần quét (.dll/.exe).

    Returns:
        Dict ``{tên thành phần: số lần khớp}``, chỉ chứa mục có khớp.
    """
    try:
        data = binary.read_bytes()
    except OSError:
        return {}
    hits: dict[str, int] = {}
    for name, markers in _GPL_COMPONENT_MARKERS.items():
        count = sum(data.count(marker) for marker in markers)
        if count:
            hits[name] = count
    return hits


def _check_libmpv() -> tuple[str, bool]:
    """Kiểm license ``vendor/mpv/libmpv-2.dll`` bằng QUÉT BINARY.

    [v3.23.308] THAY ĐỔI QUAN TRỌNG: bản trước chỉ chạy ``mpv --version`` và khi không
    có CLI thì **mặc định coi là an toàn** kèm ghi chú "bản Subtitle Edit mirror thường
    là LGPL". Đó là GIẢ ĐỊNH CHƯA KIỂM CHỨNG và đã được chứng minh SAI: bản libmpv-2.dll
    phổ biến (shinchiro/mpv-winbuild-cmake) có nhúng x265/x264/rubberband — đều GPL.

    Nay quét trực tiếp dấu vết thành phần GPL trong DLL. Không đoán, không mặc định an toàn.

    Returns:
        Cặp ``(mô tả, is_safe)``.
    """
    mpv_dir = Path(__file__).resolve().parent.parent / "vendor" / "mpv"
    binary = next(
        (mpv_dir / n for n in ("libmpv-2.dll", "mpv-2.dll") if (mpv_dir / n).is_file()),
        None,
    )
    if binary is None:
        return (
            "libmpv: không có vendor/mpv/ — bỏ qua.\n"
            "  ⓘ Không có DLL thì trình phát/xem trước video sẽ không chạy.",
            True,
        )

    hits = _scan_binary_for_gpl_components(binary)
    size_mb = binary.stat().st_size / 1024 / 1024
    lines = [
        "libmpv (binary vendored)",
        f"  Tệp        : {binary}",
        f"  Kích thước : {size_mb:.1f} MB",
        f"  Cách kiểm  : quét dấu vết thành phần GPL trong binary",
    ]
    if not hits:
        lines.append("  Thành phần GPL: không phát hiện ✓")
        return "\n".join(lines), True

    lines.append("  Thành phần GPL PHÁT HIỆN:")
    for name, count in sorted(hits.items(), key=lambda kv: -kv[1]):
        lines.append(f"      - {name}: {count} dấu vết")
    lines.append(
        "  ⓘ GPL — ĐÚNG CHỦ ĐÍCH (dự án là mã nguồn mở).\n"
        "     libmpv là trình phát tốt nhất cho nhu cầu này (hwdec, vo=gpu-next).\n"
        "     Nghĩa vụ khi phân phối: kèm mã nguồn + thông báo bản quyền."
    )
    return "\n".join(lines), True


def _read_embedded_ffmpeg_config(binary: Path) -> str | None:
    """[v3.23.307] Đọc chuỗi ``configuration:`` NHÚNG SẴN trong binary ffmpeg.

    FFmpeg nhúng nguyên văn dòng cấu hình build vào file thực thi. Đọc trực tiếp
    ưu việt hơn chạy ``ffmpeg -version`` vì: không cần thực thi file lạ (an toàn hơn),
    kiểm được binary Windows từ Linux/CI, không phụ thuộc máy chạy được exe hay không.

    Args:
        binary: Đường dẫn file ffmpeg/ffprobe.

    Returns:
        Chuỗi cấu hình (bắt đầu bằng ``--prefix=``) nếu tìm thấy; ``None`` nếu không.
    """
    try:
        data = binary.read_bytes()
    except OSError:
        return None
    match = re.search(rb"--prefix=[ -~]{100,}", data)
    if match is None:
        return None
    return match.group().decode("ascii", errors="replace")


def _analyse_ffmpeg_config(config: str) -> tuple[bool, bool, list[str]]:
    """Phân tích chuỗi cấu hình build của ffmpeg.

    Args:
        config: Chuỗi cấu hình (dòng ``configuration:`` hoặc chuỗi nhúng).

    Returns:
        Bộ ba ``(has_gpl, has_nonfree, enabled_gpl_components)``.

    Notes:
        [v3.23.307] SỬA LỖI BÁO NHẦM: bản trước tìm ``"libx264" in output`` nên bản
        LGPL có cờ ``--disable-libx264`` cũng bị liệt kê là "thành phần GPL". Nay chỉ
        tính khi thực sự ``--enable-<tên>``.
    """
    has_gpl = "--enable-gpl" in config
    has_nonfree = "--enable-nonfree" in config
    gpl_components = [
        name
        for name in (
            "libx264", "libx265", "libxvid", "librubberband", "frei0r", "libvidstab",
        )
        if f"--enable-{name}" in config
    ]
    return has_gpl, has_nonfree, gpl_components


def _check_vendored_ffmpeg() -> tuple[str, bool | None]:
    """Kiểm license binary ``vendor/ffmpeg/ffmpeg.exe`` (nếu có).

    KHÁC với :func:`_check_ffmpeg_via_pyav`: hàm kia kiểm libav **nhúng trong PyAV**,
    còn hàm này kiểm **binary CLI đặt trong vendor/**. Hai thứ độc lập — bundle một bản
    ffmpeg.exe GPL sẽ làm hỏng tính license-clean dù PyAV vẫn LGPL.

    Returns:
        Cặp ``(mô tả, is_safe)``. ``is_safe=None`` khi KHÔNG có binary vendored.
    """
    vendor_dir = Path(__file__).resolve().parent.parent / "vendor" / "ffmpeg"
    candidates = [
        vendor_dir / name
        for name in ("ffmpeg.exe", "ffmpeg", "ffprobe.exe", "ffprobe")
    ]
    binary = next((p for p in candidates if p.is_file()), None)
    if binary is None:
        return (
            "FFmpeg (binary vendored)\n"
            "  Không có vendor/ffmpeg/ — LICENSE AN TOÀN ✓\n"
            "  ⓘ Nhưng lưu ý cho bản STANDALONE: máy người dùng cuối thường KHÔNG có\n"
            "    ffmpeg trên PATH, nên các tính năng sau sẽ không chạy được:\n"
            "      - Ngữ cảnh video khi dịch (cắt đoạn gửi Gemini)\n"
            "      - Trích phụ đề nhúng từ video\n"
            "      - Sóng âm (waveform) trong Trình sửa\n"
            "    (Lõi OCR hardsub + trình phát video KHÔNG bị ảnh hưởng — dùng PyAV/libmpv.)",
            None,
        )

    config = _read_embedded_ffmpeg_config(binary)
    method = "đọc chuỗi cấu hình nhúng trong binary"

    if config is None:
        method = "chạy ffmpeg -version"
        try:
            result = subprocess.run(
                [str(binary), "-version"],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=30, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return (
                f"FFmpeg (binary vendored)\n  Tệp: {binary}\n"
                f"  Không đọc được cấu hình và không chạy được: {exc}\n"
                f"  ⚠️ KHÔNG xác nhận được license — kiểm tra thủ công.",
                False,
            )
        output = f"{result.stdout}\n{result.stderr}"
        config = next(
            (ln for ln in output.splitlines() if ln.strip().startswith("configuration:")),
            "",
        )

    if not config:
        return (
            f"FFmpeg (binary vendored)\n  Tệp: {binary}\n"
            f"  ⚠️ Không tìm thấy chuỗi cấu hình — kiểm tra thủ công.",
            False,
        )

    has_gpl, has_nonfree, gpl_components = _analyse_ffmpeg_config(config)
    # [v3.23.313] Dự án là mã nguồn mở (GPL) -> --enable-gpl KHÔNG còn là vấn đề.
    # Chỉ `nonfree` mới thực sự nguy hiểm: bản nonfree KHÔNG được phân phối lại
    # dưới BẤT KỲ hình thức nào, kể cả kèm mã nguồn.
    is_safe = not has_nonfree
    lines = [
        "FFmpeg (binary vendored)",
        f"  Tệp             : {binary}",
        f"  Cách kiểm       : {method}",
        f"  --enable-gpl    : {'CÓ ⚠️' if has_gpl else 'KHÔNG ✓'}",
        f"  --enable-nonfree: {'CÓ ⚠️' if has_nonfree else 'KHÔNG ✓'}",
        f"  Thành phần GPL BẬT: {', '.join(gpl_components) if gpl_components else 'không có ✓'}",
    ]
    if "--enable-version3" in config:
        lines.append("  --enable-version3: CÓ (LGPL v3) ✓")
    if not is_safe:
        # [v3.23.313] Dự án đã chuyển sang MÃ NGUỒN MỞ (GPL) nên GPL là CHỦ ĐÍCH,
        # không còn là lỗi cần sửa. Chỉ nhắc nghĩa vụ kèm mã nguồn.
        lines.append(
            "  🚫 BẢN NONFREE — KHÔNG ĐƯỢC PHÂN PHỐI LẠI dù kèm mã nguồn.\n"
            "     Phải thay bằng bản gpl hoặc lgpl."
        )
    elif has_gpl:
        lines.append(
            "  ⓘ GPL — ĐÚNG CHỦ ĐÍCH (dự án là mã nguồn mở).\n"
            "     Nghĩa vụ khi phân phối: kèm mã nguồn + giữ nguyên thông báo bản quyền\n"
            "     (vendor/ffmpeg/LICENSE-ffmpeg-GPLv3.txt)."
        )
    return "\n".join(lines), is_safe


def main() -> int:
    print("=" * 60)
    print("  KIỂM LICENSE BINARY MEDIA (FFmpeg + libmpv)")
    print("=" * 60)

    ffmpeg_desc, ffmpeg_safe = _check_ffmpeg_via_pyav()
    print("\n" + ffmpeg_desc)

    # [v3.23.305] Binary vendored là nguồn rủi ro license ĐỘC LẬP với PyAV.
    vendored_desc, vendored_safe = _check_vendored_ffmpeg()
    print("\n" + vendored_desc)

    mpv_desc, mpv_safe = _check_libmpv()
    print("\n" + mpv_desc)

    # vendored_safe=None nghĩa là không có binary -> không tính vào kết luận.
    all_safe = ffmpeg_safe and mpv_safe and (vendored_safe is not False)

    print("\n" + "=" * 60)
    if all_safe:
        print("  ✓ KẾT LUẬN: đã ghi nhận license mọi thành phần media.")
        print("     Dự án là MÃ NGUỒN MỞ (GPL) — nhớ kèm mã nguồn khi phân phối.")
        print("=" * 60)
        return 0
    print("  ⓘ Có thành phần chưa xác nhận được license — xem chi tiết ở trên.")
    print("     Dự án là MÃ NGUỒN MỞ (GPL): nhớ kèm mã nguồn khi phân phối.")
    print("=" * 60)
    return 1


if __name__ == "__main__":
    sys.exit(main())
