"""[v3.23.541] Tải binary native (libmpv, ffmpeg) MỚI NHẤT vào thư mục ``vendor/``.

Vì sao cần script này: ``vendor/`` chứa DLL và tệp thực thi nặng hàng trăm MB nên KHÔNG nằm trong
mã nguồn. Bản clone mới hoặc gói mã nguồn đều thiếu nó, khiến trình phát video không chạy và hai
bộ kiểm thử phải bỏ qua. Trước đây chưa có cách lấy tự động — phải tải tay, dễ lấy nhầm bản.

Nguồn (đều là bản dựng chính thức, công khai trên GitHub Releases):
- libmpv: ``shinchiro/mpv-winbuild-cmake`` — tài sản ``mpv-dev-x86_64-<ngày>-git-<hash>.7z``
- ffmpeg: ``BtbN/FFmpeg-Builds`` — tài sản ``ffmpeg-master-latest-win64-gpl-shared.zip``

Giấy phép: mpv và ffmpeg là phần mềm tự do (LGPL/GPL tuỳ bản dựng). Bản ffmpeg ``gpl`` kéo theo
ràng buộc GPL cho sản phẩm phân phối kèm — dùng ``--ffmpeg-variant lgpl`` nếu cần tránh điều đó.
Chạy ``tools/generate_license_report.py`` sau khi tải để cập nhật báo cáo giấy phép.

Cách dùng::

    python tools/fetch_vendor_binaries.py              # tải cả hai, bỏ qua thứ đã có
    python tools/fetch_vendor_binaries.py --force      # tải lại kể cả đã có
    python tools/fetch_vendor_binaries.py --only mpv   # chỉ libmpv
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger("fetch_vendor")

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
VENDOR_DIR: Final[Path] = PROJECT_ROOT / "vendor"

MPV_REPO_URL: Final[str] = "https://github.com/shinchiro/mpv-winbuild-cmake"
MPV_RELEASES_API: Final[str] = (
    "https://api.github.com/repos/shinchiro/mpv-winbuild-cmake/releases?per_page=5"
)
FFMPEG_RELEASE_BASE: Final[str] = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest"
)
REQUEST_TIMEOUT_SEC: Final[float] = 120.0
USER_AGENT: Final[str] = "SubtitlesExtractor-vendor-fetcher"

#: Tệp BẮT BUỘC phải có sau khi giải nén — dùng để xác nhận tải đúng gói.
MPV_REQUIRED: Final[tuple[str, ...]] = ("libmpv-2.dll",)
FFMPEG_REQUIRED: Final[tuple[str, ...]] = ("ffmpeg.exe", "ffprobe.exe")


class FetchError(RuntimeError):
    """Không lấy được binary (mạng lỗi, không tìm thấy tài sản, giải nén hỏng…)."""


def _http_get(url: str) -> bytes:
    """Tải nội dung một URL.

    Args:
        url: Địa chỉ cần tải.

    Returns:
        Nội dung nhị phân.

    Raises:
        FetchError: Khi mạng lỗi hoặc máy chủ trả mã lỗi.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SEC) as response:
            return response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        raise FetchError(f"Không tải được {url}: {exc}") from exc


def resolve_latest_mpv_asset() -> tuple[str, str]:
    """Tìm tài sản libmpv mới nhất trên GitHub Releases.

    Duyệt vài bản phát hành gần nhất thay vì chỉ bản đầu, vì kho này phát hành hằng ngày và thỉnh
    thoảng có bản thiếu tài sản ``mpv-dev``.

    Returns:
        Cặp ``(tên tệp, địa chỉ tải)``.

    Raises:
        FetchError: Khi không bản nào có tài sản phù hợp.
    """
    try:
        payload: Any = json.loads(_http_get(MPV_RELEASES_API))
    except FetchError as exc:
        # [v3.23.541] API GitHub giới hạn theo IP (60 lượt/giờ cho khách). Mạng công ty, VPN hay
        # máy dùng chung IP rất dễ chạm trần và nhận 403. Khi đó chuyển sang đọc trang phát hành —
        # không qua API nên không bị giới hạn.
        logger.info("API GitHub không dùng được (%s) → chuyển sang đọc trang phát hành.", exc)
        return _resolve_mpv_asset_via_html()
    if not isinstance(payload, list):
        raise FetchError(f"GitHub trả dữ liệu lạ: {str(payload)[:200]}")
    for release in payload:
        for asset in release.get("assets", []):
            name = str(asset.get("name", ""))
            # Lấy đúng bản dev (có header + import lib) cho kiến trúc x86_64 thường,
            # KHÔNG lấy biến thể "-v3" vì nó yêu cầu CPU mới (AVX2).
            if name.startswith("mpv-dev-x86_64-") and "-v3-" not in name:
                return name, str(asset["browser_download_url"])
    raise FetchError("Không thấy tài sản mpv-dev-x86_64 trong các bản phát hành gần nhất.")


def _resolve_mpv_asset_via_html() -> tuple[str, str]:
    """Tìm tài sản libmpv bằng cách đọc trang phát hành (không qua API, không bị giới hạn).

    Đi hai bước: nguồn cấp ATOM cho biết các thẻ phát hành mới nhất, rồi trang danh sách tài sản
    của từng thẻ cho biết tên tệp kèm mã git — thứ không đoán trước được.

    Returns:
        Cặp ``(tên tệp, địa chỉ tải)``.

    Raises:
        FetchError: Khi không thẻ nào gần đây có tài sản phù hợp.
    """
    feed = _http_get(f"{MPV_REPO_URL}/releases.atom").decode("utf-8", "replace")
    tags = re.findall(r"/releases/tag/([0-9]{8})", feed)
    for tag in dict.fromkeys(tags):  # giữ thứ tự, bỏ trùng
        page = _http_get(f"{MPV_REPO_URL}/releases/expanded_assets/{tag}")
        links = re.findall(
            r'href="(/[^"]*?/(mpv-dev-x86_64-[0-9]{8}-git-[0-9a-f]+\.7z))"',
            page.decode("utf-8", "replace"),
        )
        for href, name in links:
            if "-v3-" in name:
                continue
            return name, f"https://github.com{href}"
    raise FetchError("Không thấy tài sản mpv-dev-x86_64 ở các bản phát hành gần nhất.")


def _extract_7z(archive: Path, target_dir: Path) -> None:
    """Giải nén tệp .7z, ưu tiên py7zr rồi mới tới lệnh hệ thống.

    Raises:
        FetchError: Khi không có công cụ nào giải nén được.
    """
    try:
        import py7zr  # type: ignore
    except ImportError:
        py7zr = None
    if py7zr is not None:
        try:
            with py7zr.SevenZipFile(archive, mode="r") as handle:
                handle.extractall(path=target_dir)
            return
        except Exception as exc:  # noqa: BLE001 — mọi lỗi đều chuyển sang 7-Zip hệ thống
            # [v3.23.541] py7zr KHÔNG hỗ trợ bộ lọc BCJ2 mà gói mpv-dev đang dùng, và nó hỏng
            # GIỮA CHỪNG sau khi đã ghi ra một phần tệp. Nếu để lỗi thoát ra thì thư mục còn lại
            # dở dang mà lần chạy sau lại tưởng "đã có" rồi bỏ qua. Nay dọn sạch rồi thử 7-Zip.
            logger.info("py7zr không giải nén được (%s) → thử 7-Zip hệ thống.", exc)
            shutil.rmtree(target_dir, ignore_errors=True)
            target_dir.mkdir(parents=True, exist_ok=True)
    for binary in ("7z", "7za", "7zr"):  # 7-Zip hệ thống hỗ trợ đủ bộ lọc
        if shutil.which(binary):
            import subprocess

            subprocess.run(
                [binary, "x", "-y", f"-o{target_dir}", str(archive)],
                check=True, capture_output=True, timeout=600,
                stdin=subprocess.DEVNULL,
            )
            return
    raise FetchError(
        "Cần py7zr hoặc 7-Zip để giải nén .7z. Cài: pip install py7zr"
    )


def _verify(target_dir: Path, required: tuple[str, ...], label: str) -> None:
    """Xác nhận các tệp bắt buộc đã có sau khi giải nén.

    Raises:
        FetchError: Khi thiếu tệp bắt buộc.
    """
    missing = [
        name for name in required if not any(target_dir.rglob(name))
    ]
    if missing:
        raise FetchError(f"Gói {label} thiếu tệp bắt buộc: {', '.join(missing)}")


def fetch_mpv(force: bool) -> None:
    """Tải libmpv vào ``vendor/mpv``."""
    target = VENDOR_DIR / "mpv"
    if not force and any(target.glob("*.dll")):
        logger.info("Đã có vendor/mpv — bỏ qua (dùng --force để tải lại).")
        return
    name, url = resolve_latest_mpv_asset()
    logger.info("Tải libmpv: %s", name)
    data = _http_get(url)
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / name
        archive.write_bytes(data)
        _extract_7z(archive, target)
    _verify(target, MPV_REQUIRED, "libmpv")
    logger.info("Xong libmpv → %s", target)


def fetch_ffmpeg(force: bool, variant: str) -> None:
    """Tải ffmpeg/ffprobe vào ``vendor/ffmpeg``.

    Args:
        force: Tải lại kể cả khi đã có.
        variant: ``gpl`` (đầy đủ codec) hoặc ``lgpl`` (ràng buộc giấy phép nhẹ hơn).
    """
    target = VENDOR_DIR / "ffmpeg"
    if not force and any(target.rglob("ffmpeg.exe")):
        logger.info("Đã có vendor/ffmpeg — bỏ qua (dùng --force để tải lại).")
        return
    name = f"ffmpeg-master-latest-win64-{variant}-shared.zip"
    logger.info("Tải ffmpeg: %s", name)
    data = _http_get(f"{FFMPEG_RELEASE_BASE}/{name}")
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / name
        archive.write_bytes(data)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(target)
    _verify(target, FFMPEG_REQUIRED, "ffmpeg")
    logger.info("Xong ffmpeg → %s", target)


def main(argv: list[str] | None = None) -> int:
    """Điểm vào dòng lệnh.

    Returns:
        0 nếu thành công, 1 nếu có lỗi tải/giải nén.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Tải lại kể cả khi đã có.")
    parser.add_argument(
        "--only", choices=("mpv", "ffmpeg"), help="Chỉ tải một thành phần."
    )
    parser.add_argument(
        "--ffmpeg-variant", choices=("gpl", "lgpl"), default="gpl",
        help="Bản dựng ffmpeg: gpl (mặc định, đủ codec) hoặc lgpl.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        if args.only in (None, "mpv"):
            fetch_mpv(args.force)
        if args.only in (None, "ffmpeg"):
            fetch_ffmpeg(args.force, args.ffmpeg_variant)
    except FetchError as exc:
        logger.error("Thất bại: %s", exc)
        return 1
    logger.info("Hoàn tất. Chạy tools/generate_license_report.py để cập nhật giấy phép.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
