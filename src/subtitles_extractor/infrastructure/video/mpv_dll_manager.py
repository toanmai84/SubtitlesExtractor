"""Quản lý tải/phát hiện thư viện libmpv DLL trên Windows.

Tương tự Subtitle Edit, ứng dụng tự kiểm tra ``libmpv-2.dll``/``mpv-2.dll``:
    1. Nếu tệp đã tồn tại trong thư mục dữ liệu app → set ``PATH`` rồi bỏ qua.
    2. Nếu chưa có → tải ``libmpv2-64.zip`` từ GitHub mirror của Subtitle Edit
       về thư mục dữ liệu, giải nén ``mpv-2.dll`` và set ``PATH``.

Thư mục dữ liệu (``app_data_dir``):
    * Windows: ``%APPDATA%\\SubtitlesExtractor\\mpv``
    * Linux/macOS: ``~/.local/share/SubtitlesExtractor/mpv``

Trên Linux/macOS module này KHÔNG làm gì — mpv được cài qua package
manager (``apt install libmpv-dev`` / ``brew install mpv``).
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import sys
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from subtitles_extractor.infrastructure.vendor import vendor_subdir

logger = logging.getLogger(__name__)


# ── Hằng số ───────────────────────────────────────────────────────────────


# Mirror chính thức của Subtitle Edit support files — file ổn định cho Windows.
_LIBMPV_DOWNLOAD_URL: str = (
    "https://github.com/SubtitleEdit/support-files/raw/master/mpv/libmpv2-64.zip"
)
# Tên file DLL bên trong zip — python-mpv tìm theo tên này.
_DLL_FILENAMES: tuple[str, ...] = ("mpv-2.dll", "libmpv-2.dll")
# Kích thước file zip kỳ vọng — dùng để verify download.
_MIN_DOWNLOAD_BYTES: int = 5_000_000  # ~5 MB; thực tế ~22 MB
_DOWNLOAD_TIMEOUT_SEC: int = 120


@dataclass(frozen=True, slots=True)
class MpvDllStatus:
    """Trạng thái cài đặt libmpv DLL.

    Attributes:
        is_available: Tệp DLL đã sẵn sàng cho python-mpv import.
        dll_path:     Đường dẫn DLL (None nếu chưa có).
        platform_supported: False trên Linux/macOS — không cần manage DLL.
    """

    is_available: bool
    dll_path: Path | None
    platform_supported: bool


class MpvDllError(RuntimeError):
    """Raised khi không tải/giải nén được DLL."""


class MpvDllManager:
    """Quản lý vòng đời libmpv DLL.

    Args:
        app_data_dir: Thư mục dữ liệu app (sẽ tự tạo subdir ``mpv/``).
    """

    def __init__(self, app_data_dir: Path) -> None:
        self._dll_dir: Path = app_data_dir / "mpv"

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def dll_dir(self) -> Path:
        """Thư mục chứa DLL — UI dùng để hiển thị cho user."""
        return self._dll_dir

    def ensure_available(self) -> MpvDllStatus:
        """Đảm bảo DLL sẵn sàng. Trả về :class:`MpvDllStatus`.

        KHÔNG raise nếu chưa có — caller (UI) quyết định download hay không.
        """
        if not _is_windows():
            # Linux/macOS: dựa vào package manager. Trả "available" giả định.
            return MpvDllStatus(
                is_available=True, dll_path=None, platform_supported=False
            )

        existing = self._find_existing_dll()
        if existing is not None:
            self._inject_path(existing.parent)
            return MpvDllStatus(
                is_available=True,
                dll_path=existing,
                platform_supported=True,
            )

        return MpvDllStatus(
            is_available=False,
            dll_path=None,
            platform_supported=True,
        )

    def download_and_install(
        self,
        progress_callback=None,
    ) -> MpvDllStatus:
        """Tải zip → giải nén DLL → set PATH.

        Args:
            progress_callback: Optional ``Callable[[int, int], None]`` —
                nhận ``(bytes_downloaded, total_bytes)``. Để hiển thị
                progress bar trong UI.

        Raises:
            MpvDllError: Khi download/extract/test thất bại.
        """
        if not _is_windows():
            raise MpvDllError(
                "Tự tải DLL chỉ áp dụng cho Windows. "
                "Linux: cài 'libmpv-dev' qua apt; macOS: cài 'mpv' qua brew."
            )

        self._dll_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self._dll_dir / "libmpv2-64.zip"

        try:
            self._download_with_progress(
                _LIBMPV_DOWNLOAD_URL, zip_path, progress_callback
            )
        except (urllib.error.URLError, OSError) as exc:
            raise MpvDllError(
                f"Không tải được libmpv từ {_LIBMPV_DOWNLOAD_URL}: {exc}"
            ) from exc

        if zip_path.stat().st_size < _MIN_DOWNLOAD_BYTES:
            zip_path.unlink(missing_ok=True)
            raise MpvDllError(
                f"Tệp tải về quá nhỏ ({zip_path.stat().st_size} bytes) — "
                "có thể bị lỗi mạng hoặc URL không khả dụng."
            )

        try:
            self._extract_dll(zip_path)
        except (zipfile.BadZipFile, KeyError, OSError) as exc:
            raise MpvDllError(f"Không giải nén được libmpv: {exc}") from exc
        finally:
            zip_path.unlink(missing_ok=True)

        existing = self._find_existing_dll()
        if existing is None:
            raise MpvDllError(
                f"Sau khi giải nén không tìm thấy DLL trong {self._dll_dir}."
            )
        self._inject_path(existing.parent)
        logger.info("Đã cài libmpv DLL thành công tại %s.", existing)
        return MpvDllStatus(
            is_available=True,
            dll_path=existing,
            platform_supported=True,
        )

    def remove(self) -> None:
        """Xoá DLL đã tải — buộc tải lại lần sau."""
        for filename in _DLL_FILENAMES:
            target = self._dll_dir / filename
            target.unlink(missing_ok=True)
        logger.info("Đã xoá libmpv DLL khỏi %s.", self._dll_dir)

    # ── Private helpers ─────────────────────────────────────────────────

    # Biến môi trường cho phép chỉ định thư mục chứa libmpv DLL (ưu tiên cao nhất).
    _MPV_DIR_ENV: str = "SUBEXT_MPV_DIR"

    def _iter_candidate_dirs(self) -> Iterator[Path]:
        """Sinh các thư mục có thể chứa libmpv DLL, theo THỨ TỰ ƯU TIÊN.

        [v3.23.300] Tập trung: ưu tiên ``vendor/mpv`` (nguồn duy nhất, dùng chung cơ
        chế với ffmpeg), rồi tới các vị trí dự phòng. Thứ tự:
            1. Biến môi trường ``SUBEXT_MPV_DIR`` (escape hatch riêng cho mpv).
            2. ``vendor/mpv`` tập trung (env SUBEXT_VENDOR_DIR / bundle / gốc dự án).
            3. Cạnh file thực thi (frozen): thư mục chứa ``.exe`` (+ ``mpv/``).
            4. Thư mục tải về mặc định (``self._dll_dir``) — nơi bản tải runtime lưu.

        Yields:
            Các :class:`~pathlib.Path` thư mục ứng viên (có thể chưa tồn tại).
        """
        override = os.environ.get(self._MPV_DIR_ENV, "").strip()
        if override:
            override_dir = Path(override)
            yield override_dir
            yield override_dir / "mpv"

        vendor_mpv = vendor_subdir("mpv")
        if vendor_mpv is not None:
            yield vendor_mpv

        if getattr(sys, "frozen", False):
            exe_dir = Path(sys.executable).resolve().parent
            yield exe_dir
            yield exe_dir / "mpv"

        yield self._dll_dir

    @staticmethod
    def _find_dll_in_dir(directory: Path) -> Path | None:
        """Tìm file DLL libmpv hợp lệ (> 1MB) trong một thư mục cụ thể."""
        for filename in _DLL_FILENAMES:
            candidate = directory / filename
            try:
                if candidate.is_file() and candidate.stat().st_size > 1_000_000:
                    return candidate
            except OSError:
                continue
        return None

    def _find_existing_dll(self) -> Path | None:
        """Tìm libmpv DLL ở mọi vị trí ứng viên, ưu tiên bản có sẵn (không cần tải)."""
        seen: set[Path] = set()
        for directory in self._iter_candidate_dirs():
            try:
                resolved = directory.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            found = self._find_dll_in_dir(resolved)
            if found is not None:
                logger.info("Tìm thấy libmpv DLL: %s.", found)
                return found
        return None

    def _inject_path(self, dll_parent_dir: Path) -> None:
        """Thêm thư mục chứa DLL vào ``PATH`` để ``ctypes.CDLL`` tìm thấy.

        Phải gọi TRƯỚC khi ``import mpv``. Lý do: python-mpv dùng
        ``ctypes.find_library('mpv')`` nội bộ — nó scan ``PATH``.
        """
        absolute = str(dll_parent_dir.resolve())
        current_path = os.environ.get("PATH", "")
        if absolute not in current_path.split(os.pathsep):
            os.environ["PATH"] = absolute + os.pathsep + current_path
            logger.info("Đã inject %s vào PATH cho libmpv.", absolute)

        # Python 3.8+ trên Windows: thêm DLL search dir tường minh.
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(absolute)  # type: ignore[attr-defined]
            except (FileNotFoundError, OSError) as exc:
                logger.warning("add_dll_directory thất bại: %s.", exc)

    @staticmethod
    def _download_with_progress(
        url: str, target: Path, callback
    ) -> None:
        """Tải vào file ``.tmp`` rồi atomic rename — an toàn nếu bị ngắt.

        Nếu user đóng app giữa chừng, file ``.tmp`` còn lại sẽ bị ghi
        đè ở lần thử tiếp theo, không ảnh hưởng DLL hợp lệ trước đó.
        """
        tmp_target = target.with_suffix(target.suffix + ".tmp")
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "subtitles-extractor/2.7 (+mpv installer)"
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=_DOWNLOAD_TIMEOUT_SEC
            ) as response:
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 64 * 1024  # 64 KB
                with tmp_target.open("wb") as fp:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        fp.write(chunk)
                        downloaded += len(chunk)
                        if callback is not None:
                            callback(downloaded, total)
            # Atomic rename: chỉ "commit" khi download hoàn tất sạch sẽ.
            tmp_target.replace(target)
            logger.info(
                "Đã tải xong libmpv (%d bytes) → %s.", downloaded, target
            )
        except (OSError, RuntimeError):
            # Dọn file tạm nếu có lỗi I/O hoặc HTTP.
            tmp_target.unlink(missing_ok=True)
            raise

    def _extract_dll(self, zip_path: Path) -> None:
        with zipfile.ZipFile(zip_path) as archive:
            # Tìm DLL trong zip (tên có thể có path prefix khác nhau).
            for member in archive.namelist():
                base = Path(member).name.lower()
                if base in {f.lower() for f in _DLL_FILENAMES}:
                    target = self._dll_dir / Path(member).name
                    with archive.open(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    # Đặt tên chính tắc thành "mpv-2.dll" nếu cần.
                    if target.name.lower() == "libmpv-2.dll":
                        canonical = self._dll_dir / "mpv-2.dll"
                        if canonical != target:
                            shutil.copy(target, canonical)
                    logger.info("Đã giải nén DLL: %s.", target.name)
                    return
        raise KeyError(
            f"Không tìm thấy file DLL trong zip. Các thành viên: "
            f"{archive.namelist()[:5]}"
        )


# ── Module-level helpers ──────────────────────────────────────────────────


def _is_windows() -> bool:
    return platform.system().lower() == "windows" or sys.platform == "win32"


__all__ = ["MpvDllError", "MpvDllManager", "MpvDllStatus"]
