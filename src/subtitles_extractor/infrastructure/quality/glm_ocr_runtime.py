"""[v3.23.745] GLM-OCR làm người đọc kiểm định — chạy bằng ``llama-server`` (llama.cpp), TẢI KHI DÙNG LẦN ĐẦU.

Chủ dự án (18/11/2026): exe không được nặng thêm ~3 GB; chỉ tải mô hình khi dùng. Đo 457 câu hai phim (57 câu bất đồng
đã xem ảnh + 400 câu ngẫu nhiên): GLM-OCR 57/57 + 397/400 (PaddleOCR-VL 50/57 + 398/400); bản GGUF Q8_0 qua llama.cpp
57/57 + 158/159 — giữ nguyên chất lượng bản gốc.

* ``llama-server`` là chương trình C++ độc lập (không PyTorch, không đụng bộ chặn torch / Paddle trong tiến trình chính),
  chạy ở TIẾN TRÌNH RIÊNG, GPU NVIDIA qua CUDA (``-ngl``), nạp mô hình MỘT lần cho cả lượt; xong lượt thì tắt → trả VRAM.
  Trên Windows tiến trình được gắn vào Job Object "đóng là giết" → ứng dụng chết bất thường thì server chết theo.
* Tải về ``<thư mục dữ liệu>/models/glm_ocr/`` (≈ 1,9 GB một lần): bản llama.cpp **b12387** (đúng commit đã đo chất
  lượng) cho Windows CUDA 12.4 + DLL CUDA; mô hình GGUF ``GLM-OCR-Q8_0`` + phần thị giác ``mmproj`` (kiểm SHA-256).
  Hai gói llama.cpp từ GitHub chưa ghim được mã băm (máy dựng bản không tải được trang phát hành) — kiểm bằng danh sách
  tệp bắt buộc + số bản dựng ``--version``.
* Chọn GPU mà llama-server không đưa được mô hình lên GPU (thiếu DLL CUDA, trình điều khiển cũ) → lỗi, không âm thầm
  chạy CPU. Mọi lỗi (mạng, giải nén, server không lên) → :class:`GlmOcrError`; lượt trích xuất lùi về PaddleOCR-VL.
"""

from __future__ import annotations

import base64
import contextlib
import functools
import hashlib
import http.client
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np

from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.ocr_model_names import (
    GLM_OCR_QA_MODEL,
)
from subtitles_extractor.infrastructure.process.hidden_process import (
    no_window_kwargs,
)

if TYPE_CHECKING:
    from subtitles_extractor.domain.ports.ocr_engine_port import OcrEnginePort

logger = logging.getLogger(__name__)

#: Bản llama.cpp đã đo chất lượng GLM-OCR (tag ↔ commit a956a85).
LLAMA_CPP_BUILD: Final[str] = "b12387"
LLAMA_CPP_COMMIT: Final[str] = "a956a85"
_GITHUB: Final[str] = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_BUILD}"
#: Phiên bản kho mô hình trên Hugging Face (ghim — tệp đổi thì mã băm không khớp).
GLM_GGUF_REVISION: Final[str] = "65a42de1148dbed2297e922b5dbc7d9b70c36578"
_HF: Final[str] = f"https://huggingface.co/ggml-org/GLM-OCR-GGUF/resolve/{GLM_GGUF_REVISION}"
#: Nhãn ghi vào gói kiểm định (hằng ở tầng domain — tầng ứng dụng dùng chung).
GLM_OCR_LABEL: Final[str] = GLM_OCR_QA_MODEL
#: Lời nhắc nhận dạng chữ của GLM-OCR (đúng lời nhắc đã đo).
GLM_PROMPT: Final[str] = "Text Recognition:"

SERVER_START_TIMEOUT_SEC: Final[float] = 240.0
REQUEST_TIMEOUT_SEC: Final[float] = 120.0
#: Bỏ cả lượt GLM-OCR (lùi về PaddleOCR-VL) khi chừng này ảnh liên tiếp đọc lỗi; lỗi lẻ → ảnh đó rỗng.
MAX_CONSECUTIVE_READ_ERRORS: Final[int] = 5
#: [v3.23.759] Bước soát câu (giữ khoá suy luận của bộ chuyển đổi OCR): một ảnh cắt sát một câu GPU đọc < 1 s — server
#: treo phải bị bỏ sau vài chục giây, không phải 5 × 120 s.
LINE_REQUEST_TIMEOUT_GPU_SEC: Final[float] = 30.0
LINE_REQUEST_TIMEOUT_CPU_SEC: Final[float] = 120.0
LINE_MAX_CONSECUTIVE_READ_ERRORS: Final[int] = 3
_CHUNK: Final[int] = 1 << 20
#: Dòng ``llama-server --list-devices``: ``  CUDA0: NVIDIA GeForce RTX 3070 Ti (8191 MiB, 7120 MiB free)``.
_DEVICE_LINE = re.compile(r"^\s*((?:CUDA)\d+):\s*(.+?)\s*$", re.M)

#: Gọi máy chủ cục bộ KHÔNG qua proxy hệ thống (proxy công ty / Clash không bỏ qua 127.0.0.1 → treo / lỗi).
_LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class GlmOcrError(RuntimeError):
    """Không cài / không chạy được GLM-OCR (lượt trích xuất lùi về PaddleOCR-VL)."""


@dataclass(frozen=True, slots=True)
class Asset:
    """Một tệp cần tải: ``sha256`` rỗng = tệp nén (kiểm bằng danh sách tệp bắt buộc bên trong)."""

    name: str
    url: str
    size: int
    sha256: str = ""
    required_members: tuple[str, ...] = ()


MODEL_ASSETS: Final[tuple[Asset, ...]] = (
    Asset("GLM-OCR-Q8_0.gguf", f"{_HF}/GLM-OCR-Q8_0.gguf", 950_433_408,
          "45bc244a6446aff850521dc41f18bc8d7105ad5f0c2c8c28af04e7cc4f4d50b1"),
    Asset("mmproj-GLM-OCR-Q8_0.gguf", f"{_HF}/mmproj-GLM-OCR-Q8_0.gguf", 484_403_648,
          "9c4b58e33e316ed142eb5dcb41abec3844d3e6e5dc361ffb782c3fa9d175141f"),
)
#: Bản Windows x64 CUDA 12.4 (trình điều khiển NVIDIA ≥ 550): zip llama.cpp đã gộp sẵn llama-server + ggml-cuda.dll;
#: zip DLL CUDA (cudart / cublas) đặt cạnh.
WINDOWS_SERVER_ASSETS: Final[tuple[Asset, ...]] = (
    Asset(f"llama-{LLAMA_CPP_BUILD}-bin-win-cuda-12.4-x64.zip",
          f"{_GITHUB}/llama-{LLAMA_CPP_BUILD}-bin-win-cuda-12.4-x64.zip", 0,
          required_members=("llama-server.exe", "ggml-cuda.dll")),
    Asset("cudart-llama-bin-win-cuda-12.4-x64.zip", f"{_GITHUB}/cudart-llama-bin-win-cuda-12.4-x64.zip", 0,
          required_members=("cudart64_12.dll", "cublas64_12.dll")),
)

#: ``progress(thông điệp, 0..1)`` → ``False`` để huỷ.
ProgressFn = Callable[[str, float], bool]


def _report(progress: ProgressFn, label: str, fraction: float) -> bool:
    return progress(label, fraction)


class GlmOcrInstaller:
    """Tải / kiểm / giải nén GLM-OCR + llama-server vào ``root`` (mặc định ``<dữ liệu>/models/glm_ocr``)."""

    def __init__(self, root: Path, *, server_assets: Sequence[Asset] | None = None,
                 model_assets: Sequence[Asset] = MODEL_ASSETS, server_binary: Path | None = None) -> None:
        self.root = root
        self._server_assets = tuple(WINDOWS_SERVER_ASSETS if server_assets is None else server_assets)
        self._model_assets = tuple(model_assets)
        self._server_binary = server_binary

    @property
    def bin_dir(self) -> Path:
        return self.root / f"llama-{LLAMA_CPP_BUILD}"

    @property
    def model_path(self) -> Path:
        return self.root / self._model_assets[0].name

    @property
    def mmproj_path(self) -> Path:
        return self.root / self._model_assets[1].name

    @property
    def server_binary(self) -> Path:
        if self._server_binary is not None:
            return self._server_binary
        return self.bin_dir / ("llama-server.exe" if sys.platform == "win32" else "llama-server")

    @property
    def _marker(self) -> Path:
        return self.root / "installed.json"

    def _server_asset_ok(self, asset: Asset) -> bool:
        """Gói llama.cpp đủ khi MỌI tệp bắt buộc của nó có trong ``bin_dir`` (cài dở → thiếu → tải lại gói đó)."""
        return all((self.bin_dir / member).is_file() for member in asset.required_members)

    def _model_ok(self, asset: Asset) -> bool:
        path = self.root / asset.name
        return path.is_file() and path.stat().st_size == asset.size

    def _server_ok(self) -> bool:
        if self._server_binary is not None:
            return self._server_binary.is_file()
        return all(self._server_asset_ok(asset) for asset in self._server_assets)

    def is_installed(self) -> bool:
        if not (self._server_ok() and self._marker.is_file()):
            return False
        try:
            marker = json.loads(self._marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return marker.get("build") == LLAMA_CPP_BUILD and all(self._model_ok(asset) for asset in self._model_assets)

    def install(self, progress: ProgressFn) -> None:
        """Tải những gì còn thiếu. Ném :class:`GlmOcrError` khi lỗi / bị huỷ (tệp dở dang bị xoá)."""
        if self.is_installed():
            return
        if self._server_binary is None and sys.platform != "win32" and self._server_assets == WINDOWS_SERVER_ASSETS:
            # Chỉ có bản Windows CUDA được ghim — máy khác (máy phát triển) không tải ~1,9 GB rồi mới báo lỗi.
            raise GlmOcrError("GLM-OCR (llama-server tải sẵn) chỉ hỗ trợ Windows x64.")
        self.root.mkdir(parents=True, exist_ok=True)
        assets = [a for a in self._model_assets if not self._model_ok(a)]
        if self._server_binary is None:
            assets += [a for a in self._server_assets if not self._server_asset_ok(a)]
        for number, asset in enumerate(assets, start=1):
            label = f"Đang tải GLM-OCR ({number}/{len(assets)}: {asset.name})"
            target = self.root / asset.name
            self._download(asset, target, functools.partial(_report, progress, label))
            if asset.required_members:
                try:
                    self._extract(target, asset.required_members)
                finally:
                    target.unlink(missing_ok=True)
        if not self._server_ok():
            raise GlmOcrError(f"Không thấy {self.server_binary.name} sau khi giải nén.")
        self._marker.write_text(json.dumps({"build": LLAMA_CPP_BUILD, "commit": LLAMA_CPP_COMMIT,
                                            "gguf_revision": GLM_GGUF_REVISION}), encoding="utf-8")
        logger.info("Đã cài GLM-OCR (llama.cpp %s) vào %s.", LLAMA_CPP_BUILD, self.root)

    @staticmethod
    def _download(asset: Asset, target: Path, progress: Callable[[float], bool]) -> None:
        part = target.with_name(target.name + ".part")
        digest = hashlib.sha256()
        request = urllib.request.Request(asset.url, headers={"User-Agent": "subtitles-extractor (glm-ocr installer)"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as handle:  # noqa: S310
                total = int(response.headers.get("Content-Length") or asset.size or 0)
                done = 0
                while True:
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    if progress(min(done / total, 1.0) if total else 0.0) is False:
                        raise GlmOcrError("Đã huỷ tải GLM-OCR.")
                if total and done != total:
                    raise GlmOcrError(f"{asset.name}: tải được {done} byte, cần {total}.")
        except GlmOcrError:
            part.unlink(missing_ok=True)
            raise
        except BaseException as exc:            # mạng đứt giữa chừng (IncompleteRead…), đĩa đầy, huỷ luồng: xoá tệp dở
            part.unlink(missing_ok=True)
            if isinstance(exc, (OSError, http.client.HTTPException, ValueError)):
                raise GlmOcrError(f"Tải {asset.name} lỗi: {exc}") from exc
            raise
        if asset.size and part.stat().st_size != asset.size:
            size = part.stat().st_size
            part.unlink(missing_ok=True)
            raise GlmOcrError(f"{asset.name}: tải được {size} byte, cần {asset.size}.")
        if asset.sha256 and digest.hexdigest() != asset.sha256:
            part.unlink(missing_ok=True)
            raise GlmOcrError(f"{asset.name}: mã băm SHA-256 không khớp — tệp hỏng hoặc bị thay.")
        part.replace(target)

    def _extract(self, archive: Path, required: Sequence[str]) -> None:
        """Giải nén PHẲNG (chỉ tên tệp) vào thư mục tạm rồi chuyển vào ``bin_dir`` — giải nén hỏng giữa chừng không để
        lại gói "đủ tệp bắt buộc" giả."""
        try:
            with zipfile.ZipFile(archive) as bundle:
                bad = bundle.testzip()
                if bad is not None:
                    raise GlmOcrError(f"{archive.name}: tệp hỏng trong gói ({bad}).")
                names: dict[str, str] = {}
                for name in bundle.namelist():
                    base = Path(name.replace("\\", "/")).name
                    if name.endswith(("/", "\\")) or base in ("", ".", ".."):
                        continue
                    names[base] = name                    # tên trùng ở thư mục con khác: bản sau đè (README, LICENSE)
                lower = {base.lower() for base in names}
                missing = [m for m in required if m.lower() not in lower]
                if missing:
                    raise GlmOcrError(f"{archive.name}: thiếu {', '.join(missing)}.")
                self.bin_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=self.root) as staging:
                    for base, name in names.items():
                        with bundle.open(name) as source, (Path(staging) / base).open("wb") as sink:
                            shutil.copyfileobj(source, sink)
                    for base in names:
                        os.replace(Path(staging) / base, self.bin_dir / base)
        except zipfile.BadZipFile as exc:
            raise GlmOcrError(f"{archive.name}: không phải tệp zip hợp lệ.") from exc
        except OSError as exc:
            raise GlmOcrError(f"Giải nén {archive.name} lỗi: {exc}") from exc
        if sys.platform != "win32" and self.server_binary.is_file():
            self.server_binary.chmod(0o755)


def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def server_command(
    binary: Path, model: Path, mmproj: Path, port: int, *, use_gpu: bool, device: str = "",
) -> list[str]:
    """Dòng lệnh ``llama-server`` (hàm thuần): nghe 127.0.0.1, một luồng hội thoại, không giao diện web.

    GPU: ``--device <CUDA đã dò>`` + ``-ngl 999`` — thiết bị không dùng được thì server THOÁT (lùi về PaddleOCR-VL)
    thay vì âm thầm chạy CPU. CPU: ``--device none``.
    """
    return [
        str(binary), "-m", str(model), "--mmproj", str(mmproj), "--host", "127.0.0.1", "--port", str(port),
        "-c", "4096", "-np", "1", "--no-webui",
        "--device", device if use_gpu and device else "none", "-ngl", "999" if use_gpu else "0",
    ]


def cuda_devices(listing: str) -> list[tuple[str, str]]:
    """``[(tên, mô tả)]`` thiết bị CUDA trong kết quả ``llama-server --list-devices`` (hàm thuần).

    [v3.23.748] Thay cách đọc dòng "offloaded N/M layers to GPU" trong log (3.23.745): llama-server KHÔNG in dòng đó ở
    mức log mặc định → máy chủ dự án (RTX 3070 Ti) bị báo nhầm "không đưa được mô hình lên GPU" (log 19/11 00:16).
    """
    return [(name, description) for name, description in _DEVICE_LINE.findall(listing or "")]


def _job_kernel32() -> Any:
    """[v3.23.747] kernel32 với ``argtypes`` đủ: mặc định ctypes truyền số nguyên 32 bit → HANDLE 64 bit bị cắt cụt (Job
    Object / gắn tiến trình thất bại âm thầm)."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined,unused-ignore]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    return kernel32


class _KillOnCloseJob:
    """Windows: Job Object ``KILL_ON_JOB_CLOSE`` — tiến trình con chết theo khi ứng dụng thoát / bị giết (không mồ côi
    giữ VRAM + khoá tệp). Nơi khác: không làm gì. Lỗi API → bỏ qua (vẫn tắt server bình thường khi xong lượt)."""

    def __init__(self) -> None:
        self._handle: int | None = None
        if sys.platform != "win32":
            return
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = _job_kernel32()
            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                return

            class _Basic(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                            ("SchedulingClass", wintypes.DWORD)]

            class _Io(ctypes.Structure):
                _fields_ = [(name, ctypes.c_uint64) for name in ("ReadOps", "WriteOps", "OtherOps", "ReadBytes",
                                                                  "WriteBytes", "OtherBytes")]

            class _Extended(ctypes.Structure):
                _fields_ = [("BasicLimitInformation", _Basic), ("IoInfo", _Io),
                            ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                            ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

            info = _Extended()
            info.BasicLimitInformation.LimitFlags = 0x2000          # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if kernel32.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):  # Extended = 9
                self._handle = handle
            else:
                kernel32.CloseHandle(handle)
        except Exception as exc:  # noqa: BLE001 — chỉ là lớp bảo vệ thêm
            logger.debug("Không tạo được Job Object cho llama-server: %s", exc)

    def assign(self, process: subprocess.Popen[bytes]) -> None:
        if self._handle is None:
            return
        try:
            _job_kernel32().AssignProcessToJobObject(self._handle, int(process._handle))  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            logger.debug("Không gắn được llama-server vào Job Object: %s", exc)

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            _job_kernel32().CloseHandle(handle)


class LlamaServer:
    """Tiến trình ``llama-server`` của một lượt: bật → đợi ``/health`` → (GPU: kiểm mô hình đã lên GPU) → dùng → tắt."""

    def __init__(self, installer: GlmOcrInstaller, *, use_gpu: bool) -> None:
        self._installer = installer
        self._use_gpu = use_gpu
        self._process: subprocess.Popen[bytes] | None = None
        self._log_handle: Any = None
        self._job: _KillOnCloseJob | None = None
        self.port = 0
        self.device_note = "CPU"

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def log_path(self) -> Path:
        return self._installer.root / "llama-server.log"

    def start(self, cancelled: Callable[[], bool] = lambda: False) -> None:
        """Bật server; lỗi bất kỳ (kể cả ngoại lệ lạ) → tắt server rồi ném lại — không để tiến trình mồ côi."""
        try:
            self._start(cancelled)
        except BaseException:
            self.stop()
            raise

    def _start(self, cancelled: Callable[[], bool]) -> None:
        self.port = _free_port()
        self._installer.root.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("wb")              # ghi ra tệp — không đọc ống (tránh nghẽn ống đầy)
        env = dict(os.environ)
        env["PATH"] = str(self._installer.server_binary.parent) + os.pathsep + env.get("PATH", "")
        device = self._pick_device(env) if self._use_gpu else ""
        command = server_command(self._installer.server_binary, self._installer.model_path,
                                 self._installer.mmproj_path, self.port, use_gpu=self._use_gpu, device=device)
        self._job = _KillOnCloseJob()
        try:
            self._process = subprocess.Popen(  # noqa: S603 — tệp do bộ cài kiểm (mã băm / danh sách tệp)
                command, stdin=subprocess.DEVNULL, stdout=self._log_handle, stderr=subprocess.STDOUT,
                cwd=str(self._installer.server_binary.parent), env=env, **no_window_kwargs(),
            )
        except OSError as exc:
            raise GlmOcrError(f"Không chạy được llama-server: {exc}") from exc
        self._job.assign(self._process)
        deadline = time.monotonic() + SERVER_START_TIMEOUT_SEC
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise GlmOcrError(f"llama-server thoát (mã {self._process.returncode}): {self._log_tail()}")
            if cancelled():
                raise GlmOcrError("Đã huỷ.")
            try:
                with _LOCAL.open(f"{self.base_url}/health", timeout=2) as response:
                    if response.status == 200:
                        logger.info("llama-server (GLM-OCR) sẵn sàng ở cổng %d — %s.", self.port, self.device_note)
                        return
            except (OSError, http.client.HTTPException):
                pass
            time.sleep(0.5)
        raise GlmOcrError(f"llama-server không sẵn sàng sau {SERVER_START_TIMEOUT_SEC:.0f} s: {self._log_tail()}")

    def _pick_device(self, env: dict[str, str]) -> str:
        """Dò thiết bị CUDA bằng ``llama-server --list-devices``; không có → :class:`GlmOcrError` (không âm thầm CPU)."""
        try:
            listing = subprocess.run(  # noqa: S603 — tệp do bộ cài kiểm
                [str(self._installer.server_binary), "--list-devices"], capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=120, check=False, env=env,
                cwd=str(self._installer.server_binary.parent), **no_window_kwargs(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GlmOcrError(f"Không dò được thiết bị của llama-server: {exc}") from exc
        output = (listing.stdout or "") + (listing.stderr or "")
        devices = cuda_devices(output)
        if not devices:
            raise GlmOcrError("llama-server không thấy GPU CUDA (thiếu DLL CUDA / trình điều khiển NVIDIA cũ hơn "
                              f"CUDA 12.4?): {output.strip()[-400:]}")
        name, description = devices[0]
        self.device_note = f"GPU {name}: {description}"
        return name

    def _log_tail(self, size: int = 600) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")[-size:].strip()
        except OSError:
            return ""

    def stop(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        if self._job is not None:
            self._job.close()
            self._job = None


def clean_reading(text: str) -> str:
    """Chữ GLM-OCR trả về → một dòng (bỏ xuống dòng / khoảng trắng thừa, rào mã ``` nếu có) (hàm thuần)."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip() and not line.strip().startswith("```")]
    return "".join(lines)


def _png_base64(image_rgb: np.ndarray) -> str:
    import cv2

    ok, encoded = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise GlmOcrError("Không mã hoá được ảnh.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


class GlmOcrEngine:
    """Bộ nhận dạng cho :class:`~subtitles_extractor.infrastructure.quality.quality_auditor.VlQualityAuditor`: mỗi ảnh
    (đã cắt sát câu phụ đề) → một hộp chữ phủ cả ảnh. Một ảnh đọc lỗi → rỗng (ghi log); nhiều ảnh liên tiếp lỗi → ném
    :class:`GlmOcrError` (server hỏng)."""

    vl_error = None

    def __init__(self, server: LlamaServer) -> None:
        self._server = server
        self._failures = 0
        self.read_errors = 0

    def read(self, image_rgb: np.ndarray, timeout: float = REQUEST_TIMEOUT_SEC) -> str:
        body = {
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _png_base64(image_rgb)}},
                {"type": "text", "text": GLM_PROMPT},
            ]}],
            "temperature": 0, "max_tokens": 64,
        }
        request = urllib.request.Request(f"{self._server.base_url}/v1/chat/completions", json.dumps(body).encode("utf-8"),
                                         {"Content-Type": "application/json"})
        with _LOCAL.open(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return clean_reading(str(payload["choices"][0]["message"]["content"] or ""))

    def _read_tolerant(self, image_rgb: np.ndarray, timeout: float = REQUEST_TIMEOUT_SEC,
                       max_failures: int = MAX_CONSECUTIVE_READ_ERRORS) -> str:
        try:
            text = self.read(image_rgb, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 — [v3.23.759] gồm cả cv2.error khi mã hoá ảnh: tính là MỘT ảnh lỗi
            self._failures += 1
            self.read_errors += 1
            logger.warning("GLM-OCR đọc một ảnh lỗi (%d lần liên tiếp): %s", self._failures, exc)
            if self._failures >= max_failures:
                raise GlmOcrError(f"GLM-OCR đọc lỗi {self._failures} ảnh liên tiếp: {exc}") from exc
            return ""
        self._failures = 0
        return text

    def infer_batch(
        self, images_rgb: list[np.ndarray], frame_indices: list[int], timestamps_sec: list[float],
        config: object | None = None,
    ) -> list[OcrFrameResult]:
        del config                                          # cấu hình PaddleOCR không áp cho GLM-OCR
        results = []
        for image, index, stamp in zip(images_rgb, frame_indices, timestamps_sec, strict=True):
            text = self._read_tolerant(image)
            height, width = image.shape[:2]
            boxes = [OcrTextBox(text=text, confidence=Confidence(0.99),
                                polygon=[(0, 0), (width, 0), (width, height), (0, height)])] if text else []
            results.append(OcrFrameResult(frame_index=index, timestamp_sec=stamp, text_boxes=boxes))
        return results


@contextlib.contextmanager
def glm_ocr_auditor(
    installer: GlmOcrInstaller, *, use_gpu: bool, progress: ProgressFn, cancelled: Callable[[], bool],
) -> Iterator[Any]:
    """Cài (nếu thiếu) → bật ``llama-server`` → trả bộ kiểm định đọc bằng GLM-OCR → tắt server khi xong (mọi đường)."""
    from subtitles_extractor.infrastructure.quality.quality_auditor import (
        VlQualityAuditor,
    )

    installer.install(progress)
    server = LlamaServer(installer, use_gpu=use_gpu)
    try:
        server.start(cancelled)
        engine = GlmOcrEngine(server)
        yield VlQualityAuditor(cast("OcrEnginePort", engine))   # đủ giao diện infer_batch bộ kiểm định dùng
        if engine.read_errors:
            logger.warning("GLM-OCR: %d ảnh đọc lỗi (để rỗng trong gói kiểm định).", engine.read_errors)
    finally:
        server.stop()


class GlmLineReader:
    """[v3.23.758] Bộ đọc DÒNG bằng GLM-OCR, cùng giao diện bộ đọc PaddleOCR-VL của bộ chuyển đổi OCR
    (``read(ảnh BGR, device)`` → chữ, ``release``, ``model_name``) — để bước SOÁT CÂU đọc lại bằng GLM-OCR khi người dùng
    chọn "Bộ đọc kiểm định: GLM-OCR" (đo 457 câu: GLM-OCR 57/57 câu khó, PaddleOCR-VL 50/57). Server hỏng (nhiều ảnh
    liên tiếp lỗi) → ném :class:`GlmOcrError`; bộ chuyển đổi lùi về PaddleOCR-VL."""

    def __init__(self, engine: GlmOcrEngine, *, use_gpu: bool = True,
                 cancelled: Callable[[], bool] = lambda: False) -> None:
        self._engine = engine
        self._timeout = LINE_REQUEST_TIMEOUT_GPU_SEC if use_gpu else LINE_REQUEST_TIMEOUT_CPU_SEC
        self._cancelled = cancelled

    @property
    def model_name(self) -> str:
        return GLM_OCR_LABEL

    @property
    def read_errors(self) -> int:
        return self._engine.read_errors

    def read(self, crops_bgr: Sequence[np.ndarray], device: str = "gpu") -> list[str]:
        del device                                          # thiết bị do llama-server quyết khi bật (CUDA0 / CPU)
        texts: list[str] = []
        for crop in crops_bgr:
            if self._cancelled():                           # [v3.23.759] người dùng huỷ: ảnh còn lại giữ bản nháp PP-OCR
                texts.append("")
                continue
            texts.append(self._engine._read_tolerant(np.ascontiguousarray(crop[:, :, ::-1]), timeout=self._timeout,
                                                     max_failures=LINE_MAX_CONSECUTIVE_READ_ERRORS))
        return texts

    def release(self) -> None:
        """Server do khối ``with`` của :func:`glm_ocr_line_reader` tắt — ở đây không làm gì."""


@contextlib.contextmanager
def glm_ocr_line_reader(
    installer: GlmOcrInstaller, *, use_gpu: bool, progress: ProgressFn, cancelled: Callable[[], bool],
) -> Iterator[GlmLineReader]:
    """[v3.23.758] Cài (nếu thiếu) → bật ``llama-server`` → trả bộ đọc dòng GLM-OCR → tắt server khi xong (mọi đường)."""
    installer.install(progress)
    server = LlamaServer(installer, use_gpu=use_gpu)
    try:
        server.start(cancelled)
        reader = GlmLineReader(GlmOcrEngine(server), use_gpu=use_gpu, cancelled=cancelled)
        yield reader
        if reader.read_errors:
            logger.warning("GLM-OCR soát câu: %d ảnh đọc lỗi (giữ bản đọc PP-OCR cho ảnh đó).", reader.read_errors)
    finally:
        server.stop()


__all__ = [
    "GLM_OCR_LABEL", "LLAMA_CPP_BUILD", "MODEL_ASSETS", "WINDOWS_SERVER_ASSETS", "Asset", "GlmOcrEngine", "GlmOcrError",
    "GlmLineReader", "GlmOcrInstaller", "LlamaServer", "clean_reading", "cuda_devices", "glm_ocr_auditor",
    "glm_ocr_line_reader", "server_command",
]
