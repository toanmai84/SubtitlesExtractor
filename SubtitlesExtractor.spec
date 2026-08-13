# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec cho SubtitlesExtractor (Windows).

Cách dùng:
    pyinstaller SubtitlesExtractor.spec --noconfirm              # MẶC ĐỊNH: 1 file .exe nhỏ
    set SUBEXT_ONEDIR=1 && pyinstaller SubtitlesExtractor.spec --noconfirm   # chế độ thư mục

Các công tắc môi trường (env) khi build:
- SUBEXT_ONEDIR=1        → dựng chế độ THƯ MỤC (mặc định: MỘT file .exe duy nhất).
- SUBEXT_BUNDLE_PADDLE=1 → nhúng sẵn lõi paddle ~810MB (mặc định: KHÔNG nhúng, tải lúc chạy).
- SUBEXT_BUNDLE_CUDA=1   → nhúng sẵn CUDA ~2.3GB (mặc định: KHÔNG, tải lúc chạy để bật GPU).

Xem docs/BUILD_PLAN.md để hiểu các quyết định đóng gói. Nguyên tắc:
- Bản NHỎ 1-file để dễ lưu trữ/chia sẻ: KHÔNG nhúng paddle/CUDA/model — tải lúc chạy vào
  models/ CẠNH exe (bền cho cả one-file, xem infrastructure/model_store._frozen_runtime_root).
- KHÔNG bundle whisperx/torch/cupy: nặng + cần CUDA, cài ở môi trường riêng whisperx_env.
- KHÔNG UPX cho paddle: UPX nén DLL paddle gây crash.
"""

# [v3.23.323] PHẢI import ở ĐẦU tệp: các khối phía dưới dùng `_os` từ dòng ~164,
# nếu import muộn hơn thì build chết ngay với NameError.
import os as _os

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

# ── Thu thập trọn gói các phụ thuộc phức tạp ──────────────────────────────────
# collect_all trả về (datas, binaries, hiddenimports) — gom hết resource + native lib.
datas = []
binaries = []
hiddenimports = []


# [v3.23.303] LỌC SUBMODULE VÔ ÍCH khỏi hiddenimports "ép buộc".
# Log build v3.23.302 cho thấy collect_all() kéo vào TOÀN BỘ test suite của scipy
# (scipy.stats.tests.*, scipy.linalg.tests.*, scipy.sparse.tests.* … hàng trăm module),
# cùng các hệ con app KHÔNG DÙNG: paddlex.repo_apis.* (API huấn luyện Paddle3D/Clas/
# Detection/NLP/Seg/TS/Video), paddle.distributed.* (huấn luyện phân tán),
# onnxruntime.transformers|tools.* (công cụ chuyển đổi model), paddleocr._doc2md.*
# (doc→markdown, cần lxml). Chúng phình bundle rất nhiều và góp phần gây LỖI HẾT ĐĨA.
#
# AN TOÀN: đây chỉ bỏ việc ÉP đưa vào. Module nào THỰC SỰ được import lúc chạy vẫn
# được PyInstaller phát hiện qua phân tích phụ thuộc bình thường. Ta chỉ ngừng ép
# nhét những thứ không ai import.
_SKIP_SUBMODULE_PATTERNS: tuple[str, ...] = (
    ".tests.",              # bộ test của scipy/librosa/…
    ".test_",               # module test lẻ
    "._precompute",         # script sinh bảng của scipy (cần mpmath)
    "paddlex.repo_apis.",   # API huấn luyện — app chỉ suy luận (inference)
    "paddle.distributed.",  # huấn luyện phân tán
    "paddle.incubate.",     # API thử nghiệm
    "onnxruntime.transformers.",  # công cụ chuyển đổi/tối ưu model
    "onnxruntime.tools.",
    "onnxruntime.quantization",
    "paddleocr._doc2md.",   # doc→markdown (cần lxml, app không dùng)
    "huggingface_hub.cli.",  # giao diện dòng lệnh HF
    "huggingface_hub.inference._mcp.",
    "vieneu.serve",         # demo server gradio của tác giả — app không dùng
    "vieneu.v3_turbo_serve",
)


def _is_useless_submodule(module_name: str) -> bool:
    """True nếu submodule chắc chắn không cần lúc chạy (test/huấn luyện/CLI)."""
    if module_name.endswith(".tests") or module_name.endswith(".test"):
        return True
    return any(pattern in module_name for pattern in _SKIP_SUBMODULE_PATTERNS)


def _add_all(package_name: str) -> None:
    """Gom toàn bộ package (data + binary + submodule) vào build, bỏ phần vô ích."""
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package_name)
    datas.extend(pkg_datas)
    binaries.extend(pkg_binaries)
    kept_hidden = [name for name in pkg_hidden if not _is_useless_submodule(name)]
    skipped = len(pkg_hidden) - len(kept_hidden)
    if skipped:
        print(f"[spec] {package_name}: bỏ {skipped} submodule vô ích (test/train/CLI).")
    hiddenimports.extend(kept_hidden)


# [v3.23.387] Gate SUBEXT_BUNDLE_PADDLE (MẶC ĐỊNH OFF): KHÔNG nhúng lõi paddle (~810MB) —
# thành phần NẶNG NHẤT. App tải lõi paddle LÚC CHẠY vào models/paddle_runtime/ (xem
# infrastructure/ocr/paddle_runtime_plan.py + nút "Tải lõi OCR" ở Cài đặt). paddleocr/paddlex
# (pure-python, nhỏ, dùng chung) VẪN nhúng để pipeline OCR sẵn sàng; chúng import paddle lúc
# chạy — nếu chưa tải, adapter OCR sẽ báo cần tải (xử lý mượt ở Bước 4).
# Đặt SUBEXT_BUNDLE_PADDLE=1 để nhúng sẵn lõi paddle (bản đầy đủ, dung lượng lớn — hành vi cũ).
_BUNDLE_PADDLE = _os.environ.get("SUBEXT_BUNDLE_PADDLE", "").strip().lower() in (
    "1", "true", "yes", "on",
)
_paddle_packages = (
    ("paddle", "paddleocr", "paddlex") if _BUNDLE_PADDLE else ("paddleocr", "paddlex")
)
for _pkg in _paddle_packages:
    try:
        _add_all(_pkg)
    except Exception as exc:  # noqa: BLE001 — build script, log & tiếp tục
        print(f"[spec] Cảnh báo: không gom được {_pkg}: {exc}")

# [v3.23.398] Khi LOẠI paddle khỏi bundle, PyInstaller không thu đủ các dep DÙNG CHUNG mà
# paddle/paddleocr import ĐỘNG. Pillow là thủ phạm điển hình: thiếu PIL.ImageEnhance →
# "cannot import name 'ImageEnhance' from 'PIL'" lúc chạy OCR (và làm paddle init lỗi dây
# chuyền → libpaddle.pir). Thu TRỌN PIL (submodule + data + binary) để đủ mọi submodule.
try:
    _add_all("PIL")
except Exception as exc:  # noqa: BLE001
    print(f"[spec] Cảnh báo: không gom được PIL: {exc}")
if _BUNDLE_PADDLE:
    print("[spec] SUBEXT_BUNDLE_PADDLE=1 → NHÚNG lõi paddle (~810MB, bản đầy đủ).")
else:
    print(
        "[spec] MẶC ĐỊNH KHÔNG nhúng lõi paddle (build nhỏ). App tải lúc chạy qua nút "
        "'Tải lõi OCR (paddle)' ở Cài đặt. Đặt SUBEXT_BUNDLE_PADDLE=1 để nhúng sẵn."
    )

# [v3.23.302] CUDA RUNTIME: paddlepaddle-gpu KHÔNG chứa sẵn thư viện CUDA — chúng là
# các gói pip RIÊNG (nvidia-cudnn-cu12 768MB, nvidia-cublas-cu12 553MB, cusparse 363MB,
# cusolver 320MB, cufft 200MB, curand 69MB, nvjitlink 36MB, cuda-runtime 4MB ≈ 2.3GB).
# Paddle nạp chúng lúc chạy qua os.add_dll_directory/ctypes — KHÔNG phải import Python —
# nên PyInstaller KHÔNG tự phát hiện. Nếu thiếu, bản đóng gói sẽ không dùng được GPU và
# (nhờ v3.23.295) âm thầm lùi về CPU => OCR chậm hẳn mà không báo lỗi.
#
# [v3.23.375] MẶC ĐỊNH KHÔNG NHÚNG (~2.3GB) để bản build đủ nhỏ up lên GitHub Releases.
# App vẫn CHẠY NGAY bằng CPU (paddle tự lùi CPU khi thiếu CUDA). Người dùng bấm nút
# "Bật tăng tốc GPU (OCR)" để TẢI CUDA runtime về ``models/cuda_runtime/`` lúc chạy — khớp
# cơ chế tải-lúc-chạy của WhisperX/VieNeu. Đặt SUBEXT_BUNDLE_CUDA=1 nếu muốn build offline
# đầy đủ (nhúng sẵn CUDA, dung lượng lớn).
_BUNDLE_CUDA = _os.environ.get("SUBEXT_BUNDLE_CUDA", "").strip().lower() in (
    "1", "true", "yes", "on",
)
_NVIDIA_CUDA_PACKAGES = (
    "nvidia.cuda_runtime",
    "nvidia.cudnn",
    "nvidia.cublas",
    "nvidia.cufft",
    "nvidia.curand",
    "nvidia.cusolver",
    "nvidia.cusparse",
    "nvidia.nvjitlink",
)
_nvidia_collected = 0
if _BUNDLE_CUDA:
    for _nv_pkg in _NVIDIA_CUDA_PACKAGES:
        try:
            _nv_binaries = collect_dynamic_libs(_nv_pkg)
        except Exception as exc:  # noqa: BLE001 — gói không cài / tên khác -> bỏ qua
            print(f"[spec] Bỏ qua CUDA lib {_nv_pkg}: {exc}")
            continue
        if _nv_binaries:
            binaries.extend(_nv_binaries)
            _nvidia_collected += len(_nv_binaries)
    print(f"[spec] SUBEXT_BUNDLE_CUDA=1 → nhúng {_nvidia_collected} thư viện CUDA (~2.3GB).")
else:
    print(
        "[spec] MẶC ĐỊNH KHÔNG nhúng CUDA (build nhỏ). App chạy CPU; bấm 'Bật tăng tốc GPU' "
        "để tải CUDA runtime lúc chạy. Đặt SUBEXT_BUNDLE_CUDA=1 để nhúng sẵn."
    )

# [v3.23.304] TensorRT (TUY CHON — mac dinh KHONG cai; xem build_windows.bat).
# Neu build_env co goi 'tensorrt' (do SUBEXT_ENABLE_TENSORRT=1), gom native lib de
# tuy chon "use_tensorrt" trong Cai dat > Phan cung hoat dong o ban dong goi.
# LICENSE: TensorRT la phan mem DOC QUYEN cua NVIDIA (EULA rieng, khong phai giay phep
# mo). Chi bat khi da kiem tra dieu khoan phan phoi lai cua NVIDIA.
#
# [v3.23.305] Kiem module TON TAI truoc khi goi collect_dynamic_libs. Ban truoc goi
# thang nen PyInstaller in 3 canh bao thua moi lan build khi khong cai TensorRT:
#   "collect_dynamic_libs - skipping library collection for module 'tensorrt' as it
#    is not a package."  (collect_dynamic_libs CHI LOG canh bao, KHONG raise, nen
#    khoi try/except khong bat duoc.)
import importlib.util as _importlib_util

_trt_collected = 0
for _trt_pkg in ("tensorrt", "tensorrt_libs", "tensorrt_bindings"):
    try:
        _trt_spec = _importlib_util.find_spec(_trt_pkg)
    except (ImportError, ValueError, ModuleNotFoundError):
        continue
    # submodule_search_locations chi co o PACKAGE (khong co o module don) -> tranh
    # dung canh bao "as it is not a package".
    if _trt_spec is None or not _trt_spec.submodule_search_locations:
        continue
    try:
        _trt_binaries = collect_dynamic_libs(_trt_pkg)
    except Exception as exc:  # noqa: BLE001 — build script: log & tiep tuc
        print(f"[spec] Bỏ qua TensorRT lib {_trt_pkg}: {exc}")
        continue
    if _trt_binaries:
        binaries.extend(_trt_binaries)
        _trt_collected += len(_trt_binaries)
if _trt_collected:
    print(f"[spec] Gom {_trt_collected} thư viện TensorRT (tùy chọn, license NVIDIA).")
else:
    print("[spec] Không có TensorRT (mặc định) — paddle chạy GPU qua cuDNN/cuBLAS.")

# [v3.23.277] librosa + scipy có data files/lazy submodule + numba — collect_all để chắc
# chắn đầy đủ (librosa ISC thay pedalboard GPL cho time-stretch).
for _pkg in ("librosa", "scipy", "soundfile"):
    try:
        _add_all(_pkg)
    except Exception as exc:  # noqa: BLE001
        print(f"[spec] Cảnh báo: không gom được {_pkg}: {exc}")

# [v3.23.312] NÂNG CẤP CHẤT LƯỢNG — các gói trước đây bị loại vì license thương mại
# hoặc dung lượng. Người dùng đã quyết định KHÔNG thương mại hoá nên bật bản tốt nhất.
# Tất cả đều có native binary / data file nên phải collect_all, không thể chỉ hiddenimport.
# Gói nào chưa cài thì bỏ qua an toàn — app đã có sẵn cơ chế fallback cho từng cái.
# [v3.23.313] qfluentwidgets (GPL v3) — chỉ gom khi bật SUBEXT_USE_QFLUENTWIDGETS=1.
# Gói này có file .qss/resource nên phải collect_all. Không bật -> app dùng bản Qt thuần
# tương thích API trong presentation/fluent_compat (không cần gì thêm).
if _os.environ.get("SUBEXT_USE_QFLUENTWIDGETS", "").strip().lower() in (
    "1", "true", "yes", "on"
):
    for _qf_pkg in ("qfluentwidgets",):
        try:
            _add_all(_qf_pkg)
            print(f"[spec] Đã gom {_qf_pkg} (giao diện Fluent gốc, GPL v3).")
        except Exception as exc:  # noqa: BLE001 — chưa cài -> dùng bản Qt thuần
            print(f"[spec] Bỏ qua {_qf_pkg} (chưa cài, dùng fluent_compat): {exc}")
else:
    print("[spec] Giao diện: bản Qt thuần fluent_compat (đặt SUBEXT_USE_QFLUENTWIDGETS=1 để đổi).")

_QUALITY_PACKAGES = (
    "pedalboard",     # GPL v3 — kéo giãn thời gian chất lượng cao (nhân Rubber Band)
    "edge_tts",       # GPL v3 — giọng neural Microsoft
    "fastembed",      # Apache 2.0 — nhúng ngữ nghĩa cho Bộ nhớ Dịch
    "sklearn",        # BSD-3 — gom cụm DBSCAN cho tự động dò ROI
)
for _pkg in _QUALITY_PACKAGES:
    try:
        _add_all(_pkg)
        print(f"[spec] Đã gom gói nâng cấp: {_pkg}.")
    except Exception as exc:  # noqa: BLE001 — chưa cài -> app tự fallback
        print(f"[spec] Bỏ qua {_pkg} (chưa cài, app sẽ dùng phương án dự phòng): {exc}")

# [v3.23.279] paddlex kiểm dependency qua importlib.metadata (dist-info) lúc tạo pipeline.
# PyInstaller KHÔNG bundle metadata mặc định → "A dependency error occurred during pipeline
# creation". Bundle metadata của paddlex/paddleocr/paddle + 18 required deps paddlex kiểm.
# Lưu ý chardet: module bị exclude (LGPL, thay shim MIT) nhưng METADATA vẫn bundle (chỉ là
# text dist-info, không phải code) để paddlex check pass; shim cung cấp module lúc runtime.
_METADATA_PACKAGES = (
    # [v3.23.303] Bỏ "paddlepaddle" — bản GPU cài dưới tên "paddlepaddle-gpu", tên trần
    # không tồn tại nên gây warning "No package metadata was found for paddlepaddle".
    "paddlex", "paddleocr", "paddlepaddle-gpu",
    # 18 required deps của paddlex (đọc từ paddlex.utils.deps.REQUIRED_DEP_SPECS):
    "PyYAML", "aistudio-sdk", "chardet", "colorlog", "filelock", "huggingface-hub",
    "modelscope", "numpy", "packaging", "pandas", "pillow", "prettytable",
    "py-cpuinfo", "pydantic", "requests", "ruamel.yaml", "typing-extensions", "ujson",
    # [v3.23.286] Dep nhóm 'ocr-core' (paddleocr[ocr-core]) — paddlex kiểm metadata cua
    # chung khi tạo pipeline OCR. Chẩn đoán v285 cho thấy lỗi ở gói mở rộng. paddleocr mặc
    # định chỉ dùng ocr-core (6 dep), không phải nhóm 'ocr' đầy đủ.
    "imagesize", "opencv-contrib-python", "pyclipper", "pypdfium2", "python-bidi", "shapely",
)
for _meta_pkg in _METADATA_PACKAGES:
    try:
        # recursive=True gom metadata cua ca dependency con -> chac chan du.
        datas.extend(copy_metadata(_meta_pkg, recursive=True))
    except TypeError:
        # PyInstaller cu khong ho tro recursive -> fallback khong recursive.
        try:
            datas.extend(copy_metadata(_meta_pkg))
        except Exception as exc:  # noqa: BLE001
            print(f"[spec] Bỏ qua metadata {_meta_pkg}: {exc}")
    except Exception as exc:  # noqa: BLE001 — package không cài thì bỏ
        print(f"[spec] Bỏ qua metadata {_meta_pkg}: {exc}")

# [v3.23.267] qfluentwidgets ĐÃ BỎ (GPL) — thay bằng fluent_compat tự xây (Qt thuần).
# Không cần collect. PySide6 tự có hook PyInstaller.

# ── Data files của ứng dụng ──────────────────────────────────────────────────
# [v3.23.370] Đóng gói TOÀN BỘ thư mục data. Trước đây chỉ thêm strings_vi.json nên
# THIẾU: strings_en.json (đa ngôn ngữ tiếng Anh KHÔNG chạy trong bản build) và
# app.ico/app.png (KHÔNG có icon cửa sổ/thanh tác vụ lúc runtime). Vòng lặp này gom mọi
# tệp trong data → tự bao gồm cả file ngôn ngữ/tài nguyên thêm sau này.
_data_src_dir = "src/subtitles_extractor/data"
for _data_file in sorted(_os.listdir(_data_src_dir)):
    _full = _os.path.join(_data_src_dir, _data_file)
    if _os.path.isfile(_full):
        datas.append((_full, "subtitles_extractor/data"))
# Worker edge-tts subprocess (chạy bằng Python NGOÀI để cách ly GPL — xem LICENSE_ANALYSIS).
datas.append(("src/subtitles_extractor/infrastructure/tts/edge_tts_subprocess.py",
              "subtitles_extractor/infrastructure/tts"))
# [v3.23.340] Worker WhisperX subprocess — BỊ BỎ SÓT cho tới nay.
# Adapter chạy `python <đường dẫn>/whisperx_subprocess.py`. Nếu tệp KHÔNG có trong
# bundle thì Python báo "can't open file" và thoát **mã 2** — đúng lỗi người dùng gặp.
# (Đã đo: thiếu tệp script -> mã 2, y hệt argparse sai đối số, nên rất dễ chẩn đoán nhầm.)
datas.append(("src/subtitles_extractor/infrastructure/stt/whisperx_subprocess.py",
              "subtitles_extractor/infrastructure/stt"))
# [v3.23.344] Worker VieNeu GPU — chạy bằng Python của môi trường riêng (nơi CÓ torch).
datas.append(("src/subtitles_extractor/infrastructure/tts/vieneu_gpu_subprocess.py",
              "subtitles_extractor/infrastructure/tts"))
# [v3.23.269] Kèm thông báo license bên thứ ba (BẮT BUỘC khi phân phối thương mại).
for _lic_file in ("THIRD_PARTY_LICENSES.md",):
    if _os.path.exists(_lic_file):
        datas.append((_lic_file, "."))

# [v3.23.301] KHO MODEL TẬP TRUNG: nhúng NGUYÊN CÂY models/ vào bundle.
#   models/paddle/official_models/...  -> models/paddle/...      (PaddleOCR offline)
#   models/huggingface/hub/...         -> models/huggingface/... (VieNeu-TTS, fastembed)
# Prefetch trước khi build:
#     python tools/prefetch_ocr_models.py          (-> models/paddle)
#     python tools/prefetch_hf_models.py           (-> models/huggingface)
# Runtime: infrastructure/model_store.py trỏ PADDLE_PDX_CACHE_HOME / HF_HOME về đây.
# BỎ QUA an toàn nếu models/ chưa có -> app tải model theo yêu cầu như cũ.
from pathlib import Path as _Path

_MODELS_DIR = _Path("models")
if _MODELS_DIR.is_dir():
    _model_file_count = 0
    for _model_file in _MODELS_DIR.rglob("*"):
        if _model_file.is_file() and _model_file.name != "README.md":
            # Giữ nguyên cấu trúc models/<hệ sinh thái>/<đường dẫn con>.
            datas.append((str(_model_file), _model_file.parent.as_posix()))
            _model_file_count += 1
    print(f"[spec] Nhúng {_model_file_count} file model từ models/ (tập trung).")
else:
    print(
        "[spec] Bỏ qua nhúng models/: chưa thấy thư mục. Chạy tools/prefetch_ocr_models.py "
        "và tools/prefetch_hf_models.py nếu muốn model offline lần đầu."
    )

# [v3.23.300] STANDALONE: nhúng NGUYÊN CÂY vendor/ (binary native tập trung) vào bundle.
#   vendor/mpv/{libmpv-2,mpv-2}.dll  -> vendor/mpv/    (libmpv LGPL)
#   vendor/ffmpeg/{ffmpeg,ffprobe}   -> vendor/ffmpeg/ (PHAI la ban LGPL --disable-gpl!)
# Runtime: infrastructure/vendor.py phân giải sys._MEIPASS/vendor -> ffmpeg_locator &
# mpv_dll_manager ưu tiên bản vendored này -> chạy offline, không cần cài/PATH/tải.
# BỎ QUA an toàn nếu vendor/ trống. CẢNH BÁO LICENSE: ffmpeg.exe đa số bản dựng là GPL.
_VENDOR_DIR = _Path("vendor")
if _VENDOR_DIR.is_dir():
    _vendor_file_count = 0
    for _vendor_file in _VENDOR_DIR.rglob("*"):
        if _vendor_file.is_file() and _vendor_file.name != "README.md":
            # Đích trong bundle: giữ nguyên cấu trúc vendor/<nhóm>/<file>.
            _rel_parent = _vendor_file.parent.as_posix()
            # [v3.23.397] Python embeddable (vendor/python_embed) chứa hỗn hợp exe/dll/py/pyd
            # (gồm pip). Đưa vào DATAS (chỉ copy, KHÔNG để PyInstaller phân tích .py của pip)
            # thay vì binaries — tránh nhiễu/rủi ro. Các binary vendored khác (mpv/ffmpeg DLL)
            # giữ ở binaries như cũ.
            if "python_embed" in _vendor_file.parts:
                datas.append((str(_vendor_file), _rel_parent))
            else:
                binaries.append((str(_vendor_file), _rel_parent))
            _vendor_file_count += 1
    print(f"[spec] Nhúng {_vendor_file_count} file vendored từ vendor/ (tập trung).")
else:
    print("[spec] Bỏ qua nhúng vendor/: chưa thấy thư mục vendor.")

# ── Hidden imports (import động PyInstaller không tự thấy) ────────────────────
hiddenimports.extend([
    # GUI (PySide6 LGPL — thay PyQt6 GPL)
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtSvg",
    "PySide6.QtSvgWidgets",
    # [v3.23.310] QtMultimedia: QAudioSink phát tiếng cho trình phát PyAV (thay libmpv
    # GPL). Khai báo tường minh vì nó chỉ được import trong một module con — nếu thiếu,
    # bản đóng gói sẽ phát không tiếng mà không báo lỗi rõ ràng.
    "PySide6.QtMultimedia",
    "shiboken6",
    # Khoa học dữ liệu / audio (C-ext)
    "scipy",
    "scipy.signal",
    "scipy.special",
    "scipy.io",
    "scipy.ndimage",
    "soundfile",
    "librosa",       # time-stretch ISC (thay pedalboard GPL)
    "pydub",         # MP3->WAV worker Edge TTS
    "pyloudnorm",    # đo LUFS (tùy chọn)
    "av",
    "numpy",
    # Cấu hình
    "pydantic",
    "pydantic_settings",
    # Tiện ích
    "rjieba",
    "json_repair",
    "charset_normalizer",  # thay chardet LGPL (shim MIT)
    "loguru",
    # TTS engine nhẹ (bundle mặc định)
    "google.genai",
    # [v3.23.292] VieNeu-TTS offline (bundle mặc định — Apache 2.0)
    "onnxruntime",
    "vieneu",
    "sea_g2p",
    "soxr",
    "tokenizers",
    "huggingface_hub",
    # [v3.23.396] paddle bị LOẠI khỏi bundle → PyInstaller không tự thấy các import gián tiếp
    # của paddle. paddle import NHIỀU submodule setuptools.command.* (easy_install, install,
    # build_ext…) cho JIT/cpp_extension — các module này bị XÓA dần khỏi setuptools>=80, nên
    # requirements ghim setuptools<80. THU TOÀN BỘ submodule setuptools (không liệt kê từng cái,
    # tránh whack-a-mole từng module thiếu) + pkg_resources.
])
# [v3.23.399] Thu toàn bộ submodule setuptools/pkg_resources NHƯNG bỏ *.tests (cần pytest,
# gây 100+ cảnh báo build vô ích + không cần khi chạy).
def _no_tests(mods):
    return [m for m in mods if ".tests" not in m and not m.endswith(".tests")]

hiddenimports.extend(_no_tests(collect_submodules("setuptools")))
hiddenimports.extend(_no_tests(collect_submodules("pkg_resources")))
hiddenimports.append("pkg_resources")

# [v3.23.292] VieNeu-TTS: engine TTS tieng Viet OFFLINE — BUNDLE MAC DINH (Apache 2.0).
# CPU torch-free (chi ONNX). collect_all vi sea_g2p co binary Rust (.pyd) + vieneu co config
# data. Model ONNX tai runtime lan dau (KHONG bundle — nang). Neu build_env khong cai (ban
# CPU toi gian) thi bo qua an toan.
for _vieneu_pkg in (
    "vieneu", "sea_g2p", "onnxruntime", "soxr", "tokenizers", "huggingface_hub",
):
    try:
        _add_all(_vieneu_pkg)
    except Exception as exc:  # noqa: BLE001
        print(f"[spec] Bỏ qua {_vieneu_pkg} (không cài trong build_env): {exc}")

# [v3.23.287] PyNvVideoCodec: NVDEC hardware decode. File .pyd chinh co VERSION trong ten
# (vd PyNvVideoCodec_130.cp312-win_amd64.pyd — 130 = NVENC/driver version) nen PyInstaller
# KHONG tu nhat duoc (chi thay VersionCheck.pyd). Dung collect_all + collect_dynamic_libs
# de gom HET .pyd/.dll trong package. App tu fallback PyAV/CPU neu thieu.
try:
    from PyInstaller.utils.hooks import collect_dynamic_libs

    _add_all("PyNvVideoCodec")
    # Gom moi .pyd/.dll (ke ca file version-specific PyInstaller khong tu thay).
    binaries.extend(collect_dynamic_libs("PyNvVideoCodec"))
    # Gom moi data file (.pyd doi khi bi coi la data, khong phai binary).
    datas.extend(collect_data_files("PyNvVideoCodec", include_py_files=True))
except Exception as exc:  # noqa: BLE001 — khong cai (may khong GPU) thi bo qua
    print(f"[spec] Bỏ qua PyNvVideoCodec (không cài): {exc}")

# ── Loại trừ các package nặng KHÔNG bundle ───────────────────────────────────
# [v3.23.333] WhisperX + torch LUÔN bị loại khỏi bundle — kể cả khi bật
# SUBEXT_ENABLE_WHISPERX. Chúng được cài vào môi trường RIÊNG `whisperx_env` cạnh thư
# mục dự án, vì ba lý do (đã tra metadata thật của whisperx 3.8.6):
#   1. whisperx ghim `huggingface-hub<1.0.0` còn app dùng 1.24.0 -> cài chung sẽ hạ cấp
#      và có thể làm hỏng VieNeu-TTS/PaddleOCR.
#   2. torch nạp CUDA riêng, dễ xung đột DLL với paddle.
#   3. Gom vào bundle sẽ phình thêm ~3GB.
# Adapter tự tìm `whisperx_env` lúc chạy nên không cần bundle.
_heavy_stt_excludes = ["whisperx", "torch", "torchvision", "torchaudio"]
if _os.environ.get("SUBEXT_ENABLE_WHISPERX", "").strip().lower() in (
    "1", "true", "yes", "on"
):
    print(
        "[spec] SUBEXT_ENABLE_WHISPERX=1 — WhisperX cài ở môi trường riêng "
        "'whisperx_env', KHÔNG gom vào bundle (tránh xung đột huggingface-hub/CUDA)."
    )

excludes = [
    *_heavy_stt_excludes,
    # charset_normalizer (MIT) KHÔNG chỉ an toàn license mà còn nhận diện mã hoá CHÍNH
    # XÁC HƠN chardet — giữ shim này kể cả khi không còn ràng buộc thương mại.
    "chardet",
    # [v3.23.279] PyNvVideoCodec + cupy KHÔNG còn exclude: máy build có GPU NVIDIA
    # (requirements dùng paddlepaddle-gpu + pynvvideocodec) — nếu cài trong build_env
    # sẽ được bundle để NVDEC hoạt động; nếu không cài thì PyInstaller tự bỏ qua.
                     # Đặt SUBEXT_DISABLE_PEDALBOARD=1 lúc chạy nếu muốn ép dùng librosa.
    "pytest",        # test framework
    "matplotlib",    # không dùng trong runtime GUI
    "IPython",
    "notebook",
    # ── [v3.23.302] ỨNG VIÊN GIẢM DUNG LƯỢNG (chưa bật — CẦN KIỂM CHỨNG) ──────
    # `vieneu` kéo theo gradio>=5.49.1 (gradio 6.x + fastapi + uvicorn + starlette…),
    # vốn chỉ phục vụ web demo UI của tác giả — app KHÔNG dùng. Bỏ đi tiết kiệm
    # hàng trăm MB. RỦI RO: nếu vieneu/__init__.py import gradio ở top-level thì
    # `from vieneu import Vieneu` sẽ hỏng => TTS VieNeu chết.
    # CÁCH KIỂM CHỨNG trong build_env trước khi bật (phải in ra OK):
    #     build_env\Scripts\python.exe -c "import sys; sys.modules['gradio']=None; from vieneu import Vieneu; print('OK - khong can gradio')"
    # Nếu in "OK" -> bỏ dấu # ở 3 dòng dưới để giảm dung lượng bundle.
    # "gradio",
    # "gradio_client",
    # "uvicorn",
]

# [v3.23.387] Khi KHÔNG nhúng paddle (mặc định): LOẠI hẳn 'paddle' khỏi bundle để phân tích
# phụ thuộc của PyInstaller không kéo lõi ~810MB vào qua import của paddleocr/paddlex.
# Lúc chạy, bootstrap thêm models/paddle_runtime/ (lõi tải về) vào sys.path → 'import paddle'
# lấy từ đó. paddleocr/paddlex vẫn được nhúng (nhỏ); nếu paddle chưa tải, adapter OCR báo cần
# tải. Metadata 'paddlepaddle-gpu' vẫn được bundle ở _METADATA_PACKAGES (chỉ là text dist-info)
# để paddlex kiểm phụ thuộc pass. Đặt SUBEXT_BUNDLE_PADDLE=1 để nhúng lõi paddle như cũ.
if not _BUNDLE_PADDLE:
    excludes.append("paddle")
    print("[spec] Loại 'paddle' khỏi bundle (tải lúc chạy). paddleocr/paddlex vẫn nhúng.")

# ── Phân tích ─────────────────────────────────────────────────────────────────
a = Analysis(
    ["main.py"],
    pathex=["src"],  # để import subtitles_extractor.* mà không cần pip install
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# [v3.23.387] Icon dùng chung cho cả hai chế độ (PyInstaller bỏ qua nếu thiếu).
_icon_path = (
    "src/subtitles_extractor/data/app.ico"
    if __import__("pathlib").Path("src/subtitles_extractor/data/app.ico").exists()
    else None
)

# [v3.23.389] MẶC ĐỊNH ONE-FILE: dựng MỘT file .exe duy nhất (dễ lưu trữ/chia sẻ) — theo yêu
# cầu. Đặt SUBEXT_ONEDIR=1 để quay lại chế độ thư mục (onedir) khi cần chẩn đoán/khởi động
# nhanh. Khi one-file, PyInstaller nhét a.binaries + a.datas THẲNG vào EXE
# (exclude_binaries=False) và KHÔNG dùng COLLECT. Lưu ý: one-file giải nén ra thư mục tạm mỗi
# lần chạy (khởi động chậm hơn onedir); nhưng nhờ v3.23.384, dữ liệu tải-lúc-chạy
# (paddle/CUDA/model) nằm ở models/ CẠNH exe nên KHÔNG bị mất. UPX vẫn TẮT (paddle DLL dễ hỏng).
_FORCE_ONEDIR = _os.environ.get("SUBEXT_ONEDIR", "").strip().lower() in (
    "1", "true", "yes", "on",
)
_ONEFILE = not _FORCE_ONEDIR

if _ONEFILE:
    print("[spec] MẶC ĐỊNH one-file → dựng MỘT file .exe duy nhất. Đặt SUBEXT_ONEDIR=1 để dùng thư mục.")
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        exclude_binaries=False,   # onefile: gói mọi binary/data vào trong EXE
        name="SubtitlesExtractor",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,                # TẮT UPX — paddle DLL dễ hỏng với UPX
        runtime_tmpdir=None,      # giải nén vào %TEMP% mặc định của hệ
        console=False,            # GUI app — không hiện cửa sổ console
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=_icon_path,
    )
    # KHÔNG COLLECT ở chế độ onefile — EXE đã tự chứa hết.
else:
    print("[spec] SUBEXT_ONEDIR=1 → chế độ thư mục (onedir). Bỏ biến này để về one-file mặc định.")
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="SubtitlesExtractor",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,          # TẮT UPX toàn cục — paddle DLL dễ hỏng với UPX
        console=False,      # GUI app — không hiện cửa sổ console
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=_icon_path,
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,          # TẮT UPX — an toàn cho paddle
        upx_exclude=[],
        name="SubtitlesExtractor",
    )
