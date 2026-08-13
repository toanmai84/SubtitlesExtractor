"""Kho model tập trung: định vị ``models/`` và trỏ các thư viện về đó (offline-first).

Ý tưởng
=======
Song song với ``vendor/`` (binary native), MỌI model mà app cần tải về được gom vào
MỘT thư mục ``models/`` ở gốc dự án, có cấu trúc con theo từng hệ sinh thái::

    models/
        paddle/        official_models/<model>/...   ← PaddleOCR (PaddleX cache)
        huggingface/   hub/models--<org>--<repo>/... ← VieNeu-TTS, fastembed
        README.md

Lợi ích: tập trung một chỗ, dễ thêm/bớt/cập nhật, gốc dự án sạch, dễ ``.gitignore``
model lớn, và ``.spec`` chỉ cần nhúng nguyên cây ``models/``.

Cách hoạt động
--------------
Các thư viện (PaddleX, huggingface_hub) đọc thư mục cache từ **biến môi trường lúc
import**. Module này set các biến đó trỏ về ``models/`` TRƯỚC khi thư viện được import
(gọi sớm trong ``main()``), nên chúng dùng model có sẵn thay vì tải từ Internet.

Thứ tự phân giải gốc ``models``
-------------------------------
1. Biến môi trường ``SUBEXT_MODELS_DIR`` (escape hatch, trỏ gốc models tuỳ ý).
2. Bản đóng gói: ``sys._MEIPASS/models`` (do ``.spec`` nhúng).
3. Chạy nguồn: ``<gốc dự án>/models`` (cạnh ``main.py``).

Nguyên tắc an toàn
------------------
* **Chỉ set biến môi trường khi model THỰC SỰ có sẵn** — nếu chưa prefetch thì không
  đụng gì, để thư viện dùng cache mặc định và tải như cũ (không phá luồng hiện có).
* **Tôn trọng biến môi trường người dùng đã đặt** — không ghi đè.
* **Không bật chế độ offline cứng** (``HF_HUB_OFFLINE``) — model chưa có vẫn tải được.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Tên thư mục kho model + biến môi trường ghi đè gốc kho.
_MODELS_DIRNAME: str = "models"
_MODELS_DIR_ENV: str = "SUBEXT_MODELS_DIR"

# Thư mục con theo hệ sinh thái (khớp cấu trúc models/ và tools/prefetch_*).
PADDLE_SUBDIR: str = "paddle"
HUGGINGFACE_SUBDIR: str = "huggingface"

# Biến môi trường của PaddleX điều khiển gốc cache model (đã xác minh từ source:
# paddlex/utils/cache.py -> CACHE_DIR = os.environ.get("PADDLE_PDX_CACHE_HOME", ...)).
_PADDLE_CACHE_HOME_ENV: str = "PADDLE_PDX_CACHE_HOME"
# PaddleX kỳ vọng model nằm ở <CACHE_HOME>/official_models/<tên model>/.
_PADDLE_OFFICIAL_MODELS_SUBDIR: str = "official_models"

# Biến môi trường gốc cache của huggingface_hub (VieNeu-TTS, fastembed dùng chung).
# HF_HOME là gốc; hub cache mặc định là <HF_HOME>/hub.
_HF_HOME_ENV: str = "HF_HOME"


def _frozen_runtime_root() -> Path | None:
    """Trả về kho runtime GHI ĐƯỢC, CỐ ĐỊNH cạnh file thực thi (chỉ khi đóng gói).

    Đây là chỗ lưu MỌI thứ tải-lúc-chạy (CUDA runtime, paddlepaddle-gpu, model…) cho
    bản đóng gói. Dùng ``<thư mục chứa .exe>/models`` — bền vững qua các lần chạy và
    **di chuyển được** cùng file thực thi.

    Vì sao KHÔNG dùng ``sys._MEIPASS``:
        * Bản **onefile**: ``_MEIPASS`` là thư mục TẠM, bị xoá sau khi thoát → thứ tải
          về sẽ mất, phải tải lại mỗi lần khởi động.
        * Bản **onedir**: ``_MEIPASS`` (= ``_internal/``) là nơi chứa tài nguyên nhúng,
          không nên trộn dữ liệu tải-lúc-chạy vào đó.

    Returns:
        Đường dẫn ``<exe_dir>/models`` nếu đang chạy đóng gói (``sys.frozen``); ngược lại
        ``None`` (chạy từ nguồn).
    """
    if not getattr(sys, "frozen", False):
        return None
    try:
        exe_dir = Path(sys.executable).resolve().parent
    except (OSError, ValueError):
        return None
    return exe_dir / _MODELS_DIRNAME


def model_store_root() -> Path | None:
    """Trả về gốc thư mục ``models`` đang dùng, theo thứ tự ưu tiên.

    Returns:
        :class:`~pathlib.Path` tới gốc kho model tồn tại đầu tiên, hoặc ``None`` nếu
        không có ở đâu (khi đó các thư viện dùng cache mặc định của chúng).
    """
    override = os.environ.get(_MODELS_DIR_ENV, "").strip()
    if override:
        override_dir = Path(override)
        if override_dir.is_dir():
            return override_dir
        logger.warning(
            "%s trỏ tới thư mục không tồn tại: %s — bỏ qua.",
            _MODELS_DIR_ENV,
            override,
        )

    # Bản đóng gói: model NHÚNG (nếu có) nằm ở _MEIPASS/models — ưu tiên cho bản onedir
    # còn nhúng sẵn model. Tương thích ngược với các build cũ.
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        bundled = Path(sys._MEIPASS) / _MODELS_DIRNAME
        if bundled.is_dir():
            return bundled

    # Kho runtime CỐ ĐỊNH cạnh exe — nơi chứa model/thư viện TẢI-LÚC-CHẠY.
    frozen_root = _frozen_runtime_root()
    if frozen_root is not None and frozen_root.is_dir():
        return frozen_root

    # Chạy nguồn: model_store.py ở src/subtitles_extractor/infrastructure/
    # → parents[3] = gốc dự án (cạnh main.py).
    try:
        source_root = Path(__file__).resolve().parents[3] / _MODELS_DIRNAME
    except IndexError:
        return None
    if source_root.is_dir():
        return source_root

    return None


def ensure_model_store_root() -> Path:
    """Trả về gốc kho ``models`` GHI ĐƯỢC, TẠO thư mục nếu chưa có.

    Khác :func:`model_store_root` (chỉ trả khi đã tồn tại): hàm này dùng cho các tính năng
    TẢI-LÚC-CHẠY (CUDA runtime, paddlepaddle-gpu…) cần một chỗ ghi ỔN ĐỊNH cạnh app.

    Với bản đóng gói, LUÔN trả về kho cố định cạnh exe (:func:`_frozen_runtime_root`) —
    KHÔNG dùng ``sys._MEIPASS`` (tạm với onefile). Nhờ vậy thứ tải về giữ được qua các
    lần chạy, kể cả bản one-file.

    Returns:
        Đường dẫn thư mục ``models`` GHI ĐƯỢC đã tạo.
    """
    override = os.environ.get(_MODELS_DIR_ENV, "").strip()
    if override:
        root = Path(override)
    else:
        frozen_root = _frozen_runtime_root()
        if frozen_root is not None:
            root = frozen_root
        else:
            root = Path(__file__).resolve().parents[3] / _MODELS_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def model_store_subdir(name: str) -> Path | None:
    """Trả về thư mục con ``models/<name>`` nếu tồn tại.

    Args:
        name: Tên hệ sinh thái (vd ``"paddle"``, ``"huggingface"``).

    Returns:
        :class:`~pathlib.Path` tới ``models/<name>`` nếu có, ngược lại ``None``.
    """
    root = model_store_root()
    if root is None:
        return None
    subdir = root / name
    return subdir if subdir.is_dir() else None


def _set_env_if_absent(name: str, value: Path) -> bool:
    """Set biến môi trường ``name`` = ``value`` nếu chưa được đặt sẵn.

    Args:
        name: Tên biến môi trường.
        value: Đường dẫn sẽ gán.

    Returns:
        True nếu đã set; False nếu biến đã có giá trị từ trước (tôn trọng người dùng).
    """
    existing = os.environ.get(name)
    if existing:
        logger.debug("%s đã được đặt sẵn (%s) — tôn trọng, không ghi đè.", name, existing)
        return False
    os.environ[name] = str(value)
    return True


def configure_paddle_model_store() -> Path | None:
    """Trỏ ``PADDLE_PDX_CACHE_HOME`` để model OCR (PP-OCRv6 det/rec) nằm cạnh app.

    Thứ tự:
        1. Nếu ĐÃ prefetch (``models/paddle/official_models/`` tồn tại) → trỏ tới đó (offline).
        2. [v3.23.396] Nếu đang chạy ĐÓNG GÓI mà chưa prefetch (one-file / onedir nhỏ) → TẠO
           kho cố định ``<exe_dir>/models/paddle`` và trỏ ``PADDLE_PDX_CACHE_HOME`` vào đó để
           PaddleX TẢI model det/rec vào ``official_models/`` CẠNH exe (portable, bền — không
           rơi vào ``~/.paddlex`` tách rời hay ``_MEIPASS`` tạm).
        3. Chạy từ nguồn mà chưa prefetch → trả ``None`` (để PaddleX dùng cache mặc định của dev).

    Returns:
        Đường dẫn cache đã áp dụng, hoặc ``None`` nếu không áp dụng.
    """
    paddle_dir = model_store_subdir(PADDLE_SUBDIR)
    prefetched = (
        paddle_dir is not None
        and (paddle_dir / _PADDLE_OFFICIAL_MODELS_SUBDIR).is_dir()
    )
    if not prefetched:
        if not getattr(sys, "frozen", False):
            # Chạy nguồn chưa prefetch: để PaddleX tự dùng cache mặc định.
            return None
        # Bản đóng gói: dựng chỗ GHI ĐƯỢC, CỐ ĐỊNH cạnh exe cho model tải-lúc-chạy.
        paddle_dir = ensure_model_store_root() / PADDLE_SUBDIR
        paddle_dir.mkdir(parents=True, exist_ok=True)

    if paddle_dir is None:
        return None
    if not _set_env_if_absent(_PADDLE_CACHE_HOME_ENV, paddle_dir):
        return None

    logger.info("OCR model: %s -> %s.", _PADDLE_CACHE_HOME_ENV, paddle_dir)
    return paddle_dir


def configure_huggingface_model_store() -> Path | None:
    """Trỏ ``HF_HOME`` về ``models/huggingface`` để model VieNeu/fastembed nằm cạnh app.

    Ảnh hưởng tới mọi thư viện dùng ``huggingface_hub``: VieNeu-TTS (model ONNX) và
    fastembed (model embedding). Không bật ``HF_HUB_OFFLINE`` để model chưa có vẫn
    tải được bình thường.

    Thứ tự:
        1. Nếu ``models/huggingface`` ĐÃ tồn tại (bundled onedir hoặc đã tải trước) → dùng.
        2. [v3.23.393] Nếu đang chạy ĐÓNG GÓI mà chưa có (one-file / onedir không prefetch) →
           TẠO kho cố định ``<exe_dir>/models/huggingface`` để model tải-lúc-chạy nằm CẠNH exe
           (portable), KHÔNG rơi vào cache người dùng tách rời hay ``_MEIPASS`` tạm.
        3. Chạy từ nguồn mà chưa có → trả ``None`` (để dev dùng cache mặc định).

    Returns:
        Đường dẫn đã áp dụng, hoặc ``None`` nếu không áp dụng.
    """
    hf_dir = model_store_subdir(HUGGINGFACE_SUBDIR)
    if hf_dir is None and getattr(sys, "frozen", False):
        # Bản đóng gói: đảm bảo có chỗ GHI ĐƯỢC, CỐ ĐỊNH cạnh exe cho model tải-lúc-chạy.
        hf_dir = ensure_model_store_root() / HUGGINGFACE_SUBDIR
        hf_dir.mkdir(parents=True, exist_ok=True)
    if hf_dir is None:
        return None

    if not _set_env_if_absent(_HF_HOME_ENV, hf_dir):
        return None

    logger.info("Model HuggingFace: %s -> %s.", _HF_HOME_ENV, hf_dir)
    return hf_dir


def configure_all_model_stores() -> dict[str, Path]:
    """Cấu hình TẤT CẢ kho model tập trung (gọi sớm trong ``main()``).

    PHẢI gọi **trước** khi import bất kỳ thư viện nào đọc các biến môi trường này
    (paddle/paddlex, huggingface_hub, vieneu, fastembed).

    Returns:
        Dict ánh xạ tên hệ sinh thái -> đường dẫn đã áp dụng (rỗng nếu không áp dụng gì).
    """
    applied: dict[str, Path] = {}

    paddle_dir = configure_paddle_model_store()
    if paddle_dir is not None:
        applied[PADDLE_SUBDIR] = paddle_dir

    hf_dir = configure_huggingface_model_store()
    if hf_dir is not None:
        applied[HUGGINGFACE_SUBDIR] = hf_dir

    if not applied:
        logger.debug(
            "Không có kho model tập trung nào được áp dụng — dùng cache mặc định "
            "/ tải theo yêu cầu."
        )
    return applied


__all__ = [
    "HUGGINGFACE_SUBDIR",
    "PADDLE_SUBDIR",
    "configure_all_model_stores",
    "configure_huggingface_model_store",
    "configure_paddle_model_store",
    "model_store_root",
    "model_store_subdir",
]
