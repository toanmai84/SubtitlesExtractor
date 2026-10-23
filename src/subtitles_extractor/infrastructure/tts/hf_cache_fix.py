"""[v3.23.628] Gỡ symlink khỏi cache HuggingFace để onnxruntime mới đọc được model VieNeu.

onnxruntime (đo trên 1.31.0) kiểm tra ``external data`` của graph ONNX: đường dẫn sau khi
``realpath`` PHẢI nằm trong thư mục của model. Cache HuggingFace lại để ``snapshots/<rev>/x.data``
là SYMLINK tới ``blobs/<hash>`` — vừa tải xong đã chết với::

    onnxruntime ... Fail: External data path validation failed for initializer: ...
    External data path escapes model directory.

(đo thật 20/10/2026 khi nạp VieNeu-TTS v3 Turbo: ``vieneu_backbone_shared.data``). Windows bật
chế độ nhà phát triển / Linux / macOS đều dính; Windows thường không có symlink nên được cứu bởi
việc HF chép tệp. Cách sửa: sau mỗi lần ``hf_hub_download``, thay symlink trong ``snapshots/``
bằng HARD LINK tới đúng blob (không tốn thêm dung lượng; không cùng ổ thì chép) — cache HF vẫn
hợp lệ vì tệp snapshot vẫn tồn tại.
"""

from __future__ import annotations

import functools
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_subext_materialize_patch"


def materialize_symlink(path: str | os.PathLike[str]) -> str:
    """Nếu ``path`` là symlink thì thay bằng hard link (hoặc bản chép) tới đúng đích.

    Args:
        path: Đường dẫn tệp trong ``snapshots/`` của cache HF.

    Returns:
        ``str(path)`` (luôn giữ nguyên đường dẫn người gọi nhận).
    """
    link = Path(path)
    try:
        if not link.is_symlink():
            return str(path)
        target = link.resolve(strict=True)
        temp = link.with_name(link.name + ".materialize_tmp")
        temp.unlink(missing_ok=True)
        try:
            os.link(target, temp)
        except OSError:
            shutil.copy2(target, temp)
        os.replace(temp, link)
    except OSError as exc:
        logger.warning("Không gỡ được symlink cache HuggingFace %s: %s", path, exc)
    return str(path)


def install_hf_materialize_patch() -> bool:
    """Bọc ``huggingface_hub.hf_hub_download`` để mọi tệp tải về đều không còn là symlink.

    Idempotent. Lời gọi ``from huggingface_hub import hf_hub_download`` thực hiện SAU khi vá (VieNeu
    import lười bên trong hàm) nhận bản đã vá.

    Returns:
        ``True`` nếu đã vá (hoặc đã vá từ trước); ``False`` khi không có huggingface_hub.
    """
    try:
        import huggingface_hub
    except ImportError:
        return False
    original = huggingface_hub.hf_hub_download
    if getattr(original, _PATCH_FLAG, False):
        return True

    @functools.wraps(original)
    def patched(*args, **kwargs):
        result = original(*args, **kwargs)
        if isinstance(result, (str, os.PathLike)):
            materialize_symlink(result)
        return result

    setattr(patched, _PATCH_FLAG, True)
    huggingface_hub.hf_hub_download = patched
    return True


__all__ = ["install_hf_materialize_patch", "materialize_symlink"]
