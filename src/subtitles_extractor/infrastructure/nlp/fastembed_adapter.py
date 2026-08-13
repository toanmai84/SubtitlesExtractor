"""Adapter nhúng Vector Ngữ nghĩa sử dụng FastEmbed & ONNX Runtime.

Changelog v3.4:
    * [BUG B019] Bỏ ``@lru_cache`` trên instance method (memory leak risk).
      Thay bằng OrderedDict-based LRU cache instance-level — cache_clear()
      hoạt động đúng theo per-instance (Singleton scope).
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)

_ENCODE_CACHE_MAX_SIZE: int = 16384

# [v3.23.305] Cờ chống lặp log 'fastembed chưa cài' (mỗi phiên chỉ báo 1 lần).
_MISSING_FASTEMBED_LOGGED: bool = False


class FastEmbedAdapter:
    """Tạo Vector Embeddings siêu tốc không cần PyTorch (Singleton)."""

    _instance: FastEmbedAdapter | None = None
    _lock = threading.RLock()

    def __new__(cls) -> FastEmbedAdapter:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._model = None
                cls._instance.enabled = False
                cls._instance.mode = "hybrid"
                cls._instance._model_name = "BAAI/bge-small-zh-v1.5"
                cls._instance._encode_cache = OrderedDict()
                cls._instance._cache_lock = threading.Lock()
            return cls._instance

    def configure(self, enabled: bool, model_name: str, mode: str) -> None:
        """Nhận cấu hình từ ApplicationContainer mỗi khi khởi động hoặc lưu Settings."""
        with self._lock:
            self.enabled = enabled
            self.mode = mode
            if self._model_name != model_name:
                self._model_name = model_name
                self._model = None
                self._clear_cache()
                # [Zombie Threads Leak] ONNX Runtime (C++) sinh thread ngầm; đổi model
                # liên tục mà không thu gom làm cạn File Descriptor → sập app. Ép GC
                # để thư viện C++ huỷ các luồng cũ, giải phóng tài nguyên ngay.
                import gc

                gc.collect()
                logger.info("Cấu hình NLP thay đổi: Model chuyển sang %s.", model_name)

    def _clear_cache(self) -> None:
        """Xoá toàn bộ cache embedding — gọi khi đổi model."""
        with self._cache_lock:
            self._encode_cache.clear()

    @staticmethod
    def get_supported_models() -> list[str]:
        """Tải danh sách mô hình được hỗ trợ chính thức bởi FastEmbed."""
        try:
            from fastembed import TextEmbedding
            return [m['model'] for m in TextEmbedding.list_supported_models()]
        except (ImportError, AttributeError, RuntimeError):
            return ["BAAI/bge-small-zh-v1.5", "BAAI/bge-m3"]

    def _initialize(self) -> None:
        if self._model is not None:
            return

        try:
            from fastembed import TextEmbedding
            logger.info("Đang nạp mô hình NLP FastEmbed: %s...", self._model_name)

            try:
                self._model = TextEmbedding(model_name=self._model_name, threads=4)
            except ValueError as val_err:
                logger.warning(
                    "Mô hình %r không được hỗ trợ trên ONNX: %s.",
                    self._model_name, val_err,
                )
                logger.warning("Đang fallback về mô hình chuẩn 'BAAI/bge-small-zh-v1.5'...")
                self._model_name = "BAAI/bge-small-zh-v1.5"
                self._model = TextEmbedding(model_name=self._model_name, threads=4)

            logger.info("Nạp mô hình NLP thành công.")
        except ImportError as exc:
            # [v3.23.290] fastembed là TÙY CHỌN (semantic embed cho Translation Memory).
            # Thiếu nó KHÔNG phải lỗi — app tự bỏ qua tính năng này. Dùng info thay vì
            # exception để không gây hiểu nhầm là lỗi nghiêm trọng trong log.
            # [v3.23.305] CHỈ log MỘT LẦN mỗi phiên: log chạy thực tế cho thấy thông
            # điệp này lặp 4 lần liên tiếp (mỗi lần truy vấn Translation Memory) gây
            # nhiễu log mà không thêm thông tin gì.
            global _MISSING_FASTEMBED_LOGGED
            if not _MISSING_FASTEMBED_LOGGED:
                _MISSING_FASTEMBED_LOGGED = True
                logger.info(
                    "fastembed chưa cài — bỏ qua embedding ngữ nghĩa (tùy chọn). "
                    "Cài 'pip install fastembed' nếu muốn dùng Translation Memory "
                    "ngữ nghĩa. (Chỉ báo một lần mỗi phiên.)"
                )
            raise RuntimeError("Thiếu thư viện fastembed") from exc
        except (RuntimeError, OSError, AttributeError, TypeError) as exc:
            logger.exception("Lỗi khởi tạo mô hình NLP: %s.", exc)
            raise

    def encode_text(self, text: str) -> np.ndarray | None:
        """Encode text thành vector embedding với cache LRU instance-level."""
        if not self.enabled or not text.strip():
            return None

        # Cache lookup nhanh — không cần lock model.
        with self._cache_lock:
            if text in self._encode_cache:
                self._encode_cache.move_to_end(text)
                return self._encode_cache[text]

        with self._lock:
            if self._model is None:
                self._initialize()
            active_model = self._model

        try:
            embeddings = list(active_model.embed([text]))
            result = embeddings[0]
        except (RuntimeError, ValueError, IndexError, AttributeError) as exc:
            logger.debug("Lỗi khi encode văn bản %r: %s.", text, exc)
            return None

        # Lưu vào cache với LRU eviction.
        with self._cache_lock:
            self._encode_cache[text] = result
            if len(self._encode_cache) > _ENCODE_CACHE_MAX_SIZE:
                # [LỖI 3 RACE CONDITION FIX]: Check dict rỗng trước khi pop
                if self._encode_cache:
                    self._encode_cache.popitem(last=False)
        return result

    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        if not self.enabled or not text_a or not text_b:
            return 0.0
        if text_a == text_b:
            return 1.0

        vec_a = self.encode_text(text_a)
        vec_b = self.encode_text(text_b)
        if vec_a is None or vec_b is None:
            return 0.0

        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        denominator = norm_a * norm_b
        if denominator < 1e-8:
            return 0.0

        return float(np.clip(dot_product / denominator, 0.0, 1.0))


__all__ = ["FastEmbedAdapter"]
