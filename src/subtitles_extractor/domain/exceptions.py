"""Cây phân cấp ngoại lệ cho tầng nghiệp vụ.

Mọi lỗi do tầng *domain*/*application* phát ra phải kế thừa từ
:class:`SubtitlesExtractorError`. Tầng *infrastructure* được phép map các
ngoại lệ thư viện thấp hơn (``OSError``, ``RuntimeError`` của Paddle, …)
sang một trong các lớp ở đây để tầng cao không phải biết chi tiết.

Quy tắc:
    * Thông điệp luôn bằng tiếng Việt, hướng tới lập trình viên/end-user.
    * Ưu tiên tạo *subclass* cụ thể thay vì raise ``SubtitlesExtractorError``
      trực tiếp.
"""

from __future__ import annotations


class SubtitlesExtractorError(Exception):
    """Lớp gốc cho mọi ngoại lệ trong ứng dụng."""


# ── Cấu hình ──────────────────────────────────────────────────────────────

class ConfigurationError(SubtitlesExtractorError):
    """Cấu hình không hợp lệ (giá trị ngoài phạm vi, key thiếu, định dạng sai)."""


# ── Video ─────────────────────────────────────────────────────────────────

class VideoError(SubtitlesExtractorError):
    """Lớp gốc cho mọi lỗi liên quan đến video."""


class VideoNotFoundError(VideoError):
    """File video không tồn tại hoặc không thể đọc."""


class VideoDecodeError(VideoError):
    """Không giải mã được khung hình hoặc luồng video."""


# ── OCR ───────────────────────────────────────────────────────────────────

class OcrError(SubtitlesExtractorError):
    """Lớp gốc cho mọi lỗi của tầng OCR."""


class OcrModelLoadError(OcrError):
    """Không nạp được mô hình PaddleOCR (thiếu file, GPU không khả dụng…)."""


class OcrInferenceError(OcrError):
    """Lỗi khi chạy suy luận trên một hoặc nhiều ảnh."""


# ── Subtitle ──────────────────────────────────────────────────────────────

class SubtitleExportError(SubtitlesExtractorError):
    """Lỗi khi ghi tệp phụ đề ra đĩa (SRT/ASS)."""


class SubtitleImportError(SubtitlesExtractorError):
    """Lỗi khi đọc/phân tích tệp phụ đề (SRT/ASS sai cú pháp, không tìm thấy…)."""



class SpeechToTextError(SubtitlesExtractorError):
    """Lỗi khi phiên âm giọng nói (engine STT thiếu, model lỗi, audio hỏng…)."""


__all__ = [
    "ConfigurationError",
    "OcrError",
    "OcrInferenceError",
    "OcrModelLoadError",
    "SpeechToTextError",
    "SubtitleExportError",
    "SubtitleImportError",
    "SubtitlesExtractorError",
    "VideoDecodeError",
    "VideoError",
    "VideoNotFoundError",
]
