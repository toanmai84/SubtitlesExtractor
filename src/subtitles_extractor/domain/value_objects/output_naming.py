"""Quy ước đặt tên file đầu ra cho từng khâu trong quy trình Auto-Dubbing.

Trình tự công việc: trích xuất → chỉnh sửa → biên dịch → chỉnh sửa → TTS.
Mỗi khâu lưu file đặt theo TÊN VIDEO gốc (hoặc tên file dữ liệu nguồn khi xử
lý độc lập), với phần mở rộng/hậu tố tương ứng:

    - Trích xuất / chỉnh sửa:  <tên>.original.srt  hoặc  <tên>.original.ass
    - Biên dịch:               <tên>.translate.<lang>.srt / .ass
    - TTS (đã chỉnh giờ):      <tên>.tts.<lang>.srt
    - TTS (âm thanh):          <tên>.wav

Giúp các file của cùng một video nằm cạnh nhau, dễ nhận biết và liên thông.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path


class SubtitleFormat(str, Enum):
    """Định dạng phụ đề người dùng chọn trong cài đặt."""

    SRT = "srt"
    ASS = "ass"

    @classmethod
    def from_str(cls, value: str | None) -> "SubtitleFormat":
        """Phân giải chuỗi (không phân biệt hoa thường) về enum, mặc định SRT."""
        if value and value.strip().lower() == "ass":
            return cls.ASS
        return cls.SRT


def _base_stem(source_path: str | Path) -> tuple[Path, str]:
    """Trả về (thư mục, tên gốc không phần mở rộng) của file nguồn."""
    path = Path(source_path)
    return path.parent, path.stem


def extracted_subtitle_path(
    source_path: str | Path, fmt: SubtitleFormat = SubtitleFormat.SRT
) -> Path:
    """Đường dẫn phụ đề gốc (trích xuất/biên tập): ``<tên>.original.srt``.

    Hậu tố ``.original.`` tách bạch khỏi bản dịch ``.translate.`` và bản TTS
    ``.tts.`` để các bước KHÔNG vô tình ghi đè file của nhau.
    """
    parent, stem = _base_stem(source_path)
    return parent / f"{stem}.original.{fmt.value}"


def translated_subtitle_path(
    source_path: str | Path,
    target_lang: str,
    fmt: SubtitleFormat = SubtitleFormat.SRT,
) -> Path:
    """Đường dẫn phụ đề đã dịch: ``<tên>.translate.<lang>.srt``.

    Hậu tố ``.translate.`` để KHÔNG ghi đè phụ đề gốc ``<tên>.original.srt``.

    Args:
        target_lang: Mã ngôn ngữ đích (vd "vi", "en"). Lấy phần gốc trước dấu
            gạch (vd "vi-VN" → "vi") để hậu tố gọn.
    """
    parent, stem = _base_stem(source_path)
    lang_code = (target_lang or "vi").split("-")[0].lower()
    return parent / f"{stem}.translate.{lang_code}.{fmt.value}"


def raw_ocr_cache_path(source_path: str | Path) -> Path:
    """Đường dẫn cache OCR thô: ``<tên>.seraw.json.gz`` (nén sẵn).

    [v3.23.322] OCR là khâu TỐN THỜI GIAN NHẤT (hàng chục phút mỗi tập), còn dựng câu
    phụ đề chỉ mất vài giây. Lưu lại kết quả OCR cho phép đổi tham số dựng câu rồi
    **dựng lại tức thì** thay vì chạy lại OCR từ đầu.

    Dùng ``.gz`` vì dữ liệu OCR một tập rất lớn; serializer tự nén khi thấy đuôi này.
    """
    parent, stem = _base_stem(source_path)
    return parent / f"{stem}.seraw.json.gz"


def tts_subtitle_path(source_path: str | Path, target_lang: str = "") -> Path:
    """Đường dẫn phụ đề đã chỉnh giờ khớp giọng TTS: ``<tên>.tts.<lang>.srt``.

    Tách khỏi phụ đề gốc/bản dịch để không ghi đè lẫn nhau. Khi biết ngôn ngữ,
    chèn mã ngôn ngữ để phân biệt bản TTS của từng ngôn ngữ.

    Args:
        target_lang: Mã ngôn ngữ (vd "vi", "en-US"). Rỗng → ``<tên>.tts.srt``.
    """
    parent, stem = _base_stem(source_path)
    lang_code = (target_lang or "").split("-")[0].lower()
    if lang_code:
        return parent / f"{stem}.tts.{lang_code}.srt"
    return parent / f"{stem}.tts.srt"


def tts_audio_path(source_path: str | Path) -> Path:
    """Đường dẫn file âm thanh TTS: ``<tên>.wav``."""
    parent, stem = _base_stem(source_path)
    return parent / f"{stem}.wav"


__all__ = [
    "SubtitleFormat",
    "extracted_subtitle_path",
    "translated_subtitle_path",
    "tts_subtitle_path",
    "tts_audio_path",
]
