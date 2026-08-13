"""Chọn ĐÚNG tệp phụ đề để xuất bản (thuần, không phụ thuộc giao diện).

VẤN ĐỀ ĐƯỢC SỬA (v3.23.318)
===========================
Khâu TTS không chỉ tạo giọng đọc — nó còn **chỉnh lại mốc thời gian** phụ đề cho khớp
lời thoại đã tổng hợp (nén/giãn câu, dời câu khi thiếu chỗ), rồi ghi ra tệp riêng
``<tên>.tts.<lang>.srt`` (xem :func:`...output_naming.tts_subtitle_path`).

Nhưng trang Xuất bản lại lấy phụ đề từ **trang Dịch** — tức mốc thời gian GỐC, chưa
chỉnh. Hệ quả: xuất phim kèm cả phụ đề lẫn thuyết minh thì **phụ đề lệch với giọng
nói**, càng về sau càng lệch (drift tích luỹ).

Module này quyết định nguồn phụ đề theo thứ tự ưu tiên đúng đắn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from subtitles_extractor.domain.value_objects.output_naming import tts_subtitle_path

logger = logging.getLogger(__name__)


class SubtitleSource(StrEnum):
    """Nguồn phụ đề được chọn để xuất bản."""

    TTS_SYNCED = "tts_synced"
    """Phụ đề đã chỉnh giờ khớp giọng TTS — ưu tiên cao nhất khi đã tạo giọng đọc."""

    TRANSLATED = "translated"
    """Bản dịch với mốc thời gian gốc — dùng khi chưa chạy TTS."""

    NONE = "none"
    """Không có phụ đề nào."""


@dataclass(frozen=True, slots=True)
class SubtitleChoice:
    """Kết quả chọn nguồn phụ đề.

    Attributes:
        path: Đường dẫn tệp phụ đề nên dùng (``None`` nếu không có).
        source: Nguồn đã chọn.
        warning: Cảnh báo cần hiển thị cho người dùng (``None`` nếu không có).
    """

    path: Path | None
    source: SubtitleSource
    warning: str | None = None


def find_tts_synced_subtitle(
    tts_audio_path: Path | None, target_language: str = ""
) -> Path | None:
    """Tìm tệp phụ đề đã chỉnh giờ khớp giọng TTS, nếu có.

    [v3.23.323] SỬA LỖI THẬT: bản trước ĐOÁN mã ngôn ngữ từ ``ProjectRecord.target_lang``
    rồi chỉ thử đúng hai tên tệp. Khi ``target_lang`` rỗng (dự án chưa qua khâu Dịch
    trong phiên đó), hàm chỉ thử ``<tên>.tts.srt`` và BỎ SÓT tệp thật
    ``<tên>.tts.vi.srt`` — dẫn tới xuất bản dùng phụ đề lệch mà vẫn báo "không tìm thấy".

    Nay QUÉT MẪU ``<tên>.tts.*.srt`` nên không phụ thuộc việc biết trước mã ngôn ngữ.

    Args:
        tts_audio_path: Đường dẫn tệp âm thanh TTS (``None`` nếu chưa chạy TTS).
        target_language: Mã ngôn ngữ đích — nếu biết thì được ƯU TIÊN khi có nhiều tệp.

    Returns:
        Đường dẫn tệp phụ đề đồng bộ TTS nếu tồn tại; ``None`` nếu không.
    """
    if tts_audio_path is None:
        return None

    # 1) Ưu tiên đúng mã ngôn ngữ nếu biết, rồi tới bản không mã.
    preferred: list[Path] = []
    if target_language:
        preferred.append(tts_subtitle_path(tts_audio_path, target_language))
    preferred.append(tts_subtitle_path(tts_audio_path, ""))
    for candidate in preferred:
        try:
            if candidate.is_file():
                return candidate
        except OSError as exc:
            logger.debug("Bỏ qua ứng viên phụ đề TTS %s: %s.", candidate, exc)

    # 2) Không đoán được -> QUÉT mọi biến thể ngôn ngữ thực có trên đĩa.
    reference = tts_subtitle_path(tts_audio_path, "")
    stem = reference.name[: -len(".tts.srt")]
    try:
        matches = sorted(reference.parent.glob(f"{stem}.tts.*.srt"))
    except OSError as exc:
        logger.debug("Không quét được thư mục %s: %s.", reference.parent, exc)
        return None
    if matches:
        if len(matches) > 1:
            logger.info(
                "Có %d biến thể phụ đề đồng bộ TTS, dùng %s.",
                len(matches), matches[0].name,
            )
        return matches[0]
    return None


def choose_publish_subtitle(
    *,
    tts_audio_path: Path | None,
    translated_subtitle_path: Path | None,
    target_language: str = "",
    audio_will_be_used: bool = False,
) -> SubtitleChoice:
    """Chọn tệp phụ đề nên dùng khi xuất bản.

    Thứ tự ưu tiên:
        1. **Phụ đề đồng bộ TTS** — khi đã chạy TTS và tệp tồn tại. Đây là bản duy nhất
           khớp với giọng đọc.
        2. **Bản dịch** — khi chưa chạy TTS.

    Args:
        tts_audio_path: Tệp âm thanh TTS đã tạo (``None`` nếu chưa chạy).
        translated_subtitle_path: Tệp phụ đề bản dịch (mốc thời gian gốc).
        target_language: Mã ngôn ngữ đích, để tìm đúng biến thể tệp.
        audio_will_be_used: ``True`` khi bản xuất có kèm giọng đọc (thuyết minh hoặc
            track tiếng Việt) — lúc đó dùng nhầm phụ đề gốc sẽ gây LỆCH rõ rệt.

    Returns:
        :class:`SubtitleChoice` gồm đường dẫn, nguồn và cảnh báo (nếu có).
    """
    synced = find_tts_synced_subtitle(tts_audio_path, target_language)
    if synced is not None:
        return SubtitleChoice(synced, SubtitleSource.TTS_SYNCED)

    if translated_subtitle_path is not None:
        warning = None
        if audio_will_be_used and tts_audio_path is not None:
            # Có giọng đọc nhưng KHÔNG tìm thấy phụ đề đã chỉnh giờ -> nguy cơ lệch.
            warning = (
                "Không tìm thấy phụ đề đã chỉnh giờ theo giọng đọc. Đang dùng bản dịch "
                "với mốc thời gian gốc — phụ đề có thể LỆCH so với giọng thuyết minh. "
                "Hãy chạy lại khâu TTS để tạo tệp phụ đề đồng bộ."
            )
        return SubtitleChoice(translated_subtitle_path, SubtitleSource.TRANSLATED, warning)

    return SubtitleChoice(None, SubtitleSource.NONE)


__all__ = [
    "SubtitleChoice",
    "SubtitleSource",
    "choose_publish_subtitle",
    "find_tts_synced_subtitle",
]
