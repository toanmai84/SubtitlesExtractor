"""Lập kế hoạch TỔNG HỢP GIỌNG ĐỌC HÀNG LOẠT cho phim bộ (thuần, không phụ thuộc Qt).

VÌ SAO cần
==========
Sau v3.23.329, xử lý hàng loạt phủ được khâu Trích xuất và Xuất bản. Khâu **TTS** vẫn
phải làm thủ công từng tập — với bộ 84 tập là 84 lần lặp thao tác y hệt.

TTS đáng tự động hoá vì:

* Đầu vào là TỆP phụ đề đã dịch, có sẵn sau khâu Dịch.
* Chạy cục bộ (VieNeu) nên không vướng giới hạn tốc độ dịch vụ ngoài.
* **Mô hình chỉ nạp một lần** rồi dùng cho cả bộ — chạy hàng loạt còn nhanh hơn nhiều
  so với mở lại từng tập (mỗi lần mở lại phải nạp mô hình ~15 giây).

Module này dò các tập đã có bản dịch và lập kế hoạch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from subtitles_extractor.application.services.batch_publish_plan import (
    find_episode_videos,
)
from subtitles_extractor.domain.value_objects.output_naming import (
    tts_audio_path,
    translated_subtitle_path,
)

logger = logging.getLogger(__name__)

#: Phần mở rộng tệp giọng đọc mà khâu TTS có thể đã ghi ra.
_AUDIO_SUFFIXES: tuple[str, ...] = (".flac", ".wav", ".m4a", ".mp3")


class TtsItemStatus(StrEnum):
    """Trạng thái một tập trong kế hoạch TTS."""

    READY = "ready"
    """Có bản dịch, sẵn sàng tổng hợp."""

    MISSING_TRANSLATION = "missing_translation"
    """Chưa có bản dịch — cần chạy khâu Dịch trước."""

    ALREADY_DONE = "already_done"
    """Đã có tệp giọng đọc — bỏ qua."""


@dataclass(frozen=True, slots=True)
class TtsItem:
    """Một tập trong kế hoạch TTS hàng loạt.

    Attributes:
        video_path: Video của tập (dùng để suy tên các tệp liên quan).
        subtitle_path: Phụ đề bản dịch (``None`` nếu chưa có).
        output_path: Tệp giọng đọc sẽ ghi ra.
        status: Trạng thái.
        note: Ghi chú cho người dùng.
    """

    video_path: Path
    subtitle_path: Path | None
    output_path: Path
    status: TtsItemStatus
    note: str = ""

    @property
    def will_run(self) -> bool:
        """``True`` nếu tập này sẽ được tổng hợp."""
        return self.status is TtsItemStatus.READY


def find_translated_subtitle(
    video_path: Path, target_language: str = "vi"
) -> Path | None:
    """Tìm phụ đề BẢN DỊCH của một tập.

    Thử đúng mã ngôn ngữ trước, rồi quét mẫu ``*.translate.*.srt`` — cùng bài học với
    v3.23.323: đừng phụ thuộc việc biết trước mã ngôn ngữ, vì trường ``target_lang``
    của dự án có thể rỗng khi chưa qua khâu Dịch trong phiên đó.

    Args:
        video_path: Video của tập.
        target_language: Mã ngôn ngữ đích.

    Returns:
        Đường dẫn phụ đề bản dịch, hoặc ``None``.
    """
    preferred = translated_subtitle_path(video_path, target_language or "vi")
    try:
        if preferred.is_file():
            return preferred
    except OSError:
        pass

    try:
        matches = sorted(preferred.parent.glob(f"{video_path.stem}.translate.*.srt"))
    except OSError as exc:
        logger.debug("Không quét được thư mục %s: %s", preferred.parent, exc)
        return None
    return matches[0] if matches else None


def find_existing_audio(video_path: Path) -> Path | None:
    """Tìm tệp giọng đọc đã tổng hợp trước đó (nếu có).

    Args:
        video_path: Video của tập.

    Returns:
        Đường dẫn tệp giọng đọc, hoặc ``None``.
    """
    base = tts_audio_path(video_path)
    for suffix in _AUDIO_SUFFIXES:
        candidate = base.with_suffix(suffix)
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def build_tts_plan(
    videos: list[Path],
    *,
    target_language: str = "vi",
    output_suffix: str = ".wav",
    skip_existing: bool = True,
) -> list[TtsItem]:
    """Lập kế hoạch tổng hợp giọng đọc cho danh sách tập.

    Args:
        videos: Các video nguồn.
        target_language: Mã ngôn ngữ của bản dịch cần tìm.
        output_suffix: Phần mở rộng tệp giọng đọc sẽ ghi (``.wav`` hoặc ``.flac``).
        skip_existing: Bỏ qua tập đã có giọng đọc.

    Returns:
        Danh sách :class:`TtsItem` theo đúng thứ tự đầu vào.
    """
    plan: list[TtsItem] = []

    for video in videos:
        output = tts_audio_path(video).with_suffix(output_suffix)
        subtitle = find_translated_subtitle(video, target_language)

        if skip_existing:
            existing = find_existing_audio(video)
            if existing is not None:
                plan.append(
                    TtsItem(video, subtitle, existing, TtsItemStatus.ALREADY_DONE,
                            f"Đã có {existing.name}")
                )
                continue

        if subtitle is None:
            plan.append(
                TtsItem(video, None, output, TtsItemStatus.MISSING_TRANSLATION,
                        "Chưa có bản dịch — chạy khâu Dịch cho tập này trước.")
            )
            continue

        plan.append(TtsItem(video, subtitle, output, TtsItemStatus.READY))

    return plan


def summarise_tts_plan(plan: list[TtsItem]) -> str:
    """Tóm tắt kế hoạch thành một dòng để hiển thị.

    Args:
        plan: Kế hoạch đã lập.

    Returns:
        Chuỗi dạng ``"12 tập sẽ tổng hợp · 3 chưa dịch · 2 đã có"``.
    """
    ready = sum(1 for item in plan if item.will_run)
    missing = sum(
        1 for item in plan if item.status is TtsItemStatus.MISSING_TRANSLATION
    )
    done = sum(1 for item in plan if item.status is TtsItemStatus.ALREADY_DONE)

    parts = [f"{ready} tập sẽ tổng hợp"]
    if missing:
        parts.append(f"{missing} chưa dịch")
    if done:
        parts.append(f"{done} đã có")
    return " · ".join(parts)


def estimate_batch_minutes(plan: list[TtsItem], seconds_per_episode: float = 60.0) -> float:
    """Ước lượng thời gian chạy cả lô (phút).

    Ước lượng thô để người dùng biết có nên chạy qua đêm hay không. KHÔNG cộng thời
    gian nạp mô hình vì chạy hàng loạt chỉ nạp MỘT LẦN cho cả bộ.

    Args:
        plan: Kế hoạch đã lập.
        seconds_per_episode: Thời gian trung bình mỗi tập (giây).

    Returns:
        Số phút ước tính.
    """
    return sum(1 for item in plan if item.will_run) * seconds_per_episode / 60.0


__all__ = [
    "TtsItem",
    "TtsItemStatus",
    "build_tts_plan",
    "estimate_batch_minutes",
    "find_episode_videos",
    "find_existing_audio",
    "find_translated_subtitle",
    "summarise_tts_plan",
]
