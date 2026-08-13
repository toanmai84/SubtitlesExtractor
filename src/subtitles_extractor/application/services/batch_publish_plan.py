"""Lập kế hoạch XUẤT BẢN HÀNG LOẠT cho phim bộ (thuần, không phụ thuộc giao diện).

VÌ SAO cần
==========
Xử lý hàng loạt hiện chỉ có ở khâu **trích xuất** (v3.23.319). Bốn khâu còn lại —
Biên tập, Dịch, TTS, Xuất bản — vẫn phải làm thủ công từng tập. Với bộ 84 tập, riêng
khâu xuất bản là 84 lần lặp thao tác y hệt nhau.

Xuất bản là khâu **đáng tự động hoá nhất** trong bốn khâu đó:

* Mọi đầu vào đã là TỆP có sẵn khi tới bước này (video, phụ đề TTS, giọng đọc).
* Chạy rất nhanh khi ghép mềm + thuyết minh (không mã hoá lại hình).
* Không cần khoá API, không nạp mô hình — nên chạy nền hàng chục tập là an toàn.

Module này dò các tập đã sẵn sàng theo quy ước đặt tên của dự án và lập kế hoạch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from subtitles_extractor.domain.value_objects.output_naming import (
    tts_audio_path,
    tts_subtitle_path,
)

logger = logging.getLogger(__name__)

#: Phần mở rộng video mà ứng dụng xử lý.
VIDEO_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".mp4", ".mkv", ".avi", ".mov", ".ts", ".webm", ".m4v", ".flv"}
)

#: Phần mở rộng tệp giọng đọc có thể có (TTS ghi ra .flac hoặc .wav).
_AUDIO_SUFFIXES: Final[tuple[str, ...]] = (".flac", ".wav", ".m4a", ".mp3")

#: Tệp do chính ứng dụng xuất ra — KHÔNG được coi là video nguồn.
_GENERATED_MARKERS: Final[tuple[str, ...]] = (
    "_phude", "_phudechay", "_thuyetminh", "_tiengviet", "_xuatban",
)


class PublishItemStatus(StrEnum):
    """Trạng thái một tập trong kế hoạch xuất bản."""

    READY = "ready"
    """Đủ dữ liệu, sẵn sàng xuất."""

    MISSING_SUBTITLE = "missing_subtitle"
    """Chưa có phụ đề (cần chạy khâu Dịch/TTS trước)."""

    MISSING_AUDIO = "missing_audio"
    """Chưa có giọng đọc (cần chạy khâu TTS trước)."""

    ALREADY_DONE = "already_done"
    """Đã có tệp đích — bỏ qua để khỏi làm lại."""


@dataclass(frozen=True, slots=True)
class PublishItem:
    """Một tập trong kế hoạch xuất bản hàng loạt.

    Attributes:
        video_path: Video nguồn.
        subtitle_path: Phụ đề tìm được (``None`` nếu không có).
        audio_path: Giọng đọc tìm được (``None`` nếu không có).
        output_path: Tệp đích sẽ ghi ra.
        status: Trạng thái.
        note: Ghi chú hiển thị cho người dùng.
    """

    video_path: Path
    subtitle_path: Path | None
    audio_path: Path | None
    output_path: Path
    status: PublishItemStatus
    note: str = ""

    @property
    def will_run(self) -> bool:
        """``True`` nếu tập này sẽ được xuất."""
        return self.status is PublishItemStatus.READY


def is_source_video(path: Path) -> bool:
    """``True`` nếu tệp là video NGUỒN (không phải bản do app xuất ra).

    Quét thư mục sẽ bắt gặp cả tệp đã xuất bản (``Tap01_phude_thuyetminh.mkv``); nếu
    coi chúng là nguồn thì lần chạy sau sẽ xuất bản chồng lên chính kết quả cũ.

    Args:
        path: Đường dẫn cần xét.

    Returns:
        ``True`` nếu là video nguồn.
    """
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        return False
    stem = path.stem.lower()
    return not any(marker in stem for marker in _GENERATED_MARKERS)


def find_episode_videos(folder: Path) -> list[Path]:
    """Tìm mọi video nguồn trong thư mục, sắp theo tên.

    Args:
        folder: Thư mục chứa các tập.

    Returns:
        Danh sách video nguồn đã sắp xếp; rỗng nếu thư mục không đọc được.
    """
    try:
        entries = [p for p in folder.iterdir() if p.is_file() and is_source_video(p)]
    except OSError as exc:
        logger.warning("Không đọc được thư mục %s: %s", folder, exc)
        return []
    return sorted(entries, key=lambda p: p.name)


def find_tts_subtitle(video_path: Path, target_language: str = "") -> Path | None:
    """Tìm phụ đề đã chỉnh giờ theo giọng đọc của một tập.

    Dùng cùng quy tắc với :mod:`...publish_subtitle_selector`: thử biến thể có mã ngôn
    ngữ trước, rồi quét mẫu ``*.tts.*.srt`` để không phụ thuộc việc biết trước mã.

    Args:
        video_path: Video của tập.
        target_language: Mã ngôn ngữ đích (nếu biết).

    Returns:
        Đường dẫn phụ đề, hoặc ``None``.
    """
    candidates: list[Path] = []
    if target_language:
        candidates.append(tts_subtitle_path(video_path, target_language))
    candidates.append(tts_subtitle_path(video_path, ""))
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue

    reference = tts_subtitle_path(video_path, "")
    stem = reference.name[: -len(".tts.srt")]
    try:
        matches = sorted(reference.parent.glob(f"{stem}.tts.*.srt"))
    except OSError:
        return None
    return matches[0] if matches else None


def find_tts_audio(video_path: Path) -> Path | None:
    """Tìm tệp giọng đọc của một tập (thử các phần mở rộng TTS hay dùng).

    Args:
        video_path: Video của tập.

    Returns:
        Đường dẫn giọng đọc, hoặc ``None``.
    """
    preferred = tts_audio_path(video_path)
    try:
        if preferred.is_file():
            return preferred
    except OSError:
        pass
    for suffix in _AUDIO_SUFFIXES:
        candidate = preferred.with_suffix(suffix)
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def build_publish_plan(
    videos: list[Path],
    *,
    output_suffix: str,
    needs_subtitle: bool,
    needs_audio: bool,
    target_language: str = "",
    skip_existing: bool = True,
) -> list[PublishItem]:
    """Lập kế hoạch xuất bản cho danh sách tập.

    Args:
        videos: Các video nguồn.
        output_suffix: Hậu tố tên tệp đích (vd ``"_phude_thuyetminh"``).
        needs_subtitle: Lựa chọn hiện tại có cần phụ đề không.
        needs_audio: Lựa chọn hiện tại có cần giọng đọc không.
        target_language: Mã ngôn ngữ đích, để tìm đúng biến thể phụ đề.
        skip_existing: Bỏ qua tập đã có tệp đích.

    Returns:
        Danh sách :class:`PublishItem` theo đúng thứ tự đầu vào.
    """
    plan: list[PublishItem] = []

    for video in videos:
        output = video.with_name(f"{video.stem}{output_suffix}.mkv")
        subtitle = find_tts_subtitle(video, target_language) if needs_subtitle else None
        audio = find_tts_audio(video) if needs_audio else None

        if skip_existing and output.is_file():
            plan.append(
                PublishItem(video, subtitle, audio, output,
                            PublishItemStatus.ALREADY_DONE,
                            f"Đã có {output.name}")
            )
            continue

        if needs_subtitle and subtitle is None:
            plan.append(
                PublishItem(video, None, audio, output,
                            PublishItemStatus.MISSING_SUBTITLE,
                            "Chưa có phụ đề đồng bộ — chạy khâu TTS cho tập này trước.")
            )
            continue

        if needs_audio and audio is None:
            plan.append(
                PublishItem(video, subtitle, None, output,
                            PublishItemStatus.MISSING_AUDIO,
                            "Chưa có giọng đọc — chạy khâu TTS cho tập này trước.")
            )
            continue

        plan.append(PublishItem(video, subtitle, audio, output, PublishItemStatus.READY))

    return plan


def summarise_publish_plan(plan: list[PublishItem]) -> str:
    """Tóm tắt kế hoạch thành một dòng để hiển thị.

    Args:
        plan: Kế hoạch đã lập.

    Returns:
        Chuỗi dạng ``"12 tập sẽ xuất · 3 thiếu giọng đọc · 2 đã có"``.
    """
    ready = sum(1 for item in plan if item.will_run)
    done = sum(1 for item in plan if item.status is PublishItemStatus.ALREADY_DONE)
    no_subtitle = sum(
        1 for item in plan if item.status is PublishItemStatus.MISSING_SUBTITLE
    )
    no_audio = sum(1 for item in plan if item.status is PublishItemStatus.MISSING_AUDIO)

    parts = [f"{ready} tập sẽ xuất"]
    if no_subtitle:
        parts.append(f"{no_subtitle} thiếu phụ đề")
    if no_audio:
        parts.append(f"{no_audio} thiếu giọng đọc")
    if done:
        parts.append(f"{done} đã có")
    return " · ".join(parts)


__all__ = [
    "VIDEO_SUFFIXES",
    "PublishItem",
    "PublishItemStatus",
    "build_publish_plan",
    "find_episode_videos",
    "find_tts_audio",
    "find_tts_subtitle",
    "is_source_video",
    "summarise_publish_plan",
]
