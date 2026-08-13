"""Test tên tệp đích của trang Xuất bản — v3.23.326.

HAI LỖI THẬT ĐƯỢC SỬA:

1. **Đổi chế độ mà tên không đổi.** Log thực tế: xuất ``voice_over`` nhưng tệp ra tên
   ``第19集_phude.mkv`` (hậu tố của kiểu phụ đề rời).
2. **Đổi video gốc mà tên đích không đổi** — xuất phim B vào tệp mang tên A; bấm xuất
   lần nữa sẽ GHI ĐÈ kết quả của phim A.

Logic đặt tên được mô phỏng ở đây (không cần Qt) đúng như trang cài đặt.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.video.video_render_command import (
    AudioMode,
    SubtitleMode,
)

_SUBTITLE_TAGS = {
    SubtitleMode.NONE: "",
    SubtitleMode.SOFT: "_phude",
    SubtitleMode.BURNED: "_phudechay",
}
_AUDIO_TAGS = {
    AudioMode.ORIGINAL: "",
    AudioMode.VOICE_OVER: "_thuyetminh",
    AudioMode.REPLACE_TRACK: "_tiengviet",
}


def default_output_path(
    video_text: str, subtitle_mode: SubtitleMode, audio_mode: AudioMode
) -> str:
    """Tên tệp đích gợi ý — cùng quy tắc với trang Xuất bản."""
    if not video_text:
        return ""
    video = Path(video_text)
    suffix = f"{_SUBTITLE_TAGS[subtitle_mode]}{_AUDIO_TAGS[audio_mode]}" or "_xuatban"
    return str(video.with_name(f"{video.stem}{suffix}.mkv"))


def all_suggested_paths(video_text: str) -> set[str]:
    """Mọi tên tệp hệ thống có thể đã gợi ý cho một video."""
    if not video_text:
        return set()
    video = Path(video_text)
    results = {
        str(video.with_name(f"{video.stem}{s}{a}.mkv"))
        for s in _SUBTITLE_TAGS.values()
        for a in _AUDIO_TAGS.values()
    }
    results.add(str(video.with_name(f"{video.stem}_xuatban.mkv")))
    results.add(str(video.with_name(f"{video.stem}_phude_chay.mkv")))
    return results


class _PageModel:
    """Mô phỏng phần trạng thái tên tệp của trang Xuất bản."""

    def __init__(self) -> None:
        self.video = ""
        self.output = ""
        self.output_source = ""
        self.subtitle_mode = SubtitleMode.SOFT
        self.audio_mode = AudioMode.ORIGINAL

    def refresh(self) -> None:
        if self.output and self.output not in all_suggested_paths(self.output_source):
            return  # người dùng tự gõ -> giữ nguyên
        self.output = default_output_path(
            self.video, self.subtitle_mode, self.audio_mode
        )
        self.output_source = self.video

    def set_video(self, video: str) -> None:
        self.video = video
        self.refresh()

    def set_modes(self, subtitle_mode: SubtitleMode, audio_mode: AudioMode) -> None:
        self.subtitle_mode = subtitle_mode
        self.audio_mode = audio_mode
        self.refresh()


# ── Lỗi 1: đổi chế độ ────────────────────────────────────────────────────────
def test_changing_mode_updates_filename() -> None:
    page = _PageModel()
    page.set_video("D:/Phim/第19集.mp4")
    assert Path(page.output).name == "第19集_phude.mkv"

    page.set_modes(SubtitleMode.NONE, AudioMode.VOICE_OVER)
    assert Path(page.output).name == "第19集_thuyetminh.mkv"


def test_combined_mode_gets_combined_name() -> None:
    """Tổ hợp mới (phụ đề + thuyết minh) phải có tên phản ánh cả hai."""
    page = _PageModel()
    page.set_video("D:/Phim/Tap01.mp4")
    page.set_modes(SubtitleMode.BURNED, AudioMode.VOICE_OVER)
    assert Path(page.output).name == "Tap01_phudechay_thuyetminh.mkv"


# ── Lỗi 2: đổi video gốc ─────────────────────────────────────────────────────
def test_changing_source_video_updates_filename() -> None:
    """Đổi video mà tên đích không đổi sẽ GHI ĐÈ kết quả của phim trước."""
    page = _PageModel()
    page.set_video("D:/Phim/第19集.mp4")
    page.set_modes(SubtitleMode.SOFT, AudioMode.VOICE_OVER)
    assert "第19集" in page.output

    page.set_video("D:/Phim/第21集.mp4")
    assert "第21集" in page.output
    assert "第19集" not in page.output
    assert Path(page.output).name == "第21集_phude_thuyetminh.mkv"


def test_switching_between_episodes_never_reuses_name() -> None:
    """Chạy hàng loạt nhiều tập: mỗi tập phải ra một tên riêng."""
    page = _PageModel()
    page.set_modes(SubtitleMode.SOFT, AudioMode.VOICE_OVER)
    seen: set[str] = set()
    for index in range(1, 6):
        page.set_video(f"D:/Phim/Tap{index:02d}.mp4")
        assert page.output not in seen
        seen.add(page.output)
    assert len(seen) == 5


# ── Tên người dùng tự đặt phải được tôn trọng ────────────────────────────────
def test_user_typed_name_survives_mode_change() -> None:
    page = _PageModel()
    page.set_video("D:/Phim/Tap01.mp4")
    page.output = "D:/Xuat/ban_cuoi.mkv"
    page.set_modes(SubtitleMode.BURNED, AudioMode.VOICE_OVER)
    assert page.output == "D:/Xuat/ban_cuoi.mkv"


def test_user_typed_name_survives_video_change() -> None:
    page = _PageModel()
    page.set_video("D:/Phim/Tap01.mp4")
    page.output = "D:/Xuat/ban_cuoi.mkv"
    page.set_video("D:/Phim/Tap02.mp4")
    assert page.output == "D:/Xuat/ban_cuoi.mkv"


# ── Không tổ hợp nào trùng tên ───────────────────────────────────────────────
def test_every_combination_has_distinct_name() -> None:
    """Trùng tên nghĩa là hai kiểu xuất ghi đè lên nhau."""
    names = [
        default_output_path("D:/Phim/Tap01.mp4", subtitle_mode, audio_mode)
        for subtitle_mode in SubtitleMode
        for audio_mode in AudioMode
    ]
    assert len(names) == 9
    assert len(set(names)) == 9


def test_no_video_gives_empty_name() -> None:
    assert default_output_path("", SubtitleMode.SOFT, AudioMode.ORIGINAL) == ""


@pytest.mark.parametrize("legacy", ["_phude_chay", "_phude", "_thuyetminh"])
def test_legacy_suggested_names_are_replaceable(legacy: str) -> None:
    """Tên do bản CŨ gợi ý cũng phải được coi là tự sinh (thay được)."""
    video = "D:/Phim/Tap01.mp4"
    assert str(Path(video).with_name(f"Tap01{legacy}.mkv")) in all_suggested_paths(video)
