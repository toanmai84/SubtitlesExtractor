"""Test lập kế hoạch xuất bản hàng loạt cho phim bộ — v3.23.329.

KHOẢNG TRỐNG ĐƯỢC LẤP: xử lý hàng loạt trước đây CHỈ có ở khâu trích xuất. Bốn khâu
sau vẫn thủ công từng tập — với bộ 84 tập thì riêng khâu xuất bản là 84 lần lặp.

Bẫy quan trọng nhất được canh giữ ở đây: **không được nhận nhầm tệp do chính ứng dụng
xuất ra làm video nguồn** — nếu nhầm, lần chạy sau sẽ xuất bản chồng lên kết quả cũ.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.application.services.batch_publish_plan import (
    PublishItemStatus,
    build_publish_plan,
    find_episode_videos,
    find_tts_audio,
    find_tts_subtitle,
    is_source_video,
    summarise_publish_plan,
)


@pytest.fixture
def series(tmp_path: Path) -> Path:
    """Thư mục phim bộ 6 tập ở các mức hoàn thiện khác nhau."""
    for index in range(1, 7):
        (tmp_path / f"第{index}集.mp4").write_bytes(b"\x00")
    # Tập 1–4: đã có phụ đề TTS và giọng đọc.
    for index in range(1, 5):
        (tmp_path / f"第{index}集.tts.vi.srt").write_text("1\n", encoding="utf-8")
        (tmp_path / f"第{index}集.flac").write_bytes(b"\x00")
    # Tập 5: có phụ đề nhưng thiếu giọng đọc.
    (tmp_path / "第5集.tts.vi.srt").write_text("1\n", encoding="utf-8")
    # Tập 2: đã xuất bản rồi.
    (tmp_path / "第2集_phude_thuyetminh.mkv").write_bytes(b"\x00")
    # Nhiễu: tệp do app xuất ra + tệp không phải video.
    (tmp_path / "第1集_thuyetminh.mkv").write_bytes(b"\x00")
    (tmp_path / "ghichu.txt").write_text("x", encoding="utf-8")
    return tmp_path


# ── Nhận diện video nguồn ────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Tap01.mp4", True),
        ("第19集.mkv", True),
        ("phim.avi", True),
        ("Tap01_phude.mkv", False),          # do app xuất
        ("Tap01_thuyetminh.mkv", False),
        ("Tap01_phudechay_thuyetminh.mkv", False),
        ("Tap01_tiengviet.mkv", False),
        ("Tap01_xuatban.mkv", False),
        ("ghichu.txt", False),
        ("Tap01.srt", False),
    ],
)
def test_source_video_detection(name: str, expected: bool) -> None:
    """Tệp app tự xuất KHÔNG được coi là nguồn — nếu nhầm sẽ xuất chồng lên kết quả."""
    assert is_source_video(Path(name)) is expected


def test_find_videos_skips_generated_files(series: Path) -> None:
    videos = find_episode_videos(series)
    assert len(videos) == 6
    assert all("_thuyetminh" not in v.name for v in videos)


def test_find_videos_is_sorted(series: Path) -> None:
    """Thứ tự ổn định để người dùng theo dõi được tiến độ."""
    videos = find_episode_videos(series)
    assert videos == sorted(videos, key=lambda p: p.name)


def test_missing_folder_returns_empty(tmp_path: Path) -> None:
    assert find_episode_videos(tmp_path / "khong-co") == []


# ── Tìm phụ đề và giọng đọc ──────────────────────────────────────────────────
def test_finds_subtitle_without_knowing_language(series: Path) -> None:
    """Quét mẫu ``*.tts.*.srt`` — không phụ thuộc việc biết trước mã ngôn ngữ."""
    found = find_tts_subtitle(series / "第1集.mp4", "")
    assert found is not None
    assert found.name == "第1集.tts.vi.srt"


def test_finds_audio_by_extension(series: Path) -> None:
    found = find_tts_audio(series / "第1集.mp4")
    assert found is not None
    assert found.suffix == ".flac"


def test_returns_none_when_absent(series: Path) -> None:
    assert find_tts_subtitle(series / "第6集.mp4", "vi") is None
    assert find_tts_audio(series / "第5集.mp4") is None


# ── Lập kế hoạch ─────────────────────────────────────────────────────────────
def test_plan_classifies_every_episode(series: Path) -> None:
    plan = build_publish_plan(
        find_episode_videos(series), output_suffix="_phude_thuyetminh",
        needs_subtitle=True, needs_audio=True,
    )
    statuses = {item.video_path.name: item.status for item in plan}
    assert statuses["第1集.mp4"] is PublishItemStatus.READY
    assert statuses["第2集.mp4"] is PublishItemStatus.ALREADY_DONE
    assert statuses["第5集.mp4"] is PublishItemStatus.MISSING_AUDIO
    assert statuses["第6集.mp4"] is PublishItemStatus.MISSING_SUBTITLE


def test_only_ready_items_run(series: Path) -> None:
    plan = build_publish_plan(
        find_episode_videos(series), output_suffix="_phude_thuyetminh",
        needs_subtitle=True, needs_audio=True,
    )
    assert sum(1 for item in plan if item.will_run) == 3  # tập 1, 3, 4


def test_audio_not_required_widens_eligibility(series: Path) -> None:
    """Chỉ ghép phụ đề thì tập thiếu giọng đọc vẫn xuất được."""
    plan = build_publish_plan(
        find_episode_videos(series), output_suffix="_phude",
        needs_subtitle=True, needs_audio=False,
    )
    assert sum(1 for item in plan if item.will_run) == 5


def test_can_force_rerun_existing(series: Path) -> None:
    plan = build_publish_plan(
        find_episode_videos(series), output_suffix="_phude_thuyetminh",
        needs_subtitle=True, needs_audio=True, skip_existing=False,
    )
    statuses = {item.video_path.name: item.status for item in plan}
    assert statuses["第2集.mp4"] is PublishItemStatus.READY


def test_output_paths_are_unique(series: Path) -> None:
    """Trùng tên đích nghĩa là tập này ghi đè tập kia."""
    plan = build_publish_plan(
        find_episode_videos(series), output_suffix="_phude_thuyetminh",
        needs_subtitle=True, needs_audio=True, skip_existing=False,
    )
    outputs = [item.output_path for item in plan]
    assert len(set(outputs)) == len(outputs)


def test_plan_preserves_input_order(series: Path) -> None:
    videos = find_episode_videos(series)
    plan = build_publish_plan(
        videos, output_suffix="_phude", needs_subtitle=True, needs_audio=False,
    )
    assert [item.video_path for item in plan] == videos


def test_summary_counts_each_category(series: Path) -> None:
    plan = build_publish_plan(
        find_episode_videos(series), output_suffix="_phude_thuyetminh",
        needs_subtitle=True, needs_audio=True,
    )
    summary = summarise_publish_plan(plan)
    assert "3 tập sẽ xuất" in summary
    assert "thiếu phụ đề" in summary
    assert "thiếu giọng đọc" in summary
    assert "đã có" in summary


def test_missing_items_explain_next_action(series: Path) -> None:
    """Tập chưa sẵn sàng phải nói RÕ cần làm gì, không chỉ báo thiếu."""
    plan = build_publish_plan(
        find_episode_videos(series), output_suffix="_phude_thuyetminh",
        needs_subtitle=True, needs_audio=True,
    )
    blocked = [item for item in plan if not item.will_run and item.note]
    assert blocked
    assert any("TTS" in item.note for item in blocked)
