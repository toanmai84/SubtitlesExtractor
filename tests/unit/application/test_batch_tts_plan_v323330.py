"""Test lập kế hoạch TTS hàng loạt cho phim bộ — v3.23.330.

Sau v3.23.329, hàng loạt phủ được Trích xuất và Xuất bản. Khâu TTS vẫn thủ công từng
tập — với bộ 84 tập là 84 lần lặp, mỗi lần còn phải nạp lại mô hình (~15 giây).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.application.services.batch_tts_plan import (
    TtsItemStatus,
    build_tts_plan,
    estimate_batch_minutes,
    find_episode_videos,
    find_existing_audio,
    find_translated_subtitle,
    summarise_tts_plan,
)


@pytest.fixture
def series(tmp_path: Path) -> Path:
    """Thư mục 7 tập ở các mức hoàn thiện khác nhau."""
    for index in range(1, 8):
        (tmp_path / f"第{index}集.mp4").write_bytes(b"\x00")
    for index in range(1, 6):
        (tmp_path / f"第{index}集.translate.vi.srt").write_text("1\n", encoding="utf-8")
    (tmp_path / "第2集.flac").write_bytes(b"\x00")   # đã có giọng đọc (.flac)
    (tmp_path / "第3集.wav").write_bytes(b"\x00")    # đã có giọng đọc (.wav)
    (tmp_path / "第6集.translate.en.srt").write_text("1\n", encoding="utf-8")
    (tmp_path / "第1集_thuyetminh.mkv").write_bytes(b"\x00")  # tệp app tự xuất
    return tmp_path


# ── Tìm tệp ──────────────────────────────────────────────────────────────────
def test_finds_translation_with_matching_language(series: Path) -> None:
    found = find_translated_subtitle(series / "第1集.mp4", "vi")
    assert found is not None
    assert found.name == "第1集.translate.vi.srt"


def test_finds_translation_of_other_language(series: Path) -> None:
    """Quét mẫu ``*.translate.*.srt`` — không phụ thuộc biết trước mã ngôn ngữ."""
    found = find_translated_subtitle(series / "第6集.mp4", "vi")
    assert found is not None
    assert found.name == "第6集.translate.en.srt"


def test_no_translation_returns_none(series: Path) -> None:
    assert find_translated_subtitle(series / "第7集.mp4", "vi") is None


@pytest.mark.parametrize(("episode", "suffix"), [("第2集", ".flac"), ("第3集", ".wav")])
def test_finds_existing_audio_of_any_supported_format(
    series: Path, episode: str, suffix: str
) -> None:
    """Khâu TTS ghi ra .flac hoặc .wav tuỳ cấu hình — phải nhận cả hai."""
    found = find_existing_audio(series / f"{episode}.mp4")
    assert found is not None
    assert found.suffix == suffix


def test_no_audio_returns_none(series: Path) -> None:
    assert find_existing_audio(series / "第1集.mp4") is None


# ── Lập kế hoạch ─────────────────────────────────────────────────────────────
def test_plan_classifies_every_episode(series: Path) -> None:
    plan = build_tts_plan(find_episode_videos(series), target_language="vi")
    statuses = {item.video_path.name: item.status for item in plan}
    assert statuses["第1集.mp4"] is TtsItemStatus.READY
    assert statuses["第2集.mp4"] is TtsItemStatus.ALREADY_DONE
    assert statuses["第3集.mp4"] is TtsItemStatus.ALREADY_DONE
    assert statuses["第6集.mp4"] is TtsItemStatus.READY
    assert statuses["第7集.mp4"] is TtsItemStatus.MISSING_TRANSLATION


def test_only_ready_items_run(series: Path) -> None:
    plan = build_tts_plan(find_episode_videos(series))
    assert sum(1 for item in plan if item.will_run) == 4


def test_skips_generated_files_as_sources(series: Path) -> None:
    """Tệp do app xuất KHÔNG được coi là tập cần tổng hợp."""
    videos = find_episode_videos(series)
    assert all("_thuyetminh" not in v.name for v in videos)
    assert len(videos) == 7


def test_can_force_rerun(series: Path) -> None:
    plan = build_tts_plan(find_episode_videos(series), skip_existing=False)
    assert sum(1 for item in plan if item.will_run) == 6


def test_output_paths_are_unique(series: Path) -> None:
    plan = build_tts_plan(find_episode_videos(series), skip_existing=False)
    outputs = [item.output_path for item in plan]
    assert len(set(outputs)) == len(outputs)


def test_output_suffix_is_respected(series: Path) -> None:
    plan = build_tts_plan(
        find_episode_videos(series), output_suffix=".flac", skip_existing=False
    )
    assert all(item.output_path.suffix == ".flac" for item in plan)


def test_plan_preserves_order(series: Path) -> None:
    videos = find_episode_videos(series)
    plan = build_tts_plan(videos)
    assert [item.video_path for item in plan] == videos


# ── Trình bày ────────────────────────────────────────────────────────────────
def test_summary_counts_each_category(series: Path) -> None:
    summary = summarise_tts_plan(build_tts_plan(find_episode_videos(series)))
    assert "4 tập sẽ tổng hợp" in summary
    assert "chưa dịch" in summary
    assert "đã có" in summary


def test_untranslated_item_explains_next_action(series: Path) -> None:
    plan = build_tts_plan(find_episode_videos(series))
    blocked = [
        item for item in plan if item.status is TtsItemStatus.MISSING_TRANSLATION
    ]
    assert blocked
    assert "Dịch" in blocked[0].note


def test_estimate_counts_only_runnable_items(series: Path) -> None:
    """Ước lượng KHÔNG được tính tập bị bỏ qua."""
    plan = build_tts_plan(find_episode_videos(series))
    assert estimate_batch_minutes(plan, seconds_per_episode=60.0) == pytest.approx(4.0)


def test_estimate_is_zero_for_empty_plan() -> None:
    assert estimate_batch_minutes([]) == 0.0
