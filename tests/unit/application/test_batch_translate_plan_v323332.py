"""Test lập kế hoạch dịch hàng loạt — v3.23.332.

KHÂU NÀY KHÁC HẲN ba khâu đã tự động hoá trước: Trích xuất / TTS / Xuất bản chạy cục
bộ nên cứ chạy tới hết, còn Dịch gọi **dịch vụ ngoài có hạn mức ngày (RPD)** chỉ từ 20
đến 500 request tuỳ mô hình. Với bộ 84 tập, rất dễ hết hạn mức giữa chừng.

Nên kế hoạch phải ước lượng được số request và cảnh báo TRƯỚC khi chạy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.application.services.batch_translate_plan import (
    TranslateItemStatus,
    build_translate_plan,
    count_subtitle_lines,
    estimate_requests,
    find_episode_videos,
    find_existing_translation,
    find_source_subtitle,
    quota_warning,
    summarise_translate_plan,
)


def _srt(lines: int) -> str:
    return "".join(
        f"{i}\n00:00:{i % 60:02d},000 --> 00:00:{(i + 1) % 60:02d},000\nCâu {i}\n\n"
        for i in range(1, lines + 1)
    )


@pytest.fixture
def series(tmp_path: Path) -> Path:
    """8 tập: 1–6 đã trích xuất, tập 2 đã dịch, 7–8 chưa trích xuất."""
    for index in range(1, 9):
        (tmp_path / f"第{index}集.mp4").write_bytes(b"\x00")
    for index in range(1, 7):
        (tmp_path / f"第{index}集.original.srt").write_text(_srt(45), encoding="utf-8")
    (tmp_path / "第2集.translate.vi.srt").write_text("1\n", encoding="utf-8")
    (tmp_path / "第1集_thuyetminh.mkv").write_bytes(b"\x00")  # tệp app tự xuất
    return tmp_path


# ── Đếm câu ──────────────────────────────────────────────────────────────────
def test_counts_srt_lines(tmp_path: Path) -> None:
    path = tmp_path / "a.srt"
    path.write_text(_srt(45), encoding="utf-8")
    assert count_subtitle_lines(path) == 45


def test_counts_ass_dialogues(tmp_path: Path) -> None:
    path = tmp_path / "a.ass"
    path.write_text(
        "[Events]\nFormat: Layer, Start, End\n"
        "Dialogue: 0,0:00:01.00,0:00:02.00,,Xin chào\n"
        "Dialogue: 0,0:00:03.00,0:00:04.00,,Tạm biệt\n",
        encoding="utf-8",
    )
    assert count_subtitle_lines(path) == 2


def test_unreadable_file_counts_zero(tmp_path: Path) -> None:
    assert count_subtitle_lines(tmp_path / "khong-co.srt") == 0


# ── Tìm tệp ──────────────────────────────────────────────────────────────────
def test_finds_source_subtitle(series: Path) -> None:
    found = find_source_subtitle(series / "第1集.mp4")
    assert found is not None
    assert found.name == "第1集.original.srt"


def test_finds_existing_translation(series: Path) -> None:
    found = find_existing_translation(series / "第2集.mp4", "vi")
    assert found is not None
    assert found.name == "第2集.translate.vi.srt"


def test_no_translation_returns_none(series: Path) -> None:
    assert find_existing_translation(series / "第1集.mp4", "vi") is None


# ── Lập kế hoạch ─────────────────────────────────────────────────────────────
def test_plan_classifies_every_episode(series: Path) -> None:
    plan = build_translate_plan(find_episode_videos(series))
    statuses = {item.video_path.name: item.status for item in plan}
    assert statuses["第1集.mp4"] is TranslateItemStatus.READY
    assert statuses["第2集.mp4"] is TranslateItemStatus.ALREADY_DONE
    assert statuses["第7集.mp4"] is TranslateItemStatus.MISSING_SOURCE


def test_only_ready_items_run(series: Path) -> None:
    plan = build_translate_plan(find_episode_videos(series))
    assert sum(1 for item in plan if item.will_run) == 5


def test_line_count_recorded_for_ready_items(series: Path) -> None:
    """Số câu cần cho việc ước lượng request — phải đọc thật, không đoán."""
    plan = build_translate_plan(find_episode_videos(series))
    ready = [item for item in plan if item.will_run]
    assert all(item.line_count == 45 for item in ready)


def test_skips_generated_files(series: Path) -> None:
    videos = find_episode_videos(series)
    assert all("_thuyetminh" not in v.name for v in videos)


def test_can_force_retranslate(series: Path) -> None:
    plan = build_translate_plan(find_episode_videos(series), skip_existing=False)
    assert sum(1 for item in plan if item.will_run) == 6


def test_output_paths_are_unique(series: Path) -> None:
    plan = build_translate_plan(find_episode_videos(series), skip_existing=False)
    outputs = [item.output_path for item in plan]
    assert len(set(outputs)) == len(outputs)


# ── Ước lượng request và hạn mức ─────────────────────────────────────────────
def test_estimate_scales_with_batch_size(series: Path) -> None:
    plan = build_translate_plan(find_episode_videos(series))
    small = estimate_requests(plan, batch_size=20)
    large = estimate_requests(plan, batch_size=100)
    assert small > large  # lô nhỏ -> nhiều request hơn


def test_estimate_counts_only_runnable(series: Path) -> None:
    plan = build_translate_plan(find_episode_videos(series))
    # 5 tập × 45 câu ÷ 40 câu/lô = 5 × 2 = 10 request.
    assert estimate_requests(plan, batch_size=40) == 10


def test_estimate_is_zero_for_empty_plan() -> None:
    assert estimate_requests([]) == 0


def test_estimate_handles_zero_batch_size(series: Path) -> None:
    """Kích thước lô 0 không được gây chia cho 0."""
    plan = build_translate_plan(find_episode_videos(series))
    assert estimate_requests(plan, batch_size=0) > 0


def test_warns_when_exceeding_daily_limit() -> None:
    warning = quota_warning(estimated_requests=168, daily_limit=20)
    assert warning is not None
    assert "168" in warning and "20" in warning
    # Phải trấn an: tập đã xong vẫn được giữ.
    assert "giữ" in warning


def test_no_warning_within_limit() -> None:
    assert quota_warning(estimated_requests=168, daily_limit=500) is None


@pytest.mark.parametrize("limit", [None, 0, -1])
def test_no_warning_when_limit_unknown(limit: int | None) -> None:
    assert quota_warning(estimated_requests=9999, daily_limit=limit) is None


def test_large_series_estimate_is_realistic(tmp_path: Path) -> None:
    """Bộ 84 tập: kiểm con số ước lượng có hợp lý để cảnh báo đúng."""
    for index in range(1, 85):
        (tmp_path / f"第{index}集.mp4").write_bytes(b"\x00")
        (tmp_path / f"第{index}集.original.srt").write_text(_srt(45), encoding="utf-8")

    plan = build_translate_plan(find_episode_videos(tmp_path))
    requests = estimate_requests(plan, batch_size=40)

    assert requests == 168  # 84 tập × ceil(45/40)=2
    assert quota_warning(requests, 500) is None   # vừa hạn mức flash-lite 3.1
    assert quota_warning(requests, 20) is not None  # vượt xa flash-lite 2.5


# ── Trình bày ────────────────────────────────────────────────────────────────
def test_summary_includes_line_total(series: Path) -> None:
    summary = summarise_translate_plan(build_translate_plan(find_episode_videos(series)))
    assert "5 tập sẽ dịch" in summary
    assert "225 câu" in summary
    assert "chưa trích xuất" in summary
    assert "đã dịch" in summary


def test_missing_source_explains_next_action(series: Path) -> None:
    plan = build_translate_plan(find_episode_videos(series))
    blocked = [
        item for item in plan if item.status is TranslateItemStatus.MISSING_SOURCE
    ]
    assert blocked
    assert "Trích xuất" in blocked[0].note
