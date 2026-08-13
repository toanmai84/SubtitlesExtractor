"""Test lập kế hoạch trích xuất hàng loạt nhiều tập — v3.23.319.

Ứng dụng phục vụ phim bộ CJK (hàng chục tập, vị trí phụ đề gần như giống nhau) nhưng
trang Trích xuất chỉ mở được MỘT video mỗi lần.

Cái bẫy được kiểm kỹ nhất ở đây: :class:`Roi` lưu toạ độ TUYỆT ĐỐI. Nếu các tập khác
độ phân giải, dùng lại ROI nguyên xi sẽ cắt sai vùng — OCR ra rác mà KHÔNG báo lỗi.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.application.services.batch_extraction_plan import (
    BatchItemStatus,
    build_batch_plan,
    roi_fits_in,
    scale_roi,
    summarise_plan,
)
from subtitles_extractor.domain.value_objects.roi import Roi

_REFERENCE_SIZE = (720, 1280)  # định dạng CJK drama dọc
_REFERENCE_ROI = Roi(x=100, y=1000, width=520, height=200)


def _make_videos(directory: Path, count: int) -> list[Path]:
    videos = []
    for index in range(1, count + 1):
        path = directory / f"Tap{index:02d}.mp4"
        path.write_bytes(b"\x00")
        videos.append(path)
    return videos


# ── Co giãn ROI ──────────────────────────────────────────────────────────────
def test_same_resolution_keeps_roi_identical() -> None:
    scaled = scale_roi(_REFERENCE_ROI, from_size=_REFERENCE_SIZE, to_size=_REFERENCE_SIZE)
    assert scaled is not None
    assert (scaled.x, scaled.y, scaled.width, scaled.height) == (100, 1000, 520, 200)


def test_doubling_resolution_doubles_roi() -> None:
    scaled = scale_roi(_REFERENCE_ROI, from_size=(720, 1280), to_size=(1440, 2560))
    assert scaled is not None
    assert (scaled.x, scaled.y, scaled.width, scaled.height) == (200, 2000, 1040, 400)


@pytest.mark.parametrize(
    "target", [(1080, 1920), (360, 640), (1280, 720), (480, 854), (2160, 3840)]
)
def test_scaled_roi_always_fits_target_frame(target: tuple[int, int]) -> None:
    """ROI sau co giãn PHẢI nằm trọn trong khung hình đích — nếu không OCR sẽ lỗi."""
    scaled = scale_roi(_REFERENCE_ROI, from_size=_REFERENCE_SIZE, to_size=target)
    assert scaled is not None
    assert roi_fits_in(scaled, target)


def test_scaling_preserves_roi_metadata() -> None:
    """Căn lề và hướng chữ phải giữ nguyên — chúng không phụ thuộc độ phân giải."""
    scaled = scale_roi(_REFERENCE_ROI, from_size=_REFERENCE_SIZE, to_size=(1080, 1920))
    assert scaled is not None
    assert scaled.alignment is _REFERENCE_ROI.alignment
    assert scaled.orientation is _REFERENCE_ROI.orientation


@pytest.mark.parametrize(
    ("from_size", "to_size"), [((0, 1280), (720, 1280)), ((720, 1280), (0, 0))]
)
def test_invalid_sizes_return_none(
    from_size: tuple[int, int], to_size: tuple[int, int]
) -> None:
    assert scale_roi(_REFERENCE_ROI, from_size=from_size, to_size=to_size) is None


# ── Lập kế hoạch ─────────────────────────────────────────────────────────────
def test_plan_keeps_input_order(tmp_path: Path) -> None:
    videos = _make_videos(tmp_path, 4)
    plan = build_batch_plan(
        videos, reference_roi=_REFERENCE_ROI, reference_size=_REFERENCE_SIZE
    )
    assert [item.video_path for item in plan] == videos


def test_output_paths_use_original_suffix(tmp_path: Path) -> None:
    """Tệp ra phải là ``<tên>.original.srt`` để không ghi đè bản dịch/TTS."""
    videos = _make_videos(tmp_path, 1)
    plan = build_batch_plan(
        videos, reference_roi=_REFERENCE_ROI, reference_size=_REFERENCE_SIZE
    )
    assert plan[0].output_path.name == "Tap01.original.srt"


def test_skips_videos_that_already_have_output(tmp_path: Path) -> None:
    videos = _make_videos(tmp_path, 3)
    (tmp_path / "Tap02.original.srt").write_text("1\n", encoding="utf-8")

    plan = build_batch_plan(
        videos, reference_roi=_REFERENCE_ROI, reference_size=_REFERENCE_SIZE,
        skip_existing=True,
    )

    assert plan[1].status is BatchItemStatus.SKIPPED_EXISTS
    assert not plan[1].will_run
    assert plan[0].will_run and plan[2].will_run


def test_can_force_rerun_existing(tmp_path: Path) -> None:
    videos = _make_videos(tmp_path, 2)
    (tmp_path / "Tap01.original.srt").write_text("1\n", encoding="utf-8")

    plan = build_batch_plan(
        videos, reference_roi=_REFERENCE_ROI, reference_size=_REFERENCE_SIZE,
        skip_existing=False,
    )
    assert plan[0].will_run


def test_marks_items_needing_roi_scaling(tmp_path: Path) -> None:
    """Tập khác độ phân giải phải được ĐÁNH DẤU để người dùng kiểm lại."""
    videos = _make_videos(tmp_path, 2)
    plan = build_batch_plan(
        videos, reference_roi=_REFERENCE_ROI, reference_size=_REFERENCE_SIZE,
        video_sizes={videos[1]: (1080, 1920)},
    )

    assert plan[0].status is BatchItemStatus.READY
    assert plan[1].status is BatchItemStatus.ROI_SCALED
    assert plan[1].will_run
    assert plan[1].note is not None and "co giãn" in plan[1].note
    assert plan[1].roi != _REFERENCE_ROI  # đã đổi thật, không phải chỉ gắn nhãn


def test_missing_video_is_invalid(tmp_path: Path) -> None:
    plan = build_batch_plan(
        [tmp_path / "khong-co.mp4"],
        reference_roi=_REFERENCE_ROI, reference_size=_REFERENCE_SIZE,
    )
    assert plan[0].status is BatchItemStatus.INVALID
    assert not plan[0].will_run


def test_without_reference_roi_each_video_auto_detects(tmp_path: Path) -> None:
    """Không có ROI mẫu -> để pipeline tự dò cho từng tập, vẫn chạy được."""
    videos = _make_videos(tmp_path, 2)
    plan = build_batch_plan(videos, reference_roi=None, reference_size=None)
    for item in plan:
        assert item.roi is None
        assert item.will_run


def test_summary_counts_every_category(tmp_path: Path) -> None:
    videos = _make_videos(tmp_path, 3)
    (tmp_path / "Tap02.original.srt").write_text("1\n", encoding="utf-8")
    plan = build_batch_plan(
        videos + [tmp_path / "thieu.mp4"],
        reference_roi=_REFERENCE_ROI, reference_size=_REFERENCE_SIZE,
        video_sizes={videos[2]: (1080, 1920)},
    )

    summary = summarise_plan(plan)
    assert "2 tập sẽ chạy" in summary
    assert "co giãn" in summary
    assert "bỏ qua" in summary
    assert "lỗi" in summary
