"""Test kiểm tra trước khi trích xuất và chẩn đoán kết quả rỗng — v3.23.325.

Hai lỗ hổng được sửa:
    * ``start_extraction`` KHÔNG kiểm gì — ROI lệch ra ngoài khung hay tệp đã bị di
      chuyển chỉ lộ ra sau khi chờ xong.
    * Ra 0 câu vẫn báo màu xanh ``"✓ Hoàn tất! 0 câu … → Đã lưu vào Database"`` —
      vừa sai (không lưu gì) vừa khiến người dùng tưởng phim không có phụ đề.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.application.services.extraction_preflight import (
    IssueLevel,
    check_before_extraction,
    diagnose_empty_result,
    has_blocker,
    summarise_issues,
)

_SIZE = (720, 1280)  # định dạng CJK drama dọc
_GOOD_ROI = (100, 1050, 520, 150)  # dải phụ đề điển hình


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "Tap01.mp4"
    path.write_bytes(b"\x00")
    return path


# ── Trường hợp chạy được ─────────────────────────────────────────────────────
def test_normal_setup_has_no_issues(video: Path) -> None:
    issues = check_before_extraction(
        video_path=video, video_size=_SIZE, roi=_GOOD_ROI, duration_sec=67.0
    )
    assert issues == []


def test_auto_roi_is_allowed(video: Path) -> None:
    """Không vẽ ROI là hợp lệ — hệ thống sẽ tự dò."""
    issues = check_before_extraction(
        video_path=video, video_size=_SIZE, roi=None, duration_sec=67.0
    )
    assert not has_blocker(issues)


# ── Trường hợp phải CHẶN ─────────────────────────────────────────────────────
def test_missing_video_blocks(tmp_path: Path) -> None:
    issues = check_before_extraction(
        video_path=tmp_path / "khong-co.mp4", video_size=_SIZE, roi=_GOOD_ROI
    )
    assert has_blocker(issues)


def test_no_video_blocks() -> None:
    issues = check_before_extraction(video_path=None, video_size=_SIZE, roi=None)
    assert has_blocker(issues)


def test_unreadable_frame_size_blocks(video: Path) -> None:
    issues = check_before_extraction(video_path=video, video_size=(0, 0), roi=None)
    assert has_blocker(issues)


def test_roi_outside_frame_blocks(video: Path) -> None:
    """Hay gặp khi đổi sang video khác độ phân giải mà quên vẽ lại ROI."""
    issues = check_before_extraction(
        video_path=video, video_size=_SIZE, roi=(100, 1200, 520, 300)
    )
    assert has_blocker(issues)
    assert any("ngoài khung hình" in issue.message for issue in issues)


@pytest.mark.parametrize("roi", [(100, 100, 0, 50), (100, 100, 50, 0)])
def test_zero_sized_roi_blocks(video: Path, roi: tuple[int, int, int, int]) -> None:
    issues = check_before_extraction(video_path=video, video_size=_SIZE, roi=roi)
    assert has_blocker(issues)


# ── Trường hợp chỉ CẢNH BÁO (vẫn cho chạy) ───────────────────────────────────
def test_tiny_roi_warns_but_allows(video: Path) -> None:
    """ROI bé xíu thường do kéo trượt tay — cảnh báo nhưng không chặn."""
    issues = check_before_extraction(
        video_path=video, video_size=_SIZE, roi=(100, 1000, 10, 6)
    )
    assert issues
    assert not has_blocker(issues)
    assert all(issue.level is IssueLevel.WARNING for issue in issues)


def test_full_frame_roi_warns(video: Path) -> None:
    """ROI phủ gần hết khung chạy rất chậm và dễ bắt nhầm chữ trong cảnh phim."""
    issues = check_before_extraction(
        video_path=video, video_size=_SIZE, roi=(0, 100, 720, 1000)
    )
    assert issues and not has_blocker(issues)


def test_unknown_duration_warns_only(video: Path) -> None:
    issues = check_before_extraction(
        video_path=video, video_size=_SIZE, roi=_GOOD_ROI, duration_sec=0.0
    )
    assert issues and not has_blocker(issues)


# ── Trình bày ────────────────────────────────────────────────────────────────
def test_summary_is_empty_when_no_issues() -> None:
    assert summarise_issues([]) == ""


def test_summary_shows_message_and_hint(video: Path) -> None:
    """Mỗi vấn đề phải kèm HƯỚNG XỬ LÝ, không chỉ báo lỗi."""
    issues = check_before_extraction(
        video_path=video, video_size=_SIZE, roi=(100, 1200, 520, 300)
    )
    summary = summarise_issues(issues)
    assert "⛔" in summary
    assert "→" in summary  # phần gợi ý


# ── Chẩn đoán kết quả rỗng ───────────────────────────────────────────────────
def test_zero_frames_points_to_file_problem() -> None:
    """Không đọc được khung nào là lỗi TỆP, không phải lỗi ROI."""
    message = diagnose_empty_result(
        frames_processed=0, roi=_GOOD_ROI, ocr_language="ch"
    )
    assert "Không đọc được khung hình nào" in message
    assert "ROI" not in message  # không đổ lỗi nhầm cho ROI


def test_empty_result_lists_causes_by_likelihood() -> None:
    message = diagnose_empty_result(
        frames_processed=1240, roi=_GOOD_ROI, ocr_language="ch"
    )
    assert "1,240 khung hình" in message
    assert "ROI" in message
    assert "ngôn ngữ" in message
    assert "track phụ đề nhúng" in message  # gợi ý đường vào khác


def test_auto_roi_failure_has_different_advice() -> None:
    """Tự dò ROI thất bại thì lời khuyên phải khác với ROI vẽ tay sai."""
    message = diagnose_empty_result(
        frames_processed=500, roi=None, ocr_language="ch"
    )
    assert "Tự dò ROI không tìm được" in message


def test_probe_mode_suggests_other_segment() -> None:
    """Thử nhanh ra rỗng có thể chỉ vì đoạn đó không có thoại."""
    message = diagnose_empty_result(
        frames_processed=200, roi=_GOOD_ROI, ocr_language="ch", probed=True
    )
    assert "đoạn khác" in message


def test_language_is_named_in_diagnosis() -> None:
    message = diagnose_empty_result(
        frames_processed=100, roi=_GOOD_ROI, ocr_language="japan"
    )
    assert "japan" in message
