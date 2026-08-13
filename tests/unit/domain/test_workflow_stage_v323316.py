"""Test quy trình công việc của dự án (:class:`WorkflowStage`) — v3.23.316.

Trước bản này, quy trình có 2 lỗ hổng:
    * KHÔNG có mốc kết thúc — nhìn Thư viện không biết phim nào đã xuất bản xong.
    * Ứng dụng KHÔNG gợi ý bước tiếp theo ở bất kỳ đâu; người dùng phải tự biết thứ tự.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.domain.entities.project_record import (
    ProjectRecord,
    WorkflowStage,
)

_ORDER = [
    WorkflowStage.NEW,
    WorkflowStage.EXTRACTED,
    WorkflowStage.EDITED,
    WorkflowStage.TRANSLATED,
    WorkflowStage.TTS_DONE,
    WorkflowStage.PUBLISHED,
]


def test_stages_are_strictly_increasing() -> None:
    """Các khâu phải tăng dần để so sánh ``record.stage < X`` hoạt động đúng."""
    values = [stage.value for stage in _ORDER]
    assert values == sorted(values)
    assert len(set(values)) == len(values)


@pytest.mark.parametrize("stage", list(WorkflowStage))
def test_every_stage_has_vietnamese_label(stage: WorkflowStage) -> None:
    assert stage.label_vi
    assert stage.label_vi.strip() == stage.label_vi


@pytest.mark.parametrize("stage", list(WorkflowStage))
def test_every_stage_has_next_action(stage: WorkflowStage) -> None:
    """Mọi khâu phải nói được việc tiếp theo — kể cả khâu cuối."""
    assert stage.next_action_vi


def test_next_page_chain_follows_workflow_order() -> None:
    """Chuỗi trang phải đúng thứ tự: Trích xuất → Biên tập → Dịch → TTS → Xuất bản."""
    assert [stage.next_page_key for stage in _ORDER] == [
        "extractPage",
        "editorPage",
        "translatePage",
        "ttsPage",
        "publishPage",
        None,
    ]


def test_only_final_stage_is_complete() -> None:
    for stage in _ORDER[:-1]:
        assert not stage.is_complete
    assert WorkflowStage.PUBLISHED.is_complete
    assert WorkflowStage.PUBLISHED.next_page_key is None


def test_progress_ratio_spans_zero_to_one() -> None:
    assert WorkflowStage.NEW.progress_ratio == 0.0
    assert WorkflowStage.PUBLISHED.progress_ratio == 1.0
    ratios = [stage.progress_ratio for stage in _ORDER]
    assert ratios == sorted(ratios)


def test_record_defaults_to_new_stage() -> None:
    record = ProjectRecord(video_hash="abc")
    assert record.stage is WorkflowStage.NEW
    assert record.published_video_path == ""


def test_record_can_store_published_path() -> None:
    """Đường dẫn tệp hoàn chỉnh phải lưu được để mở lại từ Thư viện."""
    record = ProjectRecord(video_hash="abc")
    record.stage = WorkflowStage.PUBLISHED
    record.published_video_path = "D:/out/phim_vi.mkv"
    assert record.stage.is_complete
    assert record.published_video_path.endswith(".mkv")
