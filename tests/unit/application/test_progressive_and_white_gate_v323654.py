"""[v3.23.654] Lọc chữ trắng bật mặc định (một lần cho cấu hình cũ); gộp phụ đề hiện dần có bật/tắt."""

from __future__ import annotations

from pathlib import Path

from subtitles_extractor.application.services.subtitle_pipeline.event_refinement import merge_progressive_events
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.settings.settings_service import SettingsService


def _ev(index: int, text: str, start: float, end: float) -> SubtitleEvent:
    return SubtitleEvent(index=index, text=text, interval=TimeInterval(start, end), frame_count=10)


def test_progressive_pairs_from_real_episodes_are_merged() -> None:
    # Ba cặp thật đo trên tập 12/02/04 (mốc và độ dài lấy từ kết quả dựng phụ đề).
    events = [
        _ev(1, "我", 34.68, 35.32), _ev(2, "我只是", 35.32, 36.9),
        _ev(3, "师", 80.52, 80.97), _ev(4, "师叔", 81.0, 82.1), _ev(5, "师叔你", 82.2, 83.0),
        _ev(6, "我一定", 165.5, 166.4), _ev(7, "好吧", 166.6, 167.5),
    ]
    merged, absorbed = merge_progressive_events(events)
    assert absorbed == 3
    assert [(e.text, e.start_sec) for e in merged] == [
        ("我只是", 34.68), ("师叔你", 80.52), ("我一定", 165.5), ("好吧", 166.6),
    ]
    assert merged[0].frame_count == 20


def test_not_merged_when_far_apart_or_shorter() -> None:
    events = [_ev(1, "我", 1.0, 1.5), _ev(2, "我只是", 2.5, 3.0), _ev(3, "舒服吗", 3.1, 4.0), _ev(4, "舒服", 4.0, 5.0)]
    assert merge_progressive_events(events) == (events, 0)


def test_builder_option_off_by_default() -> None:
    from subtitles_extractor.application.dtos.extract_subtitles_dto import SubtitleBuilderConfig
    from subtitles_extractor.infrastructure.settings.application_settings import ApplicationSettings

    assert SubtitleBuilderConfig().merge_progressive is False
    assert ApplicationSettings().post_process.merge_progressive is False




def test_white_gate_default_on_and_old_config_flipped_once(tmp_path: Path) -> None:
    from subtitles_extractor.infrastructure.settings.application_settings import ApplicationSettings

    assert ApplicationSettings().frame.white_text_gate is True
    old = {"white_text_gate": False, "white_text_luma": 200, "sample_step_sec": 0.04}
    migrated = SettingsService._migrate({"frame": old})
    assert migrated["frame"]["white_text_gate"] is True and migrated["frame"]["white_text_gate_v654"] is True
    # Người dùng tắt sau khi đã nâng cấp: lựa chọn được giữ.
    chosen = {**migrated["frame"], "white_text_gate": False}
    assert SettingsService._migrate({"frame": chosen})["frame"]["white_text_gate"] is False
