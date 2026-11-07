"""[v3.23.674] "Thời lượng tối đa" (Cài đặt → hậu xử lý) nay có tác dụng.

Trước đây ô này được lưu, nạp, truyền vào cấu hình dựng phụ đề nhưng không đoạn mã nào đọc. Nay là TRẦN mỗi khi
ứng dụng TỰ KÉO DÀI một dòng: đệm thời gian khi trích xuất, kéo dài khung chớp, giãn giờ đọc ở trang Dịch. Dòng vốn
hiện lâu trên video KHÔNG bị cắt. Đo phát lại tập 1-5 phim mẫu (429 dòng đáp án, dòng dài nhất 4,48 s): kết quả dựng
phụ đề giống hệt bản cũ với mặc định 7 s.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.application.dtos.extract_subtitles_dto import SubtitleBuilderConfig
from subtitles_extractor.application.services.flicker_absorber import absorb_flickers
from subtitles_extractor.application.services.subtitle_builder import FrameGroup, SubtitleBuilder
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

ROOT = Path(__file__).resolve().parents[3] / "src" / "subtitles_extractor"


def _group(text: str, start: float, end: float) -> FrameGroup:
    return FrameGroup(reconstructed_text=text, start_timestamp_sec=start, end_timestamp_sec=end,
                      accumulated_confidence=0.9, total_frames_count=5)


def _events(max_dur: float, padding: float, groups: list[FrameGroup]) -> list[SubtitleEvent]:
    builder = SubtitleBuilder(SubtitleBuilderConfig(temporal_padding_sec=padding,
                                                    max_duration_sec=max_dur))
    return builder._convert_groups_to_events(groups)


def test_dem_thoi_gian_khong_vuot_tran() -> None:
    ev = _events(1.2, 0.5, [_group("一", 0.0, 1.0)])[0]
    assert ev.end_sec == pytest.approx(1.2)


def test_khong_cat_dong_von_hien_lau() -> None:
    ev = _events(2.0, 0.5, [_group("一", 0.0, 3.0)])[0]
    assert ev.end_sec == pytest.approx(3.0)


def test_tran_rong_thi_dem_nhu_cu() -> None:
    ev = _events(7.0, 0.5, [_group("一", 0.0, 1.0)])[0]
    assert ev.end_sec == pytest.approx(1.5)


def _ev(i: int, text: str, start: float, end: float) -> SubtitleEvent:
    return SubtitleEvent(index=i, text=text, interval=TimeInterval(start, end),
                         confidence=Confidence(0.9), frame_count=3)


def test_keo_dai_khung_chop_khong_vuot_tran() -> None:
    events = [_ev(1, "你好啊", 0.0, 0.3), _ev(2, "再见了", 5.0, 6.0)]
    capped = absorb_flickers(events, min_duration_sec=1.0, similarity_threshold=0.75,
                             merge_gap_sec=0.2, max_duration_sec=0.6)
    plain = absorb_flickers(events, min_duration_sec=1.0, similarity_threshold=0.75,
                            merge_gap_sec=0.2)
    assert capped[0].end_sec == pytest.approx(0.6)
    assert plain[0].end_sec == pytest.approx(1.0)


def test_cau_hinh_cu_nang_mot_lan() -> None:
    from subtitles_extractor.infrastructure.settings.settings_service import SettingsService

    up = SettingsService._migrate({"post_process": {"max_duration_sec": 5.0}})
    assert up["post_process"]["max_duration_sec"] == 7.0
    assert up["post_process"]["defaults_v674"] is True
    # Người dùng tự chọn giá trị khác → giữ nguyên; đã nâng rồi → không nâng lại.
    assert SettingsService._migrate(
        {"post_process": {"max_duration_sec": 4.0}})["post_process"]["max_duration_sec"] == 4.0
    again = SettingsService._migrate(
        {"post_process": {"max_duration_sec": 5.0, "defaults_v674": True}})
    assert again["post_process"]["max_duration_sec"] == 5.0


def test_mac_dinh_moi_bang_tran_cu_cua_gian_gio_doc() -> None:
    from subtitles_extractor.application.services.reading_time_extender import (
        DEFAULT_MAX_DURATION_SEC,
    )
    from subtitles_extractor.infrastructure.settings.application_settings import (
        PostProcessSettings,
    )

    assert PostProcessSettings().max_duration_sec == DEFAULT_MAX_DURATION_SEC == 7.0


def test_gian_gio_doc_dung_tran_trong_cai_dat() -> None:
    src = (ROOT / "presentation" / "view_models" / "translate_page_view_model.py").read_text(
        encoding="utf-8")
    assert "post_process.max_duration_sec" in src
    assert "max_duration_sec=max_duration" in src


def test_toi_thieu_lon_hon_toi_da_chi_chan_phan_keo_dai() -> None:
    """Đặt tối thiểu 1,0 s > tối đa 0,6 s: việc PHÂN LOẠI khung chớp / ngưỡng loại vẫn theo tối thiểu như cũ."""
    events = [_ev(1, "你好啊", 0.0, 0.8), _ev(2, "再见了", 5.0, 6.0)]
    capped = absorb_flickers(events, min_duration_sec=1.0, similarity_threshold=0.75,
                             merge_gap_sec=0.2, max_duration_sec=0.6)
    # Dòng 0,8 s (< tối thiểu) vẫn được coi là ngắn nhưng không bị cắt bớt xuống 0,6 s.
    assert len(capped) == 2
    assert capped[0].end_sec == pytest.approx(0.8)


def test_dem_khi_dau_dong_bi_day_lui() -> None:
    """Đầu dòng bị đẩy lùi để khỏi đè dòng trước: trần tính từ đầu dòng THẬT, không cắt dòng gốc."""
    events = _events(1.0, 0.5, [_group("一", 0.0, 1.0), _group("二", 0.9, 1.6)])
    assert events[1].start_sec > events[0].end_sec - 1e-9
    assert events[1].end_sec >= 1.6 - 1e-9
    assert events[1].end_sec - events[1].start_sec <= 1.0 + 1e-6


@pytest.mark.parametrize(("stored", "want"), [(0.2, 7.0), ("abc", 7.0), (12.0, 12.0)])
def test_nang_cau_hinh_gia_tri_la(stored, want) -> None:
    from subtitles_extractor.infrastructure.settings.settings_service import SettingsService

    up = SettingsService._migrate({"post_process": {"max_duration_sec": stored}})
    assert up["post_process"]["max_duration_sec"] == want
