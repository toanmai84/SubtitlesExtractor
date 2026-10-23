"""[v3.23.604] Khung OCR chớp nhoáng (< 0,3 s) được kéo tới thời lượng đọc được.

Chuẩn đoán thật 11/10 16:01 (灭门惨案II, VieNeu 2 giọng): 2/2 chỗ chồng tiếng và 3/4 dòng
đọc ≥ 2,2x là khung OCR chỉ 0,065-0,1 s — dòng 916 "我........" (4081,267-4081,332)
có 1,33 s lặng phía trước nhưng sát dòng 917 phía sau, nên bản cũ (chỉ giãn đuôi) không
giãn được và lồng tiếng đọc "Tôi..." ở 2,5x. Mốc giờ trong test là mốc thật của bản dịch
cuối.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from subtitles_extractor.application.services.reading_time_extender import (
    GLITCH_MIN_DISPLAY_SEC,
    compute_glitch_window,
    extend_events_for_reading,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

CPS = 21.0


def _event(index: int, text: str, start: float, end: float) -> SubtitleEvent:
    return SubtitleEvent(index=index, text=text, interval=TimeInterval(start, end))


#: Dòng 906-917 thật.
SCENE_916 = [
    _event(906, "Đừng quấy, để tao nghỉ một tí.", 4040.933, 4042.867),
    _event(907, "Nghỉ gì chứ? Xuống đây đi.", 4042.933, 4044.2),
    _event(908, "Đến con rồi.", 4045.133, 4046.2),
    _event(909, "Tao chín mươi cái, cũng khá đấy chứ?", 4049.533, 4051.733),
    _event(910, "Đúng là lão già vô dụng, xem con đây này.", 4051.8, 4054.067),
    _event(911, "Để tao cho mày nếm mùi lợi hại nhé.", 4056.6, 4058.533),
    _event(912, "Con trai.", 4064.8, 4065.733),
    _event(913, "Con khốn chết tiệt, dám đánh tao à?", 4074.6, 4076.267),
    _event(914, "Quá đáng thật, dám đánh người à!", 4076.933, 4078.667),
    _event(915, "Gan mày to thật đấy.", 4078.733, 4079.933),
    _event(916, "Tôi...", 4081.267, 4081.332),
    _event(917, "Tôi kiện các người!", 4081.333, 4082.2),
]


def test_dong_916_muon_khoang_lang_phia_truoc() -> None:
    start, end = compute_glitch_window(6, 4081.267, 4081.332, 4079.933, 4081.333, CPS)
    assert end == pytest.approx(4081.332)
    assert end - start == pytest.approx(GLITCH_MIN_DISPLAY_SEC)


def test_dong_504_chua_khoang_cach_voi_cau_truoc() -> None:
    # Dòng 503 (đã giãn) kết thúc 2070,543; dòng 505 bắt đầu 2071,467.
    start, end = compute_glitch_window(
        len("Đừng mà anh Kiếm!"), 2071.333, 2071.433, 2070.543, 2071.467, CPS
    )
    assert end == pytest.approx(2071.433)
    assert start == pytest.approx(2070.543 + 0.1)


def test_dong_356_muon_phia_sau_truoc() -> None:
    start, end = compute_glitch_window(2, 1517.6, 1517.867, 1516.81, 1520.067, CPS)
    assert start == pytest.approx(1517.6)
    assert end == pytest.approx(1517.6 + GLITCH_MIN_DISPLAY_SEC)


def test_dong_200_ket_giua_hai_cau_giu_nguyen() -> None:
    # Dòng 199 kết thúc 841,667; dòng 201 bắt đầu 841,8 — không còn chỗ.
    assert compute_glitch_window(14, 841.733, 841.799, 841.667, 841.8, CPS) == (
        841.733, 841.799,
    )


def test_khung_thuong_khong_bi_dong_toi() -> None:
    assert compute_glitch_window(20, 10.0, 10.5, 8.0, 20.0, CPS) == (10.0, 10.5)


def test_giai_doan_gian_gio_doc_khong_tao_chong_dong() -> None:
    out, count = extend_events_for_reading(SCENE_916, CPS)
    assert count >= 1
    by_index = {event.index: event for event in out}
    assert by_index[916].start_sec == pytest.approx(4081.332 - GLITCH_MIN_DISPLAY_SEC)
    assert by_index[916].end_sec == pytest.approx(4081.332)
    assert by_index[917].start_sec == pytest.approx(4081.333)
    for earlier, later in pairwise(out):
        assert later.start_sec >= earlier.end_sec
    assert SCENE_916[10].start_sec == pytest.approx(4081.267)  # không sửa danh sách gốc


def test_long_tieng_van_ghep_dung_cau_goc() -> None:
    from subtitles_extractor.application.services.dubbing_cast import align_to_source

    out, _count = extend_events_for_reading(SCENE_916, CPS)
    spans = tuple((event.start_sec, event.end_sec) for event in SCENE_916)
    assert align_to_source(out, spans) == list(range(len(SCENE_916)))
