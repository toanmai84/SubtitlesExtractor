"""[v3.23.676] Lỗi do lượt rà độc lập các bản 3.23.670-675 tìm ra (đã xác nhận bằng chạy thật).

* OCR lỡ 2 khung liền giữa một câu dài (bước 0,04s = 80 ms) → câu bị tách thành HAI câu trùng (dịch, lồng tiếng hai
  lần). Câu lặp thật cách nhau 3-8 khung trống (đo 12 tập) nên ở bước dày nay cần ≥ 3 khung trống.
* Pass tách câu lặp bỏ chữ ĐÃ BẦU của group, lấy chuỗi thô nhiều nhất (có thể là biến thể nhiễu).
* Đổi chữ Hán sót sang âm Hán-Việt chèn dấu cách sau ngoặc mở: "(蛊)" → "( cổ)".
* Bước nghe giọng: âm thanh bị bỏ (nhà cung cấp tuỳ chỉnh / trả rỗng) mà vẫn hỏi → model bịa nhân vật.
* Bước nghe giọng (tuỳ chọn) ném OSError/ValueError → hỏng CẢ bước phân tích ngữ cảnh.
* Dò "người chưa có trong bảng": 老爷爷 (ông cụ) bị đếm là 老爷 (lão gia); 白痴/马上… bị coi là tên.
* "vi_VN" không được coi là tiếng Việt; lý do lỗi Edge TTS lộ tên/mật khẩu proxy.

Kiểm chứng không hồi quy: phát lại dựng phụ đề tập 1-5 (429 dòng đáp án) — 428/429 khớp, y như bản trước; 12 tập (838
dòng) — kết quả giống hệt bản trước ở cả bước 0,04s và 0,12s.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.unit.application.test_ocr_line_boundaries_v323672 import _build, _events

LONG = "他就是掌门弟子"


def test_ocr_lo_hai_khung_lien_khong_tach_cau() -> None:
    # Bước 0,04s: khung 4,00 và 4,04 trống (OCR lỡ) giữa một câu dài 2,5 s.
    timeline = [(3.0, 4.0, LONG, 0.99), (4.08, 5.5, LONG, 0.99)]
    assert _build(timeline, step=0.04) == [LONG]
    assert _build([(3.0, 4.0, LONG, 0.99), (4.1, 5.5, LONG, 0.99)], step=0.05) == [LONG]


def test_cau_lap_that_van_tach() -> None:
    # Hai lần "师叔" cách 3 khung trống (tập 2 phim mẫu) — vẫn là hai câu.
    timeline = [(3.0, 3.72, "师叔", 0.99), (3.88, 4.6, "师叔", 0.99)]
    assert _build(timeline, step=0.04) == ["师叔", "师叔"]


def test_tach_cau_lap_giu_chu_da_bau() -> None:
    from subtitles_extractor.application.services.subtitle_builder import FrameGroup
    from subtitles_extractor.application.services.subtitle_pipeline.event_filters import (
        split_repeat_gap_groups,
    )

    step = 0.04
    members = [(round(3.0 + i * step, 2), "他就是掌鬥" if i % 3 else "他就是掌门") for i in range(20)]
    members += [(round(4.2 + i * step, 2), "他就是掌鬥" if i % 3 else "他就是掌门") for i in range(20)]
    textful = [m[0] for m in members]
    group = FrameGroup(reconstructed_text="他就是掌门", start_timestamp_sec=3.0,
                       end_timestamp_sec=5.0, accumulated_confidence=40.0, total_frames_count=40,
                       member_texts=members)
    parts = split_repeat_gap_groups([group], step, textful)
    assert [p.reconstructed_text for p in parts] == ["他就是掌门", "他就是掌门"]


def test_tach_khoi_on_dinh_giu_mep_ngoai_cua_group() -> None:
    from subtitles_extractor.application.services.subtitle_builder import FrameGroup
    from subtitles_extractor.application.services.subtitle_pipeline.event_filters import (
        split_stable_variant_groups,
    )

    step = 0.04
    members = [(round(1.0 + i * step, 2), "舒服吗") for i in range(25)]
    members += [(round(2.0 + i * step, 2), "舒服") for i in range(25)]
    group = FrameGroup(reconstructed_text="舒服吗", start_timestamp_sec=0.9,   # đã nới (lọc mềm)
                       end_timestamp_sec=3.1, accumulated_confidence=50.0, total_frames_count=50,
                       member_texts=members)
    parts = split_stable_variant_groups([group])
    assert len(parts) == 2
    assert parts[0].start_timestamp_sec == pytest.approx(0.9)
    assert parts[-1].end_timestamp_sec == pytest.approx(3.1)


def test_events_khong_doi_voi_cau_ngan_lien_cau_dai() -> None:
    # Ca thật tập 5 (v3.23.672) vẫn đúng.
    timeline = [(3.0, 3.68, "师叔", 0.99), (3.72, 5.0, "哎呦师叔", 0.99)]
    assert [e.text for e in _events(timeline)] == ["师叔", "哎呦师叔"]


# ── Âm Hán-Việt ────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("text", "want"), [
    ("Đây là (蛊) mà", "Đây là (cổ) mà"),
    ("—蛊 thôi", "—Cổ thôi"),
    ("Lâm先生 đến", "Lâm Tiên Sinh đến"),
])
def test_khong_chen_cach_sau_ngoac_mo(text: str, want: str) -> None:
    from subtitles_extractor.application.services.cjk_transliteration import (
        transliterate_leftover_cjk,
    )
    from subtitles_extractor.application.services.name_reading_check import NameReadings

    readings = NameReadings.from_encoded({"蛊": "cổ", "先": "tiên", "生": "sinh"})
    assert transliterate_leftover_cjk(text, readings) == want


def test_ma_vi_vn_la_tieng_viet() -> None:
    from subtitles_extractor.application.services.cjk_transliteration import is_vietnamese_target

    assert is_vietnamese_target("vi_VN") and is_vietnamese_target("vi-VN")
    assert not is_vietnamese_target("English")


# ── Bước nghe giọng ────────────────────────────────────────────────────────────────────────


def test_nha_cung_cap_tuy_chinh_khong_hoi_khi_can_am_thanh() -> None:
    from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
        GeminiSubtitleTranslator,
    )

    class _Custom(GeminiSubtitleTranslator):   # lớp con: không đụng thuộc tính của lớp thật
        uses_custom_provider = True

    tr = _Custom.__new__(_Custom)
    tr._api_keys = ["k"]
    assert tr._call_gemini("m", "p", None, lambda _d: None, video_files=[object()],
                           require_media=True) == {}


@pytest.mark.parametrize("error", [OSError("không tách được âm thanh"), ValueError("JSON hỏng")])
def test_nghe_giong_loi_bat_ky_giu_bang_nhan_vat(error: Exception) -> None:
    from subtitles_extractor.application.use_cases.analyze_subtitle_context import (
        AnalyzeSubtitleContextUseCase,
    )

    def listen(*_a, **_k):
        raise error

    roster = SimpleNamespace(characters="叶然 — nam chính", source_lang="zh")
    uc = AnalyzeSubtitleContextUseCase.__new__(AnalyzeSubtitleContextUseCase)
    uc._translator = SimpleNamespace(listen_roster_voices=listen)
    assert uc._listen_voices(roster, [], "Vietnamese", "m", "/v.mp4", None) is roster


# ── Dò người chưa có trong bảng ────────────────────────────────────────────────────────────


def test_tu_thuong_khong_bi_coi_la_nguoi() -> None:
    from subtitles_extractor.application.services.roster_coverage import find_unlisted_mentions

    texts = ["白痴", "白痴", "马上", "马上", "老爷爷你好", "老爷爷你好", "叶然", "叶然", "师叔", "师叔"]
    lines = [SimpleNamespace(index=i + 1, text=t) for i, t in enumerate(texts)]
    terms = {m.term for m in find_unlisted_mentions("", lines)}
    assert terms == {"叶然", "师叔"}


def test_che_mat_khau_proxy_trong_ly_do_loi() -> None:
    from subtitles_extractor.infrastructure.tts import edge_tts_adapter as mod

    mod._note_edge_failure("ProxyError: http://bob:s3cr3t@10.0.0.1:3128 refused")
    assert "s3cr3t" not in mod.EdgeTTSAdapter._last_failure
    assert "http://***@10.0.0.1:3128" in mod.EdgeTTSAdapter._last_failure


# ── [rà độc lập lần 2] ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("voted", "majority", "want"), [
    ("掌门", "掌鬥", True), ("师叔", "师父", True), ("师叔", "卑鄙", False),
    ("他就是掌门", "他就是掌鬥", True), ("你好", "再见了吧", False),
])
def test_chu_da_bau_cau_ngan(voted: str, majority: str, want: bool) -> None:
    from subtitles_extractor.application.services.subtitle_pipeline.event_filters import (
        _is_voted_variant,
    )

    assert _is_voted_variant(voted, majority) is want


def test_nhay_thang_la_dau_dong_khong_them_cach() -> None:
    from subtitles_extractor.application.services.cjk_transliteration import (
        transliterate_leftover_cjk,
    )
    from subtitles_extractor.application.services.name_reading_check import NameReadings

    readings = NameReadings.from_encoded({"先": "tiên", "生": "sinh"})
    assert transliterate_leftover_cjk('"Lâm"先生 đến', readings) == '"Lâm" Tiên Sinh đến'


@pytest.mark.parametrize("raw", ["http://bob:p@ss@h:1 x", "http://bob:p/ss@h", "socks5://u:pw@h"])
def test_che_mat_khau_proxy_moi_dang(raw: str) -> None:
    from subtitles_extractor.infrastructure.tts import edge_tts_adapter as mod

    mod._note_edge_failure(raw)
    assert "ss" not in mod.EdgeTTSAdapter._last_failure.split("@")[0]
    assert "pw" not in mod.EdgeTTSAdapter._last_failure


def test_khung_khoa_khong_don_tre() -> None:
    from subtitles_extractor.infrastructure.video.decoders.pyav_frame_sampler import _next_target

    assert _next_target(10.0, 19.5, 20.0, True) == 30.0       # trễ 9,5 s không cộng dồn
    assert _next_target(10.0, 29.0, 20.0, True) == 39.0       # không sát mẫu vừa lấy
    assert _next_target(10.0, 10.1, 0.12, False) == pytest.approx(10.22)
