"""[v3.23.672] Ranh giới câu OCR: câu ngắn liền ngay câu dài chứa nó, câu lặp sau khoảng trống ngắn.

Ca thật (phụ đề gốc người dùng gửi, phim mẫu):
* Tập 5 dòng 10-11: "师叔" (0,68s) rồi NGAY "哎呦师叔" — OCR đọc đúng từng khung, nhưng bị
  gộp thành một câu "哎呦师叔". Hai gốc: bộ gom Viterbi (mặc định) không ghi chữ từng khung nên
  pass tách "hai khối ổn định" không bao giờ chạy; pass gộp cuối chỉ giữ riêng cặp chênh ĐÚNG
  một ký tự.
* Tập 2 dòng 122-123: "师叔" hai lần cách nhau 0,12s trống — bị gộp thành một câu.

Các ca còn lại khoá hành vi KHÔNG được đổi: OCR mất một ký tự giữa câu, OCR chớp một khung, chữ
hiện dần một ký tự, rác Latin/số và chữ Latin thật.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    SubtitleBuilderConfig,
)
from subtitles_extractor.application.services.subtitle_builder import SubtitleBuilder
from subtitles_extractor.application.services.subtitle_pipeline.event_filters import (
    is_repeat_utterance,
    is_stable_distinct_pair,
)
from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.roi import Roi

ROI = Roi(x=57, y=819, width=603, height=119)
CHAR_W = 62


def _box(text: str, conf: float) -> OcrTextBox:
    width = CHAR_W * max(1, len(text))
    x1 = 301 - width // 2
    return OcrTextBox(text=text, confidence=Confidence(conf),
                      polygon=[(x1, 18), (x1 + width, 18), (x1 + width, 90), (x1, 90)])


def _frames(timeline: list[tuple[float, float, str | None, float]], step: float,
            total: float | None = None) -> list[OcrFrameResult]:
    """``timeline``: (từ, đến, chữ hoặc None = trống, conf). Ngoài các đoạn là khung trống."""
    total = total if total is not None else max(end for _s, end, _t, _c in timeline) + 1.0
    frames = []
    for index in range(round(total / step)):
        ts = round(index * step, 3)
        boxes = [
            _box(text, conf) for start, end, text, conf in timeline
            if text and start - 1e-9 <= ts < end - 1e-9
        ]
        frames.append(OcrFrameResult(frame_index=index, timestamp_sec=ts, text_boxes=boxes[:1]))
    return frames


#: Câu thoại rõ nét trước/sau để phim có ngữ cảnh "CJK chủ đạo" như thật.
CONTEXT = [(0.5, 1.7, "这都修仙了", 0.999), (1.7, 2.6, "怎么还尿急啊", 0.999)]
TAIL = [(20.0, 21.2, "无耻逆徒", 0.999), (21.2, 22.4, "我是你师叔", 0.999)]


def _events(timeline, step=0.04, viterbi=True, padding=0.0, merge_progressive=False):
    config = SubtitleBuilderConfig(sample_step_sec=step, temporal_padding_sec=padding,
                                   use_viterbi=viterbi, merge_progressive=merge_progressive)
    frames = _frames(CONTEXT + timeline + TAIL, step)
    skip = {t for _a, _b, t, _c in CONTEXT + TAIL}
    return [e for e in SubtitleBuilder(config=config).build(frames, roi=ROI) if e.text not in skip]


def _build(timeline, step=0.04, viterbi=True, **kw):
    return [e.text for e in _events(timeline, step, viterbi, **kw)]


STEPS = [0.04, 0.12]
MODES = [True, False]


# ── Ca thật ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("viterbi", MODES)
def test_cau_ngan_lien_ngay_cau_dai_chua_no_la_hai_cau(step, viterbi) -> None:
    timeline = [(10.0, 10.76, "师叔", 0.9999), (10.80, 11.68, "哎呦师叔", 0.9997)]
    assert _build(timeline, step, viterbi) == ["师叔", "哎呦师叔"]


@pytest.mark.parametrize("viterbi", MODES)
@pytest.mark.parametrize("padding", [0.0, 0.1])
def test_cau_lap_sau_khoang_trong_ngan_la_hai_cau(viterbi, padding) -> None:
    timeline = [(10.0, 10.72, "师叔", 0.9999), (10.84, 11.56, "师叔", 0.9999)]
    assert _build(timeline, 0.04, viterbi, padding=padding) == ["师叔", "师叔"]


def test_buoc_thua_mot_khung_trong_khong_tach_cau_lap() -> None:
    # Bước 0,12s: một khung trống không phân biệt được với OCR lỡ → giữ hành vi cũ (một câu).
    timeline = [(10.0, 10.72, "师叔", 0.9999), (10.84, 11.56, "师叔", 0.9999)]
    assert _build(timeline, 0.12) == ["师叔"]


@pytest.mark.parametrize("step", STEPS)
def test_cau_dai_roi_cau_ngan_chua_trong_no_la_hai_cau(step) -> None:
    # "不过什么" rồi "不过" (chênh ở cuối, 2 ký tự) — hai câu thật như v3.23.616 mô tả.
    timeline = [(10.0, 10.9, "不过什么", 0.999), (10.9, 11.7, "不过", 0.999)]
    assert _build(timeline, step) == ["不过什么", "不过"]


# ── Không được đổi ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("step", STEPS)
def test_ocr_mat_mot_ky_tu_dau_giua_cau_van_mot_cau(step) -> None:
    timeline = [(10.0, 10.48, "不就可以实现", 0.98), (10.48, 10.96, "就可以实现", 0.93),
                (10.96, 11.6, "不就可以实现", 0.98)]
    assert _build(timeline, step) == ["不就可以实现"]


@pytest.mark.parametrize("step", STEPS)
def test_net_manh_dau_cau_bi_mat_mot_luc_dai_van_mot_cau(step) -> None:
    # Mất 1 ký tự ĐẦU kéo dài 0,4s ("一" nét mảnh) — không phải hai câu.
    timeline = [(10.0, 10.4, "个人去吧", 0.97), (10.4, 11.4, "一个人去吧", 0.99)]
    assert _build(timeline, step) == ["一个人去吧"]


def test_ocr_lo_mot_khung_giua_cau_van_mot_cau() -> None:
    timeline = [(10.0, 10.8, "师叔你误会了", 0.999), (10.84, 11.6, "师叔你误会了", 0.999)]
    assert _build(timeline, 0.04) == ["师叔你误会了"]


@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("viterbi", MODES)
def test_khung_mo_bi_loc_giua_cau_dai_khong_cat_doi(step, viterbi) -> None:
    # Rà độc lập: 2 khung conf 0,1 (bị bộ lọc độ tin cậy loại) giữa câu 2s — vẫn MỘT câu.
    timeline = [(10.0, 11.0, "师叔你误会了", 0.999), (11.0, 11.0 + 2 * step, "师叔你误会了", 0.1),
                (11.0 + 2 * step, 12.0, "师叔你误会了", 0.999)]
    assert _build(timeline, step, viterbi) == ["师叔你误会了"]


@pytest.mark.parametrize("step", STEPS)
def test_buoc_thua_mot_khung_trong_giua_cau_dai_khong_cat_doi(step) -> None:
    timeline = [(10.0, 11.0, "师叔你误会了", 0.999), (11.0 + step, 12.0, "师叔你误会了", 0.999)]
    assert _build(timeline, step) == ["师叔你误会了"]


def test_tach_cau_lap_moi_phan_mang_chu_cua_chinh_no_va_giu_moc_ngoai() -> None:
    timeline = [(10.0, 10.72, "师叔", 0.9999), (10.84, 11.56, "师叔", 0.9999)]
    events = _events(timeline, 0.04)
    assert [e.text for e in events] == ["师叔", "师叔"]
    assert events[0].start_sec == pytest.approx(10.0, abs=0.05)
    assert events[1].end_sec == pytest.approx(11.56, abs=0.06)


@pytest.mark.parametrize("merge_progressive", [False, True])
def test_chu_mo_dan_mat_ky_tu_cuoi_van_mot_cau(merge_progressive) -> None:
    # Rà độc lập: tắt dần "他就是掌门" → "他就是掌" 0,4s; hiện dần ngược lại — một câu.
    out = _build([(10.0, 12.1, "他就是掌门", 0.999), (12.1, 12.5, "他就是掌", 0.9)], 0.04,
                 merge_progressive=merge_progressive)
    assert out == ["他就是掌门"]
    out = _build([(10.0, 10.4, "他就是掌", 0.9), (10.4, 12.5, "他就是掌门", 0.999)], 0.04,
                 merge_progressive=merge_progressive)
    assert out == ["他就是掌门"]


def test_buoc_thua_lo_mot_khung_cau_ngan_van_mot_cau() -> None:
    # Bước 0,12s: một khung trống giữa hai khối < 0,5s — không đủ chắc là câu lặp.
    timeline = [(10.0, 10.36, "师叔", 0.999), (10.48, 10.84, "师叔", 0.999)]
    assert _build(timeline, 0.12) == ["师叔"]


@pytest.mark.parametrize("step", STEPS)
def test_hien_dan_mot_ky_tu_giu_hanh_vi_cu(step) -> None:
    # Khối 1 chữ ("我" → "我只是") để tuỳ chọn "gộp phụ đề hiện dần" quyết định.
    before = _build([(10.0, 10.5, "我", 0.999), (10.5, 11.5, "我只是", 0.999)], step)
    assert before[-1] == "我只是"


# ── Hàm thuần ──────────────────────────────────────────────────────────────────────────────


def test_quy_tac_cau_lap() -> None:
    assert is_repeat_utterance(0.16, 0.04, 0.72, 0.72)          # 3 khung trống
    assert not is_repeat_utterance(0.08, 0.04, 0.72, 0.72)      # 1 khung trống: OCR chớp
    assert not is_repeat_utterance(0.16, 0.04, 0.12, 0.72)      # một phía quá ngắn
    assert is_repeat_utterance(0.40, 0.0, 0.5, 0.5)             # ngưỡng cũ 0,35s
    assert not is_repeat_utterance(0.24, 0.12, 0.6, 0.6)        # bước thưa, 1 khung trống
    assert is_repeat_utterance(0.36, 0.12, 0.6, 0.6)            # 2 khung trống


def test_quy_tac_hai_cau_on_dinh() -> None:
    assert is_stable_distinct_pair("师叔", "哎呦师叔", 0.76, 0.88, 7, 8)
    assert not is_stable_distinct_pair("就可以实现", "不就可以实现", 0.5, 1.0, 5, 9)
    assert not is_stable_distinct_pair("师叔", "哎呦师叔", 0.12, 0.88, 1, 8)
    assert not is_stable_distinct_pair("我", "我只是", 0.5, 1.0, 5, 9)
    assert not is_stable_distinct_pair("他就是掌门", "他就是掌", 2.1, 0.4, 50, 10)   # tắt dần
    assert is_stable_distinct_pair("舒服吗", "舒服", 1.0, 1.0, 20, 20)            # câu hỏi/đáp


def test_trong_that_chi_khi_khong_khung_nao_co_chu() -> None:
    from subtitles_extractor.application.services.subtitle_pipeline.event_filters import (
        blank_between,
        last_textful_at_or_before,
    )

    textful = [10.0, 10.04, 10.08, 10.20, 10.24]
    assert blank_between(textful, 10.08, 10.20)
    assert not blank_between(textful, 10.04, 10.20)
    assert not blank_between(None, 10.08, 10.20)
    assert last_textful_at_or_before(textful, 10.0, 10.20) == 10.08


# ── Rác Latin/số ───────────────────────────────────────────────────────────────────────────


def test_rac_latin_lan_so_ngan_bi_bo_chu_latin_that_duoc_giu() -> None:
    from subtitles_extractor.application.services.subtitle_pipeline.event_refinement import (
        _is_alnum_garbage,
    )

    assert _is_alnum_garbage("200 2a1s wses", 0.57, 0.19)
    assert not _is_alnum_garbage("iPhone 15", 0.95, 0.8)
    assert not _is_alnum_garbage("iPhone 15", 0.6, 1.2)       # hiện lâu: không phải mảnh lướt
    assert not _is_alnum_garbage("OK", 0.6, 0.2)              # không có chữ số
    assert not _is_alnum_garbage("第2集", 0.6, 0.2)           # có chữ CJK
