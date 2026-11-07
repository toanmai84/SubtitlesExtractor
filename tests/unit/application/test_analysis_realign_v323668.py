"""[v3.23.668] Giữ phân tích khi ReOCR thêm/bớt/tách/gộp/sửa dòng: căn dòng mới ↔ dòng cũ."""

from __future__ import annotations

import json

from subtitles_extractor.application.services.analysis_realign import (
    SourceLine,
    decode_lines,
    encode_lines,
    realign,
    remap_cues,
)

OLD = [
    SourceLine(0, 1000, "师兄，我错了"),
    SourceLine(1000, 2000, "你走吧"),
    SourceLine(2000, 3000, "我不走"),
    SourceLine(3000, 4000, "那就留下"),
    SourceLine(4000, 5000, "好"),
]
CUES = json.dumps([
    {"id": 1, "spk": "瑶瑶", "to": "叶然", "cue": ""},
    {"id": 2, "spk": "叶然", "to": "瑶瑶", "cue": ""},
    {"id": 3, "spk": "瑶瑶", "to": "叶然", "cue": ""},
    {"id": 4, "spk": "叶然", "to": "瑶瑶", "cue": ""},
    {"id": 5, "spk": "瑶瑶", "to": "叶然", "cue": ""},
], ensure_ascii=False)


def _speakers(raw: str) -> dict[int, str]:
    return {int(c["id"]): c["spk"] for c in json.loads(raw or "[]")}


def test_khong_doi_gi() -> None:
    alignment = realign(OLD, OLD)
    assert alignment.identical and alignment.reusable
    assert _speakers(remap_cues(CUES, alignment)) == _speakers(CUES)


def test_them_mot_dong_giua_khong_lech_nguoi_noi_phia_sau() -> None:
    new = [*OLD[:2], SourceLine(2000, 2400, "啊？"), SourceLine(2400, 3000, "我不走"), *OLD[3:]]
    alignment = realign(OLD, new)
    assert alignment.unmatched == (3,)
    got = _speakers(remap_cues(CUES, alignment))
    # Dòng 4..6 mới mang đúng người nói của dòng 3..5 cũ (giữ theo SỐ dòng sẽ lệch hết).
    assert [got[4], got[5], got[6]] == ["瑶瑶", "叶然", "瑶瑶"]
    assert 3 not in got  # dòng mới thật: để trống, bước "Bổ sung người nói" hỏi lại


def test_xoa_tach_gop_va_sua_chu() -> None:
    new = [
        SourceLine(0, 1000, "师兄，我错啦"),            # sửa một chữ
        SourceLine(1000, 1500, "你走"), SourceLine(1500, 2000, "吧"),   # tách
        # dòng 3 cũ bị xoá (rác)
        SourceLine(3000, 5000, "那就留下好"),          # gộp 4+5
    ]
    alignment = realign(OLD, new)
    got = _speakers(remap_cues(CUES, alignment))
    assert got[1] == "瑶瑶" and got[2] == "叶然" and got[4] == "叶然"
    assert alignment.reusable


def test_phu_de_khac_han_thi_khong_dung_lai() -> None:
    other = [SourceLine(i * 1000, i * 1000 + 900, f"完全不同{i}") for i in range(20, 25)]
    assert not realign(OLD, other).reusable


def test_ma_hoa_luu_phien_doc_lai_duoc_va_chiu_du_lieu_hong() -> None:
    assert decode_lines(encode_lines(OLD)) == OLD
    assert decode_lines("") == [] and decode_lines("{") == [] and decode_lines('[[1,"x"]]') == []


# ── View-model + kho phiên thật ─────────────────────────────────────────────

import pytest  # noqa: E402

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _events(lines: list[SourceLine]):
    from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
    from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

    return [SubtitleEvent(index=i + 1, text=ln.text,
                          interval=TimeInterval(ln.start_ms / 1000, ln.end_ms / 1000))
            for i, ln in enumerate(lines)]


def _vm():
    from tests.unit.presentation.test_translate_page_bugs_v323609 import _page

    page, container = _page()
    return page, page._view_model, container.translation_session_store


def _save_old_analysis(vm, store, video: str, with_lines: bool = True) -> None:
    from subtitles_extractor.application.services.translation_session_hashing import (
        hash_analysis_input,
    )

    store.save_analysis(
        video, characters="叶然 (Diệp Nhiên) [nam] — nhân vật chính", overview="Tóm tắt cũ",
        source_lang="zh", target_lang="Vietnamese",
        input_hash=hash_analysis_input([ln.text for ln in OLD], "Vietnamese"),
        glossary="师兄 = sư huynh", visual_cues=CUES,
        source_lines=encode_lines(OLD) if with_lines else None,
    )


def test_reocr_them_dong_van_khoi_phuc_phan_tich_va_can_nguoi_noi(app, tmp_path) -> None:
    page, vm, store = _vm()
    video = str(tmp_path / "01.mp4")
    _save_old_analysis(vm, store, video)
    new = [*OLD[:2], SourceLine(2000, 2400, "啊？"), SourceLine(2400, 3000, "我不走"), *OLD[3:]]
    vm._source_events = _events(new)
    restored = vm.restore_analysis_for_video(video, "Vietnamese")
    assert restored is not None
    assert restored.characters.startswith("叶然") and restored.overview == "Tóm tắt cũ"
    got = _speakers(restored.visual_cues)
    assert [got[4], got[5], got[6]] == ["瑶瑶", "叶然", "瑶瑶"] and 3 not in got
    assert "1 dòng mới cần hỏi lại người nói" in vm.last_restore_note
    # Đã lưu lại theo phụ đề mới → lần mở sau khôi phục thẳng, không căn lại.
    again = vm.restore_analysis_for_video(video, "Vietnamese")
    assert again is not None and vm.last_restore_note == ""
    assert _speakers(again.visual_cues) == got
    page.close()


def test_phien_cu_chua_luu_dong_hoac_khac_ngon_ngu_thi_phan_tich_lai(app, tmp_path) -> None:
    page, vm, store = _vm()
    video = str(tmp_path / "02.mp4")
    _save_old_analysis(vm, store, video, with_lines=False)
    vm._source_events = _events([*OLD[:4], SourceLine(4000, 5000, "好的")])
    assert vm.restore_analysis_for_video(video, "Vietnamese") is None
    _save_old_analysis(vm, store, video)
    assert vm.restore_analysis_for_video(video, "English") is None
    page.close()


def test_luu_phan_tich_kem_dong_phu_de(app, tmp_path) -> None:
    from subtitles_extractor.domain.ports.subtitle_translator_port import (
        SubtitleContextAnalysis,
        TranslationLine,
    )

    page, vm, store = _vm()
    video = str(tmp_path / "03.mp4")
    lines = [TranslationLine(index=i + 1, start_ms=ln.start_ms, end_ms=ln.end_ms, text=ln.text)
             for i, ln in enumerate(OLD)]
    vm._pending_analysis_video_path = video
    vm._pending_analysis_target_lang = "Vietnamese"
    vm._pending_analysis_source_lines = [ln.text for ln in lines]
    vm._pending_analysis_spans = [(ln.start_ms, ln.end_ms, ln.text) for ln in lines]
    vm._persist_analysis_result(SubtitleContextAnalysis(characters="叶然", overview="x"))
    assert decode_lines(store.get(video).analysis_lines) == OLD
    page.close()


# ── Sau rà soát độc lập ─────────────────────────────────────────────────────


def _film(n: int) -> list[SourceLine]:
    return [SourceLine(i * 2000, i * 2000 + 1500, f"第{i}句台词内容") for i in range(n)]


def test_doan_trich_ngan_khong_ghi_de_phan_tich_ca_phim() -> None:
    """Đoạn trích 60/1000 dòng: phía mới khớp 100% nhưng chỉ phủ 6% phim → không dùng lại."""
    full = _film(1000)
    alignment = realign(full, full[:60])
    assert alignment.matched_ratio == 1.0 and alignment.old_coverage < 0.1
    assert not alignment.reusable


def test_cau_ngan_lap_lai_khong_neo_sai_cho_khac() -> None:
    old = [SourceLine(0, 800, "嗯"), SourceLine(1000, 2000, "你来了"), SourceLine(2000, 3000, "坐吧"),
           SourceLine(3000, 3800, "好的"), SourceLine(4000, 5000, "走吧")]
    # "嗯" mới ở 3,0 s thay cho "好的" (OCR sửa) — trùng chữ với dòng 1 nhưng lệch 3 s.
    new = [SourceLine(1000, 2000, "你来了"), SourceLine(2000, 3000, "坐吧"), SourceLine(3000, 3800, "嗯"),
           SourceLine(4000, 5000, "走吧")]
    alignment = realign(old, new)
    assert alignment.mapping.get(3) != 1     # không mượn người nói của "嗯" ở 0 s
    assert alignment.mapping[1] == 2 and alignment.mapping[4] == 5


def test_vung_doi_lon_khong_lam_dung_giao_dien() -> None:
    import time

    old = _film(5000)
    new = [SourceLine(ln.start_ms, ln.end_ms, ln.text + "。") for ln in old]   # chuẩn hoá dấu câu cả phim
    t0 = time.perf_counter()
    alignment = realign(old, new)
    assert time.perf_counter() - t0 < 1.5
    assert alignment.reusable and len(alignment.mapping) == 5000


def test_chi_xoa_dong_rac_van_bao_dung_so_dong_doi() -> None:
    alignment = realign(OLD, [OLD[0], OLD[1], OLD[3], OLD[4]])
    assert alignment.removed == 1


def test_phan_tich_cu_duoc_ghi_bo_sung_dong_khi_mo_lai(app, tmp_path) -> None:
    page, vm, store = _vm()
    video = str(tmp_path / "04.mp4")
    _save_old_analysis(vm, store, video, with_lines=False)
    vm._source_events = _events(OLD)
    assert vm.restore_analysis_for_video(video, "Vietnamese") is not None
    assert decode_lines(store.get(video).analysis_lines) == OLD
    # Lần ReOCR sau đã căn dòng được.
    vm._source_events = _events([*OLD[:4], SourceLine(4000, 5000, "好啊")])
    assert vm.restore_analysis_for_video(video, "Vietnamese") is not None
    assert vm.last_restore_note
    page.close()


def test_bo_ket_qua_hoi_lai_nguoi_noi_khi_phu_de_doi_giua_chung(app) -> None:
    page, vm, _store = _vm()
    vm._speaker_completion_source_version = vm._source_version
    assert vm._speaker_job_matches_source()
    vm.set_source_events(_events(OLD))           # ReOCR nạp lại phụ đề trong lúc đang chạy
    assert not vm._speaker_job_matches_source()
    page.close()
