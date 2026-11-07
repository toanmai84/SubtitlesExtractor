"""[v3.23.669] Kế thừa ngữ cảnh phim bộ: gộp mệnh đề xưng hô, tóm tắt nối tiếp, tự phân tích tập chưa phân tích."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from subtitles_extractor.application.services.translation_memory import (
    merge_characters,
    recent_series_overview,
    split_episode_overviews,
    upsert_episode_overview,
)

EP1 = (
    "叶然 (Diệp Nhiên) [nam, ~20] — nhân vật chính; với Sở Dao tự xưng 'sư huynh' gọi 'sư muội'\n"
    "楚瑶 / 瑶瑶 (Sở Dao / Dao Dao) [nữ] — sư muội; với Diệp Nhiên tự xưng 'muội' gọi 'sư huynh'"
)
EP2 = (
    "叶然 (Diệp Nhiên) [nam] — đệ tử; với Nam Cung Mộng tự xưng 'đệ tử' gọi 'sư thúc'; "
    "với Dao Dao tự xưng 'huynh' gọi 'muội'\n"
    "南宫梦 (Nam Cung Mộng) [nữ] — sư thúc; với Diệp Nhiên tự xưng 'ta' gọi 'con'"
)


def test_gop_giu_dong_cu_va_them_menh_de_voi_nguoi_moi() -> None:
    merged = merge_characters(EP1, EP2).splitlines()
    ye = merged[0]
    assert ye.startswith("叶然 (Diệp Nhiên) [nam, ~20] — nhân vật chính")   # dòng cũ giữ nguyên
    assert "với Nam Cung Mộng tự xưng 'đệ tử' gọi 'sư thúc'" in ye          # quy tắc mới được thêm
    # "Dao Dao" là bí danh của Sở Dao → KHÔNG thêm mệnh đề thứ hai mâu thuẫn cho cùng cặp.
    assert "với Dao Dao" not in ye and ye.count("tự xưng") == 2
    assert any(line.startswith("南宫梦") for line in merged)
    assert len(merged) == 3


def test_gop_khong_doi_khi_khong_co_gi_moi() -> None:
    assert merge_characters(EP1, EP1) == EP1


def test_tom_tat_moi_tap_mot_khoi_theo_thu_tu_va_thay_khoi_cu() -> None:
    series = upsert_episode_overview("", "01", "Diệp Nhiên bị từ hôn.")
    series = upsert_episode_overview(series, "02", "Nam Cung Mộng xuất hiện.")
    series = upsert_episode_overview(series, "01", "Diệp Nhiên bị từ hôn (bản sửa).")
    assert [label for label, _ in split_episode_overviews(series)] == ["02", "01"]
    assert "bản sửa" in series and series.count("⟦01⟧") == 1


def test_khong_ghi_tom_tat_phim_bo_vao_khoi_cua_mot_tap() -> None:
    series = upsert_episode_overview("", "01", "A.")
    # Ô tóm tắt được điền từ chính tóm tắt phim bộ (tập chưa phân tích) → không lồng vào khối tập 02.
    assert upsert_episode_overview(series, "02", series) == series


def test_tran_do_dai_giu_tap_dau_va_cac_tap_moi_nhat() -> None:
    series = ""
    for ep in range(1, 30):
        series = upsert_episode_overview(series, f"{ep:02d}", f"Diễn biến tập {ep}. " * 40, max_chars=6000)
    labels = [label for label, _ in split_episode_overviews(series)]
    assert labels[0] == "01" and labels[-1] == "29" and len(series) <= 6000


def test_tom_tat_cu_khong_nhan_van_duoc_giu() -> None:
    series = upsert_episode_overview("Tóm tắt kiểu cũ.", "05", "Tập 5.")
    assert split_episode_overviews(series) == [("", "Tóm tắt kiểu cũ."), ("05", "Tập 5.")]


def test_dien_bien_cac_tap_truoc_bo_khoi_cua_chinh_tap() -> None:
    series = upsert_episode_overview(upsert_episode_overview("", "01", "Một."), "02", "Hai.")
    assert recent_series_overview(series, exclude_episode="02") == "⟦01⟧ Một."
    assert "Hai." in recent_series_overview(series, exclude_episode="03")


# ── View-model / trang Dịch ─────────────────────────────────────────────────

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _page():
    from tests.unit.presentation.test_translate_page_bugs_v323609 import _page as make

    return make()


def test_phan_tich_tap_sau_nhan_dien_bien_cac_tap_truoc(app, tmp_path) -> None:
    page, _container = _page()
    vm = page._view_model
    series = tmp_path / "PhimBo"
    series.mkdir()
    vm.accumulate_series_context(str(series / "01.mp4"), "", EP1, "Diệp Nhiên bị từ hôn.")
    vm.accumulate_series_context(str(series / "02.mp4"), "", EP2, "Nam Cung Mộng xuất hiện.")
    prior = vm._build_prior_series_context(str(series / "03.mp4"))
    assert "Diễn biến các tập trước" in prior
    assert "⟦01⟧ Diệp Nhiên bị từ hôn." in prior and "⟦02⟧ Nam Cung Mộng xuất hiện." in prior
    assert "với Nam Cung Mộng tự xưng 'đệ tử' gọi 'sư thúc'" in prior
    # Phân tích lại chính tập 02: không tự nhìn thấy khối của mình.
    assert "⟦02⟧" not in vm._build_prior_series_context(str(series / "02.mp4"))
    page.close()


def _batch_page(monkeypatch: pytest.MonkeyPatch, started: bool = True):
    page, _container = _page()
    calls: list[dict[str, Any]] = []

    def _start(**kwargs: Any) -> bool:
        calls.append(kwargs)
        return started

    monkeypatch.setattr(page._view_model, "start_context_analysis", _start)
    scheduled: list[Any] = []
    monkeypatch.setattr(page, "_start_batch_episode", lambda: scheduled.append(True))
    page._batch_items = [SimpleNamespace(video_path=Path("/phim/01.mp4")),
                         SimpleNamespace(video_path=Path("/phim/02.mp4"))]
    page._batch_cancelled = False
    page._reset_batch_analysis()
    page._combined_keys_text = lambda: "KEY"  # type: ignore[method-assign]
    page._source_video_path = "/phim/01.mp4"
    return page, calls, scheduled


def test_dich_ca_bo_tu_phan_tich_bang_chu_tap_chua_phan_tich(app, monkeypatch) -> None:
    from PySide6.QtWidgets import QApplication

    page, calls, scheduled = _batch_page(monkeypatch)
    page._source_analysis_restored = False
    assert page._maybe_analyze_batch_episode(0) is True
    (kwargs,) = calls
    assert kwargs["video_path"] is None and kwargs["enable_visual_cues"] is False
    assert kwargs["session_path"] == "/phim/01.mp4"
    # Phân tích xong → dịch tiếp đúng tập đó; không phân tích lại lần hai.
    assert page._batch_analysis_done(True) is True
    QApplication.processEvents()
    assert scheduled and page._batch_analyzed_count == 1
    assert page._maybe_analyze_batch_episode(0) is False
    page._batch_items = []
    page.close()


def test_tap_da_co_phan_tich_thi_khong_phan_tich_lai(app, monkeypatch) -> None:
    page, calls, _scheduled = _batch_page(monkeypatch)
    page._source_analysis_restored = True
    assert page._maybe_analyze_batch_episode(0) is False and calls == []
    page._batch_items = []
    page.close()


def test_loi_han_muc_khi_phan_tich_thi_dich_tiep_va_thoi_tu_phan_tich(app, monkeypatch) -> None:
    from PySide6.QtWidgets import QApplication

    page, calls, scheduled = _batch_page(monkeypatch)
    page._source_analysis_restored = False
    assert page._maybe_analyze_batch_episode(0)
    page._on_analyze_error("Đã hết hạn mức ngày của model ở mọi API key")
    QApplication.processEvents()
    assert scheduled and page._batch_auto_analyze is False   # tập này vẫn dịch, tập sau không thử nữa
    assert page._maybe_analyze_batch_episode(1) is False and len(calls) == 1
    page._batch_items = []
    page.close()


def test_phan_tich_bi_huy_khong_lam_treo_hang_doi(app, monkeypatch) -> None:
    from PySide6.QtWidgets import QApplication

    page, _calls, scheduled = _batch_page(monkeypatch)
    page._source_analysis_restored = False
    assert page._maybe_analyze_batch_episode(0)
    page._on_busy_changed(False)          # luồng phân tích kết thúc mà không có kết quả
    QApplication.processEvents()          # bộ hẹn giờ báo xong…
    QApplication.processEvents()          # …rồi bộ hẹn giờ dịch tiếp tập đó
    assert scheduled and page._batch_analyzing_index == -1
    page._batch_items = []
    page.close()


# ── Sau rà soát độc lập ─────────────────────────────────────────────────────


def test_tom_tat_co_dong_bat_dau_bang_ngoac_vuong_van_duoc_luu() -> None:
    series = upsert_episode_overview("", "01", "Mở đầu.")
    series = upsert_episode_overview(series, "02", "[Cảnh 1] A gặp B.\n[Hồi tưởng] B nhớ lại.")
    blocks = split_episode_overviews(series)
    assert [label for label, _ in blocks] == ["01", "02"]
    assert "[Hồi tưởng] B nhớ lại." in blocks[1][1]


def test_nhan_tap_ten_dai_va_co_ngoac_khop_khi_ghi_va_tra() -> None:
    from subtitles_extractor.application.services.translation_memory import episode_label

    a = "[SubsPlease] Ten Phim Rat Dai " + "x" * 80 + " - Episode 01 (1080p) [ABCD1234]"
    b = a.replace("Episode 01", "Episode 02")
    assert episode_label(a) != episode_label(b)
    series = upsert_episode_overview(upsert_episode_overview("", a, "Một."), b, "Hai.")
    assert recent_series_overview(series, exclude_episode=a).endswith("Hai.")
    assert "Một." not in recent_series_overview(series, exclude_episode=a)


def test_tom_tat_kieu_cu_dien_vao_o_khong_bi_ghi_thanh_cua_tap_moi() -> None:
    legacy = "Tóm tắt của tập trước (kiểu cũ, không nhãn)."
    assert upsert_episode_overview(legacy, "05", legacy) == legacy


def test_tran_do_dai_khong_lam_mat_nhan_tap_dau() -> None:
    series = upsert_episode_overview("", "01", "A" * 5000)
    series = upsert_episode_overview(series, "02", "B" * 7000, max_chars=6000)
    assert [label for label, _ in split_episode_overviews(series)] == ["01", "02"]
    assert len(series) <= 6000


def test_dong_chi_co_ten_khong_bi_doi_danh_tinh_khi_them_menh_de() -> None:
    first = merge_characters("Lâm Phong", "Lâm Phong: với Tô Tình tự xưng 'anh' gọi 'em'")
    again = merge_characters(first, "Lâm Phong: với Tô Tình tự xưng 'anh' gọi 'em'")
    assert again == first and again.count("Lâm Phong") == 1


def test_bien_the_ten_va_ngoac_kep_thang_khong_them_menh_de_trung() -> None:
    kept = "叶然 (Diệp Nhiên) [nam] — chính; với Sở Dao tự xưng 'sư huynh' gọi 'sư muội'"
    incoming = ("叶然 (Diệp Nhiên) [nam] — x; với Sở Dao (楚瑶) tự xưng \"huynh\" gọi \"muội\"; "
                "với nàng Sở Dao tự xưng 'ta' gọi 'nàng'")
    roster = kept + "\n楚瑶 (Sở Dao) [nữ] — sư muội"
    merged = merge_characters(roster, incoming).splitlines()[0]
    assert merged.count("tự xưng") == 1


def test_bat_dau_phan_tich_loi_ngay_thi_khong_dich_ngoai_hang_doi(app, monkeypatch) -> None:
    page, _container = _page()
    translated: list[bool] = []

    def _start_analysis(**_kw: Any) -> bool:
        page._batch_items = []            # lỗi cấu hình đã kết thúc hàng đợi (đồng bộ)
        return False

    monkeypatch.setattr(page._view_model, "start_context_analysis", _start_analysis)
    monkeypatch.setattr(page, "_start_translation_from_ui", lambda: translated.append(True) or True)
    page._batch_items = [SimpleNamespace(video_path=Path("/phim/01.mp4"))]
    page._batch_index = 0
    page._batch_cancelled = False
    page._reset_batch_analysis()
    page._combined_keys_text = lambda: "KEY"  # type: ignore[method-assign]
    page._source_analysis_restored = False
    page._start_batch_episode()
    assert translated == []
    page.close()


def test_het_han_muc_giua_chung_thi_thoi_tu_phan_tich(app, monkeypatch) -> None:
    from subtitles_extractor.domain.ports.subtitle_translator_port import (
        ANALYSIS_STOP_DAILY_QUOTA,
        SubtitleContextAnalysis,
    )

    page, _calls, _scheduled = _batch_page(monkeypatch)
    page._source_analysis_restored = False
    assert page._maybe_analyze_batch_episode(0)
    page._on_analyze_context_ready(SubtitleContextAnalysis(
        characters="叶然", overview="x", stop_reason=ANALYSIS_STOP_DAILY_QUOTA,
        incomplete_steps=("context",)))
    assert page._batch_auto_analyze is False
    page._batch_items = []
    page.close()
