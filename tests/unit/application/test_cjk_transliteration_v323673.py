"""[v3.23.673] Chữ Hán sót trong bản dịch → âm Hán-Việt; mục thuật ngữ dịch dở được cứu thay vì bị loại.

Chạy thật (tập 3 phim mẫu, gemini-3.5-flash-lite): thuật ngữ "九转烈阳蛊 = Cửu Chuyển Liệt
Dương C蛊" bị LOẠI khỏi prompt vì dịch dở, và dòng 16 "đã trúng Cửu Chuyển Liệt Dương C蛊" vẫn
còn chữ Hán sau khi dịch lại → giọng đọc gặp chữ Hán.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from subtitles_extractor.application.services.cjk_transliteration import (
    transliterate_leftover_cjk,
)
from subtitles_extractor.application.services.name_reading_check import NameReadings

ROOT = Path(__file__).resolve().parents[3] / "src" / "subtitles_extractor"
R = NameReadings.from_encoded({
    "蛊": "cổ", "先": "tiên", "生": "sinh", "秦": "tần", "雅": "nhã", "消": "tiêu",
    "之": "chi", "气": "khí", "公": "công", "姐": "tỷ",
})


@pytest.mark.parametrize(("text", "want"), [
    ("đã trúng Cửu Chuyển Liệt Dương C蛊", "đã trúng Cửu Chuyển Liệt Dương Cổ"),
    ("Cổ蛊 độc", "Cổ độc"),                          # model chép âm rồi chép thêm chữ
    ("Lâm先生 đến rồi", "Lâm Tiên Sinh đến rồi"),
    ("gặp 秦雅 ở đây", "gặp Tần Nhã ở đây"),
    ("thân tử đạo消", "thân tử đạo tiêu"),
    ("Cửu Chuyển Liệt Dương 蛊", "Cửu Chuyển Liệt Dương Cổ"),   # sau chữ hoa → hoa
    ("蛊 rất mạnh", "Cổ rất mạnh"),                              # đầu câu → hoa
    ("thuần dương之气", "thuần dương chi khí"),                  # dính chữ thường → thường
    ("con公", "con công"),                                       # "con" không phải phần đầu âm
    ("Phi Phi姐", "Phi Phi Tỷ"),
    ("không có chữ Hán", "không có chữ Hán"),
    ("ぁ 한", "ぁ 한"),                               # chữ Nhật/Hàn giữ nguyên
    ("", ""),
])
def test_doi_chu_han_sot(text, want) -> None:
    assert transliterate_leftover_cjk(text, R) == want


def test_chu_khong_co_am_giu_nguyen_ca_cum() -> None:
    assert transliterate_leftover_cjk("gặp 秦龘 rồi", R) == "gặp 秦龘 rồi"


def test_bang_am_rong_khong_doi_gi() -> None:
    assert transliterate_leftover_cjk("C蛊", NameReadings()) == "C蛊"


def test_bang_am_that_co_chu_trong_phim() -> None:
    from subtitles_extractor.infrastructure.translation.name_readings_loader import (
        load_name_readings,
    )

    real = load_name_readings(ROOT / "data" / "hanviet_readings.json")
    assert transliterate_leftover_cjk("Cửu Chuyển Liệt Dương C蛊", real) == (
        "Cửu Chuyển Liệt Dương Cổ")


# ── Thuật ngữ dịch dở ──────────────────────────────────────────────────────────────────────


def test_muc_thuat_ngu_dich_do_duoc_cuu_thay_vi_loai() -> None:
    from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
        _normalize_glossary,
    )

    raw = [{"term": "九转烈阳蛊", "translation": "Cửu Chuyển Liệt Dương C蛊"},
           {"term": "魔门", "translation": "Ma Môn"},
           {"term": "生死道龘", "translation": "sinh tử đạo龘"}]       # không có âm → vẫn loại
    assert "九转烈阳蛊" not in _normalize_glossary(raw)             # không có bảng âm: như cũ
    out = _normalize_glossary(raw, R)
    lines = out.splitlines()
    assert lines[0] == "九转烈阳蛊 = Cửu Chuyển Liệt Dương Cổ"       # giữ thứ tự gốc
    assert "魔门 = Ma Môn" in lines
    assert not any("龘" in line for line in lines)


# ── Bước dọn chữ Hán sót của khâu dịch ──────────────────────────────────────────────────────


def test_buoc_cuu_cuoi_cua_khau_dich(monkeypatch) -> None:
    from subtitles_extractor.application.use_cases.translate_subtitles import (
        TranslateSubtitlesUseCase,
    )
    from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine

    class Translator:
        def translate_stage(self, **kw):
            # Dịch lại vẫn còn chữ Hán (như lượt chạy thật).
            return [TranslationLine(index=1, start_ms=0, end_ms=900,
                                    text="đã trúng Cửu Chuyển Liệt Dương C蛊")]

    src = [TranslationLine(index=1, start_ms=0, end_ms=900, text="我们魔门九转烈阳蛊")]
    cur = [TranslationLine(index=1, start_ms=0, end_ms=900, text="đã trúng Cửu Chuyển C蛊")]
    request = SimpleNamespace(context=SimpleNamespace(characters="", glossary="",
                                                      target_lang="Vietnamese"),
                              video_refs=[])
    for readings, want in ((R, "đã trúng Cửu Chuyển Liệt Dương Cổ"),
                           (None, "đã trúng Cửu Chuyển C蛊")):        # không có bảng: như cũ
        uc = TranslateSubtitlesUseCase(Translator(), name_readings=readings)
        monkeypatch.setattr(uc, "_retranslation_stage", lambda *_a, **_k: object())
        out = uc._cleanup_residual_cjk(cur, src, request, None)
        assert out[0].text == want


def test_container_noi_bang_am_cho_khau_dich_va_bo_dich() -> None:
    src = (ROOT / "composition" / "container.py").read_text(encoding="utf-8")
    assert "name_readings=self.name_readings" in src
    assert "translator.name_readings = self.name_readings" in src


# ── Trang Debug: chọn nhầm tệp ─────────────────────────────────────────────────────────────


def test_chon_nham_tep_bao_loi_de_hieu(tmp_path: Path) -> None:
    from subtitles_extractor.infrastructure.serializers.raw_ocr_serializer import load_raw_ocr

    srt = tmp_path / "tap01.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\n师叔\n", encoding="utf-8")
    with pytest.raises(ValueError, match="không phải dữ liệu OCR thô"):
        load_raw_ocr(srt)


# ── [rà độc lập] Chỉ áp cho bản dịch tiếng Việt, chỉ cho phần SÓT nhỏ ──────────────────────


def _run_cleanup(monkeypatch, target_lang: str, current: str, retry: str) -> str:
    from subtitles_extractor.application.use_cases.translate_subtitles import (
        TranslateSubtitlesUseCase,
    )
    from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine

    calls: list[int] = []

    class Translator:
        def translate_stage(self, **kw):
            calls.append(len(kw["source_lines"]))
            return [TranslationLine(index=1, start_ms=0, end_ms=900, text=retry)]

    src = [TranslationLine(index=1, start_ms=0, end_ms=900, text="我爱你，你好吗？")]
    cur = [TranslationLine(index=1, start_ms=0, end_ms=900, text=current)]
    request = SimpleNamespace(context=SimpleNamespace(characters="", glossary="",
                                                      target_lang=target_lang),
                              video_refs=[])
    uc = TranslateSubtitlesUseCase(Translator(), name_readings=R_FULL)
    monkeypatch.setattr(uc, "_retranslation_stage", lambda *_a, **_k: object())
    out = uc._cleanup_residual_cjk(cur, src, request, None)[0].text
    return out, calls


R_FULL = NameReadings.from_encoded({ch: r for ch, r in zip(
    "我爱你好吗蛊", ["ngã", "ái", "nễ", "hảo", "ma", "cổ"], strict=True)})


@pytest.mark.parametrize("lang", ["Chinese (Simplified)", "Chinese (Traditional)",
                                  "Japanese", "Korean", "zh-CN"])
def test_dich_sang_tieng_trung_nhat_han_khong_dung_vao(monkeypatch, lang) -> None:
    out, calls = _run_cleanup(monkeypatch, lang, "我爱你，你好吗？", "我爱你，你好吗？")
    assert out == "我爱你，你好吗？"
    assert calls == []                                   # không gửi dịch lại vô ích


def test_dich_sang_tieng_anh_khong_chen_am_han_viet(monkeypatch) -> None:
    out, _ = _run_cleanup(monkeypatch, "English", "He used the 蛊 worm", "He used the 蛊 worm")
    assert out == "He used the 蛊 worm"


def test_ca_cau_chua_dich_khong_doi_am_de_co_qc_bao(monkeypatch) -> None:
    out, _ = _run_cleanup(monkeypatch, "Vietnamese", "我爱你，你好吗？", "我爱你，你好吗？")
    assert out == "我爱你，你好吗？"


def test_thuat_ngu_dich_tieng_anh_khong_duoc_cuu_bang_am_viet() -> None:
    from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
        GeminiSubtitleTranslator,
    )

    translator = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    translator.name_readings = R
    assert translator._readings_for("Vietnamese") is R
    assert translator._readings_for("English") is None
    assert translator._readings_for("Chinese (Simplified)") is None
