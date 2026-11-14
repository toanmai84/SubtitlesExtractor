"""[v3.23.671] Bảng nhân vật thiếu người KHÔNG TÊN có giọng riêng khi phân tích không video.

Người dùng (nghe phim): tập 1 có Diệp Nhiên, Hệ thống, Sở Dao, Nam đệ tử, Nữ đệ tử và nhóm đệ
tử phụ hoạ. Bảng lập bằng chữ chỉ có một mục "Đệ tử" (hoặc không có): nữ đệ tử 0/3 lượt. Đưa
âm thanh thẳng vào lượt lập bảng thì model nhỏ bịa tên ("林风") và đổi giới người đã có. Bước
NGHE GIỌNG riêng chỉ được THÊM người không tên — đo lại: nữ đệ tử 3/3 lượt, không thêm sai ai ở
tập 2-6.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from subtitles_extractor.application.services.roster_format import parse_roster
from subtitles_extractor.application.services.voice_roster import (
    MAX_NEW_VOICES,
    merge_voice_roster,
)
from subtitles_extractor.application.use_cases.analyze_subtitle_context import (
    AnalyzeSubtitleContextUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    ANALYSIS_FIELDS_ALL,
    ANALYSIS_FIELDS_CONTEXT,
    ANALYSIS_FIELDS_ROSTER,
    SubtitleContextAnalysis,
    SubtitleTranslationError,
    TranslationLine,
)

ROOT = Path(__file__).resolve().parents[3] / "src" / "subtitles_extractor"
ROSTER = (
    "叶然 (Diệp Nhiên) [nam, ~20] — nhân vật chính\n"
    "楚瑶 / 瑶瑶 (Sở Dao / Dao Dao) [nữ, ~18] — tiểu sư muội\n"
    "灵霄宗男弟子 (Nam đệ tử Linh Tiêu Tông) [nam] — đệ tử chế giễu Diệp Nhiên"
)
TEXT = "叶然你还不明白吗\n楚师姐都马上金丹了\n就是\n就是呀"
FEMALE = {"name": "灵霄宗女弟子", "aka": "Nữ đệ tử Linh Tiêu Tông", "gender": "nữ",
          "role": "đệ tử phụ hoạ; xưng hô: với 叶然 tự xưng 'ta' gọi 'ngươi'"}


def _names(characters: str) -> list[str]:
    return [e.name for e in parse_roster(characters)]


# ── Gộp có kiểm soát ────────────────────────────────────────────────────────────────────────


def test_them_nguoi_khong_ten_co_giong_rieng() -> None:
    out, added = merge_voice_roster(ROSTER, {"new_characters": [FEMALE]}, TEXT)
    assert added == ["灵霄宗女弟子"]
    entry = parse_roster(out)[-1]
    assert entry.name == "灵霄宗女弟子" and entry.gender == "nữ"
    assert _names(out)[:3] == _names(ROSTER)              # người đã có giữ nguyên


def test_tu_choi_ten_rieng_bia_khong_co_trong_phu_de() -> None:
    invented = {"name": "林风", "aka": "Lâm Phong", "gender": "nam", "role": "nhân vật chính"}
    out, added = merge_voice_roster(ROSTER, {"new_characters": [invented]}, TEXT)
    assert added == [] and out == ROSTER


def test_ten_rieng_co_trong_phu_de_thi_nhan() -> None:
    named = {"name": "秦风", "aka": "Tần Phong", "gender": "nam", "role": "sư huynh"}
    _out, added = merge_voice_roster(ROSTER, {"new_characters": [named]}, TEXT + "\n秦风来了")
    assert added == ["秦风"]


def test_khong_them_trung_nguoi_da_co_theo_bi_danh_hoac_phien_am() -> None:
    dupes = [
        {"name": "瑶瑶", "aka": "Dao Dao", "gender": "nữ", "role": "x"},
        {"name": "灵霄宗男弟子", "aka": "Nam đệ tử", "gender": "nam", "role": "x"},
        {"name": "女弟子", "aka": "Sở Dao", "gender": "nữ", "role": "x"},     # trùng phiên âm
    ]
    out, added = merge_voice_roster(ROSTER, {"new_characters": dupes}, TEXT + "\n瑶瑶")
    assert added == [] and out == ROSTER


def test_khong_doi_gioi_nguoi_da_co() -> None:
    patch = {"new_characters": [], "gender_fixes": [{"name": "叶然", "gender": "nữ"}]}
    out, added = merge_voice_roster(ROSTER, patch, TEXT)
    assert out == ROSTER and added == []


def test_ban_va_hong_hoac_sai_kieu_khong_lam_hong_bang() -> None:
    for patch in (None, [], "x", {"new_characters": "abc"}, {"new_characters": [None, 3, {}]},
                  {"new_characters": [{"name": "a/b", "aka": "", "gender": "", "role": ""}]}):
        out, added = merge_voice_roster(ROSTER, patch, TEXT)  # type: ignore[arg-type]
        assert out == ROSTER and added == []


def test_tran_so_nguoi_moi() -> None:
    many = [dict(FEMALE, name=f"路人{i}", aka=f"Người qua đường {i}") for i in range(20)]
    _out, added = merge_voice_roster(ROSTER, {"new_characters": many}, TEXT)
    assert len(added) == MAX_NEW_VOICES


def test_gioi_nhom_chuan_hoa() -> None:
    group = {"name": "灵霄宗众弟子", "aka": "Các đệ tử Linh Tiêu Tông", "gender": "nhóm",
             "role": "đám đông phụ hoạ"}
    out, _added = merge_voice_roster(ROSTER, {"new_characters": [group]}, TEXT)
    assert parse_roster(out)[-1].gender == "nhiều người"


# ── Luồng phân tích tách bước ───────────────────────────────────────────────────────────────

LINES = [TranslationLine(index=i, start_ms=i * 1000, end_ms=i * 1000 + 900, text=t)
         for i, t in enumerate(TEXT.splitlines(), 1)]


class Fake:
    def __init__(self, patch: Any = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.patch = {"new_characters": [FEMALE]} if patch is None else patch
        self.error = error

    def analyze_global_context(
        self, source_lines, target_lang, model_name="m", cancel_cb=None, video_refs=None,
        with_visual_cues=False, prior_context="", fields=ANALYSIS_FIELDS_ALL,
        pinned_characters="",
    ) -> SubtitleContextAnalysis:
        self.calls.append(dict(kind="global", fields=fields, pinned=pinned_characters))
        if fields == ANALYSIS_FIELDS_ROSTER:
            return SubtitleContextAnalysis(source_lang="zh", characters=ROSTER)
        if fields == ANALYSIS_FIELDS_CONTEXT:
            return SubtitleContextAnalysis(source_lang="zh", characters=pinned_characters,
                                           overview="Tóm tắt.", glossary="")
        return SubtitleContextAnalysis(source_lang="zh", characters=ROSTER, overview="gộp")

    def listen_roster_voices(self, source_lines, target_lang, model_name, characters,
                             voice_source, source_lang="", cancel_cb=None):
        self.calls.append(dict(kind="voices", characters=characters, path=voice_source,
                               lang=source_lang))
        if self.error is not None:
            raise self.error
        return self.patch


def _run(fake: Fake, path: str = "/phim/01.mp4") -> SubtitleContextAnalysis:
    uc = AnalyzeSubtitleContextUseCase(translator=fake)
    return uc.execute(LINES, "Vietnamese", model_name="m", self_audit=False,
                      voice_roster_path=path)


def test_nghe_giong_sau_bang_chu_va_truoc_tom_tat() -> None:
    fake = Fake()
    result = _run(fake)
    kinds = [c["kind"] if c["kind"] == "voices" else c["fields"] for c in fake.calls]
    assert kinds == [ANALYSIS_FIELDS_ROSTER, "voices", ANALYSIS_FIELDS_CONTEXT]
    assert fake.calls[1]["path"] == "/phim/01.mp4" and fake.calls[1]["lang"] == "zh"
    assert "灵霄宗女弟子" in fake.calls[2]["pinned"]          # bảng ghim cho bước tóm tắt
    assert "灵霄宗女弟子" in result.characters


def test_khong_co_duong_dan_thi_khong_nghe() -> None:
    fake = Fake()
    result = _run(fake, path="")
    assert all(c["kind"] != "voices" for c in fake.calls)
    assert "灵霄宗女弟子" not in result.characters


def test_loi_buoc_nghe_giong_giu_bang_chu() -> None:
    for error in (SubtitleTranslationError("503"), RuntimeError("hết hạn mức ngày")):
        fake = Fake(error=error)
        result = _run(fake)
        assert result.characters.splitlines()[:3] == ROSTER.splitlines()
        assert result.overview == "Tóm tắt."
        assert not result.incomplete_steps


def test_bo_dich_khong_ho_tro_nghe_giong_van_chay() -> None:
    class Plain(Fake):
        listen_roster_voices = None  # type: ignore[assignment]

    result = _run(Plain())
    assert result.overview == "Tóm tắt."


# ── Chỉ dẫn + nối dây giao diện ─────────────────────────────────────────────────────────────


def test_chi_dan_nghe_giong_chi_them_khong_doi_ten() -> None:
    adapter = (ROOT / "infrastructure/translation/gemini_translation_adapter.py").read_text(
        encoding="utf-8")
    assert "KHÔNG đổi, không bỏ, không đặt lại tên ai" in adapter
    assert "TUYỆT ĐỐI KHÔNG đặt tên riêng không có" in adapter
    assert "def listen_roster_voices(" in adapter
    # Người không tên tách theo từng người khi chỉ đọc chữ.
    assert "mỗi NGƯỜI nói một mục riêng" in adapter


def test_tuy_chon_giao_dien_co_chuoi_va_luu_cai_dat() -> None:
    import json

    from tests.unit.page_sources import TRANSLATE_PAGE_SOURCE

    page = TRANSLATE_PAGE_SOURCE.read_text(encoding="utf-8")  # [v3.23.694] trang + các phần đã tách
    assert "translate.opt_voice_roster" in page and '"voice_roster"' in page
    for lang in ("vi", "en"):
        data = json.loads((ROOT / f"data/strings_{lang}.json").read_text(encoding="utf-8"))
        assert data["translate"]["opt_voice_roster"] and data["translate"]["opt_voice_roster_tip"]


def test_view_model_chuyen_duong_dan_cho_worker() -> None:
    from subtitles_extractor.presentation.view_models.translate_page_view_model import (
        TranslatePageViewModel,
    )

    vm = TranslatePageViewModel.__new__(TranslatePageViewModel)
    vm.set_voice_roster(True, "/phim/01.mp4")
    assert vm._voice_roster_path == "/phim/01.mp4"
    vm.set_voice_roster(False, "/phim/01.mp4")
    assert vm._voice_roster_path == ""


def test_gioi_nhieu_nguoi_dung_mot_minh_van_la_gioi() -> None:
    entry = parse_roster("众弟子 (Các đệ tử) [nhiều người] — đám đông")[0]
    assert entry.gender == "nhiều người" and entry.age == ""
    assert parse_roster("A (B) [nam] — x")[0].gender == "nam"
    assert parse_roster("A (B) [~20] — x")[0].age == "~20"


# ── Rà độc lập: lỗ hổng bộ lọc ──────────────────────────────────────────────────────────────

R2 = "楚瑶 / 瑶瑶 (Sở Dao / Dao Dao) [nữ] — x\n门下弟子 (Đệ tử) — y"
T2 = "瑶儿你来了。我们走吧。灵霄宗 秦风来了"


def _try(name: str, gender: str = "nữ") -> list[str]:
    patch = {"new_characters": [{"name": name, "aka": "A", "gender": gender, "role": "r"}]}
    return merge_voice_roster(R2, patch, T2)[1]


def test_ten_bia_ghep_chu_le_bi_tu_choi() -> None:
    for name in ("林女", "李群", "苏声", "叶仆", "众", "女", "林风弟子", "王甲"):
        assert _try(name) == [], name


def test_dai_tu_biet_danh_va_ten_pha_cu_phap_bi_tu_choi() -> None:
    for name in ("我", "我们", "瑶儿", "女弟子 (x)", "旁白, 男", "旁白：男"):
        assert _try(name) == [], name


def test_nhan_mo_ta_hop_le_van_nhan() -> None:
    for name in ("灵霄宗女弟子", "灵霄宗男弟子2", "旁白", "众弟子", "女弟子", "秦风"):
        assert _try(name) == [name], name


def test_gioi_tinh_chuan_hoa_theo_tien_to() -> None:
    for raw, want in (("nữ giới", "nữ"), ("Nam giới", "nam"), ("nhóm người", "nhiều người"),
                      ("female", "nữ"), ("???", "")):
        out, _ = merge_voice_roster(R2, {"new_characters": [
            {"name": "灵霄宗女弟子", "aka": "A", "gender": raw, "role": "r"}]}, T2)
        assert parse_roster(out)[-1].gender == want, raw


def test_buoc_nghe_giong_bo_qua_phim_qua_dai_va_kiem_huy() -> None:
    adapter = (ROOT / "infrastructure/translation/gemini_translation_adapter.py").read_text(
        encoding="utf-8")
    body = adapter.split("def listen_roster_voices(", 1)[1].split("\n    def ", 1)[0]
    assert "MAX_DIARIZED_AUDIO_SEC" in body and "TranslationCancelledError" in body
    assert body.index("_read_duration_sec") < body.index("_dialogue_audio_part")
    assert "VOICE_AUDIO_MAX_BYTES = 14_000_000" in adapter
