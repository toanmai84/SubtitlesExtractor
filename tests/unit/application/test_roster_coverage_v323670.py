"""[v3.23.670] Phân tích không video thiếu nhân vật.

Đo tập 1-6 phim mẫu (3 lượt, gemini-3.5-flash-lite, không video): bảng nhân vật chỉ phủ 52/78
nhân vật theo đáp án — "hệ thống" có thoại bị bỏ (quy tắc cũ cấm mọi "hệ thống"), sư phụ/sư tỷ
bế quan, nhóm đệ tử chế giễu, "前辈" mất hẳn; còn bước tự kiểm tra chỉ phát hiện thiếu người qua
NHÃN người nói nên không video thì không bao giờ chạy. Bản vá "thêm cách gọi vào tên" của nó lại
bị bỏ vì khi gộp chỉ giữ tên cũ.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from subtitles_extractor.application.services.analysis_audit import (
    DECLINE_LIMIT,
    UNLISTED_MENTION_KIND,
    apply_audit_patch,
    defect_units,
    filter_declined,
    find_analysis_defects,
)
from subtitles_extractor.application.services.roster_coverage import (
    find_unlisted_mentions,
)
from subtitles_extractor.application.use_cases.analyze_subtitle_context import (
    AnalyzeSubtitleContextUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleContextAnalysis,
)

ROOT = Path(__file__).resolve().parents[3] / "src" / "subtitles_extractor"
ADAPTER = (ROOT / "infrastructure" / "translation" / "gemini_translation_adapter.py").read_text(
    encoding="utf-8")

YE = "叶然 (Diệp Nhiên) [nam] — nhân vật chính; xưng hô: với Nam Cung Mộng tự xưng 'ta' gọi 'nàng'"
NAN = "南宫梦 (Nam Cung Mộng) [nữ] — sư thúc; xưng hô: với Diệp Nhiên tự xưng 'ta' gọi 'tiền bối'"


def _lines(*texts: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(index=i, text=t, original_text=t) for i, t in enumerate(texts, 1)]


def _terms(characters: str, lines) -> list[str]:
    return [m.term for m in find_unlisted_mentions(characters, lines)]


# ── Dò cách gọi chưa có trong bảng ──────────────────────────────────────────────────────────


def test_he_thong_co_thoai_chua_co_trong_bang_bi_phat_hien() -> None:
    lines = _lines("系统：恭喜宿主完成任务", "系统奖励已发放", "叶然你在干什么")
    assert _terms(YE, lines) == ["系统"]


def test_cach_goi_da_co_qua_phien_am_thi_khong_bao() -> None:
    lines = _lines("系统：恭喜宿主", "系统奖励已发放")
    roster = YE + "\n系统 (Hệ thống) — hệ thống đồng hành"
    assert _terms(roster, lines) == []
    # Chỉ có PHIÊN ÂM (không chữ gốc) cũng tính là đã có.
    assert _terms(YE + "\nThống tử / Hệ thống — hệ thống", lines) == []


def test_ho_cong_chuc_danh_lay_dung_mot_chu_ho() -> None:
    lines = _lines("今日和秦师兄约好", "秦师兄来了")
    assert _terms(YE, lines) == ["秦师兄"]


def test_ho_kep_va_khong_dem_chuc_danh_tron_cung_cho() -> None:
    lines = _lines("南宫师叔来了", "见过南宫师叔")
    assert _terms(NAN, lines) == []
    assert _terms(YE, lines) == ["南宫师叔"]


def test_tu_dem_than_tu_va_canh_gioi_khong_phai_ten() -> None:
    lines = _lines("呦师叔", "呦师叔", "合体境", "合体境", "什么", "什么")
    # 师叔 vẫn là một người cụ thể (chức danh), nhưng "呦" không phải họ, "合体境" không phải tên.
    assert _terms(YE, lines) == ["师叔"]


def test_ten_dung_rieng_thanh_loi_goi_la_bi_danh_chua_ghi() -> None:
    lines = _lines("瑶瑶", "瑶瑶啊", "你怎么来了")
    assert _terms("楚瑶 (Sở Dao) [nữ] — tiểu sư muội", lines) == ["瑶瑶"]


def test_mot_lan_le_khong_bao() -> None:
    assert _terms(YE, _lines("师父他老人家", "走吧")) == []


# ── Tự kiểm tra chạy được khi KHÔNG có nhãn người nói ───────────────────────────────────────


def test_khong_video_van_tim_ra_thieu_nguoi() -> None:
    lines = _lines("昨晚救我的前辈呢", "前辈", "要不是有那位前辈")
    analysis = SubtitleContextAnalysis(source_lang="zh", characters=NAN, overview="x")
    defects = find_analysis_defects(analysis, lines)
    kinds = [d.kind for d in defects]
    assert UNLISTED_MENTION_KIND in kinds
    mention = next(d for d in defects if d.kind == UNLISTED_MENTION_KIND)
    assert mention.line_ids and "'前辈'" in mention.detail


def test_don_vi_khiem_khuyet_theo_cach_goi_va_bi_bo_qua_du_lan_thi_thoi() -> None:
    lines = _lines("前辈", "前辈呢")
    analysis = SubtitleContextAnalysis(source_lang="zh", characters=NAN, overview="x")
    defect = next(d for d in find_analysis_defects(analysis, lines)
                  if d.kind == UNLISTED_MENTION_KIND)
    assert defect_units(defect) == [f"{UNLISTED_MENTION_KIND}:前辈"]
    assert filter_declined([defect], {f"{UNLISTED_MENTION_KIND}:前辈": DECLINE_LIMIT}) == []
    assert filter_declined([defect], {}) == [defect]


# ── Gộp bản vá: hợp bí danh ─────────────────────────────────────────────────────────────────


def test_ban_va_them_cach_goi_vao_ten_duoc_giu() -> None:
    analysis = SubtitleContextAnalysis(source_lang="zh", characters=YE, overview="x")
    patched, changes = apply_audit_patch(
        analysis, {"characters": ["叶然 / 前辈 (Diệp Nhiên / Tiền bối) [nam] — nhân vật chính"]},
        set())
    assert changes == 1
    head = patched.characters.splitlines()[0]
    assert "前辈" in head and "Tiền bối" in head
    assert "xưng hô: với Nam Cung Mộng" in head          # vai trò/xưng hô cũ giữ nguyên
    lines = _lines("前辈", "前辈呢")
    assert not [d for d in find_analysis_defects(patched, lines) if d.kind == UNLISTED_MENTION_KIND]


def test_ban_va_dat_cach_goi_moi_len_truoc_khong_tao_muc_trung() -> None:
    analysis = SubtitleContextAnalysis(source_lang="zh", characters=YE, overview="x")
    patched, _ = apply_audit_patch(
        analysis, {"characters": ["前辈 / 叶然 (Tiền bối / Diệp Nhiên) [nam] — nhân vật chính"]},
        set())
    rows = [r for r in patched.characters.splitlines() if r.strip()]
    assert len(rows) == 1
    assert rows[0].startswith("叶然")                      # tên chính giữ ở đầu
    assert "前辈" in rows[0]


def test_ban_va_khong_doi_gi_thi_khong_tinh_thay_doi() -> None:
    analysis = SubtitleContextAnalysis(source_lang="zh", characters=YE, overview="x")
    _patched, changes = apply_audit_patch(
        analysis, {"characters": ["叶然 (Diệp Nhiên) [nam] — khác"]}, set())
    assert changes == 0


# ── Trọn luồng: phân tích chữ → tự kiểm tra bổ sung người thiếu ─────────────────────────────


class _Translator:
    def __init__(self) -> None:
        self.audits: list[str] = []

    def audit_analysis(self, lines, lang, model, characters, glossary, overview, labels,
                       defects_json, focus_ids, cancel_cb=None):
        self.audits.append(defects_json)
        return {"characters": ["师父 (Sư phụ) [nữ] — sư phụ của Diệp Nhiên, đang bế quan"]}


def test_tu_kiem_tra_khong_video_bo_sung_nguoi_thieu() -> None:
    lines = _lines("师父闭关了", "等师父出关", "叶然你好")
    translator = _Translator()
    use_case = AnalyzeSubtitleContextUseCase(translator, audit_ledger=None)
    analysis = SubtitleContextAnalysis(source_lang="zh", characters=YE, overview="Tóm tắt.")
    result = use_case.refine(analysis, lines, "Vietnamese", "m")
    assert len(translator.audits) == 1 and "师父" in translator.audits[0]
    assert "师父 (Sư phụ)" in result.characters
    assert use_case.last_audit_report.remaining == 0


# ── Chỉ dẫn lập bảng nhân vật ───────────────────────────────────────────────────────────────


def test_chi_dan_bang_nhan_vat_gom_thuc_the_biet_noi_va_quy_trinh_ra_soat() -> None:
    assert "CHỈ LIỆT KÊ CON NGƯỜI. TUYỆT ĐỐI KHÔNG đưa vào đây: hệ thống" not in ADAPTER
    for needle in ("thực thể biết NÓI", "QUY TRÌNH RÀ SOÁT", "mỗi NGƯỜI nói một mục riêng",
                   "NGỮ CẢNH CÁC TẬP TRƯỚC", "KHÔNG có audio/video thì suy giới tính",
                   "'unlisted_mention'"):
        assert needle in ADAPTER, needle


def test_ban_va_khong_tu_gan_nguoi_noi_khi_phan_tich_chua_co_nhan() -> None:
    analysis = SubtitleContextAnalysis(source_lang="zh", characters=YE, overview="x")
    patched, changes = apply_audit_patch(
        analysis, {"cue_fixes": [{"id": 1, "spk": "叶然", "to": "南宫梦"}]}, {1, 2})
    assert changes == 0 and not patched.visual_cues


def test_ban_va_van_sua_nguoi_noi_khi_da_co_nhan() -> None:
    from subtitles_extractor.application.services.visual_cue_serializer import (
        serialize_visual_cues,
    )
    from subtitles_extractor.domain.ports.subtitle_translator_port import VisualCue

    analysis = SubtitleContextAnalysis(
        source_lang="zh", characters=YE + "\n" + NAN, overview="x",
        visual_cues=serialize_visual_cues([VisualCue(1, "南宫梦", "叶然")]))
    patched, changes = apply_audit_patch(
        analysis, {"cue_fixes": [{"id": 1, "spk": "叶然", "to": "南宫梦"}]}, {1, 2})
    assert changes == 1 and "叶然" in patched.visual_cues.split("南宫梦")[0]


# ── Rà độc lập: nhiễu & gộp nhầm người ─────────────────────────────────────────────────────


def test_tu_thuong_dung_rieng_mot_dong_khong_bi_coi_la_ten() -> None:
    words = ["住手", "放心", "奇怪", "加油", "抱歉", "站住", "爸爸", "妈妈", "老公", "老婆", "宝贝",
             "哥哥", "姐姐", "帅哥", "表哥"]
    lines = _lines(*[w for w in words for _ in range(2)], "人总要长大", "人总是这样",
                   "事情总会过去的", "事情总有办法", "师傅去机场", "师傅快点")
    assert _terms(YE, lines) == []


def test_phu_de_nhat_va_ngon_ngu_khac_khong_do() -> None:
    ja = _lines("先生、ありがとう", "先生です", "殿下が来た", "殿下")
    assert _terms(YE, ja) == []
    analysis = SubtitleContextAnalysis(source_lang="ja", characters=YE, overview="x")
    kanji = _lines("先輩", "先輩", "殿下", "殿下")
    assert not [d for d in find_analysis_defects(analysis, kanji)
                if d.kind == UNLISTED_MENTION_KIND]


def test_phon_the_duoc_do_nhu_gian_the() -> None:
    lines = _lines("師父閉關了", "等師父出關")
    assert _terms(YE, lines) == ["師父"]
    assert _terms(YE + "\n师父 (Sư phụ) — sư phụ", lines) == []      # khớp qua cách đọc


def test_thuat_ngu_da_xep_vao_glossary_khong_bao_lai() -> None:
    lines = _lines("宠妻系统已激活", "宠妻系统奖励")
    assert find_unlisted_mentions(YE, lines, glossary="宠妻系统 = Hệ thống sủng thê") == []


def test_hai_nguoi_khong_bi_gop_lam_mot_qua_bi_danh_chung() -> None:
    from subtitles_extractor.application.services.roster_format import parse_roster

    roster = ("叶然 / 前辈 (Diệp Nhiên / Tiền bối) [nam] — nhân vật chính\n"
              "苏白 (Tô Bạch) [nữ] — sư tỷ")
    analysis = SubtitleContextAnalysis(source_lang="zh", characters=roster, overview="x")
    patched, _ = apply_audit_patch(
        analysis, {"characters": ["苏白 / 前辈 (Tô Bạch / Tiền bối) [nữ] — sư tỷ"]}, set())
    entries = parse_roster(patched.characters)
    assert [e.name.split("/")[0].strip() for e in entries] == ["叶然", "苏白"]
    assert entries[1].gender == "nữ"


def test_muc_giu_cho_chuc_danh_khong_nuot_nguoi_co_ten() -> None:
    analysis = SubtitleContextAnalysis(
        source_lang="zh", characters="师父 (Sư phụ) [nữ] — sư phụ của A", overview="x")
    patched, _ = apply_audit_patch(
        analysis, {"characters": ["青云子 / 师父 (Thanh Vân Tử / Sư phụ) [nam] — sư phụ của B"]},
        set())
    rows = [r for r in patched.characters.splitlines() if r.strip()]
    assert rows[0].startswith("师父 (Sư phụ) [nữ]")
    assert any(r.startswith("青云子") and "[nam" in r for r in rows)


def test_ban_va_chep_lai_dong_khong_doi_phien_am_cung_khong_tinh_thay_doi() -> None:
    analysis = SubtitleContextAnalysis(source_lang="zh", characters=YE, overview="x")
    patched, changes = apply_audit_patch(
        analysis, {"characters": ["叶然 (Ye Ran) [nam] — nhân vật chính"]}, set())
    assert changes == 0 and "Ye Ran" not in patched.characters
    spaced = SubtitleContextAnalysis(source_lang="zh", characters="胖子/三胖子 (Mập) — bạn", overview="x")
    _p, changes = apply_audit_patch(spaced, {"characters": ["胖子 (Mập) — bạn"]}, set())
    assert changes == 0


def test_them_bi_danh_chi_them_cach_doc_tuong_ung() -> None:
    analysis = SubtitleContextAnalysis(
        source_lang="zh", characters="叶然 / 叶子 (Diệp Nhiên / Diệp Tử) [nam] — x", overview="x")
    patched, changes = apply_audit_patch(
        analysis, {"characters": ["叶然 / 前辈 (Ye Ran / Tiền bối) [nam] — x"]}, set())
    head = patched.characters.splitlines()[0]
    assert changes == 1
    assert head.startswith("叶然 / 叶子 / 前辈 (Diệp Nhiên / Diệp Tử / Tiền bối)")


def test_dong_tu_truoc_chuc_danh_khong_phai_ho() -> None:
    terms = _terms(NAN, _lines("我去找师叔", "快找师叔来"))
    assert "找师叔" not in terms and terms == ["师叔"]   # 师叔 trơn vẫn là một người cụ thể
