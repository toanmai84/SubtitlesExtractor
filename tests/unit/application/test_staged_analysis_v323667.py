"""[v3.23.667] Phân tích ngữ cảnh TÁCH BƯỚC: nhân vật → tóm tắt/thuật ngữ → người nói.

Đo 54 lượt trên tập 1-3 phim mẫu: lời gọi gộp (video + cues) cho bảng nhân vật/xưng hô 73%,
lập riêng bằng chữ 84%; quy tắc xưng hô 38% → 62%. Các test ở đây khoá luồng mới: video chỉ
đi tới bước người nói, bảng nhân vật đã chốt được ghim vào bước tóm tắt, người nói lạ do bước
người nói tìm ra được thêm vào bảng nhân vật.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from subtitles_extractor.application.services.cue_label_canonicalizer import (
    add_cue_speakers_to_roster,
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
    VisualCue,
)

ROSTER = "叶然 (Diệp Nhiên) [nam, ~20] — nhân vật chính; với Dao Dao tự xưng 'sư huynh' gọi 'sư muội'"
LINES = [
    TranslationLine(index=i, start_ms=i * 1000, end_ms=i * 1000 + 900, text=f"台词{i}")
    for i in range(1, 7)
]


class StagedFake:
    """Bộ dịch giả có ``fields`` tường minh (như adapter thật)."""

    def __init__(self, cue_error: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.cue_error = cue_error

    def analyze_global_context(
        self, source_lines, target_lang, model_name="m", cancel_cb=None, video_refs=None,
        with_visual_cues=False, prior_context="", fields=ANALYSIS_FIELDS_ALL,
        pinned_characters="",
    ) -> SubtitleContextAnalysis:
        self.calls.append(dict(kind="global", fields=fields, video_refs=video_refs,
                               with_visual_cues=with_visual_cues, pinned=pinned_characters,
                               prior=prior_context))
        if fields == ANALYSIS_FIELDS_ROSTER:
            return SubtitleContextAnalysis(source_lang="zh", characters=ROSTER)
        if fields == ANALYSIS_FIELDS_CONTEXT:
            return SubtitleContextAnalysis(source_lang="zh", characters=pinned_characters,
                                           overview="Diệp Nhiên tu luyện.", glossary="师兄 = sư huynh")
        return SubtitleContextAnalysis(source_lang="zh", characters="GỘP", overview="gộp")

    def analyze_visual_cues(self, source_lines, target_lang, model_name="m", video_refs=None,
                            batch_size=150, cancel_cb=None, progress_cb=None, characters="",
                            **_kw) -> list[VisualCue]:
        self.calls.append(dict(kind="cues", video_refs=video_refs, characters=characters))
        if self.cue_error:
            raise SubtitleTranslationError("503 UNAVAILABLE")
        speakers = ["叶然", "魔修", "魔修", "魔修", "Giọng nam chưa rõ", "叶然"]
        return [VisualCue(line_no=ln.index, speaker=spk, addressee="", scene="")
                for ln, spk in zip(source_lines, speakers, strict=True)]

    def analyze_text_cues(self, source_lines, target_lang, model_name="m", characters="",
                          cancel_cb=None, progress_cb=None, **_kw) -> list[VisualCue]:
        self.calls.append(dict(kind="text_cues", characters=characters))
        return [VisualCue(line_no=ln.index, speaker="叶然", addressee="", scene="")
                for ln in source_lines]


def _uc(fake: Any, staged: bool = True) -> AnalyzeSubtitleContextUseCase:
    return AnalyzeSubtitleContextUseCase(translator=fake, staged=staged)


def test_video_chi_toi_buoc_nguoi_noi_va_bang_nhan_vat_duoc_ghim() -> None:
    fake = StagedFake()
    video = [SimpleNamespace(remote_name="files/1", start_sec=0.0, end_sec=60.0)]
    result = _uc(fake).execute(LINES, "Vietnamese", model_name="gemini-x", video_refs=video,
                               enable_visual_cues=True, self_audit=False, prior_context="# cũ")
    kinds = [(c["kind"], c.get("fields")) for c in fake.calls]
    assert kinds[:3] == [("global", ANALYSIS_FIELDS_ROSTER), ("global", ANALYSIS_FIELDS_CONTEXT),
                         ("cues", None)]
    roster_call, context_call, cue_call = fake.calls[:3]
    # Hai bước chữ KHÔNG mang video, không đòi cues; ngữ cảnh phim bộ tới cả hai.
    assert roster_call["video_refs"] is None and context_call["video_refs"] is None
    assert not roster_call["with_visual_cues"] and not context_call["with_visual_cues"]
    assert roster_call["prior"] == context_call["prior"] == "# cũ"
    # Bảng nhân vật đã chốt được ghim vào bước tóm tắt và đưa cho bước người nói.
    assert "叶然" in context_call["pinned"] and "叶然" in cue_call["characters"]
    assert cue_call["video_refs"] == video
    assert result.overview == "Diệp Nhiên tu luyện." and "师兄" in result.glossary
    # Người lạ "魔修" (3 dòng) được thêm vào bảng nhân vật; nhãn giọng thì không.
    assert "魔修" in result.characters and "Giọng nam" not in result.characters
    cues = {c["id"]: c["spk"] for c in json.loads(result.visual_cues)}
    assert cues[2] == "魔修"


def test_khong_video_thi_gan_nguoi_noi_bang_chu_voi_bang_da_chot() -> None:
    fake = StagedFake()
    result = _uc(fake).execute(LINES, "Vietnamese", model_name="gemini-x", self_audit=False,
                               text_cues=True)
    kinds = [c["kind"] for c in fake.calls]
    assert kinds == ["global", "global", "text_cues"]
    assert "叶然" in fake.calls[2]["characters"]
    assert json.loads(result.visual_cues)


def test_buoc_nguoi_noi_loi_van_giu_nhan_vat_va_tom_tat() -> None:
    fake = StagedFake(cue_error=True)
    video = [SimpleNamespace(remote_name="files/1", start_sec=0.0, end_sec=60.0)]
    result = _uc(fake).execute(LINES, "Vietnamese", model_name="gemini-x", video_refs=video,
                               enable_visual_cues=True, self_audit=False)
    assert result.characters.startswith("叶然") and result.overview
    assert not result.visual_cues


def test_tat_tach_buoc_thi_dung_loi_goi_gop_cu() -> None:
    fake = StagedFake()
    result = _uc(fake, staged=False).execute(LINES, "Vietnamese", model_name="gemini-x",
                                             self_audit=False)
    assert [c["fields"] for c in fake.calls] == [ANALYSIS_FIELDS_ALL]
    assert result.characters == "GỘP"


def test_bo_dich_cu_khong_co_fields_thi_khong_hong() -> None:
    """Bộ dịch (vd đồ giả cũ) chỉ nhận **kwargs → không đoán được, dùng lời gọi gộp."""

    class Legacy:
        def __init__(self) -> None:
            self.kwargs: list[dict[str, Any]] = []

        def analyze_global_context(self, **kwargs: Any) -> SubtitleContextAnalysis:
            self.kwargs.append(kwargs)
            return SubtitleContextAnalysis(source_lang="zh", characters=ROSTER, overview="x")

    fake = Legacy()
    _uc(fake).execute(LINES, "Vietnamese", model_name="gemini-x", self_audit=False)
    assert len(fake.kwargs) == 1 and "fields" not in fake.kwargs[0]


@pytest.mark.parametrize("label,count,added", [
    ("魔修", 2, True), ("魔修", 1, False), ("Giọng nam chưa rõ", 5, False),
    ("Đám đông", 5, False), ("叶然", 5, False), ("Diệp Nhiên", 5, False),
])
def test_them_nguoi_noi_moi_vao_bang(label: str, count: int, added: bool) -> None:
    cues = [VisualCue(line_no=i, speaker=label, addressee="", scene="") for i in range(count)]
    characters, new = add_cue_speakers_to_roster(ROSTER, cues)
    assert bool(new) is added
    if added:
        assert characters.startswith(ROSTER) and characters.split("\n")[-1].startswith(label)
    else:
        assert characters == ROSTER


# ── Adapter: schema thu hẹp, không video, bảng nhân vật ghim ──────────────


def _adapter(captured: list[dict[str, Any]]) -> Any:
    from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
        GeminiSubtitleTranslator,
    )

    tr = GeminiSubtitleTranslator(api_key="KEY_A", retry_count=1)
    tr._ensure_available = lambda: None  # type: ignore[method-assign]
    tr._resolve_handles_with_heal = lambda refs, handles: list(refs or [])  # type: ignore[method-assign]

    def _fake_call(model, prompt, config, validator, **kw):
        captured.append(dict(config=config, kw=kw, prompt=prompt))
        return {"source_lang": "zh", "characters": [{"name": "叶然", "aka": "Diệp Nhiên"}],
                "overview": "ov", "glossary": [{"term": "师兄", "translation": "sư huynh"}]}

    tr._call_gemini = _fake_call  # type: ignore[method-assign]
    if tr._types_module is None:  # máy không có google-genai
        from subtitles_extractor.infrastructure.translation.openai_compat_client import (
            FALLBACK_TYPES,
        )

        tr._types_module = FALLBACK_TYPES
    return tr


def _schema_keys(config: Any) -> set[str]:
    schema = getattr(config, "response_schema", None)
    return set((schema or {}).get("properties", {}))


def test_adapter_chi_lap_bang_nhan_vat_khong_gui_video() -> None:
    captured: list[dict[str, Any]] = []
    tr = _adapter(captured)
    video = [SimpleNamespace(remote_name="files/1", start_sec=0.0, end_sec=60.0)]
    out = tr.analyze_global_context(LINES, "Vietnamese", model_name="gemini-x", video_refs=video,
                                    with_visual_cues=True, fields=ANALYSIS_FIELDS_ROSTER)
    (call,) = captured
    assert _schema_keys(call["config"]) == {"source_lang", "characters"}
    assert not call["kw"].get("video_files") and not call["kw"].get("video_refs")
    assert "叶然" in out.characters and not out.visual_cues


def test_adapter_buoc_tom_tat_ghim_bang_nhan_vat() -> None:
    captured: list[dict[str, Any]] = []
    tr = _adapter(captured)
    out = tr.analyze_global_context(LINES, "Vietnamese", model_name="gemini-x",
                                    fields=ANALYSIS_FIELDS_CONTEXT, pinned_characters=ROSTER)
    (call,) = captured
    assert _schema_keys(call["config"]) == {"source_lang", "overview", "glossary"}
    assert ROSTER in str(getattr(call["config"], "system_instruction", ""))
    # Kết quả giữ ĐÚNG bảng đã chốt (không lấy bảng model tự viết lại).
    assert out.characters == ROSTER and out.overview == "ov" and "师兄" in out.glossary


def test_adapter_mac_dinh_van_la_loi_goi_gop() -> None:
    captured: list[dict[str, Any]] = []
    tr = _adapter(captured)
    tr.analyze_global_context(LINES, "Vietnamese", model_name="gemini-x")
    assert {"characters", "overview", "glossary"} <= _schema_keys(captured[0]["config"])


# ── Sau rà soát độc lập ─────────────────────────────────────────────────────


class QuotaExhaustedError(RuntimeError):
    """Cùng tên/cùng gốc với lỗi hết hạn mức ngày của tầng hạ tầng."""


class FailingFake(StagedFake):
    def __init__(self, fail_at: str, error: Exception, partial: int = 0) -> None:
        super().__init__()
        self.fail_at, self.error, self.partial = fail_at, error, partial

    def analyze_global_context(
        self, source_lines, target_lang, model_name="m", cancel_cb=None, video_refs=None,
        with_visual_cues=False, prior_context="", fields=ANALYSIS_FIELDS_ALL,
        pinned_characters="",
    ) -> SubtitleContextAnalysis:
        if fields == self.fail_at:
            self.calls.append(dict(kind="global", fields=fields))
            raise self.error
        return super().analyze_global_context(
            source_lines, target_lang, model_name, cancel_cb, video_refs, with_visual_cues,
            prior_context, fields, pinned_characters)

    def analyze_visual_cues(self, source_lines, *args, **kwargs):
        self.calls.append(dict(kind="cues"))
        if self.fail_at == "cues":
            if self.partial:
                self.error.partial_cues = [  # type: ignore[attr-defined]
                    VisualCue(line_no=ln.index, speaker="叶然", addressee="", scene="")
                    for ln in source_lines[: self.partial]]
            raise self.error
        return super().analyze_visual_cues(source_lines, *args, **kwargs)


def _video() -> list:
    return [SimpleNamespace(remote_name="files/1", start_sec=0.0, end_sec=60.0)]


def test_het_han_muc_o_buoc_nguoi_noi_giu_hai_buoc_dau_va_lo_da_quet() -> None:
    from subtitles_extractor.application.services.analysis_completeness import (
        describe_incomplete_analysis,
    )
    from subtitles_extractor.domain.ports.subtitle_translator_port import (
        ANALYSIS_STEP_SPEAKERS,
        ANALYSIS_STOP_DAILY_QUOTA,
    )

    fake = FailingFake("cues", QuotaExhaustedError("hết hạn mức ngày"), partial=3)
    result = _uc(fake).execute(LINES, "Vietnamese", model_name="gemini-x", video_refs=_video(),
                               enable_visual_cues=True, self_audit=False)
    assert result.characters.startswith("叶然") and result.overview
    assert result.incomplete_steps == (ANALYSIS_STEP_SPEAKERS,)
    assert result.stop_reason == ANALYSIS_STOP_DAILY_QUOTA
    assert len(json.loads(result.visual_cues)) == 3          # 3 dòng đã quét được giữ lại
    message = describe_incomplete_analysis(result)
    assert "nhận diện người nói" in message and "Bổ sung người nói" in message


def test_qua_tai_o_buoc_tom_tat_giu_bang_nhan_vat_va_khong_quet_nguoi_noi() -> None:
    from subtitles_extractor.domain.ports.subtitle_translator_port import (
        ANALYSIS_STEP_CONTEXT,
        ANALYSIS_STEP_SPEAKERS,
        ANALYSIS_STOP_OVERLOAD,
        ModelOverloadedError,
    )

    fake = FailingFake(ANALYSIS_FIELDS_CONTEXT, ModelOverloadedError("503", model_name="gemini-x"))
    result = _uc(fake).execute(LINES, "Vietnamese", model_name="gemini-x", video_refs=_video(),
                               enable_visual_cues=True, self_audit=False)
    assert result.characters.startswith("叶然")
    assert result.incomplete_steps == (ANALYSIS_STEP_CONTEXT, ANALYSIS_STEP_SPEAKERS)
    assert result.stop_reason == ANALYSIS_STOP_OVERLOAD
    assert "cues" not in [c["kind"] for c in fake.calls]     # không đốt thêm lượt


def test_huy_van_lan_ra_ngoai() -> None:
    from subtitles_extractor.domain.ports.subtitle_translator_port import (
        TranslationCancelledError,
    )

    fake = FailingFake("cues", TranslationCancelledError("huỷ"))
    with pytest.raises(TranslationCancelledError):
        _uc(fake).execute(LINES, "Vietnamese", model_name="gemini-x", video_refs=_video(),
                          enable_visual_cues=True, self_audit=False)


@pytest.mark.parametrize("label", [
    "叶然 / 魔修", "苏晴 / 林逸", "Tô Tình, Lâm Dật", "Nam đệ tử - người gác", "Thuyết minh",
    "Hệ thống", "Mọi người", "Giọng nam mới chưa rõ",
])
def test_khong_them_nhan_gop_nhieu_nguoi_hoac_khong_phai_nguoi(label: str) -> None:
    roster = ROSTER + "\n苏晴 (Tô Tình) [nữ] — nữ chính; với Lâm Dật tự xưng 'em' gọi 'anh'"
    cues = [VisualCue(line_no=i, speaker=label, addressee="", scene="") for i in range(5)]
    characters, added = add_cue_speakers_to_roster(roster, cues)
    assert added == [] and characters == roster


def test_adapter_dinh_kem_lo_da_quet_khi_loi_giua_chung() -> None:
    from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
        GeminiSubtitleTranslator,
    )

    tr = GeminiSubtitleTranslator(api_key="KEY_A", retry_count=1)
    tr._ensure_available = lambda: None  # type: ignore[method-assign]
    tr._resolve_handles_with_heal = lambda refs, handles: []  # type: ignore[method-assign]
    if tr._types_module is None:
        from subtitles_extractor.infrastructure.translation.openai_compat_client import (
            FALLBACK_TYPES,
        )

        tr._types_module = FALLBACK_TYPES
    calls = {"n": 0}

    def _fake_call(model, prompt, config, validator, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise QuotaExhaustedError("hết hạn mức ngày")
        return {"cues": [{"id": i, "spk": "叶然", "to": "", "cue": ""} for i in range(1, 4)]}

    tr._call_gemini = _fake_call  # type: ignore[method-assign]
    many = [TranslationLine(index=i, start_ms=i * 1000, end_ms=i * 1000 + 900, text=f"台词{i}")
            for i in range(1, 7)]
    with pytest.raises(QuotaExhaustedError) as info:
        tr.analyze_visual_cues(many, "Vietnamese", model_name="gemini-x", batch_size=3,
                               sleep_between_s=0.0)
    assert [c.line_no for c in info.value.partial_cues] == [1, 2, 3]  # type: ignore[attr-defined]
