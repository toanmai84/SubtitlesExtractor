"""[v3.23.637] Tự kiểm tra: bảng nhân vật KHÔNG có mệnh đề xưng hô nào → khiếm khuyết + vá.

Ca thật: chẩn đoán phim 86 tập (19 nhân vật, ``address_rules.clauses = 0``); tái hiện 1/6 lượt phân
tích phim ghép 12 tập (gemini-3.8-flash viết ``tự xưng 'ta' / gọi 'ngươi' (với 叶然)``, không đọc
được). Sau tự kiểm tra: 0 → 10-12 mệnh đề, tất cả tra được người nghe (3/3 lượt thật).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from subtitles_extractor.application.services.analysis_audit import (
    NO_ADDRESS_RULES_KIND,
    apply_audit_patch,
    find_analysis_defects,
)
from subtitles_extractor.application.services.roster_format import address_clauses
from subtitles_extractor.domain.ports.subtitle_translator_port import SubtitleContextAnalysis

LINES = [SimpleNamespace(index=i, text="你好", original_text="你好") for i in range(1, 4)]
BAD = (
    "叶然 (Diệp Nhiên) [nam, ~20] — nhân vật chính, đệ tử; tự xưng 'ta' / gọi 'sư thúc' (với 南宫梦)\n"
    "南宫梦 (Nam Cung Mộng) [nữ, ~30] — sư thúc của 叶然; tự xưng 'ta' / gọi 'ngươi'"
)


def _analysis(characters: str = BAD) -> SubtitleContextAnalysis:
    return SubtitleContextAnalysis(
        source_lang="zh", characters=characters, overview="Tóm tắt.", glossary="")


def _patch(*characters: str) -> dict:
    return {"characters": list(characters), "remove_characters": [], "glossary": [],
            "remove_glossary": [], "overview": "", "cue_fixes": []}


def test_roster_without_readable_clause_is_a_defect() -> None:
    defects = [d for d in find_analysis_defects(_analysis(), LINES)
               if d.kind == NO_ADDRESS_RULES_KIND]
    assert len(defects) == 1
    assert "叶然" in defects[0].detail and "南宫梦" in defects[0].detail


def test_roster_with_clauses_is_not_flagged() -> None:
    good = BAD.replace("; tự xưng 'ta' / gọi 'sư thúc' (với 南宫梦)",
                       "; xưng hô: với 南宫梦 tự xưng 'đệ tử' gọi 'sư thúc'")
    kinds = {d.kind for d in find_analysis_defects(_analysis(good), LINES)}
    assert NO_ADDRESS_RULES_KIND not in kinds


def test_patch_adds_clauses_and_keeps_old_role_text() -> None:
    patch = _patch(
        "叶然 (Diệp Nhiên) [nam, ~20] — vai trò model viết lại; xưng hô: với 南宫梦 tự xưng "
        "'đệ tử' gọi 'sư thúc'",
        "南宫梦 (Nam Cung Mộng) [nữ, ~30] — x; xưng hô: với 叶然 tự xưng 'ta' gọi 'ngươi'",
    )
    out, changes = apply_audit_patch(_analysis(), patch, {1, 2, 3})
    assert changes >= 2
    assert "nhân vật chính, đệ tử" in out.characters           # vai trò cũ còn nguyên
    assert "vai trò model viết lại" not in out.characters
    clauses = address_clauses(out.characters)
    assert len(clauses) == 2 and all(c.listeners for c in clauses)


def test_patch_never_overwrites_an_existing_clause() -> None:
    mine = "叶然 (Diệp Nhiên) [nam, ~20] — chính; xưng hô: với 南宫梦 tự xưng 'con' gọi 'sư thúc'\n" \
           "南宫梦 (Nam Cung Mộng) [nữ, ~30] — sư thúc"
    patch = _patch("叶然 (Diệp Nhiên) [nam, ~20] — chính; xưng hô: với 南宫梦 tự xưng 'ta' gọi 'ngươi'")
    out, _ = apply_audit_patch(_analysis(mine), patch, {1})
    assert "tự xưng 'con' gọi 'sư thúc'" in out.characters
    assert "tự xưng 'ta' gọi 'ngươi'" not in out.characters


def test_patch_without_readable_clause_changes_nothing() -> None:
    patch = _patch("叶然 (Diệp Nhiên) [nam, ~20] — viết lại không có xưng hô")
    out, changes = apply_audit_patch(_analysis(), patch, {1})
    assert changes == 0 and out.characters == BAD


class TestAdapter:
    def _adapter(self, monkeypatch):
        from tests.unit.infrastructure.gemini_call_harness import bare_analysis_adapter

        adapter = bare_analysis_adapter()
        monkeypatch.setattr(adapter, "_ensure_available", lambda: None)
        sent: dict = {}
        monkeypatch.setattr(adapter, "_build_config", lambda *a, **k: sent.update(system=a[2]) or "cfg")
        monkeypatch.setattr(
            adapter, "_call_gemini",
            lambda model, prompt, config, validator, **k: sent.update(prompt=prompt) or {},
        )
        return adapter, sent

    def test_table_level_defect_on_a_long_film_sends_a_bounded_sample(self, monkeypatch) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            AUDIT_NO_FOCUS_MAX_LINES,
        )
        film = [SimpleNamespace(index=i, text=f"câu {i}", original_text=f"句{i}", start_ms=i, end_ms=i + 1)
                for i in range(1, 3001)]
        adapter, sent = self._adapter(monkeypatch)
        adapter.audit_analysis(film, "Vietnamese", "m", BAD, "", "", {}, "[]", [])
        rows = [ln for ln in sent["prompt"].splitlines() if ln.count("|") >= 2 and ln[0].isdigit()]
        assert 0 < len(rows) <= AUDIT_NO_FOCUS_MAX_LINES
        ids = [int(r.split("|")[0]) for r in rows]
        assert min(ids) <= 5 and max(ids) >= 2990            # phủ đầu và cuối phim

    def test_audit_instruction_teaches_the_clause_format_and_seniority(self, monkeypatch) -> None:
        adapter, sent = self._adapter(monkeypatch)
        adapter.audit_analysis(LINES, "Vietnamese", "m", BAD, "", "", {}, "[]", [])
        system = sent["system"]
        assert "no_address_rules" in system and "xưng hô: với" in system
        assert "KHÔNG tự xưng 'ta'" in system                  # vai vế: đệ tử → sư thúc


class TestJunkPatchNeverCrashes:
    """[v3.23.638] Bản vá do model trả có thể sai kiểu (số, danh sách thay cho dict, null…): trước đây
    ``apply_audit_patch`` ném TypeError/AttributeError và làm mất cả kết quả phân tích."""

    @pytest.mark.parametrize("junk", [None, 5, 2.5, True, "x", [], [1, 2], {"characters": 5},
                                      {"glossary": True}, {"remove_characters": 1.5},
                                      {"remove_glossary": 3}, {"cue_fixes": 7}, {"cue_fixes": "a"},
                                      {"overview": 12}, {"overview": ["a"]},
                                      {"characters": [None, 3, {"a": 1}]},
                                      {"cue_fixes": [None, 1, "x", {"id": "z"}, {"id": 1}]}])
    def test_returns_the_analysis_unharmed(self, junk) -> None:
        analysis = _analysis()
        out, changes = apply_audit_patch(analysis, junk, {1, 2, 3}, "")
        assert out.characters == analysis.characters and out.overview == analysis.overview
        assert changes >= 0

    def test_use_case_survives_a_patch_that_cannot_be_applied(self, monkeypatch) -> None:
        from subtitles_extractor.application.use_cases import analyze_subtitle_context as mod

        class Translator:
            def audit_analysis(self, *_a, **_k):
                return {"characters": []}

        monkeypatch.setattr(mod, "apply_audit_patch", lambda *a, **k: 1 / 0)
        use_case = mod.AnalyzeSubtitleContextUseCase(Translator())
        analysis = _analysis()
        out = use_case.refine(analysis, LINES, "Vietnamese", "m")
        assert out.characters == analysis.characters
