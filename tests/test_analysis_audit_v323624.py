"""[v3.23.624] Tự kiểm tra & vá phân tích cũ: phát hiện tất định, gộp bản vá, vòng lặp use case."""

from __future__ import annotations

from types import SimpleNamespace

from subtitles_extractor.application.services.analysis_audit import (
    apply_audit_patch,
    find_analysis_defects,
)
from subtitles_extractor.application.services.visual_cue_serializer import (
    deserialize_visual_cues,
    serialize_visual_cues,
)
from subtitles_extractor.application.use_cases.analyze_subtitle_context import (
    AnalyzeSubtitleContextUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleContextAnalysis,
    VisualCue,
)

LINES = [
    SimpleNamespace(index=1, text="你好师兄", original_text="你好师兄"),
    SimpleNamespace(index=2, text="师妹有礼", original_text="师妹有礼"),
    SimpleNamespace(index=3, text="青云宗到了", original_text="青云宗到了"),
    SimpleNamespace(index=4, text="走吧", original_text="走吧"),
    SimpleNamespace(index=5, text="好", original_text="好"),
]
ROSTER = (
    "叶然 (Diệp Nhiên) [nữ] — sư muội; xưng hô: với Lý Mặc tự xưng 'muội' gọi 'sư huynh'\n"
    "李默 (Lý Mặc) [nam] — sư huynh; xưng hô: với Diệp Nhiên tự xưng 'ta' gọi 'muội'"
)


def _analysis(cues, **kw):
    base = dict(source_lang="zh", characters=ROSTER, overview="Tóm tắt.",
                glossary="青云宗 = Thanh Vân Tông\n无关词 = vô quan",
                visual_cues=serialize_visual_cues(cues))
    base.update(kw)
    return SubtitleContextAnalysis(**base)


def _cues():
    return [
        VisualCue(1, "叶然", "李默"), VisualCue(2, "李默", "叶然"),
        VisualCue(3, "苏白", "叶然"), VisualCue(4, "苏白", "李默"), VisualCue(5, "苏白", "叶然"),
    ]


def test_finds_unlisted_label_stale_glossary_and_missing_speaker():
    cues = [c for c in _cues() if c.line_no != 5]
    cues[1] = VisualCue(2, "苏白", "叶然")
    kinds = {d.kind for d in find_analysis_defects(_analysis(cues), LINES)}
    assert {"unlisted_label", "stale_glossary", "missing_speaker"} <= kinds


def test_clean_analysis_has_no_defects():
    cues = [VisualCue(i, "叶然" if i % 2 else "李默", "李默" if i % 2 else "叶然") for i in range(1, 6)]
    a = _analysis(cues, glossary="青云宗 = Thanh Vân Tông")
    assert find_analysis_defects(a, LINES) == []


def test_patch_merges_without_touching_other_entries():
    a = _analysis(_cues())
    patch = {
        "characters": ["苏白 (Tô Bạch) [nam] — sư đệ"], "remove_glossary": ["无关词"],
        "cue_fixes": [{"id": 3, "spk": "苏白", "to": "叶然"}, {"id": 99, "spk": "叶然", "to": "x"}],
        "glossary": [], "remove_characters": [], "overview": "",
    }
    out, changes = apply_audit_patch(a, patch, {1, 2, 3, 4, 5})
    assert "苏白" in out.characters and "叶然" in out.characters
    assert "无关词" not in out.glossary and "青云宗" in out.glossary
    assert out.overview == "Tóm tắt."
    assert {c.line_no for c in deserialize_visual_cues(out.visual_cues)} == {1, 2, 3, 4, 5}
    assert changes == 2


def test_use_case_patches_then_stops_when_clean():
    calls = []

    class Fake:
        def audit_analysis(self, lines, lang, model, characters, glossary, overview, labels,
                           defects_json, focus, cancel_cb=None):
            calls.append(defects_json)
            return {"characters": ["苏白 (Tô Bạch) [nam] — sư đệ"], "remove_characters": [],
                    "glossary": [], "remove_glossary": ["无关词"], "overview": "", "cue_fixes": []}

    uc = AnalyzeSubtitleContextUseCase(translator=Fake())
    out = uc.refine(_analysis(_cues()), LINES, "Vietnamese", "m")
    assert len(calls) == 1
    assert "苏白" in out.characters and "无关词" not in out.glossary


def test_use_case_does_not_call_ai_when_clean_and_survives_errors():
    class Boom:
        def audit_analysis(self, *a, **k):
            raise AssertionError("không được gọi")

    cues = [VisualCue(i, "叶然" if i % 2 else "李默", "李默" if i % 2 else "叶然") for i in range(1, 6)]
    a = _analysis(cues, glossary="青云宗 = Thanh Vân Tông")
    assert AnalyzeSubtitleContextUseCase(translator=Boom()).refine(a, LINES, "Vietnamese", "m") is a

    from subtitles_extractor.domain.ports.subtitle_translator_port import SubtitleTranslationError

    class Fail:
        def audit_analysis(self, *a, **k):
            raise SubtitleTranslationError("lỗi")

    bad = _analysis(_cues())
    assert AnalyzeSubtitleContextUseCase(translator=Fail()).refine(bad, LINES, "Vietnamese", "m") is bad


def test_stale_glossary_alone_does_not_cost_an_ai_call():
    class Boom:
        def audit_analysis(self, *a, **k):
            raise AssertionError("không được gọi")

    cues = [VisualCue(i, "叶然" if i % 2 else "李默", "李默" if i % 2 else "叶然") for i in range(1, 6)]
    a = _analysis(cues)  # glossary còn '无关词' (thừa) nhưng mọi thứ khác ổn
    assert {d.kind for d in find_analysis_defects(a, LINES)} == {"stale_glossary"}
    assert AnalyzeSubtitleContextUseCase(translator=Boom()).refine(a, LINES, "Vietnamese", "m") is a


def test_patch_protects_user_edits_and_used_characters():
    a = _analysis(_cues(), glossary="青云宗 = Thanh Vân Tông\n无关词 = vô quan")
    patch = {
        "characters": ["叶然 (Diệp Nhiên) [nữ] — vai trò do model viết lại"],
        "remove_characters": ["李默"], "glossary": [], "remove_glossary": ["青云宗", "无关词"],
        "overview": "", "cue_fixes": [],
    }
    text = "\n".join(ln.text for ln in LINES)
    out, _ = apply_audit_patch(a, patch, {1, 2, 3, 4, 5}, text)
    assert "sư muội" in out.characters and "vai trò do model" not in out.characters
    assert "李默" in out.characters            # còn được nhãn người nói nhắc tới
    assert "青云宗" in out.glossary             # có trong phụ đề → không được xoá
    assert "无关词" not in out.glossary


def test_refine_normalizes_names_after_patch():
    seen = []

    class Fake:
        def audit_analysis(self, *a, **k):
            return {"characters": ["苏白 (Tô Bạch) [nam] — sư đệ"], "remove_characters": [],
                    "glossary": [], "remove_glossary": [], "overview": "", "cue_fixes": []}

    uc = AnalyzeSubtitleContextUseCase(translator=Fake())
    uc._normalize_context = lambda r: (seen.append(r.characters) or r)  # type: ignore[method-assign]
    uc.refine(_analysis(_cues()), LINES, "Vietnamese", "m", max_rounds=1)
    assert seen and "苏白" in seen[0]


def test_refine_worker_runs_without_video():
    import pytest
    pytest.importorskip("PySide6")
    from subtitles_extractor.presentation.workers.refine_analysis_worker import (
        RefineAnalysisWorker,
    )

    class UC:
        def refine(self, current, lines, lang, model, cancel_cb=None, fallback_models=()):
            return SubtitleContextAnalysis(overview="đã vá")

    got = []
    w = RefineAnalysisWorker(UC(), LINES, "Vietnamese", "m", _analysis([]))  # type: ignore[arg-type]
    w.finished_ok.connect(got.append)
    w.run()
    assert got and got[0].overview == "đã vá"
