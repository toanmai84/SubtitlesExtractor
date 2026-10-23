"""[v3.23.640] Người dùng chọn MODEL TỰ KIỂM TRA phân tích (trước đây cố định gemini-3.1-flash-lite).

Nhật ký thật của người dùng (phim 86 tập): model phân tích ``gemini-3.8-flash`` + model dự phòng
``gemini-3.5-flash-lite`` được chọn, nhưng bước "Tự kiểm tra phân tích" luôn chạy ``gemini-3.1-flash-lite``
(``DEFAULT_AUDIT_MODEL``): ``audit_model`` không bao giờ được truyền xuống, nút "Tự kiểm tra & tối ưu"
dùng thẳng hằng số, và dòng trạng thái lại ghi model phân tích đang chọn như thể nó được dùng. Lỗi hết
hạn mức ngày của model kiểm định (``QuotaExhaustedError``) cũng không được bắt trong ``refine``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from subtitles_extractor.application.services.analysis_audit import DEFAULT_AUDIT_MODEL
from subtitles_extractor.application.use_cases.analyze_subtitle_context import (
    AnalyzeSubtitleContextUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleContextAnalysis,
    SubtitleTranslationError,
    TranslationCancelledError,
)
from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    QuotaExhaustedError,
)

LINES = [SimpleNamespace(index=i, text="你好", original_text="你好") for i in range(1, 4)]
# bảng KHÔNG có mệnh đề xưng hô → khiếm khuyết 'no_address_rules' → bước kiểm tra phải gọi model
BAD = "叶然 (Diệp Nhiên) [nam] — chính\n南宫梦 (Nam Cung) [nữ] — sư thúc"
FIXED = ("叶然 (Diệp Nhiên) [nam] — chính; xưng hô: với 南宫梦 tự xưng 'đệ tử' gọi 'sư thúc'\n"
         "南宫梦 (Nam Cung) [nữ] — sư thúc; xưng hô: với 叶然 tự xưng 'ta' gọi 'ngươi'")


def _analysis() -> SubtitleContextAnalysis:
    return SubtitleContextAnalysis(source_lang="zh", characters=BAD, overview="x", glossary="")


class _Translator:
    def __init__(self, behaviour: dict[str, object], no_quota: set[str] = frozenset()) -> None:
        self.behaviour = behaviour
        self.no_quota = set(no_quota)
        self.models_called: list[str] = []

    def has_any_daily_quota(self, model: str) -> bool:
        return model not in self.no_quota

    def audit_analysis(self, lines, lang, model, characters, glossary, overview, labels,
                       defects_json, focus_ids, cancel_cb=None):
        self.models_called.append(model)
        what = self.behaviour.get(model, "patch")
        if isinstance(what, Exception):
            raise what
        return {"characters": FIXED.splitlines(), "remove_characters": [], "glossary": [],
                "remove_glossary": [], "overview": "", "cue_fixes": []}


def _refine(translator, model="A", fallbacks=()):
    use_case = AnalyzeSubtitleContextUseCase(translator)
    return use_case.refine(_analysis(), LINES, "Vietnamese", model, fallback_models=fallbacks)


class TestRefineChain:
    def test_uses_the_chosen_model_first(self) -> None:
        translator = _Translator({})
        out = _refine(translator, "gemini-3.8-flash", ("gemini-3.5-flash-lite",))
        assert translator.models_called[0] == "gemini-3.8-flash"
        assert "xưng hô" in out.characters

    def test_daily_quota_exhausted_model_is_skipped_without_a_call(self) -> None:
        translator = _Translator({}, no_quota={"gemini-3.8-flash"})
        out = _refine(translator, "gemini-3.8-flash", ("gemini-3.5-flash-lite", DEFAULT_AUDIT_MODEL))
        assert translator.models_called[0] == "gemini-3.5-flash-lite"
        assert "xưng hô" in out.characters

    @pytest.mark.parametrize("error", [QuotaExhaustedError("hết"), SubtitleTranslationError("lỗi"),
                                       RuntimeError("boom")])
    def test_a_failing_model_falls_through_to_the_next(self, error) -> None:
        translator = _Translator({"A": error})
        out = _refine(translator, "A", ("B",))
        assert translator.models_called == ["A", "B"]
        assert "xưng hô" in out.characters

    def test_all_models_failing_keeps_the_analysis_and_does_not_raise(self) -> None:
        translator = _Translator({"A": QuotaExhaustedError("x"), "B": SubtitleTranslationError("y")})
        out = _refine(translator, "A", ("B",))
        assert out.characters == BAD

    def test_cancellation_still_propagates(self) -> None:
        translator = _Translator({"A": TranslationCancelledError("huỷ")})
        use_case = AnalyzeSubtitleContextUseCase(translator)
        with pytest.raises(TranslationCancelledError):
            use_case.refine(_analysis(), LINES, "Vietnamese", "A", cancel_cb=lambda: True,
                            fallback_models=("B",))

    def test_default_behaviour_without_fallbacks_is_unchanged(self) -> None:
        translator = _Translator({})
        _refine(translator, DEFAULT_AUDIT_MODEL)
        assert translator.models_called == [DEFAULT_AUDIT_MODEL]


class TestExecuteAndWorkers:
    def test_execute_passes_the_audit_model_and_fallbacks_down(self, monkeypatch) -> None:
        captured = {}

        class Translator(_Translator):
            def analyze_global_context(self, **_kw):
                return SubtitleContextAnalysis(source_lang="zh", characters=BAD, overview="x",
                                               glossary="")

        use_case = AnalyzeSubtitleContextUseCase(Translator({}))
        monkeypatch.setattr(
            use_case, "refine",
            lambda analysis, lines, lang, model, cancel_cb=None, max_rounds=2, fallback_models=():
            captured.update(model=model, fallbacks=tuple(fallback_models)) or analysis,
        )
        use_case.execute(LINES, "Vietnamese", model_name="gemini-3.8-flash",
                         audit_model="gemini-3.8-flash", audit_fallbacks=("gemini-3.5-flash-lite",))
        assert captured == {"model": "gemini-3.8-flash", "fallbacks": ("gemini-3.5-flash-lite",)}

    def test_execute_default_audit_model_is_the_cheap_one_not_the_analysis_model(self, monkeypatch) -> None:
        captured = {}

        class Translator(_Translator):
            def analyze_global_context(self, **_kw):
                return _analysis()

        use_case = AnalyzeSubtitleContextUseCase(Translator({}))
        monkeypatch.setattr(
            use_case, "refine",
            lambda analysis, lines, lang, model, cancel_cb=None, max_rounds=2, fallback_models=():
            captured.update(model=model) or analysis,
        )
        use_case.execute(LINES, "Vietnamese", model_name="gemini-3.8-flash")
        assert captured["model"] == DEFAULT_AUDIT_MODEL

    def test_refine_worker_uses_the_chosen_chain(self) -> None:
        pytest.importorskip("PySide6")
        from subtitles_extractor.presentation.workers.refine_analysis_worker import (
            RefineAnalysisWorker,
        )

        seen = {}

        class UseCase:
            last_audit_report = None

            def refine(self, current, lines, lang, model, cancel_cb=None, fallback_models=()):
                seen.update(model=model, fallbacks=tuple(fallback_models))
                return current

        worker = RefineAnalysisWorker(UseCase(), LINES, "Vietnamese", "gemini-3.8-flash", _analysis(),
                                      audit_model="gemini-3.5-flash-lite",
                                      audit_fallbacks=(DEFAULT_AUDIT_MODEL,))
        worker._execute(None, None)
        assert seen == {"model": "gemini-3.5-flash-lite", "fallbacks": (DEFAULT_AUDIT_MODEL,)}
        auto = RefineAnalysisWorker(UseCase(), LINES, "Vietnamese", "gemini-3.8-flash", _analysis())
        auto._execute(None, None)
        assert seen["model"] == DEFAULT_AUDIT_MODEL
