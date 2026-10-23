"""[v3.23.638] Người dùng chọn model dự phòng (Tự động / Không chuyển model / model cụ thể).

Trước đây dự phòng khi model quá tải chỉ do ứng dụng tự xếp hạng, và model dự phòng của bước phân
tích bị cố định ``gemini-3.1-flash-lite``; người dùng không có cách chọn hay tắt.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.infrastructure.translation.model_health import (
    AUTO_FALLBACK,
    NO_FALLBACK,
    ModelHealth,
)

KNOWN = ("gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.6-flash")


class TestModelHealthPreference:
    def _health(self) -> ModelHealth:
        return ModelHealth(seed_models=KNOWN)

    def test_default_is_automatic_and_unchanged(self) -> None:
        health = self._health()
        assert health.preferred_fallback == AUTO_FALLBACK
        assert health.fallbacks("gemini-3.1-flash-lite")[0] == "gemini-3.5-flash-lite"

    def test_preferred_model_comes_first(self) -> None:
        health = self._health()
        health.set_preferred_fallback("gemini-3.6-flash")
        ranked = health.fallbacks("gemini-3.1-flash-lite")
        assert ranked[0] == "gemini-3.6-flash"
        assert "gemini-3.5-flash-lite" in ranked and len(set(ranked)) == len(ranked)

    def test_no_fallback_disables_failover(self) -> None:
        health = self._health()
        health.set_preferred_fallback(NO_FALLBACK)
        assert health.fallbacks("gemini-3.1-flash-lite") == []

    def test_preferred_can_be_a_model_missing_from_the_api_list(self) -> None:
        health = self._health()
        health.set_preferred_fallback("gemini-9-flash-preview")
        assert health.fallbacks("gemini-3.1-flash-lite")[0] == "gemini-9-flash-preview"

    @pytest.mark.parametrize("why", ["self", "tried", "quota"])
    def test_unusable_preferred_falls_back_to_automatic(self, why: str) -> None:
        health = self._health()
        health.set_preferred_fallback("gemini-3.6-flash")
        model = "gemini-3.6-flash" if why == "self" else "gemini-3.1-flash-lite"
        exclude = ["gemini-3.6-flash"] if why == "tried" else []
        available = (lambda m: m != "gemini-3.6-flash") if why == "quota" else (lambda _m: True)
        ranked = health.fallbacks(model, available=available, exclude=exclude)
        assert "gemini-3.6-flash" not in ranked and ranked

    def test_cooling_preferred_is_skipped(self) -> None:
        health = self._health()
        health.set_preferred_fallback("gemini-3.6-flash")
        for _ in range(3):
            health.record_overload("gemini-3.6-flash")
        assert "gemini-3.6-flash" not in health.fallbacks("gemini-3.1-flash-lite")

    def test_blank_and_none_mean_automatic(self) -> None:
        health = self._health()
        health.set_preferred_fallback("  ")
        assert health.preferred_fallback == AUTO_FALLBACK
        health.set_preferred_fallback(None)  # type: ignore[arg-type]
        assert health.preferred_fallback == AUTO_FALLBACK


def test_adapter_failover_honours_the_choice_even_across_thinking_families() -> None:
    from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
        GeminiSubtitleTranslator,
    )

    adapter = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    adapter._model_health = ModelHealth(seed_models=("gemini-2.5-flash", "gemini-3.1-flash-lite"))
    adapter._api_keys = ["k"]
    adapter._quota_manager = None
    adapter._rpd_remaining_for_key = lambda _m, _k: 100          # type: ignore[method-assign]
    adapter._model_health.set_preferred_fallback("gemini-2.5-flash")
    assert adapter._failover_model("gemini-3.1-flash-lite", set()) == "gemini-2.5-flash"
    adapter._model_health.set_preferred_fallback(NO_FALLBACK)
    assert adapter._failover_model("gemini-3.1-flash-lite", set()) is None


qt = pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class TestPage:
    def _page(self):
        from tests.unit.presentation.test_translate_page_bugs_v323609 import _page

        return _page()

    def test_combo_has_automatic_none_and_models_and_defaults_to_automatic(self, app) -> None:
        page, container = self._page()
        combo = page._fallback_model_combo
        assert combo.itemData(0) == "" and combo.itemData(1) == "-"
        assert combo.count() > 3 and page._fallback_choice() == ""
        assert container.gemini_model_health.preferred_fallback == ""

    def test_choosing_a_model_reaches_the_shared_health_and_the_analysis_fallback(self, app) -> None:
        page, container = self._page()
        combo = page._fallback_model_combo
        combo.setCurrentIndex(combo.findData("gemini-3.5-flash-lite"))
        assert container.gemini_model_health.preferred_fallback == "gemini-3.5-flash-lite"
        assert page._view_model.analysis_fallback_model() == "gemini-3.5-flash-lite"
        combo.setCurrentIndex(1)                                  # Không chuyển model
        assert container.gemini_model_health.preferred_fallback == "-"
        assert page._view_model.analysis_fallback_model() == ""
        combo.setCurrentIndex(0)
        assert page._view_model.analysis_fallback_model() == "gemini-3.1-flash-lite"

    def test_refreshing_the_model_list_keeps_the_choice_including_a_typed_model(self, app) -> None:
        page, container = self._page()
        page._fill_fallback_combo(["gemini-3.1-flash-lite"], "gemini-9-preview")
        assert page._fallback_choice() == "gemini-9-preview"
        page._on_models_ready(["gemini-3.6-flash", "gemini-3.1-flash-lite"])
        assert page._fallback_choice() == "gemini-9-preview"
        assert container.gemini_model_health.preferred_fallback == "gemini-9-preview"

    def test_choice_is_persisted_and_restored(self, app) -> None:
        page, _ = self._page()
        saved: dict = {}
        page._settings = type("S", (), {
            "setValue": lambda self, k, v: saved.__setitem__(k, v),
            "sync": lambda self: None,
        })()
        page._fallback_model_combo.setCurrentIndex(1)
        from subtitles_extractor.presentation.pages.translate_page import TranslatePage
        TranslatePage._persist_inputs(page)
        assert saved["fallback_model"] == "-"


class TestKeyStatusText:
    """[v3.23.639] Hộp thoại nhiều key: key hỏng / bị server từ chối / tổng còn lại / giờ hạn mức hồi."""

    def test_status_distinguishes_invalid_blocked_and_healthy_keys(self, app) -> None:
        page, container = TestPage()._page()
        from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
            GeminiQuotaManager,
        )

        page._all_keys = lambda: ["KEY_A_1234", "KEY_B_5678", "KEY_C_9012"]
        model = page._analyze_model_combo.currentText().strip()
        manager = container.gemini_quota_manager
        manager.mark_key_invalid(GeminiQuotaManager.key_fingerprint("KEY_A_1234"), "API key not valid")
        manager.mark_daily_exhausted(model, key_id=GeminiQuotaManager.key_fingerprint("KEY_B_5678"))
        text = page._keys_quota_status_text()
        assert "⛔" in text and "Gemini từ chối" in text and "API key not valid" in text
        assert "🔴" in text and "🟢" in text
        assert "Còn tổng cộng" in text


class TestAuditModelChoice:
    """[v3.23.640] Ô "Model tự kiểm tra" + chuỗi dự phòng + dòng trạng thái nói đúng model."""

    def test_default_chain_is_the_cheap_model(self, app) -> None:
        from subtitles_extractor.application.services.analysis_audit import DEFAULT_AUDIT_MODEL

        page, _ = TestPage()._page()
        assert page._audit_choice() == ""
        assert page._view_model.audit_chain() == (DEFAULT_AUDIT_MODEL, ())

    def test_chosen_model_and_backup_form_the_chain(self, app) -> None:
        from subtitles_extractor.application.services.analysis_audit import DEFAULT_AUDIT_MODEL

        page, _ = TestPage()._page()
        page._audit_model_combo.setCurrentIndex(page._audit_model_combo.findData("gemini-3.8-flash"))
        page._fallback_model_combo.setCurrentIndex(
            page._fallback_model_combo.findData("gemini-3.5-flash-lite"))
        assert page._view_model.audit_chain() == (
            "gemini-3.8-flash", ("gemini-3.5-flash-lite", DEFAULT_AUDIT_MODEL))
        page._fallback_model_combo.setCurrentIndex(1)                      # Không chuyển model
        assert page._view_model.audit_chain() == ("gemini-3.8-flash", (DEFAULT_AUDIT_MODEL,))

    def test_choice_survives_a_model_list_refresh_and_is_persisted(self, app) -> None:
        page, _ = TestPage()._page()
        page._fill_audit_combo(["gemini-3.1-flash-lite"], "gemini-9-preview")
        page._on_models_ready(["gemini-3.6-flash"])
        assert page._audit_choice() == "gemini-9-preview"
        saved: dict = {}
        page._settings = type("S", (), {
            "setValue": lambda self, k, v: saved.__setitem__(k, v), "sync": lambda self: None})()
        from subtitles_extractor.presentation.pages.translate_page import TranslatePage
        TranslatePage._persist_inputs(page)
        assert saved["audit_model"] == "gemini-9-preview"

    def test_status_message_names_the_model_really_used(self, app, monkeypatch) -> None:
        page, container = TestPage()._page()
        vm = page._view_model
        page._audit_model_combo.setCurrentIndex(page._audit_model_combo.findData("gemini-3.8-flash"))
        monkeypatch.setattr(container, "make_subtitle_translator",
                            lambda *a, **k: type("T", (), {"is_available": lambda self: True})())
        monkeypatch.setattr(container, "make_analyze_context_use_case", lambda *a, **k: object())
        vm.set_source_events(
            __import__("tests.unit.presentation.test_translate_page_bugs_v323609",
                       fromlist=["_events"])._events("x"))
        messages: list[str] = []
        vm.status_message.connect(messages.append)
        started: dict = {}
        from subtitles_extractor.presentation.workers import refine_analysis_worker as mod

        class FakeWorker:
            def __init__(self, *a, **k):
                started.update(k)
                for name in ("finished_ok", "cancelled", "failed", "progress_message", "finished"):
                    setattr(self, name, type("Sig", (), {"connect": lambda s, f: None})())

            def start(self):
                pass

        monkeypatch.setattr(mod, "RefineAnalysisWorker", FakeWorker)
        vm.start_analysis_refinement("k", "Vietnamese", "gemini-3.8-flash", "zh", "a", "b", "", "")
        assert started["audit_model"] == "gemini-3.8-flash"
        assert any("gemini-3.8-flash" in m and "tự kiểm tra" in m for m in messages)
