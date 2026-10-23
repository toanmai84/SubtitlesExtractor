"""[v3.23.610] Thống kê token và chi phí ước tính mỗi lượt (học từ gemini-srt-translator).

gemini-srt-translator có ``--token-stats`` / ``--token-report``: cộng token đầu vào, suy
nghĩ, đầu ra của mọi lời gọi rồi tính chi phí theo bảng giá. Ứng dụng của ta trước bản
này chỉ dùng số token để điều tiết hạn mức — người dùng không biết một lượt dịch hay phân
tích đã tiêu bao nhiêu token (và bao nhiêu trong đó là token suy nghĩ).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from subtitles_extractor.application.services.gemini_pricing import (
    PricingFormatError,
    describe_token_usage,
    estimate_cost,
    parse_pricing,
)
from subtitles_extractor.application.services.token_usage import (
    TokenUsage,
    TokenUsageLedger,
    total_usage,
)

_RAW_TABLE = {
    "currency": "USD",
    "updated": "2026-09-08",
    "source": "https://ai.google.dev/gemini-api/docs/pricing",
    "cached_input_ratio": 0.1,
    "models": {
        "gemini-3.8-flash": {"standard": {"input": 0.75, "output": 3.75}},
        "gemini-3.1-flash-lite": {"standard": {"input": 0.25, "output": 1.5}},
        "gemini-3.1-flash-lite-image": {"standard": {"input": 9.0, "output": 9.0}},
    },
}


class TestLedger:
    def test_usage_since_a_mark_only_counts_new_calls(self) -> None:
        ledger = TokenUsageLedger()
        ledger.record("gemini-3.8-flash", TokenUsage(requests=1, prompt=100, output=10))
        mark = ledger.snapshot()
        ledger.record("gemini-3.8-flash", TokenUsage(requests=1, prompt=50, thoughts=7))
        ledger.record("gemini-3.1-flash-lite", TokenUsage(requests=2, prompt=20))
        assert ledger.since(mark) == {
            "gemini-3.8-flash": TokenUsage(requests=1, prompt=50, thoughts=7),
            "gemini-3.1-flash-lite": TokenUsage(requests=2, prompt=20),
        }
        assert ledger.since(ledger.snapshot()) == {}

    def test_parallel_batches_are_all_counted(self) -> None:
        ledger = TokenUsageLedger()

        def _work() -> None:
            for _ in range(500):
                ledger.record("m", TokenUsage(requests=1, output=1))

        threads = [threading.Thread(target=_work) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert ledger.snapshot()["m"] == TokenUsage(requests=4000, output=4000)

    def test_total_adds_every_model(self) -> None:
        usage = total_usage(
            {"a": TokenUsage(1, 10, 2, 3, 4), "b": TokenUsage(2, 20, 0, 0, 6)}
        )
        assert usage == TokenUsage(3, 30, 2, 3, 10)
        assert usage.total == 30 + 3 + 10


class TestCost:
    def test_cached_input_is_cheaper_and_thinking_is_billed_as_output(self) -> None:
        table = parse_pricing(_RAW_TABLE)
        usage = {
            "gemini-3.8-flash": TokenUsage(
                requests=3, prompt=1_000_000, cached=400_000, thoughts=100_000,
                output=100_000,
            )
        }
        estimate = estimate_cost(usage, table)
        # 600k * 0.75 + 400k * 0.075 + 200k * 3.75 (per 1M)
        assert estimate.amount == pytest.approx(0.45 + 0.03 + 0.75)
        assert estimate.unpriced_models == ()

    def test_versioned_model_uses_the_closest_family_price(self) -> None:
        table = parse_pricing(_RAW_TABLE)
        lite = table.rates_for("gemini-3.1-flash-lite-preview-0612")
        assert lite is not None and lite.input_per_million == 0.25

    def test_a_cheaper_variant_is_not_priced_as_the_bigger_model(self) -> None:
        table = parse_pricing(_RAW_TABLE)
        assert table.rates_for("gemini-3.8-flash-lite") is None

    def test_unknown_model_is_reported_not_guessed(self) -> None:
        table = parse_pricing(_RAW_TABLE)
        estimate = estimate_cost({"gemini-9-ultra": TokenUsage(1, 10, 0, 0, 1)}, table)
        assert estimate.amount == 0
        assert estimate.unpriced_models == ("gemini-9-ultra",)

    def test_broken_table_is_rejected(self) -> None:
        with pytest.raises(PricingFormatError):
            parse_pricing({"models": {"m": {"standard": {"input": "rẻ"}}}})

    def test_shipped_table_is_valid_and_covers_the_default_models(self) -> None:
        from subtitles_extractor.presentation.pages import translate_page

        data = Path(translate_page.__file__).resolve().parents[2] / "data"
        raw = json.loads((data / "gemini_pricing.json").read_text(encoding="utf-8"))
        table = parse_pricing(raw)
        for model in ("gemini-3.8-flash", "gemini-3.1-flash-lite", "gemini-2.5-flash"):
            assert table.rates_for(model) is not None


class TestDescription:
    def test_report_shows_thinking_cache_and_cost(self) -> None:
        table = parse_pricing(_RAW_TABLE)
        usage = {
            "gemini-3.8-flash": TokenUsage(
                requests=37, prompt=1_100_000, cached=300_000, thoughts=80_000,
                output=54_567,
            )
        }
        text = describe_token_usage(usage, estimate_cost(usage, table))
        assert "1.234.567 token" in text
        assert "37 lượt gọi" in text
        assert "suy nghĩ 80.000" in text
        assert "300.000 lấy từ cache" in text
        assert "USD" in text

    def test_no_call_means_no_report(self) -> None:
        assert describe_token_usage({}, None) == ""

    def test_several_models_are_listed(self) -> None:
        usage = {
            "gemini-3.8-flash": TokenUsage(requests=1, prompt=900),
            "gemini-3.1-flash-lite": TokenUsage(requests=1, prompt=100),
        }
        text = describe_token_usage(usage, None)
        assert "gemini-3.8-flash: 900" in text and "gemini-3.1-flash-lite: 100" in text


def test_adapter_records_every_answered_call(monkeypatch: pytest.MonkeyPatch) -> None:
    from subtitles_extractor.infrastructure.translation import (
        gemini_translation_adapter as adapter_module,
    )


    ledger = TokenUsageLedger()
    translator = adapter_module.GeminiSubtitleTranslator(
        api_key="KEY", token_ledger=ledger
    )
    meta = SimpleNamespace(
        prompt_token_count=1200, cached_content_token_count=200,
        thoughts_token_count=300, candidates_token_count=90, total_token_count=1590,
    )
    replies = iter([None, '{"subtitles": []}'])

    def generate(*, model: str, contents: object, config: object) -> object:
        return SimpleNamespace(
            text=next(replies), usage_metadata=meta, prompt_feedback=None, candidates=None
        )

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    monkeypatch.setattr(translator, "_get_client", lambda: client)
    monkeypatch.setattr(translator, "_interruptible_sleep", lambda *_a, **_k: False)
    translator._call_gemini("gemini-3.8-flash", "p", object(), lambda _d: None)
    # Lượt trả rỗng vẫn bị máy chủ tính token — cũng phải được đếm.
    assert ledger.snapshot() == {
        "gemini-3.8-flash": TokenUsage(
            requests=2, prompt=2400, cached=400, thoughts=600, output=180
        )
    }


class TestTranslatePageReport:
    """Trang Dịch báo token của đúng lượt vừa chạy."""

    @pytest.fixture()
    def view_model(self, tmp_path: Path):
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from tests.unit.presentation.test_translate_review2_v323609 import (
            _events,
            _view_model,
        )

        view_model = _view_model(tmp_path)
        view_model.set_source_events(_events("句", 3))
        return view_model

    def test_finished_translation_reports_its_own_tokens(self, view_model) -> None:
        from tests.unit.presentation.test_translate_review2_v323609 import _finish

        ledger = view_model._container.token_usage_ledger
        ledger.record("gemini-3.8-flash", TokenUsage(requests=5, prompt=10_000))
        view_model._set_busy(True)  # bắt đầu lượt dịch: đánh dấu sổ
        ledger.record(
            "gemini-3.8-flash",
            TokenUsage(requests=2, prompt=3000, thoughts=500, output=700),
        )
        messages: list[str] = []
        view_model.status_message.connect(messages.append)
        _finish(view_model, ["a", "b", "c"])
        report = view_model.last_token_report
        assert "4.200 token qua 2 lượt gọi" in report
        assert "USD" in report
        assert report in messages[-1]

    def test_fully_cached_run_reports_nothing(self, view_model) -> None:
        from tests.unit.presentation.test_translate_review2_v323609 import _finish

        view_model._set_busy(True)
        _finish(view_model, ["a", "b", "c"])
        assert view_model.last_token_report == ""

    def test_cancelled_run_still_says_what_was_spent(self, view_model) -> None:
        view_model._set_busy(True)
        view_model._container.token_usage_ledger.record(
            "gemini-3.8-flash", TokenUsage(requests=1, prompt=800, output=200)
        )
        messages: list[str] = []
        view_model.status_message.connect(messages.append)
        view_model._on_worker_cancelled()
        assert "1.000 token" in messages[-1]


def test_translate_page_shows_the_report_when_done(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from tests.unit.presentation.test_translate_review2_v323609 import _events, _page

    page = _page(tmp_path)
    shown: list[str] = []
    page._show_success = lambda _title, text: shown.append(text)
    vm = page._view_model
    vm.set_source_events(_events("句", 2))
    vm._last_token_report = "Đã dùng 1.000 token qua 1 lượt gọi Gemini."
    page._on_translation_finished()
    assert shown and shown[-1].endswith("Đã dùng 1.000 token qua 1 lượt gọi Gemini.")
    page.close()


def test_diagnostics_export_carries_the_last_run_tokens(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from tests.unit.presentation.test_translate_review2_v323609 import (
        _events,
        _view_model,
    )

    vm = _view_model(tmp_path)
    vm.set_source_events(_events("句", 2))
    vm._set_busy(True)
    vm._container.token_usage_ledger.record(
        "gemini-3.8-flash", TokenUsage(requests=1, prompt=10, thoughts=4, output=3)
    )
    vm._on_worker_cancelled()
    out = vm.export_translation_diagnostics(tmp_path / "diag.json")
    bundle = json.loads(out.read_text(encoding="utf-8"))
    assert bundle["token_usage_last_run"]["gemini-3.8-flash"]["thoughts"] == 4
