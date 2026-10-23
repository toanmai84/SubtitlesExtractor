"""[v3.23.631] Tự kiểm tra phân tích RẺ + HỘI TỤ.

Nhật ký thật (phim 86 tập, 3649 dòng, gemini-3.8-flash 20 request/ngày/key): lượt 1 mất 10 phút
(thinking "high" 16-30 nghìn token, 504 ở mốc 300 giây vì 40 dòng nghi lỗi × cửa sổ ±40 phủ gần cả
phim vào một prompt), rồi lần chạy kế vẫn mở đầu bằng "17 khiếm khuyết" như lần trước.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from subtitles_extractor.application.services.analysis_audit import (
    AUDIT_MAX_CALLS,
    AUDIT_MAX_PROMPT_LINES,
    AUDIT_THINKING_LEVEL,
    AUDIT_WINDOW_LINES,
    DECLINE_LIMIT,
    DEFAULT_AUDIT_MODEL,
    STALE_GLOSSARY_KIND,
    AnalysisDefect,
    AuditReport,
    audit_film_id,
    defect_units,
    describe_audit_report,
    filter_declined,
    find_analysis_defects,
    note_declined,
    plan_audit_batches,
)
from subtitles_extractor.application.services.visual_cue_serializer import (
    serialize_visual_cues,
)
from subtitles_extractor.application.use_cases.analyze_subtitle_context import (
    AnalyzeSubtitleContextUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleContextAnalysis,
    VisualCue,
)
from subtitles_extractor.infrastructure.translation.audit_ledger import (
    MAX_FILMS,
    AuditLedgerStore,
)

ROSTER = (
    "叶然 (Diệp Nhiên) [nữ] — sư muội; xưng hô: với Lý Mặc tự xưng 'muội' gọi 'sư huynh'\n"
    "李默 (Lý Mặc) [nam] — sư huynh; xưng hô: với Diệp Nhiên tự xưng 'ta' gọi 'muội'"
)
FILM = 3649


def _film(n: int = FILM):
    return [SimpleNamespace(index=i, text=f"句子{i}", original_text=f"句子{i}") for i in range(1, n + 1)]


def _analysis(self_address: set[int], n: int = FILM, glossary: str = "") -> SubtitleContextAnalysis:
    cues = []
    for i in range(1, n + 1):
        speaker, target = ("叶然", "李默") if i % 2 else ("李默", "叶然")
        if i in self_address:
            target = speaker
        cues.append(VisualCue(i, speaker, target))
    return SubtitleContextAnalysis(
        source_lang="zh", characters=ROSTER, overview="Tóm tắt.", glossary=glossary,
        visual_cues=serialize_visual_cues(cues),
    )


def _spread(count: int = 40, n: int = FILM) -> set[int]:
    step = n // count
    return {1 + k * step for k in range(count)}


class TestUnits:
    def test_line_defects_split_into_one_unit_per_line(self) -> None:
        defect = AnalysisDefect("self_address", "x", (5, 9))
        assert defect_units(defect) == ["self_address:5", "self_address:9"]

    def test_label_and_table_defects(self) -> None:
        label = AnalysisDefect("unlisted_label", "'苏白' xuất hiện ở 7 dòng nhưng không có", (1, 2))
        assert defect_units(label) == ["unlisted_label:苏白"]
        assert defect_units(AnalysisDefect("empty_roster", "Bảng nhân vật rỗng.")) == [
            "empty_roster:Bảng nhân vật rỗng."
        ]

    def test_filter_keeps_lines_not_yet_given_up_on(self) -> None:
        defect = AnalysisDefect("self_address", "3 dòng", (5, 9, 12))
        declined = {"self_address:5": DECLINE_LIMIT, "self_address:9": DECLINE_LIMIT - 1}
        (kept,) = filter_declined([defect], declined)
        assert kept.line_ids == (9, 12) and kept.detail.startswith("2 dòng")
        assert filter_declined([defect], {f"self_address:{i}": 9 for i in (5, 9, 12)}) == []

    def test_filter_never_drops_stale_glossary_hints(self) -> None:
        hint = AnalysisDefect(STALE_GLOSSARY_KIND, "Thuật ngữ 'x'")
        assert filter_declined([hint], {defect_units(hint)[0]: 99}) == [hint]

    def test_note_declined_counts_only_what_survives_the_patch(self) -> None:
        sent = [AnalysisDefect("self_address", "d", (5, 9)), AnalysisDefect(STALE_GLOSSARY_KIND, "h")]
        remaining = [AnalysisDefect("self_address", "d", (9,)), AnalysisDefect(STALE_GLOSSARY_KIND, "h")]
        after = note_declined({"self_address:9": 1}, sent, remaining)
        assert after == {"self_address:9": 2}                # dòng 5 đã được sửa; gợi ý không ghi

    def test_film_id_changes_with_subtitle_and_language(self) -> None:
        film = _film(10)
        assert audit_film_id(film, "vi") == audit_film_id(film, "vi")
        assert audit_film_id(film, "vi") != audit_film_id(film, "en")
        film[3].text = film[3].original_text = "khác"
        assert audit_film_id(film, "vi") != audit_film_id(_film(10), "vi")


class TestBatches:
    def test_real_sized_film_never_sends_more_than_the_cap(self) -> None:
        defects = [AnalysisDefect("self_address", "40 dòng", tuple(sorted(_spread())))]
        order = list(range(1, FILM + 1))
        batches = plan_audit_batches(defects, order)
        assert 1 < len(batches) <= AUDIT_MAX_CALLS
        sent: list[int] = []
        for batch in batches:
            ids = sorted(i for d in batch for i in d.line_ids)
            window = {
                p for i in ids
                for p in range(max(0, i - 1 - AUDIT_WINDOW_LINES), min(FILM, i + AUDIT_WINDOW_LINES))
            }
            assert len(window) <= AUDIT_MAX_PROMPT_LINES
            sent += ids
        assert sorted(sent) == sorted(_spread())             # mỗi dòng đúng MỘT lần

    def test_old_behaviour_covered_most_of_the_film(self) -> None:
        """Cửa sổ ±40 của 40 dòng rải đều phủ ~ cả phim — nguyên nhân prompt khổng lồ."""
        covered = {
            p for i in _spread() for p in range(max(0, i - 41), min(FILM, i + 40))
        }
        assert len(covered) > 0.85 * FILM
        new = {p for i in _spread() for p in range(max(0, i - 1 - AUDIT_WINDOW_LINES),
                                                   min(FILM, i + AUDIT_WINDOW_LINES))}
        assert len(new) < 0.25 * FILM

    def test_table_level_defects_ride_with_the_first_batch(self) -> None:
        table = AnalysisDefect("empty_overview", "Tóm tắt rỗng.")
        lines = AnalysisDefect("missing_speaker", "n", (10, 2000))
        batches = plan_audit_batches([table, lines], list(range(1, FILM + 1)), max_lines=20)
        assert batches[0][0] is table and len(batches) == 2
        assert plan_audit_batches([table], list(range(1, 5))) == [[table]]
        assert plan_audit_batches([], [1, 2]) == []

    def test_detail_counts_match_each_batch(self) -> None:
        defect = AnalysisDefect("self_address", "40 dòng ghi", tuple(sorted(_spread())))
        for batch in plan_audit_batches([defect], list(range(1, FILM + 1))):
            (part,) = batch
            assert part.detail.startswith(f"{len(part.line_ids)} dòng")


class _Translator:
    """Bộ kiểm định giả: ghi lại các lời gọi; ``fix`` quyết định có sửa không."""

    def __init__(self, fix: bool = False) -> None:
        self.calls: list[dict] = []
        self.fix = fix

    def audit_analysis(self, lines, lang, model, characters, glossary, overview, labels,
                       defects_json, focus, cancel_cb=None):
        self.calls.append({"model": model, "focus": list(focus), "defects": defects_json})
        fixes = []
        if self.fix:
            for line_id in focus:
                speaker = "叶然" if line_id % 2 else "李默"
                fixes.append({"id": line_id, "spk": speaker, "to": "李默" if speaker == "叶然" else "叶然"})
        return {"characters": [], "remove_characters": [], "glossary": [], "remove_glossary": [],
                "overview": "", "cue_fixes": fixes}


class TestUseCase:
    def test_one_run_sends_small_batches_within_the_call_cap(self, tmp_path: Path) -> None:
        translator = _Translator()
        use_case = AnalyzeSubtitleContextUseCase(translator, audit_ledger=None)
        analysis = _analysis(_spread())
        use_case.refine(analysis, _film(), "Vietnamese", "m")
        assert 1 <= len(translator.calls) <= AUDIT_MAX_CALLS
        assert max(len(call["focus"]) for call in translator.calls) < 40      # không còn một cục 40
        report = use_case.last_audit_report
        assert report is not None and report.calls == len(translator.calls)

    def test_unfixable_defects_stop_being_sent_even_across_runs(self, tmp_path: Path) -> None:
        ledger = AuditLedgerStore(tmp_path / "ledger.json")
        film = _film()
        analysis = _analysis({5, 1000, 2500})
        total_calls = 0
        for run in range(DECLINE_LIMIT + 2):
            translator = _Translator(fix=False)
            use_case = AnalyzeSubtitleContextUseCase(translator, audit_ledger=ledger)
            use_case.refine(analysis, film, "Vietnamese", "m", max_rounds=1)
            total_calls += len(translator.calls)
            if run >= DECLINE_LIMIT:
                assert translator.calls == []                  # đã bỏ cuộc: KHÔNG tốn lời gọi nào
                assert use_case.last_audit_report.blocked == 3
        assert total_calls == DECLINE_LIMIT                    # mỗi lần chạy một lời gọi, ba lần

    def test_two_rounds_in_one_run_each_count_as_a_decline(self, tmp_path: Path) -> None:
        ledger = AuditLedgerStore(tmp_path / "ledger.json")
        translator = _Translator()
        use_case = AnalyzeSubtitleContextUseCase(translator, audit_ledger=ledger)
        # Mỗi lượt không đổi gì → dừng sau lượt 1 (hành vi cũ giữ nguyên), nên chỉ tính 1 lần.
        use_case.refine(_analysis({5}), _film(), "Vietnamese", "m")
        assert len(translator.calls) == 1
        assert ledger.load(audit_film_id(_film(), "Vietnamese")) == {"self_address:5": 1}

    def test_fixed_defects_are_not_blamed_on_the_model(self, tmp_path: Path) -> None:
        ledger = AuditLedgerStore(tmp_path / "ledger.json")
        use_case = AnalyzeSubtitleContextUseCase(_Translator(fix=True), audit_ledger=ledger)
        out = use_case.refine(_analysis({5, 1000}), _film(), "Vietnamese", "m")
        assert find_analysis_defects(out, _film()) == []
        assert ledger.load(audit_film_id(_film(), "Vietnamese")) == {}
        assert use_case.last_audit_report.patched >= 2 and use_case.last_audit_report.remaining == 0

    def test_a_clean_analysis_makes_no_call_and_reports_it(self) -> None:
        translator = _Translator()
        use_case = AnalyzeSubtitleContextUseCase(translator)
        use_case.refine(_analysis(set(), n=20), _film(20), "Vietnamese", "m")
        assert translator.calls == []
        assert describe_audit_report(use_case.last_audit_report) == (
            "Tự kiểm tra: không thấy khiếm khuyết nào."
        )

    def test_failed_call_keeps_the_old_analysis_and_does_not_blame(self, tmp_path: Path) -> None:
        from subtitles_extractor.domain.ports.subtitle_translator_port import (
            SubtitleTranslationError,
        )

        class Boom:
            def audit_analysis(self, *a, **k):
                raise SubtitleTranslationError("503")

        ledger = AuditLedgerStore(tmp_path / "ledger.json")
        analysis = _analysis({5})
        use_case = AnalyzeSubtitleContextUseCase(Boom(), audit_ledger=ledger)
        assert use_case.refine(analysis, _film(), "Vietnamese", "m") is analysis
        assert ledger.load(audit_film_id(_film(), "Vietnamese")) == {}

    def test_refine_button_audits_with_the_cheap_model_not_the_analysis_model(self) -> None:
        pytest.importorskip("PySide6")
        from subtitles_extractor.presentation.workers.refine_analysis_worker import (
            RefineAnalysisWorker,
        )

        seen: list[str] = []

        class UC:
            last_audit_report = AuditReport(found=3, calls=1, patched=2)

            def refine(self, current, lines, lang, model, cancel_cb=None, fallback_models=()):
                seen.append(model)
                return current

        done: list[object] = []
        worker = RefineAnalysisWorker(
            UC(), _film(5), "Vietnamese", "gemini-3.8-flash", _analysis(set(), n=5)  # type: ignore[arg-type]
        )
        worker.finished_ok.connect(done.append)
        worker.run()
        assert seen == [DEFAULT_AUDIT_MODEL] == ["gemini-3.1-flash-lite"] and done
        assert AUDIT_THINKING_LEVEL == "medium"
        assert "3 khiếm khuyết" in worker.audit_note and "vá 2" in worker.audit_note

    def test_report_text(self) -> None:
        text = describe_audit_report(AuditReport(found=17, blocked=4, calls=2, patched=9, remaining=3))
        assert "17 khiếm khuyết" in text and "2 lời gọi" in text and "vá 9" in text
        assert "bỏ qua 4" in text and "còn 3" in text
        assert describe_audit_report(None) == ""


class TestLedgerStore:
    def test_roundtrip_and_isolation_per_film(self, tmp_path: Path) -> None:
        store = AuditLedgerStore(tmp_path / "l.json")
        assert store.load("a") == {}
        assert store.save("a", {"x:1": 2}) and store.save("b", {"y:2": 1})
        assert store.load("a") == {"x:1": 2} and store.load("b") == {"y:2": 1}
        store.save("a", {})
        assert store.load("a") == {}

    def test_corrupt_file_is_an_empty_ledger(self, tmp_path: Path) -> None:
        path = tmp_path / "l.json"
        path.write_text("{hỏng", encoding="utf-8")
        assert AuditLedgerStore(path).load("a") == {}
        path.write_text('{"a": {"x": "không phải số", "y": 2, "z": true}}', encoding="utf-8")
        assert AuditLedgerStore(path).load("a") == {"y": 2}

    def test_keeps_only_the_most_recent_films(self, tmp_path: Path) -> None:
        store = AuditLedgerStore(tmp_path / "l.json")
        for i in range(MAX_FILMS + 5):
            store.save(f"film{i}", {"u": 1})
        assert store.load("film0") == {} and store.load(f"film{MAX_FILMS + 4}") == {"u": 1}


class TestAdapterPrompt:
    def _adapter(self, monkeypatch: pytest.MonkeyPatch):
        from tests.unit.infrastructure.gemini_call_harness import bare_analysis_adapter

        adapter = bare_analysis_adapter()
        monkeypatch.setattr(adapter, "_ensure_available", lambda: None)
        built: dict = {}
        sent: dict = {}
        monkeypatch.setattr(adapter, "_build_config", lambda *a, **k: built.update(k) or "cfg")
        monkeypatch.setattr(
            adapter, "_call_gemini",
            lambda model, prompt, config, validator, **k: sent.update(prompt=prompt) or {},
        )
        return adapter, built, sent

    def test_audit_uses_its_own_thinking_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter, built, _sent = self._adapter(monkeypatch)
        adapter._analysis_thinking_level = "high"            # cài đặt phân tích KHÔNG được lan sang đây
        adapter.audit_analysis(_film(50), "Vietnamese", "m", ROSTER, "", "", {}, "[]", [3])
        assert built["thinking_level"] == AUDIT_THINKING_LEVEL == "medium"

    def test_prompt_is_bounded_even_if_the_caller_forgot_to_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter, _built, sent = self._adapter(monkeypatch)
        adapter.audit_analysis(
            _film(), "Vietnamese", "m", ROSTER, "", "", {}, "[]", sorted(_spread())
        )
        rows = [ln for ln in sent["prompt"].splitlines() if ln.count("|") >= 2 and ln[0].isdigit()]
        assert 0 < len(rows) <= AUDIT_MAX_PROMPT_LINES
        assert any(row.startswith("1|") for row in rows)     # vẫn giữ dòng nghi lỗi đầu tiên
