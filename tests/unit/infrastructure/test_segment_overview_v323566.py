"""[v3.23.566] Hồi quy: tóm tắt phim gồm các bản toàn phim lặp nhau, mất đoạn giữa.

Chuẩn đoán thật v3.23.565 (gemini-3.8-flash): mỗi đoạn video model trả lại tóm tắt toàn
phim từ đầu (đúng chỉ thị "hợp nhất thành bản cập nhật đầy đủ"), bộ gộp cũ NỐI chúng lại →
tóm tắt 20 049 ký tự, 7/13 đoạn văn mở đầu y hệt "Thế giới quan đan xen giữa cuộc sống
nông thôn…", chạm trần và bị cắt MẤT phần giữa phim. Ngoài ra bảng nhân vật lẫn một mục
tiêu đề "Xưng hô" như thể là một nhân vật.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from subtitles_extractor.application.services.overview_composer import (
    TIMELINE_HEADER,
    SegmentSummary,
    compose_overview,
    trim_at_sentence,
)
from subtitles_extractor.application.services.roster_format import roster_names
from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine
from subtitles_extractor.infrastructure.translation.analysis_checkpoint import (
    AnalysisCheckpointStore,
    AnalysisProgress,
)
from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    QuotaExhaustedError,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)
from tests.unit.infrastructure.gemini_call_harness import bare_analysis_adapter

SETUP = "Thế giới quan đan xen giữa cuộc sống nông thôn, đô thị và tu chân."


class TestComposeOverview:
    def test_segments_are_ordered_by_time_not_by_run_order(self) -> None:
        text = compose_overview(
            "Toàn cảnh.",
            [
                SegmentSummary(700, 1000, "Đoạn ba."),
                SegmentSummary(0, 345, "Đoạn một."),
                SegmentSummary(345, 700, "Đoạn hai."),
            ],
        )
        body = text.split(TIMELINE_HEADER, 1)[1]
        assert body.index("Đoạn một") < body.index("Đoạn hai") < body.index("Đoạn ba")
        assert "[00:00\u201305:45]" in text

    def test_hours_are_shown_for_long_films(self) -> None:
        text = compose_overview("", [SegmentSummary(7263, 7609, "Kết phim.")])
        assert "[2:01:03\u20132:06:49]" in text

    def test_length_cap_keeps_every_segment(self) -> None:
        segments = [
            SegmentSummary(i * 345, (i + 1) * 345, "Câu dài. " * 300) for i in range(22)
        ]
        text = compose_overview("Toàn cảnh. " * 1000, segments, max_chars=20_000)
        assert len(text) <= 20_500
        assert text.count("\n[") == 22

    def test_without_segments_returns_global_only(self) -> None:
        assert compose_overview("Toàn cảnh.", []) == "Toàn cảnh."

    def test_trim_prefers_sentence_end(self) -> None:
        assert trim_at_sentence("Một câu. Hai câu dài hơn nhiều.", 20) == "Một câu."


def _ref(idx: int):
    ref = MagicMock()
    ref.chunk_index = idx
    ref.remote_name = f"files/f{idx}"
    ref.start_sec = idx * 345.0
    ref.end_sec = (idx + 1) * 345.0
    return ref


def _adapter(checkpoint_dir: Path | None = None) -> GeminiSubtitleTranslator:
    return bare_analysis_adapter(checkpoint_dir=checkpoint_dir)


class _Gemini:
    """Model giả làm đúng ca thật: 'overview' luôn là tóm tắt toàn phim từ đầu."""

    def __init__(self, with_segment_summary: bool = True, quota: int = 99) -> None:
        self.with_segment_summary = with_segment_summary
        self.quota = quota
        self.prompts: list[str] = []

    def __call__(self, _model, prompt, _config, _validator, video_refs=None, **_kw):
        if len(self.prompts) >= self.quota:
            raise QuotaExhaustedError("hết hạn mức ngày")
        self.prompts.append(prompt)
        idx = int(video_refs[0].chunk_index)
        data = {
            "source_lang": "zh",
            "characters": [{"name": "宋钱", "aka": "Tống Tiền", "role": "chính"}],
            "overview": f"{SETUP} Tới đoạn {idx + 1}, Tống Tiền đã làm {idx + 1} việc.",
            "glossary": [],
        }
        if self.with_segment_summary:
            data["segment_summary"] = f"Chi tiết riêng của đoạn {idx + 1}."
        return data


def _run(adapter: GeminiSubtitleTranslator, gemini: _Gemini, n: int = 5):
    lines = [TranslationLine(index=1, start_ms=0, end_ms=1, text="c1")]
    with (
        patch.object(
            adapter, "_resolve_video_handles", lambda r: [f"h{i}" for i in range(n)]
        ),
        patch.object(adapter, "_build_config", lambda *a, **k: MagicMock()),
        patch.object(adapter, "_call_gemini", gemini),
        patch.object(adapter, "_ensure_available", lambda: None),
    ):
        return adapter.analyze_global_context(
            lines, "Vietnamese", video_refs=[_ref(i) for i in range(n)]
        )


class TestSequentialAnalysisOverview:
    def test_setup_paragraph_is_not_repeated(self) -> None:
        result = _run(_adapter(), _Gemini())
        assert result.overview.count("Thế giới quan") == 1
        for i in range(1, 6):
            assert f"Chi tiết riêng của đoạn {i}." in result.overview

    def test_global_part_is_the_latest_version(self) -> None:
        result = _run(_adapter(), _Gemini())
        assert "Tới đoạn 5" in result.overview
        assert "Tới đoạn 1," not in result.overview

    def test_prompt_asks_for_both_fields(self) -> None:
        gemini = _Gemini()
        _run(_adapter(), gemini)
        assert all('"segment_summary"' in prompt for prompt in gemini.prompts)

    def test_previous_segment_detail_is_fed_forward(self) -> None:
        gemini = _Gemini()
        _run(_adapter(), gemini)
        assert "Chi tiết riêng của đoạn 1." in gemini.prompts[1]

    def test_model_without_segment_summary_keeps_old_merge(self) -> None:
        result = _run(_adapter(), _Gemini(with_segment_summary=False))
        assert TIMELINE_HEADER not in result.overview

    def test_resumed_run_keeps_time_order(self, tmp_path: Path) -> None:
        _run(_adapter(tmp_path), _Gemini(quota=3))
        result = _run(_adapter(tmp_path), _Gemini())
        body = result.overview.split(TIMELINE_HEADER, 1)[1]
        positions = [body.index(f"đoạn {i}.") for i in range(1, 6)]
        assert positions == sorted(positions)


class TestCheckpointKeepsSegmentSummaries:
    def test_round_trip(self, tmp_path: Path) -> None:
        store = AnalysisCheckpointStore(tmp_path)
        store.save(
            "k", AnalysisProgress(done_segments={0}, segment_summaries={0: "Một."})
        )
        loaded = store.load("k")
        assert loaded is not None
        assert loaded.segment_summaries == {0: "Một."}


class TestHeadingIsNotACharacter:
    def test_stray_heading_line_is_dropped(self) -> None:
        roster = (
            "柳老爷子 (Ông nội Liễu) [nam, ~75] — ông nội của Liễu Mộ Vân\n"
            "Xưng hô\n"
            "孙院长 (Viện trưởng Tôn) [nam, ~55] — viện trưởng"
        )
        assert roster_names(roster) == ["柳老爷子", "孙院长"]

    def test_real_name_without_details_is_kept(self) -> None:
        assert roster_names("柳老爷子 (Ông nội Liễu) — ông\n小明") == ["柳老爷子", "小明"]


class TestLegacyCheckpointKeepsOldMerge:
    def test_resume_from_pre_566_progress_does_not_drop_old_detail(
        self, tmp_path: Path
    ) -> None:
        """Tiến độ của v3.23.564/565 (chưa có diễn biến theo đoạn): giữ cách gộp cũ."""
        first = _adapter(tmp_path)
        _run(first, _Gemini(with_segment_summary=False, quota=3))
        result = _run(_adapter(tmp_path), _Gemini())
        assert TIMELINE_HEADER not in result.overview
        assert "Tới đoạn 1" in result.overview


class TestContextDiagnosticsOverviewFlags:
    @staticmethod
    def _codes(overview: str) -> set[str]:
        from types import SimpleNamespace

        from subtitles_extractor.application.services.context_diagnostics import (
            build_context_diagnostics,
        )

        ctx = SimpleNamespace(
            target_lang="Vietnamese",
            source_lang="zh",
            overview=overview,
            characters="宋钱 (Tống Tiền) — chính",
            glossary="",
            visual_cues="",
            enable_tags=False,
            include_desc=True,
        )
        report = build_context_diagnostics(context=ctx, source_events=[])
        return {flag["code"] for flag in report["quality_flags"]}

    def test_repeated_setup_paragraphs_are_flagged(self) -> None:
        overview = "\n\n".join(f"{SETUP} Tới đoạn {i}." for i in range(4))
        assert "overview_repetitive" in self._codes(overview)

    def test_truncated_overview_is_flagged(self) -> None:
        overview = "Đầu.\n\n[… lược bớt phần giữa để giữ độ dài hợp lý …]\n\nCuối."
        assert "overview_truncated" in self._codes(overview)

    def test_timeline_overview_is_clean(self) -> None:
        overview = compose_overview(
            SETUP,
            [SegmentSummary(i * 345, (i + 1) * 345, f"Đoạn {i}.") for i in range(4)],
        )
        assert not {"overview_repetitive", "overview_truncated"} & self._codes(overview)
