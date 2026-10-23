"""[v3.23.645] Đo A/B OCR trong ứng dụng: kịch bản, so sánh, đề xuất, và use case chạy tuần tự."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    ExtractSubtitlesRequest,
    SubtitleBuilderConfig,
)
from subtitles_extractor.application.services.ocr_benchmark import (
    DEFAULT_KEYS,
    DEFAULT_SCENARIOS,
    ScenarioResult,
    apply_scenario,
    compare_events,
    current_settings_values,
    recommend,
)
from subtitles_extractor.application.services.ocr_probe_window import compute_probe_window
from subtitles_extractor.application.services.subtitle_builder import SubtitleBuilder
from subtitles_extractor.application.use_cases.benchmark_ocr import BenchmarkOcrUseCase
from subtitles_extractor.application.use_cases.extract_subtitles import ExtractSubtitlesUseCase
from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.ports.frame_sampler_port import FrameSamplingConfig
from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
from tests.unit.application.test_ocr_checkpoint_v323632 import (
    DURATION,
    _Engine,
    _Reporter,
    _Sampler,
)

SC = {s.key: s for s in DEFAULT_SCENARIOS}


def _result(key: str, wall: float, events: list[tuple[float, float, str]], error: str = "") -> ScenarioResult:
    return ScenarioResult(scenario=SC[key], wall_s=wall, ocr_s=wall * 0.9, ocr_frames=100,
                          batches=10, events=tuple(events), error=error)


EVENTS = [(1.0, 2.0, "một"), (3.0, 4.0, "hai"), (5.0, 6.0, "ba")]


class TestCompare:
    def test_identical(self) -> None:
        c = compare_events(EVENTS, EVENTS)
        assert c.identical and c.same_text_ratio == 1.0 and c.n_missing == c.n_extra == 0

    def test_missing_and_extra_and_timing(self) -> None:
        other = [(1.0, 2.0, "một"), (5.0, 6.3, "ba"), (7.0, 8.0, "bốn")]
        c = compare_events(EVENTS, other)
        assert c.n_missing == 1 and c.missing == ("hai",) and c.n_extra == 1 and c.extra == ("bốn",)
        assert c.max_end_diff_s == pytest.approx(0.3) and not c.identical

    def test_small_timing_jitter_is_still_identical(self) -> None:
        jitter = [(s + 0.04, e - 0.08, t) for s, e, t in EVENTS]
        assert compare_events(EVENTS, jitter).identical

    def test_two_empty_lists(self) -> None:
        assert compare_events([], []).same_text_ratio == 1.0


class TestRecommend:
    def test_picks_the_fastest_scenario_that_does_not_change_the_result(self) -> None:
        results = [
            _result("current", 100, EVENTS), _result("no_orientation", 92, EVENTS),
            _result("det8", 60, EVENTS), _result("gate200", 30, EVENTS[:2]),     # nhanh nhất nhưng mất 1 câu
        ]
        rec = recommend(results)
        assert rec is not None and rec.key == "det8"
        assert rec.settings_patch == {"ocr": {"use_textline_orientation": False}, "hardware": {"det_batch_size": 8}}
        assert rec.speedup == pytest.approx(1.67, abs=0.01) and not rec.warnings_vi

    def test_gate_recommendation_carries_the_white_subtitle_warning(self) -> None:
        rec = recommend([_result("current", 100, EVENTS), _result("gate200", 40, EVENTS)])
        assert rec is not None and rec.key == "gate200"
        assert rec.settings_patch["frame"] == {"white_text_gate": True, "white_text_luma": 200}
        assert rec.warnings_vi and "trắng" in rec.warnings_vi[0]

    def test_lossless_wins_ties(self) -> None:
        rec = recommend([_result("current", 100, EVENTS), _result("det8", 50, EVENTS), _result("gate200", 50, EVENTS)])
        assert rec is not None and rec.key == "det8"

    def test_keeps_current_when_nothing_is_meaningfully_faster(self) -> None:
        rec = recommend([_result("current", 100, EVENTS), _result("det4", 95, EVENTS)])
        assert rec is not None and rec.key == "current" and not rec.changes_anything

    def test_no_recommendation_without_a_usable_reference(self) -> None:
        assert recommend([_result("current", 100, [])]) is None                      # đoạn không có phụ đề
        assert recommend([_result("current", 0, [], error="boom"), _result("det8", 5, EVENTS)]) is None
        assert recommend([]) is None

    def test_failed_scenarios_are_ignored(self) -> None:
        rec = recommend([_result("current", 100, EVENTS), _result("det8", 1, EVENTS, error="OOM"),
                         _result("det4", 70, EVENTS)])
        assert rec is not None and rec.key == "det4"


def _request(tmp_path: Path) -> ExtractSubtitlesRequest:
    return ExtractSubtitlesRequest(
        video_path=tmp_path / "phim.mp4", output_path=tmp_path / "out.srt",
        sampling=FrameSamplingConfig(sample_step_sec=0.1),
        ocr=OcrEngineConfig(batch_size=4, use_textline_orientation=True),
        builder=SubtitleBuilderConfig(), raw_ocr_output_path=tmp_path / "raw.json.gz", export_raw_images=True,
    )


class TestApplyScenario:
    def test_overrides_only_what_the_scenario_names_and_never_writes_files(self, tmp_path: Path) -> None:
        window = compute_probe_window(DURATION, probe_seconds=4.0)
        request = apply_scenario(_request(tmp_path), SC["gate200_det8"], window)
        assert request.ocr.det_batch_size == 8 and request.ocr.use_textline_orientation is False
        assert request.sampling.white_text_gate and request.sampling.white_text_luma == 200
        assert request.sampling.skip_intro_sec == window.skip_intro_sec
        assert request.skip_export and request.raw_ocr_output_path is None and not request.export_raw_images
        assert request.ocr.batch_size == 4 and request.sampling.sample_step_sec == 0.1       # phần còn lại giữ nguyên

    def test_current_keeps_the_user_configuration(self, tmp_path: Path) -> None:
        base = _request(tmp_path)
        base = dataclasses.replace(base, ocr=dataclasses.replace(base.ocr, det_batch_size=6))
        window = compute_probe_window(DURATION, probe_seconds=4.0)
        request = apply_scenario(base, SC["current"], window)
        assert request.ocr.det_batch_size == 6 and request.ocr.use_textline_orientation is True

    def test_default_selection_is_a_valid_subset(self) -> None:
        assert set(DEFAULT_KEYS) <= set(SC) and "current" in DEFAULT_KEYS


class _ConfigurableEngine(_Engine):
    def __init__(self) -> None:
        super().__init__()
        self.configs: list[OcrEngineConfig] = []
        self.fail_on_batch: int | None = None

    def infer_batch(self, images_rgb, frame_indices, timestamps_sec):
        if self.fail_on_batch and len(images_rgb) == self.fail_on_batch and self.configs[-1].det_batch_size > 1:
            raise RuntimeError("máy lỗi")
        return super().infer_batch(images_rgb, frame_indices, timestamps_sec)

    def initialize(self) -> None:
        pass

    def release(self) -> None:
        pass


def _use_case(tmp_path: Path, engine: _ConfigurableEngine, fail: bool = False) -> BenchmarkOcrUseCase:
    metadata = VideoMetadata(path=tmp_path / "phim.mp4", width=720, height=1280, fps=25.0,
                             total_frames=250, duration_sec=DURATION)
    reader = type("Reader", (), {"read": lambda self, path: metadata})()

    def factory(counting):
        return ExtractSubtitlesUseCase(
            metadata_reader=reader, frame_sampler=_Sampler(), ocr_engine=counting,
            builder=SubtitleBuilder(SubtitleBuilderConfig()), exporters={},
        )

    return BenchmarkOcrUseCase(factory, engine, engine.configs.append)


class TestUseCase:
    def test_runs_every_scenario_counts_frames_and_restores_the_engine_config(self, tmp_path: Path) -> None:
        (tmp_path / "phim.mp4").write_bytes(b"0" * 64)
        engine = _ConfigurableEngine()
        use_case = _use_case(tmp_path, engine)
        base = _request(tmp_path)
        window = compute_probe_window(DURATION, probe_seconds=DURATION)
        results = use_case.execute(base, [SC["det8"], SC["current"], SC["no_orientation"]], window)
        assert [r.scenario.key for r in results] == ["current", "det8", "no_orientation"]    # mốc so sánh chạy đầu
        assert all(not r.error and r.ocr_frames > 0 and r.batches > 0 and r.events for r in results)
        assert results[0].events == results[1].events == results[2].events
        assert engine.configs[-1] == base.ocr                                    # đã trả về cấu hình ban đầu
        assert [c.det_batch_size for c in engine.configs[:-1]] == [1, 8, 1]

    def test_one_failing_scenario_does_not_stop_the_others(self, tmp_path: Path) -> None:
        (tmp_path / "phim.mp4").write_bytes(b"0" * 64)
        engine = _ConfigurableEngine()
        engine.fail_on_batch = 8                                             # chỉ lô 8 ảnh (warm-up det8) lỗi
        results = _use_case(tmp_path, engine).execute(
            _request(tmp_path), [SC["current"], SC["det8"], SC["no_orientation"]],
            compute_probe_window(DURATION, probe_seconds=DURATION),
        )
        by_key = {r.scenario.key: r for r in results}
        assert by_key["det8"].error and not by_key["current"].error and not by_key["no_orientation"].error

    def test_cancel_stops_after_the_current_scenario(self, tmp_path: Path) -> None:
        (tmp_path / "phim.mp4").write_bytes(b"0" * 64)
        engine = _ConfigurableEngine()
        results = _use_case(tmp_path, engine).execute(
            _request(tmp_path), [SC["current"], SC["det8"]], compute_probe_window(DURATION, probe_seconds=DURATION),
            progress=_Reporter(cancel_after_checks=1),
        )
        assert len(results) <= 1
        assert engine.configs[-1] == _request(tmp_path).ocr

    def test_progress_is_reported_per_scenario(self, tmp_path: Path) -> None:
        (tmp_path / "phim.mp4").write_bytes(b"0" * 64)
        reporter = _Reporter()
        _use_case(tmp_path, _ConfigurableEngine()).execute(
            _request(tmp_path), [SC["current"], SC["det8"]], compute_probe_window(DURATION, probe_seconds=DURATION),
            progress=reporter,
        )
        assert any("Kịch bản 1/2" in m for m in reporter.messages) and any("Kịch bản 2/2" in m for m in reporter.messages)


def test_warm_up_uses_the_batch_size_of_the_scenario(tmp_path: Path) -> None:
    (tmp_path / "phim.mp4").write_bytes(b"0" * 64)
    seen: list[int] = []

    class _Spy(_ConfigurableEngine):
        def infer_batch(self, images_rgb, frame_indices, timestamps_sec):
            if frame_indices == list(range(len(images_rgb))) and timestamps_sec == [0.0] * len(images_rgb):
                seen.append(len(images_rgb))
            return np.array([]).tolist() or super().infer_batch(images_rgb, frame_indices, timestamps_sec)

    _use_case(tmp_path, _Spy()).execute(
        _request(tmp_path), [SC["det8"]], compute_probe_window(DURATION, probe_seconds=DURATION),
    )
    assert 8 in seen


class TestReviewFixes:
    """[v3.23.645] Ca biên do rà soát độc lập tìm ra."""

    def test_recommendation_equal_to_current_settings_changes_nothing(self, tmp_path: Path) -> None:
        base = dataclasses.replace(_request(tmp_path), ocr=OcrEngineConfig(batch_size=4, use_textline_orientation=False))
        rec = recommend(
            [_result("current", 100, EVENTS), _result("no_orientation", 80, EVENTS)],
            current_settings_values(base),
        )
        assert rec is not None and rec.key == "current" and not rec.changes_anything

    def test_patch_keeps_only_keys_that_differ_from_current_settings(self, tmp_path: Path) -> None:
        base = dataclasses.replace(
            _request(tmp_path), ocr=OcrEngineConfig(batch_size=4, use_textline_orientation=False, det_batch_size=1),
        )
        rec = recommend(
            [_result("current", 100, EVENTS), _result("det8", 40, EVENTS)], current_settings_values(base),
        )
        assert rec is not None and rec.settings_patch == {"hardware": {"det_batch_size": 8}}

    def test_scenario_equal_to_current_configuration_is_not_measured(self, tmp_path: Path) -> None:
        (tmp_path / "phim.mp4").write_bytes(b"0" * 64)
        engine = _ConfigurableEngine()
        base = dataclasses.replace(_request(tmp_path), ocr=OcrEngineConfig(batch_size=4, use_textline_orientation=False))
        results = _use_case(tmp_path, engine).execute(
            base, [SC["current"], SC["no_orientation"], SC["det8"]],
            compute_probe_window(DURATION, probe_seconds=DURATION),
        )
        assert [r.scenario.key for r in results] == ["current", "det8"]

    def test_cancelled_scenario_is_not_reported_with_partial_numbers(self, tmp_path: Path) -> None:
        (tmp_path / "phim.mp4").write_bytes(b"0" * 64)
        for checks in range(0, 40):
            results = _use_case(tmp_path, _ConfigurableEngine()).execute(
                _request(tmp_path), [SC["current"], SC["det8"]],
                compute_probe_window(DURATION, probe_seconds=DURATION), progress=_Reporter(cancel_after_checks=checks),
            )
            assert all(r.wall_s > 0 or r.error for r in results), checks

    def test_measurement_does_not_touch_the_reocr_frame_cache(self, tmp_path: Path) -> None:
        from subtitles_extractor.application.use_cases import extract_subtitles as module

        (tmp_path / "phim.mp4").write_bytes(b"0" * 64)
        cache = module._GLOBAL_FRAME_CACHE
        sentinel = [object()]
        cache._cached_key, cache._cached_frames = "giữ-lại", sentinel
        base = _request(tmp_path)
        window = compute_probe_window(DURATION, probe_seconds=DURATION)
        assert apply_scenario(base, SC["current"], window).use_frame_cache is False
        _use_case(tmp_path, _ConfigurableEngine()).execute(base, [SC["current"]], window)
        assert cache._cached_key == "giữ-lại" and cache._cached_frames is sentinel
        cache._cached_key, cache._cached_frames = "", []

    def test_expected_frames_follow_the_scanned_window(self, tmp_path: Path) -> None:
        (tmp_path / "phim.mp4").write_bytes(b"0" * 64)
        totals: set[int] = set()

        class _TotalsReporter(_Reporter):
            def report(self, current: int, total: int, message: str) -> None:
                totals.add(total)

        reporter = _TotalsReporter()
        metadata = VideoMetadata(path=tmp_path / "phim.mp4", width=720, height=1280, fps=25.0,
                                 total_frames=250, duration_sec=DURATION)
        reader = type("Reader", (), {"read": lambda self, path: metadata})()
        use_case = ExtractSubtitlesUseCase(
            metadata_reader=reader, frame_sampler=_Sampler(), ocr_engine=_ConfigurableEngine(),
            builder=SubtitleBuilder(SubtitleBuilderConfig()), exporters={},
        )
        request = dataclasses.replace(
            _request(tmp_path), skip_export=True, raw_ocr_output_path=None,
            sampling=FrameSamplingConfig(sample_step_sec=0.1, skip_intro_sec=DURATION - 2.0),
        )
        use_case.execute(request, reporter)
        assert 20 in totals and max(totals) <= 100, totals      # trước: 100 → 10 × thời lượng cả phim
