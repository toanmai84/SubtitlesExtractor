"""Unit test cho khung tự hiệu chuẩn đệ quy (tools.calibration).

Phủ: metric thuần, đặc tả không gian tham số, optimizer đệ quy + cache, ghép cặp
ground-truth, evaluator (mock builder), và warm-start của calibrator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.calibration.datasets import CalibrationDataset
from tools.calibration.exceptions import (
    EmptyDatasetError,
    InvalidParameterSpecError,
    PairingAmbiguousError,
)
from tools.calibration.ground_truth import auto_pair_seraw_to_srt, parse_srt
from tools.calibration.metrics import (
    character_error_rate,
    levenshtein_distance,
    score_subtitles,
)
from tools.calibration.optimizer import RecursiveCoordinateDescentOptimizer
from tools.calibration.recursive_calibrator import RecursiveCalibrator
from tools.calibration.search_space import ParameterSpec, SearchSpace
from tools.calibration.subtitle_evaluator import SubtitleBuildEvaluator


class TestMetrics:
    def test_levenshtein_identical_is_zero(self) -> None:
        assert levenshtein_distance("修仙", "修仙") == 0

    def test_levenshtein_one_substitution(self) -> None:
        assert levenshtein_distance("金丹", "金石") == 1

    def test_cer_ignores_punctuation_and_space(self) -> None:
        assert character_error_rate("好 啊！", "好啊") == 0.0

    def test_cer_partial(self) -> None:
        assert character_error_rate("金丹后期", "一金丹后期") == pytest.approx(0.25)

    def test_score_exact_and_spurious(self) -> None:
        ground_truth = [(0.0, 1.0, "你好"), (2.0, 3.0, "再见")]
        built = [(0.0, 1.0, "你好"), (2.0, 3.0, "再見"), (5.0, 6.0, "多余")]
        score = score_subtitles(ground_truth, built)
        assert score.exact_count == 1  # '再见' vs '再見' khác 1 ký tự
        assert score.matched_count == 2
        assert score.spurious_count == 1
        assert 0.0 <= score.quality <= 1.0


class TestSearchSpace:
    def test_invalid_low_high(self) -> None:
        with pytest.raises(InvalidParameterSpecError):
            ParameterSpec("x", 1.0, 0.0)

    def test_coerce_int_and_bool(self) -> None:
        assert ParameterSpec("n", 0, 10, "int").coerce(3.6) == 4
        assert ParameterSpec("flag", 0, 1, "bool").coerce(0.7) is True

    def test_grid_shrinks_around_center(self) -> None:
        spec = ParameterSpec("x", 0.0, 1.0, "float", 5)
        wide = spec.grid(center=0.5, span_ratio=1.0)
        narrow = spec.grid(center=0.5, span_ratio=0.2)
        assert (max(wide) - min(wide)) > (max(narrow) - min(narrow))

    def test_duplicate_names_rejected(self) -> None:
        with pytest.raises(InvalidParameterSpecError):
            SearchSpace(specs=(ParameterSpec("x", 0, 1), ParameterSpec("x", 0, 1)))


class TestRecursiveOptimizer:
    def test_converges_to_optimum(self) -> None:
        space = SearchSpace(
            specs=(ParameterSpec("x", 0.0, 1.0, "float", 5), ParameterSpec("y", 0, 6, "int", 4))
        )

        def objective(assignment: dict[str, float]) -> float:
            return -((assignment["x"] - 0.7) ** 2) - 0.1 * ((assignment["y"] - 3) ** 2)

        optimizer = RecursiveCoordinateDescentOptimizer(
            search_space=space, objective=objective, max_depth=3, span_shrink=0.4
        )
        result = optimizer.optimize()
        assert abs(result.best_assignment["x"] - 0.7) < 0.06
        assert result.best_assignment["y"] == 3

    def test_cache_avoids_recompute(self) -> None:
        space = SearchSpace(specs=(ParameterSpec("x", 0.0, 1.0, "float", 3),))
        call_counter = {"n": 0}

        def objective(assignment: dict[str, float]) -> float:
            call_counter["n"] += 1
            return -((assignment["x"] - 0.5) ** 2)

        optimizer = RecursiveCoordinateDescentOptimizer(
            search_space=space, objective=objective, max_depth=2
        )
        result = optimizer.optimize()
        # evaluations (cache size) <= số lần gọi objective thực
        assert result.evaluations == call_counter["n"]


class TestGroundTruth:
    def test_parse_srt(self, tmp_path: Path) -> None:
        srt = tmp_path / "x.srt"
        srt.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\n你好\n\n"
            "2\n00:00:03,000 --> 00:00:04,500\n世界\n",
            encoding="utf-8",
        )
        cues = parse_srt(srt)
        assert len(cues) == 2
        assert cues[0].text == "你好"
        assert cues[1].end_sec == pytest.approx(4.5)

    def test_pairing_requires_srt(self) -> None:
        with pytest.raises(PairingAmbiguousError):
            auto_pair_seraw_to_srt([Path("a_seraw.json")], [])


class TestSubtitleEvaluatorWithMock:
    def test_empty_corpus_rejected(self) -> None:
        with pytest.raises(EmptyDatasetError):
            SubtitleBuildEvaluator(
                corpus=[], builder_factory=lambda _kw: None, ocr_loader=lambda _p: ([], None)
            )

    def test_objective_with_mock_builder(self, tmp_path: Path) -> None:
        srt = tmp_path / "gt.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8")
        seraw = tmp_path / "gt_seraw.json"
        seraw.write_text("{}", encoding="utf-8")

        class _Interval:
            def __init__(self, start: float, end: float) -> None:
                self.start_sec = start
                self.end_sec = end

        class _Event:
            def __init__(self, text: str, start: float, end: float) -> None:
                self.text = text
                self.interval = _Interval(start, end)

        class _PerfectBuilder:
            def build(self, _frames, roi=None):  # noqa: ANN001
                return [_Event("你好", 0.0, 1.0)]

        evaluator = SubtitleBuildEvaluator(
            corpus=[CalibrationDataset("gt", seraw, srt)],
            builder_factory=lambda _kw: _PerfectBuilder(),
            ocr_loader=lambda _p: ([], None),
        )
        score = evaluator.score_assignment({"similarity_threshold": 0.75})
        assert score.exact_match_rate == 1.0
        assert evaluator.objective({"similarity_threshold": 0.75}) > 0.9


class TestWarmStart:
    def test_state_persists_and_warm_starts(self, tmp_path: Path) -> None:
        space = SearchSpace(specs=(ParameterSpec("x", 0.0, 1.0, "float", 5),))

        def objective(assignment: dict[str, float]) -> float:
            return -((assignment["x"] - 0.8) ** 2)

        state_path = tmp_path / "state.json"
        first = RecursiveCalibrator(
            search_space=space, objective=objective, state_path=state_path,
            optimizer_kwargs={"max_depth": 2},
        ).run()
        assert state_path.is_file()
        assert first.improved is True

        # Phiên 2: warm-start, không được thụt lùi.
        second = RecursiveCalibrator(
            search_space=space, objective=objective, state_path=state_path,
            optimizer_kwargs={"max_depth": 2},
        ).run()
        assert second.best_score >= first.best_score
        assert second.previous_best_score == pytest.approx(first.best_score)
