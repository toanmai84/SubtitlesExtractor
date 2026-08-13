"""Điểm vào dòng lệnh cho khung tự hiệu chuẩn đệ quy.

Ví dụ:
    PYTHONPATH=src:. python -m tools.calibration.cli \\
        --mode subtitle --uploads /mnt/user-data/uploads \\
        --state-dir tools/calibration/state --quick

``--quick`` giới hạn cửa sổ thời gian và số lần đánh giá để dò nhanh; bỏ đi để
hiệu chuẩn đầy đủ.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from tools.calibration.datasets import CalibrationDataset
from tools.calibration.ground_truth import auto_pair_seraw_to_srt
from tools.calibration.recursive_calibrator import RecursiveCalibrator
from tools.calibration.report import render_roi_report, render_subtitle_report
from tools.calibration.roi_evaluator import RoiCalibrationEvaluator
from tools.calibration.search_space import ParameterSpec, SearchSpace
from tools.calibration.subtitle_evaluator import SubtitleBuildEvaluator


def _ensure_app_on_path() -> None:
    """Chèn ``src`` của ứng dụng vào sys.path để import pipeline thật."""
    project_root = Path(__file__).resolve().parents[2]
    src_dir = project_root / "src"
    if src_dir.is_dir() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _build_subtitle_search_space() -> SearchSpace:
    return SearchSpace(
        specs=(
            ParameterSpec("similarity_threshold", 0.60, 0.90, "float", 5),
            ParameterSpec("line_similarity_threshold", 0.60, 0.90, "float", 5),
            ParameterSpec("merge_gap_sec", 0.20, 1.00, "float", 5),
            ParameterSpec("min_confidence", 0.20, 0.50, "float", 5),
            ParameterSpec("min_duration_sec", 0.08, 0.30, "float", 4),
            ParameterSpec("temporal_padding_sec", 0.0, 0.15, "float", 4),
            ParameterSpec("y_clustering_tolerance_ratio", 0.15, 0.45, "float", 4),
        )
    )


def _build_roi_search_space() -> SearchSpace:
    return SearchSpace(
        specs=(
            ParameterSpec("band_keep_ratio", 0.30, 0.70, "float", 5),
            ParameterSpec("band_smoothing_ratio", 0.002, 0.015, "float", 5),
        )
    )


def _collect_files(uploads_dir: Path) -> tuple[list[Path], list[Path]]:
    # Khớp cả `name.seraw.json` lẫn `name_seraw.json`.
    seraw_paths = sorted(
        {p for pattern in ("*seraw.json",) for p in uploads_dir.rglob(pattern)}
    )
    srt_paths = [
        path
        for path in sorted(uploads_dir.rglob("*.srt"))
        if not path.name.endswith("_vi.srt") and "output_tts" not in path.name
    ]
    return seraw_paths, srt_paths


def _calibrate_subtitle(args: argparse.Namespace) -> None:
    from subtitles_extractor.application.dtos.extract_subtitles_dto import (
        SubtitleBuilderConfig,
    )
    from subtitles_extractor.application.services.subtitle_pipeline.orchestrator import (
        SubtitleBuilder,
    )
    from subtitles_extractor.infrastructure.serializers.raw_ocr_serializer import (
        load_raw_ocr,
    )

    uploads = Path(args.uploads)
    seraw_paths, srt_paths = _collect_files(uploads)
    pairings = auto_pair_seraw_to_srt(seraw_paths, srt_paths)
    confident = [pair for pair in pairings if pair.is_confident]
    if not confident:
        logger.error(
            "Không có cặp seraw↔SRT nào đủ tin cậy (≥40%% & bỏ xa nhì). "
            "Cần dữ liệu ground-truth khớp đúng để hiệu chuẩn."
        )
        return

    window = (args.window_start, args.window_end) if args.quick else None
    corpus = [
        CalibrationDataset(
            label=pair.seraw_path.stem.replace("_seraw", ""),
            seraw_path=pair.seraw_path,
            srt_path=pair.srt_path,
            sample_step_sec=args.sample_step,
            time_window=window,
        )
        for pair in confident
    ]

    def builder_factory(kwargs: dict[str, float | int | bool]) -> SubtitleBuilder:
        return SubtitleBuilder(SubtitleBuilderConfig(**kwargs))

    evaluator = SubtitleBuildEvaluator(
        corpus=corpus,
        builder_factory=builder_factory,
        ocr_loader=load_raw_ocr,
        base_config_kwargs={"sample_step_sec": args.sample_step},
    )
    search_space = _build_subtitle_search_space()
    baseline_detail = evaluator.score_assignment(
        search_space.coerce_assignment(search_space.midpoint_assignment())
    )

    calibrator = RecursiveCalibrator(
        search_space=search_space,
        objective=evaluator.objective,
        state_path=Path(args.state_dir) / "subtitle_state.json",
        optimizer_kwargs={
            "max_depth": args.max_depth,
            "span_shrink": 0.4,
            "max_sweeps_per_level": 2,
            "max_evaluations": args.max_eval,
        },
    )
    outcome = calibrator.run()
    tuned_detail = evaluator.score_assignment(outcome.best_assignment)
    report = render_subtitle_report(outcome, baseline_detail, tuned_detail)
    _emit_report(report, Path(args.state_dir) / "subtitle_report.md")


def _calibrate_roi(args: argparse.Namespace) -> None:
    from subtitles_extractor.infrastructure.video.bbox_analyzer import (
        BBoxAnalyzer,
        RawBBox,
    )

    uploads = Path(args.uploads)
    seraw_paths, _srt = _collect_files(uploads)
    labelled = [(path.name.replace(".seraw.json", "").replace("_seraw.json", ""), path) for path in seraw_paths]

    def analyzer_factory(
        width: int, height: int, params: dict[str, float | int | bool]
    ) -> BBoxAnalyzer:
        # padding=0: chấm IoU trên dải KHÍT (chất lượng tinh chỉnh), không lẫn margin.
        return BBoxAnalyzer(
            frame_width=width,
            frame_height=height,
            padding=0,
            enable_band_refinement=True,
            band_keep_ratio=float(params["band_keep_ratio"]),
            band_smoothing_ratio=float(params["band_smoothing_ratio"]),
        )

    evaluator = RoiCalibrationEvaluator(
        seraw_paths=labelled,
        bbox_factory=RawBBox,
        analyzer_factory=analyzer_factory,
        labeled_bands=None,  # CHƯA có nhãn dải thật — dùng proxy
    )
    search_space = _build_roi_search_space()
    calibrator = RecursiveCalibrator(
        search_space=search_space,
        objective=evaluator.objective,
        state_path=Path(args.state_dir) / "roi_state.json",
        optimizer_kwargs={
            "max_depth": args.max_depth,
            "span_shrink": 0.4,
            "max_sweeps_per_level": 2,
            "max_evaluations": args.max_eval,
        },
    )
    outcome = calibrator.run()
    report = render_roi_report(outcome, has_labels=True)
    _emit_report(report, Path(args.state_dir) / "roi_report.md")


def _emit_report(report_text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    logger.info("Đã ghi báo cáo: {}", output_path)
    print(report_text)  # noqa: T201 — CLI cố ý in báo cáo ra stdout


def main(argv: list[str] | None = None) -> int:
    """Hàm main CLI; trả mã thoát."""
    parser = argparse.ArgumentParser(description="Hiệu chuẩn đệ quy ROI/build phụ đề.")
    parser.add_argument("--mode", choices=["subtitle", "roi", "both"], default="subtitle")
    parser.add_argument("--uploads", required=True, help="Thư mục chứa seraw.json + srt")
    parser.add_argument("--state-dir", default="tools/calibration/state")
    parser.add_argument("--sample-step", type=float, default=0.04)
    parser.add_argument("--quick", action="store_true", help="Dò nhanh (cửa sổ + trần)")
    parser.add_argument("--window-start", type=float, default=500.0)
    parser.add_argument("--window-end", type=float, default=800.0)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-eval", type=int, default=80)
    args = parser.parse_args(argv)

    _ensure_app_on_path()
    if args.mode in ("subtitle", "both"):
        _calibrate_subtitle(args)
    if args.mode in ("roi", "both"):
        _calibrate_roi(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
