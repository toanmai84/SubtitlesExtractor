"""[v3.23.644] Đo A/B tốc độ + chất lượng OCR trên MÁY CỦA BẠN (GPU thật) bằng đúng pipeline của ứng dụng.

Chạy (từ thư mục gốc dự án, cùng môi trường Python với ứng dụng)::

    python tools/bench_ocr.py --video "D:\\Phim\\phim_tron_bo.mkv" --roi 57,819,603,119 --start 600 --seconds 90

* ``--roi x,y,w,h``: vùng phụ đề (lấy ở trang Trích xuất; phim mẫu 720x1280 là ``57,819,603,119``).
* ``--start`` / ``--seconds``: đoạn phim đo (chọn đoạn có thoại dày, ~90 s là đủ; mỗi kịch bản mất 1-3 phút).
* ``--app-settings``: dùng đúng cài đặt đang lưu trong ứng dụng (bộ giải mã, tiền xử lý, batch…) làm nền.
* ``--scenarios``: danh sách kịch bản, ngăn cách bằng dấu phẩy (mặc định: tất cả).

Mỗi kịch bản chạy TOÀN BỘ ``ExtractSubtitlesUseCase`` (lấy mẫu → lọc khung → OCR theo lô → dựng phụ đề) trên cùng đoạn phim, rồi so
phụ đề dựng ra với kịch bản ``current`` (cấu hình hiện tại của bạn). Kết quả in ra bảng và ghi ``bench_ocr_result.json`` —
gửi file JSON đó lại để chọn cấu hình tối ưu. Script KHÔNG sửa cài đặt hay dữ liệu của ứng dụng.
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

#: Kịch bản: tên → (ghi đè). ``orientation`` = bộ phân loại hướng dòng chữ.
SCENARIOS: dict[str, dict[str, object]] = {
    "current": {},
    "no_orientation": {"orientation": False},
    "det4": {"orientation": False, "det_batch": 4},
    "det8": {"orientation": False, "det_batch": 8},
    "det16": {"orientation": False, "det_batch": 16},
    "gate200": {"orientation": False, "gate": True, "luma": 200},
    "gate220": {"orientation": False, "gate": True, "luma": 220},
    "gate200_det8": {"orientation": False, "gate": True, "luma": 200, "det_batch": 8},
}


class _CountingEngine:
    """Bọc engine OCR: đếm số khung / số lô thật sự đi qua bộ OCR."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.frames = 0
        self.batches = 0
        self.infer_seconds = 0.0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @property
    def is_initialized(self) -> bool:
        return self._inner.is_initialized

    def infer_batch(self, images_rgb, frame_indices, timestamps_sec):
        started = time.perf_counter()
        try:
            return self._inner.infer_batch(images_rgb, frame_indices, timestamps_sec)
        finally:
            self.infer_seconds += time.perf_counter() - started
            self.frames += len(images_rgb)
            self.batches += 1


def _parse_roi(text: str):
    from subtitles_extractor.domain.value_objects.roi import Roi

    x, y, w, h = (int(v) for v in text.split(","))
    return Roi(x=x, y=y, width=w, height=h)


def _base_configs(args, container=None):
    """Cấu hình nền (lấy mẫu + OCR + bộ lấy mẫu) — từ cài đặt ứng dụng nếu có."""
    from subtitles_extractor.domain.ports.frame_sampler_port import FrameSamplingConfig
    from subtitles_extractor.domain.ports.ocr_engine_port import DeviceKind, OcrEngineConfig

    if container is None:
        device = DeviceKind.GPU if args.device == "gpu" else DeviceKind.CPU
        ocr = OcrEngineConfig(device=device, batch_size=args.batch, use_textline_orientation=True)
        sampling = FrameSamplingConfig(sample_step_sec=args.step)
        return sampling, ocr, _make_sampler(args.decoder)

    snapshot = container.settings_service.current
    ocr = container._build_ocr_config()
    from importlib.util import find_spec

    is_nvdec = snapshot.hardware.frame_decoder_backend == "pynvvideocodec" and find_spec("cupy") is not None
    pre = snapshot.preprocess
    sampling = FrameSamplingConfig(
        sample_step_sec=snapshot.frame.sample_step_sec,
        phash_distance_threshold=snapshot.frame.phash_distance,
        pixel_diff_threshold=snapshot.frame.pixel_diff_ratio,
        apply_median_blend=pre.apply_median_blend,
        median_blend_frames=pre.median_blend_frames,
        vram_upscale_small_text=pre.upscale_small_text if is_nvdec else False,
        vram_upscale_target_height_px=pre.upscale_target_height_px,
        vram_add_border=pre.add_white_border if is_nvdec else False,
        vram_border_thickness_px=pre.border_thickness_px,
        vram_sharpen=pre.apply_sharpen if is_nvdec else False,
        vram_contrast_factor=pre.contrast_factor if (is_nvdec and pre.apply_contrast_boost) else 1.0,
        white_text_gate=False,
    )
    return sampling, ocr, container.frame_sampler


def _make_sampler(kind: str):
    from importlib.util import find_spec

    use_nvdec = kind == "pynvvideocodec" or (
        kind == "auto" and find_spec("PyNvVideoCodec") is not None and find_spec("cupy") is not None
    )
    if use_nvdec:
        from subtitles_extractor.infrastructure.video.decoders.pynvvideocodec_frame_sampler import (
            PyNvVideoCodecFrameSampler,
        )

        return PyNvVideoCodecFrameSampler(gpu_id=0)
    from subtitles_extractor.infrastructure.video.decoders.pyav_frame_sampler import PyAvFrameSampler

    return PyAvFrameSampler()


def _run_scenario(name: str, spec: dict, args, sampling, ocr, sampler, metadata, roi, workdir: Path):
    from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import PaddleOcrAdapter

    # Khóa nào kịch bản không nêu thì GIỮ giá trị của cấu hình nền (với --app-settings là cài đặt hiện tại của bạn).
    ocr_cfg = dataclasses.replace(
        ocr,
        use_textline_orientation=bool(spec.get("orientation", ocr.use_textline_orientation)),
        det_batch_size=int(spec.get("det_batch", ocr.det_batch_size)),
    )
    sampling_cfg = dataclasses.replace(
        sampling,
        sample_step_sec=sampling.sample_step_sec,
        skip_intro_sec=args.start,
        skip_outro_sec=max(0.0, metadata.duration_sec - (args.start + args.seconds)),
        white_text_gate=bool(spec.get("gate", sampling.white_text_gate)),
        white_text_luma=int(spec.get("luma", sampling.white_text_luma)),
    )
    # Mỗi kịch bản phải đọc/giải mã khung thật, không lấy từ bộ đệm RAM của kịch bản trước (làm lệch giờ).
    from subtitles_extractor.application.use_cases import extract_subtitles as extract_module

    extract_module._GLOBAL_FRAME_CACHE.clear()
    engine = _CountingEngine(PaddleOcrAdapter(ocr_cfg))
    try:
        return _measure(name, spec, args, sampling_cfg, ocr_cfg, sampler, metadata, roi, workdir, engine)
    finally:
        try:
            engine.release()
        except Exception:  # noqa: BLE001 - dọn dẹp tốt nhất có thể
            pass


def _measure(name, spec, args, sampling_cfg, ocr_cfg, sampler, metadata, roi, workdir, engine):
    from subtitles_extractor.application.dtos.extract_subtitles_dto import (
        ExtractSubtitlesRequest,
        SubtitleBuilderConfig,
    )
    from subtitles_extractor.application.services.subtitle_builder import SubtitleBuilder
    from subtitles_extractor.application.use_cases.extract_subtitles import ExtractSubtitlesUseCase

    engine.initialize()
    # khởi động nóng (nạp model, biên dịch kernel) — KHÔNG tính giờ
    import numpy as np

    warm = np.full((max(32, roi.height), max(64, roi.width), 3), 255, np.uint8)
    for _ in range(3):
        engine.infer_batch([warm], [0], [0.0])
    engine.frames = engine.batches = 0
    engine.infer_seconds = 0.0

    reader = type("Reader", (), {"read": lambda self, path: metadata})()
    use_case = ExtractSubtitlesUseCase(
        metadata_reader=reader, frame_sampler=sampler, ocr_engine=engine,
        builder=SubtitleBuilder(SubtitleBuilderConfig()), exporters={},
    )
    request = ExtractSubtitlesRequest(
        video_path=metadata.path, output_path=workdir / f"{name}.srt", roi=roi, sampling=sampling_cfg,
        ocr=ocr_cfg, skip_export=True, raw_ocr_output_path=None,
    )
    started = time.perf_counter()
    response = use_case.execute(request)
    wall = time.perf_counter() - started
    return {
        "scenario": name,
        "spec": spec,
        "wall_s": round(wall, 2),
        "ocr_s": round(engine.infer_seconds, 2),
        "other_s": round(wall - engine.infer_seconds, 2),
        "ocr_frames": engine.frames,
        "ocr_batches": engine.batches,
        "avg_batch": round(engine.frames / max(1, engine.batches), 1),
        "frames_total": response.frames_processed,
        "events": [
            {"start": round(e.start_sec, 3), "end": round(e.end_sec, 3), "text": e.text} for e in response.events
        ],
    }


def _compare(reference: list[dict], other: list[dict]) -> dict[str, object]:
    ref_texts = [e["text"] for e in reference]
    got_texts = [e["text"] for e in other]
    matcher = difflib.SequenceMatcher(None, ref_texts, got_texts, autojunk=False)
    start_diffs: list[float] = []
    end_diffs: list[float] = []
    missing: list[str] = []
    extra: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                start_diffs.append(abs(reference[i1 + k]["start"] - other[j1 + k]["start"]))
                end_diffs.append(abs(reference[i1 + k]["end"] - other[j1 + k]["end"]))
        else:
            missing += ref_texts[i1:i2]
            extra += got_texts[j1:j2]
    return {
        "events": len(other),
        "same_text_ratio": round(matcher.ratio(), 4),
        "missing": missing[:10], "n_missing": len(missing),
        "extra": extra[:10], "n_extra": len(extra),
        "max_start_diff_s": round(max(start_diffs, default=0.0), 3),
        "max_end_diff_s": round(max(end_diffs, default=0.0), 3),
    }


def main() -> int:
    for stream in (sys.stdout, sys.stderr):           # Windows: chuyển hướng ra tệp mặc định cp1252 → vỡ ký tự "▶"
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--roi", required=True, help="x,y,w,h của vùng phụ đề")
    parser.add_argument("--start", type=float, default=300.0)
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--step", type=float, default=0.04)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--decoder", choices=("auto", "pynvvideocodec", "pyav"), default="auto")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--app-settings", action="store_true", help="dùng cài đặt đang lưu trong ứng dụng làm nền")
    parser.add_argument("--out", type=Path, default=Path("bench_ocr_result.json"))
    args = parser.parse_args()

    from subtitles_extractor.infrastructure.video.pyav_metadata_reader import PyAvMetadataReader

    metadata = PyAvMetadataReader().read(args.video)
    roi = _parse_roi(args.roi)

    container = None
    if args.app_settings:
        from subtitles_extractor.composition.bootstrap import _resolve_data_dir, _resolve_user_data_dir
        from subtitles_extractor.composition.container import ApplicationContainer
        from subtitles_extractor.infrastructure.settings.qsettings_repository import QSettingsRepository

        repo = QSettingsRepository(organization="SubtitlesExtractor", application="PaddleOCR")
        container = ApplicationContainer(
            settings_repository=repo, i18n_data_dir=_resolve_data_dir(),
            user_data_dir=_resolve_user_data_dir("PaddleOCR"),
        )
    sampling, ocr, sampler = _base_configs(args, container)

    names = [n.strip() for n in args.scenarios.split(",") if n.strip()]
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        parser.error(f"kịch bản không biết: {unknown}; có: {list(SCENARIOS)}")
    if "current" not in names:
        names.insert(0, "current")                     # luôn cần mốc so sánh

    info = {
        "platform": platform.platform(), "python": sys.version.split()[0],
        "video": args.video.name, "roi": args.roi, "start": args.start, "seconds": args.seconds,
        "decoder": type(sampler).__name__, "device": args.device if container is None else str(ocr.device),
        "batch": ocr.batch_size, "step": sampling.sample_step_sec,
    }
    try:
        import paddle

        info["paddle"] = paddle.__version__
        info["gpu"] = paddle.device.cuda.get_device_name(0) if paddle.device.is_compiled_with_cuda() else "none"
    except Exception as exc:  # noqa: BLE001 - chỉ để ghi thông tin
        info["paddle"] = f"lỗi: {exc}"
    print(json.dumps(info, ensure_ascii=False, indent=2))

    results: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            print(f"\n▶ {name} {SCENARIOS[name]} …", flush=True)
            try:
                results.append(_run_scenario(name, SCENARIOS[name], args, sampling, ocr, sampler, metadata, roi, Path(tmp)))
            except Exception as exc:  # noqa: BLE001 - một kịch bản lỗi không làm mất các kịch bản khác
                print(f"   LỖI: {type(exc).__name__}: {exc}")
                results.append({"scenario": name, "error": f"{type(exc).__name__}: {exc}"})

    reference = next((r for r in results if r["scenario"] == "current" and "events" in r), None)
    print("\n" + "=" * 100)
    print(f"{'kịch bản':<16}{'tổng s':>8}{'OCR s':>8}{'khác s':>8}{'khung OCR':>11}{'lô':>6}{'TB lô':>7}"
          f"{'câu':>6}{'giống':>8}{'thiếu':>6}{'thừa':>6}{'Δstart':>8}{'Δend':>7}")
    for item in results:
        if "error" in item:
            print(f"{item['scenario']:<16}  LỖI: {item['error']}")
            continue
        comparison = _compare(reference["events"], item["events"]) if reference else {}
        item["vs_current"] = comparison
        print(f"{item['scenario']:<16}{item['wall_s']:>8}{item['ocr_s']:>8}{item['other_s']:>8}{item['ocr_frames']:>11}"
              f"{item['ocr_batches']:>6}{item['avg_batch']:>7}{comparison.get('events', len(item['events'])):>6}"
              f"{comparison.get('same_text_ratio', 1):>8}{comparison.get('n_missing', 0):>6}{comparison.get('n_extra', 0):>6}"
              f"{comparison.get('max_start_diff_s', 0):>8}{comparison.get('max_end_diff_s', 0):>7}")
    print("=" * 100)
    print("giống = tỉ lệ chuỗi câu giống kịch bản 'current'; thiếu/thừa = số câu khác biệt; Δ = lệch mốc thời gian lớn nhất (giây).")

    args.out.write_text(json.dumps({"info": info, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nĐã ghi {args.out.resolve()} — hãy gửi file này lại.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
