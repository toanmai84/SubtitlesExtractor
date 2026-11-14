"""[v3.23.714] Hồ sơ lượt chạy trong tệp OCR thô + hộp phình ở khung liền nhau + bước lấy mẫu của khâu dựng.

Lượt chạy thật của chủ dự án (3.23.711 và 3.23.712, phim nối 86 tập) lệch mốc 4 câu so với lượt dựng lại từ chính tệp
OCR thô của nó; nguyên nhân là cài đặt hậu xử lý khác mặc định mà tệp không ghi lại. Với cài đặt đó, khung có hộp chữ
phình bị lọc căn lề loại: "喜" kết thúc sớm 0,16 s, "天驹山" kết thúc sớm 0,28 s.
"""

from __future__ import annotations

import enum
import gzip
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    ExtractSubtitlesRequest,
    SubtitleBuilderConfig,
)
from subtitles_extractor.application.services.run_provenance import (
    collect_run_info,
    frame_statistics,
    summarize_result,
    to_jsonable,
)
from subtitles_extractor.application.services.subtitle_builder import SubtitleBuilder
from subtitles_extractor.application.services.subtitle_pipeline.read_fusion import tighten_inflated_boxes
from subtitles_extractor.application.use_cases.extract_subtitles import ExtractSubtitlesUseCase
from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult, OcrTextBox
from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.ports.frame_sampler_port import FrameSamplingConfig, SampledFrame
from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.roi import Roi, TextAlignment
from subtitles_extractor.infrastructure.serializers.raw_ocr_serializer import (
    RawOcrMeta,
    load_raw_ocr,
    save_raw_ocr,
)


def _read(ts: float, text: str, box: tuple[int, int, int, int], conf: float = 1.0, extra: str = "") -> OcrFrameResult:
    x0, y0, x1, y1 = box
    boxes = [OcrTextBox(text=text, confidence=Confidence(conf), polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])]
    if extra:
        boxes.insert(0, OcrTextBox(text=extra, confidence=Confidence(0.41), polygon=[(106, 52), (298, 52), (298, 140), (106, 140)]))
    return OcrFrameResult(frame_index=round(ts * 25), timestamp_sec=ts, text_boxes=boxes)


def _boxes(frames: list[OcrFrameResult]) -> list[tuple[int, int]]:
    return [(f.text_boxes[-1].polygon[0][1], f.text_boxes[-1].polygon[2][1]) for f in frames]


# ── Hộp phình ở khung liền nhau (không cùng chuỗi điểm ảnh) ─────────────────────────────────────────────────────────


def test_inflated_height_takes_the_tight_box_of_adjacent_same_text() -> None:
    """"喜" 369,16–369,36 s của lượt chạy thật: ba khung giữa có hộp cao 143 / 133 px thay vì 76–79."""
    frames = [
        _read(369.16, "喜", (322, 58, 397, 137)), _read(369.20, "喜", (321, 60, 396, 138)),
        _read(369.24, "喜", (319, -3, 399, 140), 0.99), _read(369.28, "喜", (316, 9, 401, 142)),
        _read(369.32, "喜", (316, 9, 401, 142)), _read(369.36, "喜", (320, 59, 396, 137)),
    ]
    fixed_frames, fixed = tighten_inflated_boxes(frames)
    assert fixed == 3
    assert all(bottom - top <= 80 for top, bottom in _boxes(fixed_frames))
    assert [f.timestamp_sec for f in fixed_frames] == [f.timestamp_sec for f in frames]


def test_inflated_width_is_tightened_too() -> None:
    """"天驹山" 7247,96–7248,20 s: hộp rộng 351 px (dính vật sáng bên trái) thay vì 185."""
    frames = [
        _read(7247.96, "天驹山", (103, 49, 454, 144), 0.99), _read(7248.00, "天驹山", (103, 49, 454, 144), 0.99),
        _read(7248.04, "天驹山", (102, 53, 455, 144), 0.84), _read(7248.16, "天驹山", (263, 77, 448, 141)),
        _read(7248.20, "天驹山", (263, 77, 448, 141)),
    ]
    fixed_frames, fixed = tighten_inflated_boxes(frames)
    assert fixed == 3
    assert {f.text_boxes[0].polygon[0][0] for f in fixed_frames} == {263}


def test_tightening_never_crosses_text_gap_or_multi_box_frames() -> None:
    tight, big = (320, 60, 396, 136), (316, -5, 402, 144)
    other_text = [_read(1.00, "你", tight), _read(1.04, "他", big)]                      # chữ khác: không đụng
    far_apart = [_read(2.00, "你", tight), _read(2.40, "你", big)]                       # cách 0,4 s: lượt hiện khác
    weak = [_read(3.00, "你", tight), _read(3.04, "你", big, conf=0.60)]                 # đọc yếu: không tin
    multi = [_read(4.00, "你", tight), _read(4.04, "你", big, extra="JA")]               # hai hộp: không đụng
    for case in (other_text, far_apart, weak, multi):
        fixed_frames, fixed = tighten_inflated_boxes(case)
        assert fixed == 0 and _boxes(fixed_frames) == _boxes(case)
    assert tighten_inflated_boxes([]) == ([], 0)


def test_builder_keeps_full_span_with_tight_merge_gap() -> None:
    """Cấu hình giống máy chủ dự án (khoảng gộp 0,1 s): câu không còn bị cắt đuôi vì ba khung hộp phình.

    Bộ lọc hình dạng đo cỡ chữ từ chính phim nên cần một đoạn phim đủ dài (8 câu thường) đứng trước.
    """
    frames: list[OcrFrameResult] = []
    clock = 0.0

    def push(text: str, box: tuple[int, int, int, int] | None) -> None:
        nonlocal clock
        clock = round(clock + 0.04, 2)
        frames.append(
            OcrFrameResult(frame_index=len(frames), timestamp_sec=clock, text_boxes=[]) if box is None
            else _read(clock, text, box)
        )

    for line in ("修炼一途的", "自炼气始吧", "经筑基之后", "寻求飞升啊", "检测到宿主", "即将面临了", "历史至强雷", "四灵天雷劫"):
        for _ in range(30):
            push(line, (222, 62, 498, 136))
        for _ in range(8):
            push("", None)
    start = round(clock + 0.04, 2)
    for _ in range(10):
        push("喜", (322, 60, 397, 136))
    for box in ((319, -3, 399, 140), (316, 9, 401, 142), (316, 9, 401, 142)):
        push("喜", box)
    push("喜", (320, 59, 396, 137))
    end = round(clock + 0.04, 2)
    for _ in range(8):
        push("", None)
    config = SubtitleBuilderConfig(sample_step_sec=0.04, merge_gap_sec=0.1, min_duration_sec=0.15)
    events = SubtitleBuilder(config).build(frames, roi=Roi(x=0, y=774, width=720, height=136))
    last = events[-1]
    assert last.text == "喜" and len(events) == 9
    # Mã cũ: câu kết thúc ở khung hộp phình đầu tiên (sớm 0,11 s, còn 10 khung).
    assert last.frame_count == 14 and abs(last.start_sec - start) < 0.011 and last.end_sec >= end - 0.011


# ── Hồ sơ lượt chạy ───────────────────────────────────────────────────────────────────────────────────────────────


class _Mode(enum.Enum):
    FAST = "fast"


@dataclass(frozen=True)
class _Cfg:
    path: Path
    mode: _Mode
    extra: dict


def test_to_jsonable_masks_secrets_and_flattens_types() -> None:
    data = to_jsonable(_Cfg(Path("a/b.mp4"), _Mode.FAST, {"api_key": "sk-123", "gemini_api_keys": ["k1"], "max_tokens": 9,
                                                         "nan": float("nan"), "pair": (1, 2)}))
    assert data == {"path": str(Path("a/b.mp4")), "mode": "FAST",
                    "extra": {"api_key": "***", "gemini_api_keys": "***", "max_tokens": 9, "nan": None, "pair": [1, 2]}}
    json.dumps(data)


def test_run_info_records_everything_that_decides_the_result(tmp_path: Path) -> None:
    video = tmp_path / "phim.mkv"
    video.write_bytes(b"0" * 10)
    meta = VideoMetadata(path=video, width=720, height=1280, fps=25.0, total_frames=100, duration_sec=4.0, codec="hevc")
    request = ExtractSubtitlesRequest(
        video_path=video, output_path=tmp_path / "o.srt", roi=Roi(x=0, y=774, width=720, height=136, alignment=TextAlignment.CENTER),
        sampling=FrameSamplingConfig(sample_step_sec=0.03, white_text_gate=True),
        ocr=OcrEngineConfig(detection_model_name="det", extra_kwargs={"api_key": "bí mật"}),
    )
    builder = SubtitleBuilderConfig(merge_gap_sec=0.1, alignment_center_tolerance_ratio=0.05)
    info = collect_run_info(
        run_id="abc", app_version="3.23.714", request=request, metadata=meta, builder_config=builder,
        frame_sampler=object(), ocr_engine=object(), ocr_elapsed_sec=1.23456,
        frames=[OcrFrameResult(0, 0.0, []), _read(0.04, "你", (1, 2, 3, 4)),
                OcrFrameResult(2, 0.08, [], continues_previous=True)],
    )
    text = json.dumps(info, ensure_ascii=False)
    assert "bí mật" not in text and "collect_error" not in info
    assert info["run_id"] == "abc" and info["app_version"] == "3.23.714" and info["ocr_elapsed_sec"] == 1.235
    assert info["builder"]["merge_gap_sec"] == 0.1 and info["builder"]["alignment_center_tolerance_ratio"] == 0.05
    assert info["sampling"]["white_text_gate"] is True and info["sampling"]["sample_step_sec"] == 0.03
    assert info["ocr"]["detection_model_name"] == "det" and info["ocr"]["preprocess"]["border_thickness_px"] == 8
    assert info["roi"] == {"x": 0, "y": 774, "width": 720, "height": 136, "alignment": "CENTER", "orientation": "HORIZONTAL"}
    assert info["video"]["name"] == "phim.mkv" and info["video"]["size_bytes"] == 10 and info["video"]["codec"] == "hevc"
    assert info["frames"] == {"total": 3, "with_text": 1, "empty": 2, "same_pixels_as_previous": 1}
    assert info["environment"]["python"] and "packages" in info["environment"]


def test_run_info_survives_broken_inputs() -> None:
    info = collect_run_info(run_id="x", app_version="v", request=object(), metadata=None, builder_config=None)
    assert info["run_id"] == "x" and "collect_error" in info
    assert frame_statistics([]) == {"total": 0, "with_text": 0, "empty": 0, "same_pixels_as_previous": 0}


def test_raw_file_roundtrips_run_and_result_and_reads_old_files(tmp_path: Path) -> None:
    frames = [_read(0.04, "你", (1, 2, 3, 4))]
    meta = RawOcrMeta(
        video_name="v.mp4", video_duration_sec=1.0, frame_count=1, sample_step_sec=0.03, detection_model="d",
        recognition_model="r", score_threshold=0.5, saved_at="t", run={"run_id": "abc", "builder": {"merge_gap_sec": 0.1}},
        result=summarize_result([], [{"line": 1, "outcome": "confirmed"}], build_elapsed_sec=0.5),
    )
    path = tmp_path / "a.seraw.json.gz"
    save_raw_ocr(frames, path, meta)
    _, loaded = load_raw_ocr(path)
    assert loaded.run == {"run_id": "abc", "builder": {"merge_gap_sec": 0.1}}
    assert loaded.result["verify"] == [{"line": 1, "outcome": "confirmed"}] and loaded.result["build_elapsed_sec"] == 0.5
    payload = json.loads(gzip.open(path, "rb").read().decode("utf-8"))
    payload["meta"].pop("run")
    payload["meta"].pop("result")
    old = tmp_path / "old.seraw.json"
    old.write_text(json.dumps(payload), encoding="utf-8")
    _, old_meta = load_raw_ocr(old)
    assert old_meta.run is None and old_meta.result is None


# ── Tích hợp: lượt trích xuất ghi hồ sơ + dựng bằng đúng bước lấy mẫu của lượt ───────────────────────────────────────


def _use_case(tmp_path: Path, settings_step: float) -> tuple[ExtractSubtitlesUseCase, VideoMetadata]:
    meta = VideoMetadata(path=tmp_path / "v.mp4", width=200, height=50, fps=25.0, total_frames=40, duration_sec=1.6)
    meta.path.touch()
    reader = MagicMock()
    reader.read.return_value = meta
    sampler = MagicMock()

    def _iter(*_a: object, **_k: object) -> Iterator[SampledFrame]:
        for index in range(30):
            yield SampledFrame(frame_index=index, timestamp_sec=round(index * 0.04, 2),
                               image_rgb=np.zeros((50, 200, 3), dtype=np.uint8))

    sampler.iter_frames.side_effect = _iter
    ocr = MagicMock()
    ocr.is_initialized = True
    ocr.infer_batch.side_effect = lambda images_rgb, frame_indices, timestamps_sec: [
        OcrFrameResult(
            frame_index=i, timestamp_sec=t,
            text_boxes=[] if not 5 <= i < 20 else [
                OcrTextBox(text="你好", confidence=Confidence(0.99), polygon=[(60, 10), (140, 10), (140, 40), (60, 40)]),
            ],
        )
        for i, t in zip(frame_indices, timestamps_sec, strict=True)
    ]
    use_case = ExtractSubtitlesUseCase(
        metadata_reader=reader, frame_sampler=sampler, ocr_engine=ocr,
        builder=SubtitleBuilder(SubtitleBuilderConfig(sample_step_sec=settings_step, merge_gap_sec=0.1)), exporters={},
    )
    return use_case, meta


def test_extraction_writes_run_and_result_into_the_raw_file(tmp_path: Path) -> None:
    use_case, meta = _use_case(tmp_path, settings_step=0.12)
    raw_path = tmp_path / "v.seraw.json.gz"
    request = ExtractSubtitlesRequest(
        video_path=meta.path, output_path=tmp_path / "v.srt", sampling=FrameSamplingConfig(sample_step_sec=0.04),
        ocr=OcrEngineConfig(batch_size=8), skip_export=True, raw_ocr_output_path=raw_path, verify_suspect_lines=False,
    )
    response = use_case.execute(request)
    frames, loaded = load_raw_ocr(raw_path)
    assert len(frames) == 30 and loaded.run is not None and loaded.result is not None
    assert loaded.run["builder"]["merge_gap_sec"] == 0.1
    assert loaded.run["builder"]["sample_step_sec"] == 0.04        # bước của LƯỢT CHẠY, không phải bước trong Cài đặt
    assert loaded.run["sampling"]["sample_step_sec"] == 0.04 and loaded.run["options"]["verify_suspect_lines"] is False
    assert loaded.run["frames"]["total"] == 30 and loaded.run["environment"]["ocr_engine"] == "MagicMock"
    assert [(line["t"], line["s"], line["e"]) for line in loaded.result["events"]] == [
        (event.text, round(event.start_sec, 3), round(event.end_sec, 3)) for event in response.events
    ]
    assert loaded.result["event_count"] == len(response.events) == 1 and loaded.result["verify"] == []


def test_builder_uses_the_step_of_the_run_not_of_the_settings(tmp_path: Path) -> None:
    use_case, meta = _use_case(tmp_path, settings_step=0.12)
    request = ExtractSubtitlesRequest(
        video_path=meta.path, output_path=tmp_path / "v.srt", sampling=FrameSamplingConfig(sample_step_sec=0.03),
    )
    assert use_case._builder_for(request)._config.sample_step_sec == 0.03
    assert use_case._builder_for(request)._config.merge_gap_sec == 0.1
    same = ExtractSubtitlesRequest(
        video_path=meta.path, output_path=tmp_path / "v.srt", sampling=FrameSamplingConfig(sample_step_sec=0.12),
    )
    assert use_case._builder_for(same) is use_case._subtitle_builder


def test_translation_diagnostics_carry_masked_settings() -> None:
    from subtitles_extractor.application.services.translation_diagnostics import build_diagnostics_bundle

    bundle = build_diagnostics_bundle(
        app_version="3.23.714", exported_at="t", video_path="", source_events=[], translated_events=[], session=None,
        settings={"translation": {"default_batch_size": 40, "api_key": "bí mật"}},
    )
    assert bundle["settings_at_export"] == {"translation": {"default_batch_size": 40, "api_key": "***"}}
    bare = build_diagnostics_bundle(
        app_version="v", exported_at="t", video_path="", source_events=[], translated_events=[], session=None,
    )
    assert bare["settings_at_export"] == {}
