"""[v3.23.737] Gói kiểm định chất lượng: mọi câu đọc lại bằng mô hình thứ hai trên video ROI + tiếng phim.

Chủ dự án (17/11/2026): chạy ứng dụng trên máy có GPU là ra đủ dữ liệu để đối chiếu phụ đề với hardsub — không phải dò
mắt từng dòng.
"""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
from subtitles_extractor.domain.ports.quality_audit_port import qa_paths_for
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.roi import Roi
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.quality.audio_exporter import (
    FfmpegAudioExporter,
    audio_command,
)
from subtitles_extractor.infrastructure.quality.quality_auditor import (
    VlQualityAuditor,
    audit_timestamps,
    line_text,
)

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="cần ffmpeg")


def test_paths_follow_raw_file() -> None:
    qa, audio = qa_paths_for(Path("/x/phim.seraw.json.gz"))
    assert qa == Path("/x/phim.qa.json.gz") and audio == Path("/x/phim.audio.opus")


def test_audit_timestamps_inside_line() -> None:
    assert audit_timestamps(10.0, 12.0) == [10.4, 11.0, 11.6]
    assert audit_timestamps(5.0, 5.0) == [5.0]                  # câu thời lượng 0: một mốc


def test_line_text_orders_boxes_left_to_right() -> None:
    boxes = [OcrTextBox(text="师兄", confidence=Confidence(0.9), polygon=[(300, 0), (400, 0), (400, 9), (300, 9)]),
             OcrTextBox(text="玄昊", confidence=Confidence(1.0), polygon=[(100, 0), (200, 0), (200, 9), (100, 9)])]
    text, confidence = line_text(boxes)
    assert text == "玄昊师兄" and confidence == pytest.approx(0.95)


class _EchoEngine:
    """Engine giả: đọc ra độ sáng trung bình của ảnh (để biết đúng khung) — ghi lại cấu hình được dùng."""

    def __init__(self) -> None:
        self.configs: list[OcrEngineConfig | None] = []
        self.images = 0

    def infer_batch(self, images, frame_indices, timestamps, config=None):  # noqa: ANN001, ANN201
        self.configs.append(config)
        self.images += len(images)
        return [OcrFrameResult(i, t, [OcrTextBox(text=f"{int(img.mean())}", confidence=Confidence(0.9),
                                                 polygon=[(0, 0), (9, 0), (9, 9), (0, 9)])])
                for i, t, img in zip(frame_indices, timestamps, images, strict=True)]


def _make_clip(tmp_path: Path) -> tuple[Path, Path]:
    """Video 4 s: 2 s đen rồi 2 s trắng (25 fps) → video vùng ROI thật bằng bộ xuất của ứng dụng (CPU)."""
    from subtitles_extractor.infrastructure.video.roi_clip_exporter import (
        FfmpegRoiClipExporter,
    )

    assert FFMPEG is not None
    video = tmp_path / "phim.mp4"
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=black:size=320x240:rate=25:duration=2", "-f", "lavfi", "-i",
                    "color=c=white:size=320x240:rate=25:duration=2", "-filter_complex", "[0:v][1:v]concat=n=2:v=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True, capture_output=True)
    clip = tmp_path / "phim.roi.mp4"
    result = FfmpegRoiClipExporter(lambda: FFMPEG).export(video, Roi(0, 160, 320, 60), clip, prefer_gpu=False)
    assert result is not None
    return clip, result.sidecar_path


@needs_ffmpeg
def test_auditor_rereads_every_line_from_roi_clip(tmp_path: Path) -> None:
    clip, sidecar = _make_clip(tmp_path)
    # [v3.23.744] Câu có chữ số → bản đọc (số độ sáng của engine giả) không bị bỏ như chữ Latin / số lạc từ cảnh.
    events = [SubtitleEvent(index=1, text="黑1", interval=TimeInterval(0.2, 1.6)),
              SubtitleEvent(index=2, text="白2", interval=TimeInterval(2.4, 3.8))]
    engine = _EchoEngine()
    out = tmp_path / "phim.qa.json.gz"
    config = OcrEngineConfig(recognition_model_name="PaddleOCR-VL-1.6-0.9B")
    progress: list[float] = []
    assert VlQualityAuditor(engine).audit(events, clip, sidecar, out, config=config,
                                          progress=lambda f: progress.append(f) or True) == out
    payload = json.loads(gzip.open(out, "rt", encoding="utf-8").read())
    assert payload["schema"] == 1 and payload["model"] == "PaddleOCR-VL-1.6-0.9B" and payload["frames_per_line"] == 3
    first, second = payload["lines"]
    assert first["t"] == "黑1" and len(first["ts"]) == 3 and all(int(v) < 40 for v in first["vl"])      # khung đen
    assert second["t"] == "白2" and all(int(v) > 200 for v in second["vl"])                             # khung trắng
    assert engine.images == 6 and all(c is config for c in engine.configs)


@needs_ffmpeg
def test_auditor_cancel_and_missing_clip(tmp_path: Path) -> None:
    clip, sidecar = _make_clip(tmp_path)
    events = [SubtitleEvent(index=i, text="字", interval=TimeInterval(i * 0.1, i * 0.1 + 0.1)) for i in range(40)]
    out = tmp_path / "phim.qa.json.gz"
    auditor = VlQualityAuditor(_EchoEngine())
    assert auditor.audit(events, clip, sidecar, out, config=OcrEngineConfig(), progress=lambda f: False) is None
    assert not out.exists()
    assert auditor.audit(events, tmp_path / "x.roi.mp4", sidecar, out, config=OcrEngineConfig()) is None


def test_audio_command_is_mono_16k() -> None:
    command = audio_command("ffmpeg", Path("in.mp4"), Path("out.opus"), "libopus")
    assert command[command.index("-ac") + 1] == "1" and command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-c:a") + 1] == "libopus" and command[command.index("-map") + 1] == "0:a:0"


@needs_ffmpeg
def test_audio_exported_small_mono(tmp_path: Path) -> None:
    import av

    assert FFMPEG is not None
    video = tmp_path / "phim.mp4"
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=black:size=160x120:rate=25:duration=3", "-f", "lavfi", "-i",
                    "sine=frequency=440:sample_rate=48000:duration=3", "-ac", "2", "-c:v", "libx264",
                    "-c:a", "aac", "-shortest", str(video)], check=True, capture_output=True)
    out = FfmpegAudioExporter(lambda: FFMPEG).export(video, tmp_path / "phim.audio.opus")
    assert out is not None and out.is_file()
    with av.open(str(out)) as container:
        stream = container.streams.audio[0]
        assert stream.codec_context.channels == 1
        assert float(container.duration or 0) / 1e6 == pytest.approx(3.0, abs=0.2)
    silent = tmp_path / "cam.mp4"
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=black:size=160x120:rate=25:duration=1", "-c:v", "libx264", str(silent)],
                   check=True, capture_output=True)
    assert FfmpegAudioExporter(lambda: FFMPEG).export(silent, tmp_path / "cam.audio.opus") is None


# ── nối vào lượt trích xuất ──────────────────────────────────────────────────────────────────────────────────────────
class _Reporter:
    def report(self, *_a: object) -> None: ...
    def is_cancelled(self) -> bool: return False


def test_use_case_exports_audio_and_rereads_with_vl_on_raw_images(tmp_path: Path) -> None:
    from subtitles_extractor.application.use_cases.extract_subtitles import (
        ExtractSubtitlesUseCase,
    )
    from subtitles_extractor.domain.ports.ocr_engine_port import PreprocessConfig

    raw = tmp_path / "phim.seraw.json.gz"
    (tmp_path / "phim.roi.mp4").write_bytes(b"x")
    calls: dict[str, object] = {}

    class _Auditor:
        def audit(self, events, clip, sidecar, out, *, config, progress=None, frames=None):  # noqa: ANN001, ANN201
            calls.update(clip=clip, sidecar=sidecar, out=out, config=config, n=len(events))
            return out

    class _Audio:
        def export(self, video, out):  # noqa: ANN001, ANN201
            calls.update(video=video, audio=out)
            return out

    use_case = ExtractSubtitlesUseCase.__new__(ExtractSubtitlesUseCase)
    use_case._quality_auditor, use_case._audio_exporter = _Auditor(), _Audio()
    request = SimpleNamespace(video_path=tmp_path / "phim.mp4",
                              ocr=OcrEngineConfig(recognition_model_name="PP-OCRv6_small_rec", device="gpu"))
    events = [SubtitleEvent(index=1, text="字", interval=TimeInterval(0, 1))]
    use_case._export_qa_data(request, events, raw, _Reporter())
    assert calls["audio"] == tmp_path / "phim.audio.opus" and calls["out"] == tmp_path / "phim.qa.json.gz"
    assert calls["clip"] == tmp_path / "phim.roi.mp4" and calls["sidecar"] == tmp_path / "phim.roi.json"
    config = calls["config"]
    assert isinstance(config, OcrEngineConfig)
    assert config.recognition_model_name.startswith("PaddleOCR-VL") and config.device == "gpu"   # GPU của lượt chạy
    assert config.preprocess == PreprocessConfig(upscale_small_text=False, add_white_border=False, apply_sharpen=False,
                                                 apply_contrast_boost=False, apply_clahe=False)


def test_raw_export_dialog_yes_turns_on_qa_package() -> None:
    page = Path(__file__).resolve().parents[3] / "src/subtitles_extractor/presentation/pages/extract_page.py"
    assert "export_qa_data=export_roi_clip" in page.read_text(encoding="utf-8")
    import json as _json

    data = Path(__file__).resolve().parents[3] / "src/subtitles_extractor/data"
    for lang in ("vi", "en"):
        body = _json.loads((data / f"strings_{lang}.json").read_text(encoding="utf-8"))["extract"]["dlg_roi_clip_body"]
        assert ".qa.json.gz" in body and ".audio.opus" in body


def test_echo_engine_sanity() -> None:
    engine = _EchoEngine()
    out = engine.infer_batch([np.full((4, 4, 3), 255, np.uint8)], [0], [0.0])
    assert out[0].text_boxes[0].text == "255"


@needs_ffmpeg
def test_engine_without_run_config_is_not_used(tmp_path: Path) -> None:
    # Không truyền được cấu hình VL → không phải "người đọc thứ hai" → không ghi gói giả.
    class _Legacy:
        def infer_batch(self, images, frame_indices, timestamps):  # noqa: ANN001, ANN201
            raise AssertionError("không được gọi")

    clip, sidecar = _make_clip(tmp_path)
    events = [SubtitleEvent(index=1, text="字", interval=TimeInterval(0.2, 1.0))]
    out = tmp_path / "phim.qa.json.gz"
    assert VlQualityAuditor(_Legacy()).audit(events, clip, sidecar, out, config=OcrEngineConfig()) is None
    assert not out.exists()
