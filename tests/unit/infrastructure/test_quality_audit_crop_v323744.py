"""[v3.23.744] Gói kiểm định đọc lại ẢNH CẮT SÁT hộp phụ đề, không phải cả khung video ROI.

Bộ chuyển đổi chạy bộ dò chữ trên ảnh nhận được rồi nối MỌI hộp → chữ trong cảnh lọt vào bản đọc kiểm định (豪门Z外:
"喝酒去WARMHOME", "那不就是一个报表吗BOR"; phim 86 tập: "封印" / "送给南宫梦" thay câu phụ đề). Đo 457 câu hai phim:
PaddleOCR-VL cả khung đúng 8/57 câu bất đồng → cắt sát 50/57.
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.quality.quality_auditor import (
    VlQualityAuditor,
    subtitle_crop_rect,
)


def _box(text: str, x0: int, y0: int, x1: int, y1: int) -> OcrTextBox:
    return OcrTextBox(text=text, confidence=Confidence(0.99), polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def test_scene_text_left_out_of_crop() -> None:
    boxes = [_box("喝酒去", 400, 60, 640, 110), _box("WARMHOME", 700, 10, 1000, 40)]
    assert subtitle_crop_rect(boxes, "喝酒去", 1080, 140) == (384, 44, 656, 126)      # lề 0,25 × 50 + 4 = 16


def test_scene_text_sharing_characters_elsewhere_left_out() -> None:
    # Phim 86 tập: chữ cảnh "封印" (trên cao) trùng chữ câu "修为封印" — chỉ giữ dòng phụ đề.
    boxes = [_box("封印", 100, 5, 180, 35), _box("修为封印", 400, 80, 680, 130)]
    x0, y0, x1, y1 = subtitle_crop_rect(boxes, "修为封印", 1080, 140) or (0, 0, 0, 0)
    assert x0 > 180 and y0 > 35 and x1 >= 680


def test_two_line_subtitle_kept_whole_and_offset_applied() -> None:
    boxes = [_box("师兄你", 300, 20, 600, 60), _box("等等我", 320, 70, 580, 110)]
    rect = subtitle_crop_rect(boxes, "师兄你\n等等我", 1080, 200, offset=(0, 1))
    assert rect == (286, 7, 614, 125)


def test_no_matching_box_means_whole_frame() -> None:
    assert subtitle_crop_rect([_box("福", 10, 10, 60, 60)], "你好", 1080, 140) is None
    assert subtitle_crop_rect([_box("你好", 10, 10, 60, 60)], "OK", 1080, 140) is None


class _ShapeEngine:
    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def infer_batch(self, images, frame_indices, timestamps, config=None):  # noqa: ANN001, ANN201
        self.shapes.extend(image.shape for image in images)
        return [OcrFrameResult(i, t, []) for i, t in zip(frame_indices, timestamps, strict=True)]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="cần ffmpeg")
def test_auditor_feeds_cropped_subtitle_to_recognizer(tmp_path: Path) -> None:
    from tests.unit.infrastructure.test_quality_audit_v323737 import _make_clip

    clip, sidecar = _make_clip(tmp_path)                       # video ROI 320x60
    events = [SubtitleEvent(index=1, text="黑夜", interval=TimeInterval(0.2, 1.6)),
              SubtitleEvent(index=2, text="白天", interval=TimeInterval(2.4, 3.8))]
    frames = [OcrFrameResult(k, k * 0.04, [_box("黑夜", 100, 20, 180, 44), _box("LOGO", 250, 0, 310, 15)])
              for k in range(0, 50)]                           # câu 2 không có hộp chữ thô → cả khung
    engine = _ShapeEngine()
    out = tmp_path / "phim.qa.json.gz"
    VlQualityAuditor(engine).audit(events, clip, sidecar, out, config=OcrEngineConfig(), frames=frames)
    payload = json.loads(gzip.open(out, "rt", encoding="utf-8").read())
    assert payload["input"] == "subtitle_box_crop"
    first, second = payload["lines"]
    x0, y0, x1, y1 = first["box"]
    assert (x1 - x0, y1 - y0) == (80 + 2 * 10, min(60, 44 + 10) - (20 - 10))          # lề 0,25 × 24 + 4 = 10
    assert second["box"] is None
    assert engine.shapes[:3] == [(y1 - y0, x1 - x0, 3)] * 3 and engine.shapes[3:] == [(60, 320, 3)] * 3


def test_use_case_hands_raw_frames_to_auditor(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from subtitles_extractor.application.use_cases.extract_subtitles import (
        ExtractSubtitlesUseCase,
    )

    (tmp_path / "phim.roi.mp4").write_bytes(b"x")
    seen: dict[str, object] = {}

    class _Auditor:
        def audit(self, events, clip, sidecar, out, *, config, progress=None, frames=None):  # noqa: ANN001, ANN201
            seen["frames"] = frames
            return out

    class _Reporter:
        def report(self, *_a: object) -> None: ...
        def is_cancelled(self) -> bool: return False

    use_case = ExtractSubtitlesUseCase.__new__(ExtractSubtitlesUseCase)
    use_case._quality_auditor, use_case._audio_exporter = _Auditor(), None
    frames = [OcrFrameResult(0, 0.0, [_box("字", 0, 0, 9, 9)])]
    use_case._export_qa_data(SimpleNamespace(video_path=tmp_path / "phim.mp4", ocr=OcrEngineConfig()),
                             [SubtitleEvent(index=1, text="字", interval=TimeInterval(0, 1))],
                             tmp_path / "phim.seraw.json.gz", _Reporter(), frames=frames)
    assert seen["frames"] is frames
    source = (Path(__file__).resolve().parents[3] / "src/subtitles_extractor/application/use_cases/"
              "extract_subtitles.py").read_text(encoding="utf-8")
    assert "frames=ocr_frame_results" in source


def test_crop_is_numpy_slice_of_frame() -> None:
    image = np.zeros((60, 320, 3), np.uint8)
    rect = subtitle_crop_rect([_box("你好", 100, 20, 180, 44)], "你好", image.shape[1], image.shape[0])
    assert rect is not None and image[rect[1]:rect[3], rect[0]:rect[2]].shape == (44 + 10 - 10, 100, 3)


def test_scene_text_just_above_is_not_a_second_subtitle_line() -> None:
    # Hộp thật phim 86 tập (câu 151): chữ cảnh "封印" ngay trên dòng "修为封印", lệch trục giữa 162 px.
    boxes = [_box("封印", 149, -3, 247, 56), _box("修为封印", 240, 65, 479, 134)]
    x0, y0, x1, y1 = subtitle_crop_rect(boxes, "修为封印", 720, 140) or (0, 0, 0, 0)
    assert y0 >= 65 - 21 and x0 >= 240 - 21                    # chỉ dòng phụ đề (lề 0,25 × 69 + 4 = 21)


def test_scene_latin_glued_to_subtitle_is_dropped_from_reread() -> None:
    from subtitles_extractor.infrastructure.quality.quality_auditor import (
        strip_scene_latin,
    )

    assert strip_scene_latin("林娇娇这女人 feet dream", "林娇娇这女人") == "林娇娇这女人"       # 豪门Z外 câu 874
    assert strip_scene_latin("201717上签字", "上签字") == "上签字"
    assert strip_scene_latin("腐上签字", "上签字") == "腐上签字"                                  # chữ Hán: giữ để đối chiếu
    assert strip_scene_latin("OK了", "OK了") == "OK了"                                         # câu có Latin: giữ nguyên
