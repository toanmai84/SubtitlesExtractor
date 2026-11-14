"""[v3.23.712] Gộp lần đọc theo hình học hộp chữ + lọc rác không chữ Hán lệch tâm.

Lượt chạy thật phim nối 86 tập bằng bản 3.23.711 (vùng chữ rộng cả khung 720 px, model small): 3656/3663 câu khớp.
Còn: rác "000" (dòng 3315), "其实我直都在" (rơi "一" GIỮA câu), bốn câu đọc lẫn chữ trên cùng điểm ảnh (癞/痴,
臜/攒, 吗/鸣, 狠/狼), mất câu "打" và cắt mốc "不", "爽" vì hộp chữ phình. Phát lại trên đúng dữ liệu thô đó:
3658 khớp, 0 câu thừa, 0 câu lệch mốc (còn lại: "一" đầu câu do vòng tự soát sửa lúc trích xuất, "腌臜", "诶").
"""

from __future__ import annotations

from types import SimpleNamespace

from subtitles_extractor.application.services.subtitle_pipeline.geometry_check import drop_small_non_han_events
from subtitles_extractor.application.services.subtitle_pipeline.read_fusion import (
    fuse_same_pixel_reads,
    measure_char_pitch,
    measure_text_height,
)
from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult, OcrTextBox
from subtitles_extractor.domain.value_objects.confidence import Confidence

CENTER, PITCH, HEIGHT = 359.0, 55.3, 67.0


def _box(text: str, conf: float, left: float, right: float, top: float = 66, bottom: float = 133) -> OcrTextBox:
    return OcrTextBox(text=text, confidence=Confidence(conf),
                      polygon=[(int(left), int(top)), (int(right), int(top)), (int(right), int(bottom)), (int(left), int(bottom))])


def _line(text: str, conf: float = 1.0, **kw: float) -> list[OcrTextBox]:
    width = PITCH * (len(text) + 0.35)
    return [_box(text, conf, kw.pop("left", CENTER - width / 2), kw.pop("right", CENTER + width / 2), **kw)]


def _chain(reads: list[tuple[list[OcrTextBox], int]], start: float = 0.0) -> list[OcrFrameResult]:
    frames: list[OcrFrameResult] = []
    for boxes, count in reads:
        for _ in range(count):
            frames.append(OcrFrameResult(len(frames), start + 0.04 * len(frames), list(boxes), continues_previous=bool(frames)))
    return frames


def _texts(frames: list[OcrFrameResult]) -> list[str]:
    return ["".join(box.text for box in frame.text_boxes) for frame in frames]


def _fuse(frames: list[OcrFrameResult]) -> tuple[list[OcrFrameResult], int, set[str]]:
    return fuse_same_pixel_reads(frames, CENTER, PITCH, HEIGHT)


def test_yi_dropped_mid_line_is_restored_by_box_width() -> None:
    """2119 s: "其实我一直都在" 8 khung, rồi hai hộp "其实我" + "直都在" 20 khung — cùng bề rộng 7 chữ, cùng tâm."""
    full = [_box("其实我一直都在", 1.0, 147, 571)]
    split = [_box("其实我", 1.0, 148, 350), _box("直都在", 1.0, 373, 567)]
    fused, _, _ = _fuse(_chain([(full, 2), (split, 20), (full, 6)]))
    assert set(_texts(fused)) == {"其实我一直都在"}                     # bản 3.23.711: bản 6 chữ thắng theo số khung


def test_alternating_substitutions_on_same_pixels_become_one_text() -> None:
    cases = [("这不是癞蛤蟆吗", 0.99, "这不是痴蛤蟆吗", 0.93), ("下山探风吗", 1.0, "下山探风鸣", 0.96),
             ("把你狠狠踩在脚下", 1.0, "把你狠狼踩在脚下", 0.97)]
    for good, good_conf, bad, bad_conf in cases:
        frames = _chain([(_line(good, good_conf), 13), (_line(bad, bad_conf), 10), (_line(good, good_conf), 10),
                         (_line(bad, bad_conf), 12)])                 # bản sai có khi nhiều khung hơn
        fused, _, conflicts = _fuse(frames)
        assert set(_texts(fused)) == {good} and conflicts == {good}


def test_one_way_substitution_is_not_fused() -> None:
    """A → B không quay lại A: có thể là hai câu thật khác một chữ mà bộ lọc khung không thấy đổi."""
    frames = _chain([(_line("任务一"), 12), (_line("任务二"), 12)])
    fused, fixed, conflicts = _fuse(frames)
    assert fixed == 0 and _texts(fused) == _texts(frames)
    assert conflicts == {"任务一", "任务二"}        # [v3.23.713] không gộp nhưng gắn cờ để vòng soát đọc lại phân xử


def test_inflated_box_takes_the_tight_box_of_the_same_read() -> None:
    """3087 s: "打" đọc đúng 13 khung nhưng 10 khung có hộp cao 143 px (dính nền sáng) → bị lọc hình dạng, mất câu."""
    tight = [_box("打", 1.0, 318, 399, 59, 139)]
    inflated = [_box("打", 1.0, 320, 399, -4, 139)]
    fused, fixed, _ = _fuse(_chain([(tight, 2), (inflated, 10), (tight, 1)]))
    assert fixed == 10 and {frame.text_boxes[0].polygon[0][1] for frame in fused} == {59}


def test_high_confidence_scene_text_is_not_spread_over_empty_frames() -> None:
    """Rác tin cậy cao đọc được một lần ("4" ở góc, chữ nhỏ) không được lấp sang các khung rỗng cùng chuỗi."""
    garbage = [_box("4", 0.93, 610, 640, 20, 50)]
    frames = _chain([([], 6), (garbage, 2), ([], 9)])
    fused, fixed, _ = _fuse(frames)
    assert fixed == 0 and _texts(fused) == _texts(frames)
    real = _chain([([], 2), (_line("你"), 1), ([], 10), (_line("你"), 2)])
    assert set(_texts(_fuse(real)[0])) == {"你"}                       # câu thật đúng tâm, đúng cỡ: vẫn được lấp


def test_film_metrics_need_enough_frames() -> None:
    few = _chain([(_line("怎么回事"), 20)])
    assert measure_char_pitch(few) is None and measure_text_height(few) is None
    many = _chain([(_line("怎么回事"), 30), (_line("你到底想要怎么样"), 30)])
    assert abs(measure_char_pitch(many) - PITCH) < 0.5 and measure_text_height(many) == 67


class TestOffCentreNonHan:
    @staticmethod
    def _film(extra: str, left: float, right: float, top: float, bottom: float) -> tuple[list, list]:
        events, frames, clock = [], [], 0.0
        for text, box in [("怎么回事", (238, 480, 66, 133))] * 40 + [(extra, (left, right, top, bottom))]:
            events.append(SimpleNamespace(text=text, start_sec=clock, end_sec=clock + 0.5, ocr_notes=()))
            x0, x1, y0, y1 = box
            for step in range(6):
                frames.append(SimpleNamespace(timestamp_sec=clock + 0.08 * step, text_boxes=[
                    SimpleNamespace(text=text, polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])]))
            clock += 1.0
        return events, frames

    def test_off_centre_digits_are_dropped(self) -> None:
        """Dòng 3315 lượt chạy 3.23.711: "000" tin cậy 0,98, hộp 379–453 × 90–144 (cao 0,79 lần, tâm lệch 0,84 lần)."""
        events, frames = self._film("000", 379, 453, 90, 144)
        kept, dropped = drop_small_non_han_events(events, frames)
        assert dropped == 1 and all(event.text != "000" for event in kept)     # bản 3.23.711: giữ lại

    def test_centred_full_size_latin_and_off_centre_han_are_kept(self) -> None:
        events, frames = self._film("OK", 320, 398, 68, 132)
        assert drop_small_non_han_events(events, frames)[1] == 0
        events, frames = self._film("来人", 276, 398, 62, 136)                  # câu ĐÚNG lệch nhất phim: 0,34 lần
        assert drop_small_non_han_events(events, frames)[1] == 0
        events, frames = self._film("小川", 456, 523, 66, 133)                  # chữ Hán lệch xa: không thuộc luật này
        assert drop_small_non_han_events(events, frames)[1] == 0
