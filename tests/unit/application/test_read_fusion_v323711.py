"""[v3.23.711] Gộp các lần đọc của cùng một câu (cùng điểm ảnh chữ / hơn kém chữ "一" / cặp chữ dễ lẫn hình).

Lượt chạy thật phim nối 86 tập bằng bản 3.23.710: 3660/3663 câu khớp; còn "一晚上" bị tách ba, "你还想白日宣淫不成"
bị tách đôi (白/自), "你" 1379 s mất, và ba câu một chữ bị cắt mốc vì lần đọc lại trả rỗng. Phát lại trên đúng dữ
liệu OCR thô đó: 3662/3663 (chỉ còn "诶" — mô hình OCR không khoanh được).
"""

from __future__ import annotations

import dataclasses

from subtitles_extractor.application.services.subtitle_pipeline.read_fusion import (
    fuse_same_pixel_reads,
    unify_confusable_variants,
)
from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult, OcrTextBox
from subtitles_extractor.domain.value_objects.confidence import Confidence

CENTER = 322.0
CHAR = 55.0


def _frame(ts: float, text: str = "", conf: float = 1.0, *, left: float | None = None, right: float | None = None,
           cont: bool = False) -> OcrFrameResult:
    boxes = []
    if text:
        width = CHAR * len(text) + 19
        x0 = CENTER - width / 2 if left is None else left
        x1 = CENTER + width / 2 if right is None else right
        boxes = [OcrTextBox(text=text, confidence=Confidence(conf),
                            polygon=[(int(x0), 62), (int(x1), 62), (int(x1), 134), (int(x0), 134)])]
    return OcrFrameResult(frame_index=int(ts * 25), timestamp_sec=ts, text_boxes=boxes, continues_previous=cont)


def _run(specs: list[tuple[str, int, dict]], start: float = 0.0, chained: bool = True) -> list[OcrFrameResult]:
    """``(chữ, số khung, tham số hộp)`` → chuỗi khung liền nhau; ``chained``: mọi khung sau khung đầu cùng điểm ảnh."""
    frames: list[OcrFrameResult] = []
    for text, count, extra in specs:
        for _ in range(count):
            frames.append(_frame(start + 0.04 * len(frames), text, cont=chained and bool(frames), **extra))
    return frames


def _texts(frames: list[OcrFrameResult]) -> list[str]:
    return ["".join(box.text for box in frame.text_boxes) for frame in frames]


class TestFuseSamePixelReads:
    def test_empty_rereads_take_the_text_of_the_same_pixels(self) -> None:
        """"你" 1379 s: ba lần đọc "你", rỗng, "你" trên cùng điểm ảnh — trước đây còn 3 khung có chữ → mất câu."""
        frames = _run([("", 2, {}), ("你", 1, {}), ("", 10, {}), ("你", 2, {}), ("", 3, {})])
        fused, fixed, _ = fuse_same_pixel_reads(frames, CENTER)
        assert _texts(fused) == ["你"] * 18 and fixed == 15

    def test_without_pixel_evidence_nothing_is_filled(self) -> None:
        frames = _run([("你", 1, {}), ("", 10, {}), ("你", 2, {})], chained=False)
        fused, fixed, _ = fuse_same_pixel_reads(frames, CENTER)
        assert fixed == 0 and _texts(fused) == _texts(frames)

    def test_blank_frame_breaks_the_chain(self) -> None:
        frames = _run([("哎", 5, {})]) + [_frame(0.20), _frame(0.24), _frame(0.28)]
        fused, fixed, _ = fuse_same_pixel_reads(frames, CENTER)
        assert fixed == 0 and _texts(fused)[5:] == ["", "", ""]

    def test_low_confidence_read_is_not_spread(self) -> None:
        frames = _run([("", 6, {}), ("a", 1, {"conf": 0.64}), ("", 6, {})])
        assert fuse_same_pixel_reads(frames, CENTER)[1] == 0

    def test_dropped_yi_inside_a_chain_is_restored(self) -> None:
        frames = _run([("一晚上", 13, {}), ("晚上", 11, {"left": 283, "right": 415}), ("一晚上", 11, {})])
        fused, _, _ = fuse_same_pixel_reads(frames, CENTER)
        assert set(_texts(fused)) == {"一晚上"}

    def test_different_texts_in_a_chain_are_left_alone(self) -> None:
        frames = _run([("任务一", 10, {}), ("任务二", 10, {})])
        fused, fixed, _ = fuse_same_pixel_reads(frames, CENTER)
        assert fixed == 0 and _texts(fused) == _texts(frames)


class TestUnifyVariants:
    def test_confusable_pair_is_settled_by_the_film_itself(self) -> None:
        """"白日宣淫" có ở một câu khác của phim; "自日" không có ở đâu — dù bản "自" nhiều khung hơn."""
        line = _run([("你还想白日宣淫不成", 14, {"conf": 0.97}), ("你还想自日宣淫不成", 22, {"conf": 0.96})], chained=False)
        other = _run([("白日宣淫", 20, {})], start=30.0, chained=False)
        distractor = _run([("想自毁灵台", 20, {})], start=60.0, chained=False)
        gap = [_frame(10.0), _frame(50.0)]
        frames = line + [gap[0]] + other + [gap[1]] + distractor
        unified, chosen = unify_confusable_variants(frames, CENTER)
        assert set(_texts(unified)[:36]) == {"你还想白日宣淫不成"} and chosen == {"你还想白日宣淫不成"}

    def test_alternating_reads_collapse_to_one_text(self) -> None:
        good, bad = "我跟秦执师兄清清白白的", "我跟秦执师兄清清自白的"
        frames = _run([(good, 13, {"conf": 0.95}), (bad, 11, {"conf": 0.96}), (good, 10, {"conf": 0.95}),
                       (bad, 11, {"conf": 0.96})], chained=False)
        frames += [_frame(5.0)] + _run([("我也没说你们不清白啊", 20, {})], start=5.04, chained=False)
        unified, _ = unify_confusable_variants(frames, CENTER)
        assert set(_texts(unified)[:45]) == {good}

    def test_spurious_leading_yi_loses_to_the_centred_read(self) -> None:
        """Tập 08: 2/32 khung đọc thừa "一" (hộp phình sang trái dính vật trong cảnh) — bản đúng là bản đúng tâm."""
        text = "小小教训你们一下吧"
        frames = _run([(text, 17, {}), ("一" + text, 1, {"left": 28, "right": 579}), (text, 1, {}),
                       ("一" + text, 1, {"left": 28, "right": 579}), (text, 12, {})], chained=False)
        unified, _ = unify_confusable_variants(frames, CENTER)
        assert set(_texts(unified)) == {text}

    def test_real_leading_yi_wins_even_as_minority(self) -> None:
        """"一百道雷劫一起劈下来": 27 khung thiếu "一" (lệch 24 px), 6 khung đủ chữ trong đó 2 khung đúng tâm."""
        short, full = "百道雷劫一起劈下来", "一百道雷劫一起劈下来"
        off = {"left": 108, "right": 584}                                  # tâm 346
        frames = _run([(short, 27, off), (full, 2, {"left": 104, "right": 584}), (short, 2, off),
                       (full, 2, {"left": 94, "right": 584}), (full, 2, {"left": 62, "right": 584})], chained=False)
        unified, chosen = unify_confusable_variants(frames, CENTER)
        assert set(_texts(unified)) == {full} and chosen == {full}

    def test_blank_between_keeps_two_lines(self) -> None:
        frames = _run([("一晚上", 10, {})], chained=False) + [_frame(0.4)] + _run(
            [("晚上", 10, {"left": 283, "right": 415})], start=0.44, chained=False)
        unified, chosen = unify_confusable_variants(frames, CENTER)
        assert _texts(unified) == _texts(frames) and not chosen

    def test_ordinary_consecutive_lines_are_untouched(self) -> None:
        frames = _run([("任务一", 10, {}), ("任务二", 10, {}), ("你来了", 10, {}), ("我来了", 10, {})], chained=False)
        unified, chosen = unify_confusable_variants(frames, CENTER)
        assert _texts(unified) == _texts(frames) and not chosen

    def test_polygons_of_rewritten_frames_are_kept(self) -> None:
        frames = _run([("明白了吧", 10, {}), ("明自了吧", 3, {"left": 200, "right": 440})], chained=False)
        frames += [_frame(3.0)] + _run([("我明白", 10, {})], start=3.04, chained=False)
        unified, _ = unify_confusable_variants(frames, CENTER)
        assert unified[11].text_boxes[0].text == "明白了吧" and unified[11].text_boxes[0].polygon == frames[11].text_boxes[0].polygon


def test_builder_makes_one_event_from_alternating_reads() -> None:
    """Dựng trọn vẹn: "一晚上" / "晚上" / "一晚上" liền nhau (bản 3.23.710 ra ba câu) → một câu."""
    from subtitles_extractor.application.dtos.extract_subtitles_dto import SubtitleBuilderConfig
    from subtitles_extractor.application.services.subtitle_pipeline.orchestrator import SubtitleBuilder
    from subtitles_extractor.domain.value_objects.roi import Roi

    frames = [_frame(0.04 * i) for i in range(6)]
    frames += _run([("一晚上", 13, {}), ("晚上", 11, {"left": 283, "right": 415}), ("一晚上", 11, {})], start=0.24,
                   chained=False)
    frames += [_frame(0.24 + 0.04 * (35 + i)) for i in range(6)]
    config = dataclasses.replace(SubtitleBuilderConfig(), sample_step_sec=0.04)
    events = SubtitleBuilder(config).build(frames, roi=Roi(x=35, y=774, width=644, height=136))
    assert [event.text for event in events] == ["一晚上"]


def test_low_confidence_garbage_reread_takes_the_good_read() -> None:
    """Tập 04 (trích xuất thật): "啊" 1,00 → "国sd" 0,51 → "啊" 1,00 trên cùng điểm ảnh — câu từng mất 0,9 s đầu."""
    frames = _run([("啊", 13, {}), ("国sd", 10, {"conf": 0.51}), ("啊", 6, {})])
    fused, fixed, _ = fuse_same_pixel_reads(frames, CENTER)
    assert _texts(fused) == ["啊"] * 29 and fixed == 10
