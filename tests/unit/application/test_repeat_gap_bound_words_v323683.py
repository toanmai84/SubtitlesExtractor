"""[v3.23.683] Hai lỗi cuối khi so cả phim nối 86 tập với đáp án đã đối chiếu ảnh.

* Câu lặp cách nhau MỘT khung OCR trống (bước 0,12 s): "师叔"/"师叔", "散了吧"/"散了吧" bị gộp. Dải ranh giới (điểm chữ
  sáng của mọi khung) phân biệt khoảng trống thật (0 ô) với khung OCR lỡ (số ô như hai bên).
* "觊觎" đọc thành "凯觎" ở mọi khung — sửa bằng từ "dính liền".
Số liệu chép từ dải ranh giới thật của phim (``fixed_tron_bo.seraw.json.gz``).
"""

from __future__ import annotations

from subtitles_extractor.application.dtos.extract_subtitles_dto import SubtitleBuilderConfig
from subtitles_extractor.application.services.subtitle_builder import SubtitleBuilder
from subtitles_extractor.application.services.subtitle_pipeline.boundary_refinement import pixel_confirmed_blanks
from subtitles_extractor.application.services.subtitle_pipeline.event_filters import is_repeat_utterance
from subtitles_extractor.application.services.subtitle_pipeline.text_correction import restore_bound_words
from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult, OcrTextBox
from subtitles_extractor.domain.value_objects.boundary_track import BoundaryProbe, BoundaryTrack
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.roi import Roi

STEP = 0.12
PERIOD = 0.04
ROI = Roi(x=57, y=819, width=603, height=119)


def _line(text: str) -> OcrTextBox:
    width = 68 * len(text)
    x0 = (603 - width) // 2
    return OcrTextBox(text=text, confidence=Confidence(1.0), polygon=[(x0, 20), (x0 + width, 20), (x0 + width, 88), (x0, 88)])


def _track(samples: list[tuple[float, int]], probe_pixels: dict[float, int] | None = None) -> BoundaryTrack:
    """Khung bỏ qua giữa hai khung mẫu mang số ô của khung mẫu gần nhất, trừ khi ghi trong ``probe_pixels``."""
    probe_pixels = probe_pixels or {}
    probes = []
    for (t0, p0), (t1, _p1) in zip(samples, samples[1:], strict=False):
        for k in (1, 2):
            t = round(t0 + k * PERIOD, 2)
            probes.append(BoundaryProbe(t, probe_pixels.get(t, p0), 1.0, 1.0))
    return BoundaryTrack(tuple(t for t, _ in samples), tuple(p for _, p in samples), tuple(probes), PERIOD)


def _build(spec: list[tuple[float, str]], track: BoundaryTrack | None) -> list[str]:
    frames = [
        OcrFrameResult(frame_index=i, timestamp_sec=t, text_boxes=[_line(x)] if x else [])
        for i, (t, x) in enumerate(spec)
    ]
    events = SubtitleBuilder(SubtitleBuilderConfig(sample_step_sec=STEP)).build(frames, roi=ROI, boundary_track=track)
    return [e.text for e in events]


def _repeat_spec(text: str, before: int, after: int) -> list[tuple[float, str]]:
    spec = [(round(10.0 + i * STEP, 2), "") for i in range(3)]
    t = 10.36
    for _ in range(before):
        spec.append((round(t, 2), text))
        t += STEP
    spec.append((round(t, 2), ""))
    t += STEP
    for _ in range(after):
        spec.append((round(t, 2), text))
        t += STEP
    spec += [(round(t + i * STEP, 2), "") for i in range(3)]
    return spec


def _pixels(spec: list[tuple[float, str]], blank_pixels: int, text_pixels: int = 254) -> list[tuple[float, int]]:
    """Khung có chữ = ``text_pixels`` ô; khung OCR trống giữa câu = ``blank_pixels``; ngoài câu = 0."""
    first = next(i for i, (_t, x) in enumerate(spec) if x)
    last = max(i for i, (_t, x) in enumerate(spec) if x)
    return [
        (t, text_pixels if x else (blank_pixels if first < i < last else 0))
        for i, (t, x) in enumerate(spec)
    ]


def test_pixel_confirmed_blanks_tells_real_gap_from_ocr_dropout() -> None:
    # Phim thật 388,92 s: khung mẫu 254 / 0 / 251, hai khung bỏ qua kề khung trống cũng 0 ô.
    gap = _track([(388.80, 254), (388.92, 0), (389.04, 251)], {388.96: 0, 388.88: 0})
    assert pixel_confirmed_blanks(gap) == [388.92]
    # OCR lỡ "啊" 820,08 s: 143 / 143 / 143 → chữ vẫn trên màn hình.
    assert pixel_confirmed_blanks(_track([(819.96, 143), (820.08, 143), (820.20, 143)])) == []
    # Hai bên ít chữ sáng (logo, chữ mờ) → không đủ căn cứ.
    assert pixel_confirmed_blanks(_track([(1.0, 20), (1.12, 0), (1.24, 22)], {1.04: 0, 1.08: 0, 1.16: 0})) == []
    # Chỉ đúng một khung hình trống (khung bỏ qua hai bên vẫn có chữ) → chưa đủ 2 khung.
    assert pixel_confirmed_blanks(_track([(5.0, 300), (5.12, 0), (5.24, 300)], {5.16: 300, 5.2: 300})) == []
    assert pixel_confirmed_blanks(None) == []
    # Bước dày (OCR mọi khung, không có khung bỏ qua) → không dùng bằng chứng điểm ảnh.
    assert pixel_confirmed_blanks(BoundaryTrack((1.0, 1.04, 1.08, 1.12), (300, 0, 0, 300), (), PERIOD)) == []


def test_repeat_rule_accepts_one_blank_sample_only_with_pixel_evidence() -> None:
    # 散了吧: 3 khung (0,24 s giữa khung đầu và cuối, 0,36 s trên màn hình), 1 khung trống, 6 khung.
    assert not is_repeat_utterance(0.24, STEP, 0.24, 0.6)
    assert is_repeat_utterance(0.24, STEP, 0.24, 0.6, pixel_blank=True)
    assert not is_repeat_utterance(0.24, STEP, 0.12, 0.6, pixel_blank=True)       # 2 khung = mảnh chớp


def test_builder_splits_repeat_across_single_blank_frame() -> None:
    for text, before, after in (("师叔", 6, 6), ("散了吧", 3, 6)):
        spec = _repeat_spec(text, before, after)
        blank_t = spec[3 + before][0]
        real_gap = _track(_pixels(spec, 0), {round(blank_t - PERIOD, 2): 0, round(blank_t + PERIOD, 2): 0})
        assert _build(spec, real_gap) == [text, text]
        # Khung OCR lỡ: điểm chữ sáng không đổi → vẫn một câu.
        assert _build(spec, _track(_pixels(spec, 254))) == [text]
        # Không có dải ranh giới (dữ liệu cũ) → như trước.
        assert _build(spec, None) == [text]


def test_restore_bound_words() -> None:
    assert restore_bound_words("凯觎我的女人") == "觊觎我的女人"
    assert restore_bound_words("忐志不安") == "忐忑不安"
    assert restore_bound_words("他很尬尴") == "他很尬尴"              # cả hai đều là chữ dính → không đoán
    assert restore_bound_words("人影憧憧") == "人影憧憧"              # chữ lặp
    assert restore_bound_words("螳臂挡车") == "螳臂挡车"              # 螳 không dính (đáp án phim có câu này)
    assert restore_bound_words("觊\n觎") == "觊\n觎"                   # xuống dòng không phải chữ Hán
    assert restore_bound_words("吩咐下去") == "吩咐下去"


def test_builder_fixes_bound_word() -> None:
    spec = [(round(10.0 + i * STEP, 2), "凯觎我的女人") for i in range(8)]
    assert _build(spec, None) == ["觊觎我的女人"]
