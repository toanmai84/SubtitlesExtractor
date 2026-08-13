"""Unit tests cho :mod:`raw_ocr_serializer` — round-trip và edge cases."""

from __future__ import annotations

import json
import gzip
from pathlib import Path

import pytest

from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.infrastructure.serializers.raw_ocr_serializer import (
    RawOcrMeta,
    load_raw_ocr,
    save_raw_ocr,
)


def _make_frame(
    idx: int, ts: float, text: str, conf: float = 0.92
) -> OcrFrameResult:
    box = OcrTextBox(
        text=text,
        confidence=Confidence(conf),
        polygon=[(50, 100), (300, 100), (300, 140), (50, 140)],
    )
    return OcrFrameResult(frame_index=idx, timestamp_sec=ts, text_boxes=[box])


def _make_meta(video_name: str = "test.mp4") -> RawOcrMeta:
    return RawOcrMeta(
        video_name=video_name,
        video_duration_sec=154.88,
        frame_count=10,
        sample_step_sec=0.05,
        detection_model="PP-OCRv5_mobile_det",
        recognition_model="PP-OCRv5_mobile_rec",
        score_threshold=0.45,
        saved_at="2026-05-07T10:00:00+00:00",
    )


class TestSaveLoadRoundTrip:
    """Round-trip: save → load phải cho ra dữ liệu giống hệt."""

    def test_basic_roundtrip(self, tmp_path: Path) -> None:
        frames = [
            _make_frame(0, 18.0, "恭喜宿主", conf=0.92),
            _make_frame(1, 18.05, "恭喜宿主", conf=0.91),
            _make_frame(2, 25.0, "性别", conf=0.90),
        ]
        meta = _make_meta()
        out = tmp_path / "test.seraw.json"

        save_raw_ocr(frames, out, meta)
        loaded_frames, loaded_meta = load_raw_ocr(out)

        assert len(loaded_frames) == 3
        assert loaded_frames[0].frame_index == 0
        assert abs(loaded_frames[0].timestamp_sec - 18.0) < 1e-5
        assert loaded_frames[0].text_boxes[0].text == "恭喜宿主"
        assert abs(float(loaded_frames[0].text_boxes[0].confidence) - 0.92) < 1e-4

    def test_polygon_preserved(self, tmp_path: Path) -> None:
        box = OcrTextBox(
            text="雄性",
            confidence=Confidence(0.88),
            polygon=[(10, 20), (200, 20), (200, 60), (10, 60)],
        )
        frame = OcrFrameResult(frame_index=5, timestamp_sec=25.05, text_boxes=[box])
        out = tmp_path / "poly.seraw.json"

        save_raw_ocr([frame], out, _make_meta())
        loaded, _ = load_raw_ocr(out)

        assert loaded[0].text_boxes[0].polygon == [(10, 20), (200, 20), (200, 60), (10, 60)]

    def test_empty_polygon_box(self, tmp_path: Path) -> None:
        """Box không có polygon vẫn được round-trip đúng."""
        box = OcrTextBox(text="test", confidence=Confidence(0.80), polygon=[])
        frame = OcrFrameResult(frame_index=0, timestamp_sec=1.0, text_boxes=[box])
        out = tmp_path / "no_poly.seraw.json"

        save_raw_ocr([frame], out, _make_meta())
        loaded, _ = load_raw_ocr(out)

        assert loaded[0].text_boxes[0].polygon == []

    def test_meta_preserved(self, tmp_path: Path) -> None:
        meta = RawOcrMeta(
            video_name="chinese_vid1.mp4",
            video_duration_sec=154.88,
            frame_count=1847,
            sample_step_sec=0.05,
            detection_model="PP-OCRv5_mobile_det",
            recognition_model="PP-OCRv5_mobile_rec",
            score_threshold=0.45,
            saved_at="2026-05-07T10:00:00+00:00",
            roi_xywh=[0, 640, 720, 640],
        )
        frames = [_make_frame(0, 0.0, "test")]
        out = tmp_path / "meta.seraw.json"

        save_raw_ocr(frames, out, meta)
        _, loaded_meta = load_raw_ocr(out)

        assert loaded_meta.video_name == "chinese_vid1.mp4"
        assert abs(loaded_meta.video_duration_sec - 154.88) < 0.01
        assert loaded_meta.roi_xywh == [0, 640, 720, 640]

    def test_gzip_roundtrip(self, tmp_path: Path) -> None:
        """File .seraw.json.gz nén và giải nén đúng."""
        frames = [_make_frame(i, i * 0.05, f"frame{i}") for i in range(20)]
        out = tmp_path / "test.seraw.json.gz"

        save_raw_ocr(frames, out, _make_meta())
        assert out.exists()
        assert out.stat().st_size < sum(
            len(f.text_boxes[0].text) for f in frames
        ) * 200  # Phải nhỏ hơn raw data.

        loaded, _ = load_raw_ocr(out)
        assert len(loaded) == 20

    def test_many_frames(self, tmp_path: Path) -> None:
        """100 frame — kiểm tra hiệu năng serialization."""
        frames = [
            _make_frame(i, i * 0.05, "恭喜宿主激活超神系统", conf=0.90 + (i % 5) * 0.02)
            for i in range(100)
        ]
        out = tmp_path / "many.seraw.json"

        save_raw_ocr(frames, out, _make_meta())
        loaded, _ = load_raw_ocr(out)

        assert len(loaded) == 100
        assert loaded[50].frame_index == 50
        assert abs(loaded[50].timestamp_sec - 2.5) < 1e-5

    def test_cjk_unicode_preserved(self, tmp_path: Path) -> None:
        """Ký tự CJK không bị mã hoá sai."""
        cjk_texts = ["恭喜宿主", "性别", "雄性", "超神系统", "练气三层", "你好世界"]
        frames = [_make_frame(i, float(i), t) for i, t in enumerate(cjk_texts)]
        out = tmp_path / "cjk.seraw.json"

        save_raw_ocr(frames, out, _make_meta())
        loaded, _ = load_raw_ocr(out)

        for i, expected_text in enumerate(cjk_texts):
            assert loaded[i].text_boxes[0].text == expected_text


class TestErrorHandling:
    """Kiểm tra xử lý lỗi đầu vào."""

    def test_empty_frames_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="rỗng"):
            save_raw_ocr([], tmp_path / "x.seraw.json", _make_meta())

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_raw_ocr(tmp_path / "nonexistent.seraw.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.seraw.json"
        bad_file.write_bytes(b"{invalid json}")
        with pytest.raises(ValueError, match="JSON"):
            load_raw_ocr(bad_file)

    def test_incompatible_version(self, tmp_path: Path) -> None:
        payload = {"version": "99.0", "meta": {}, "frames": []}
        bad_file = tmp_path / "v99.seraw.json"
        bad_file.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="version"):
            load_raw_ocr(bad_file)


class TestFileFormat:
    """Kiểm tra định dạng file JSON thực tế."""

    def test_compact_json_no_extra_whitespace(self, tmp_path: Path) -> None:
        """File phải là compact JSON — không có indent thừa."""
        frames = [_make_frame(0, 0.0, "test")]
        out = tmp_path / "compact.seraw.json"

        save_raw_ocr(frames, out, _make_meta())
        content = out.read_text(encoding="utf-8")

        # Compact JSON không có `\n` hoặc spaces giữa tokens.
        assert "\n" not in content
        assert '": ' not in content  # No space after colon.

    def test_short_key_names(self, tmp_path: Path) -> None:
        """Keys ngắn: 'fi', 'ts', 'boxes', 't', 'c', 'p'."""
        frames = [_make_frame(0, 0.0, "test")]
        out = tmp_path / "keys.seraw.json"

        save_raw_ocr(frames, out, _make_meta())
        data = json.loads(out.read_text())

        frame_data = data["frames"][0]
        assert "fi" in frame_data
        assert "ts" in frame_data
        assert "boxes" in frame_data
        box_data = frame_data["boxes"][0]
        assert "t" in box_data
        assert "c" in box_data

    def test_schema_version_present(self, tmp_path: Path) -> None:
        frames = [_make_frame(0, 0.0, "x")]
        out = tmp_path / "ver.seraw.json"

        save_raw_ocr(frames, out, _make_meta())
        data = json.loads(out.read_text())

        assert "version" in data
        assert data["version"] == "1.0"
