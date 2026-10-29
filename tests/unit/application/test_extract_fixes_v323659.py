"""[v3.23.659] Rà BUG khâu Trích xuất."""

from __future__ import annotations

from pathlib import Path

from subtitles_extractor.application.services.ocr_benchmark import DEFAULT_SCENARIOS
from subtitles_extractor.application.services.ocr_probe_window import compute_probe_window
from tests.unit.application.test_ocr_benchmark_v323645 import DURATION, _ConfigurableEngine, _request, _use_case

SC = {s.key: s for s in DEFAULT_SCENARIOS}


class _ReloadingEngine(_ConfigurableEngine):
    """Như PaddleOcrAdapter từ v3.23.656: cấu hình theo lượt; lượt nào cần model khác thì NẠP LẠI ngay trong lượt đó."""

    def __init__(self) -> None:
        super().__init__()
        self.loaded = None
        self.loads: list[str] = []

    def prepare(self, config) -> None:
        if config != self.loaded:
            self.loaded = config
            self.loads.append("prepare")

    def infer_batch(self, images_rgb, frame_indices, timestamps_sec, config=None):
        if config is not None and config != self.loaded:
            self.loaded = config
            self.loads.append("trong lượt tính giờ")
        return super().infer_batch(images_rgb, frame_indices, timestamps_sec)


def test_benchmark_does_not_time_model_reload(tmp_path: Path) -> None:
    """Kịch bản đổi model/hướng chữ: lần nạp lại (7–19 s) từng rơi vào lô tính giờ đầu → đề xuất lệch."""
    (tmp_path / "phim.mp4").write_bytes(b"0" * 64)
    engine = _ReloadingEngine()
    use_case = _use_case(tmp_path, engine)
    window = compute_probe_window(DURATION, probe_seconds=DURATION)
    results = use_case.execute(_request(tmp_path), [SC["current"], SC["no_orientation"], SC["det8"]], window)
    assert all(not r.error for r in results)
    assert "trong lượt tính giờ" not in engine.loads, engine.loads


def test_embedded_bitmap_reader_and_temp_cleanup(tmp_path: Path, monkeypatch) -> None:
    """Ảnh phụ đề nhúng: đọc được đường dẫn Unicode (cv2.imread hỏng trên Windows) + xoá thư mục tạm sau OCR."""
    import tempfile
    from types import SimpleNamespace

    import cv2
    import numpy as np

    from subtitles_extractor.application.use_cases import extract_embedded_subtitles as module

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    folder = Path(tempfile.mkdtemp(prefix="se_pgs_"))
    image = folder / "phụ đề 字幕.png"
    ok, encoded = cv2.imencode(".png", np.full((20, 40, 3), 200, np.uint8))
    encoded.tofile(str(image))
    assert module._read_image_bgr(image).shape == (20, 40, 3)
    module._remove_bitmap_temp_dirs([SimpleNamespace(image_path=image)])
    assert not folder.exists()
    keep = tmp_path / "khac"
    keep.mkdir()
    (keep / "a.png").write_bytes(b"x")
    module._remove_bitmap_temp_dirs([SimpleNamespace(image_path=keep / "a.png")])   # không phải thư mục tạm của app
    assert keep.exists()
