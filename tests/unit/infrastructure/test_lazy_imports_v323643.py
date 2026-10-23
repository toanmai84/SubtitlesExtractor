"""[v3.23.643] Nạp trễ sklearn / google.genai: đường chạy thật vẫn đúng và import nền không kéo chúng vào."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.tts.gemini_tts_adapter import GeminiTTSAdapter
from subtitles_extractor.infrastructure.video import bbox_analyzer
from subtitles_extractor.infrastructure.video.bbox_analyzer import BBoxAnalyzer, RawBBox

SRC = Path(__file__).resolve().parents[3] / "src"


def test_importing_bbox_analyzer_does_not_import_sklearn() -> None:
    code = (
        "import sys;"
        "import subtitles_extractor.infrastructure.video.bbox_analyzer as b;"
        "print('sklearn' in sys.modules, b.HAS_SKLEARN)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env={"PYTHONPATH": str(SRC)},
        check=True,
    ).stdout.split()
    assert out[0] == "False"


@pytest.mark.skipif(not bbox_analyzer.HAS_SKLEARN, reason="sklearn chưa cài")
def test_dbscan_path_still_clusters() -> None:
    analyzer = BBoxAnalyzer(frame_width=720, frame_height=1280, use_dbscan=True)
    assert analyzer.use_dbscan
    boxes = [RawBBox(100, 1000 + (i % 3), 600, 1050 + (i % 3), 0.95, i, i * 0.1) for i in range(60)]
    clusters = analyzer.analyze(boxes)
    assert clusters
    assert all(c.coord_y_max >= 1000 for c in clusters)


def test_dbscan_import_failure_falls_back_to_heatmap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sklearn.cluster", None)      # import → ImportError
    monkeypatch.setattr(bbox_analyzer, "HAS_SKLEARN", True)
    monkeypatch.setattr(bbox_analyzer, "SklearnDBSCAN", None)
    analyzer = BBoxAnalyzer(frame_width=720, frame_height=1280, use_dbscan=True)
    boxes = [RawBBox(100, 1000, 600, 1050, 0.95, i, i * 0.1) for i in range(60)]
    assert analyzer.analyze(boxes)


def test_gemini_tts_availability_does_not_import_genai() -> None:
    code = (
        "import sys;"
        "from subtitles_extractor.infrastructure.tts.gemini_tts_adapter import GeminiTTSAdapter;"
        "GeminiTTSAdapter('k').is_available();"
        "print('google.genai' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env={"PYTHONPATH": str(SRC)},
        check=True,
    ).stdout.split()
    assert out[-1] == "False"
    assert isinstance(GeminiTTSAdapter("k").is_available(), bool)
