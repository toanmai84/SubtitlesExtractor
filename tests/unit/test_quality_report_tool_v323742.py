"""[v3.23.742] Công cụ báo cáo chất lượng (giai đoạn 0 kế hoạch 4.0) chạy trên dữ liệu thật đã lưu."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "test data" / "tj_v3_23_741"


def _tool():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("quality_report", ROOT / "tools" / "quality_report.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_text_mask_finds_outlined_white_text_only() -> None:
    gray = np.full((40, 120), 120, np.uint8)
    gray[10:30, 20:60] = 0          # viền tối
    gray[14:26, 24:56] = 255        # nét chữ trắng có viền
    gray[10:30, 80:110] = 255       # mảng sáng không viền (áo trắng)
    mask = _tool().text_mask(gray, (0, 40))
    assert mask[:, 20:60].sum() > 0 and mask[:, 85:105].sum() == 0


@pytest.mark.skipif(not (DATA / "tj.seraw.json.gz").is_file() or shutil.which("ffmpeg") is None, reason="thiếu dữ liệu")
def test_report_on_real_clip(tmp_path: Path) -> None:
    folder = tmp_path / "run"
    folder.mkdir()
    shutil.copy(DATA / "tj.seraw.json.gz", folder / "phim.seraw.json.gz")
    shutil.copy(DATA / "tj.qa.json.gz", folder / "phim.qa.json.gz")
    shutil.copy(DATA / "clips" / "yanza_4997.roi.mp4", folder / "phim.roi.mp4")
    shutil.copy(DATA / "clips" / "yanza_4997.roi.json", folder / "phim.roi.json")
    result = subprocess.run([sys.executable, str(ROOT / "tools" / "quality_report.py"), str(folder), "--timing", "0"],
                            capture_output=True, text=True, timeout=600, check=False)
    assert result.returncode == 0, result.stderr[-1500:]
    report = json.loads((folder / "quality_report.json").read_text(encoding="utf-8"))
    assert report["events"] == 3663 and report["content"]["agree"] == 3634 and len(report["content"]["disagree"]) == 29
    assert report["missed_detector"] == []
