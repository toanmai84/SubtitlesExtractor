"""[v3.23.743] Công cụ báo cáo chất lượng báo nhầm trên phim 豪门Z外 (lượt 3.23.742, PaddleOCR-VL chạy thật).

1. Ngưỡng sót mức điểm ảnh lấy phân vị 10 % theo KHUNG (gồm khung chữ đang mờ dần) → ngưỡng ~0,0002, 50/50 ứng viên là
   cảnh không có chữ (xem ảnh), trong khi câu một chữ thật yếu nhất 0,0044. Nay: ½ mức của câu yếu nhất (phân vị 1 %
   trung vị theo CÂU).
2. Đo thời gian bằng IoU mặt nạ: nền sáng đổi sau chữ làm phần hợp tăng gấp đôi ("门没关" 2746 → 5088 điểm, IoU 0,51) dù
   chữ còn nguyên → báo nhầm mốc cuối lệch 7,6 khung. Nay: độ phủ nét chữ mẫu ≥ 60 %.

Dữ liệu: ``test data/hmzw_v3_23_742`` — seraw + gói kiểm định thật, 2 đoạn video ROI cắt từ lượt chạy.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "test data" / "hmzw_v3_23_742"

pytestmark = pytest.mark.skipif(not (DATA / "hmzw.seraw.json.gz").is_file(), reason="thiếu dữ liệu test")


def _run(tmp_path: Path, clip: str, timing: int) -> dict:
    folder = tmp_path / clip
    folder.mkdir()
    shutil.copy(DATA / "hmzw.seraw.json.gz", folder / "phim.seraw.json.gz")
    shutil.copy(DATA / "hmzw.qa.json.gz", folder / "phim.qa.json.gz")
    shutil.copy(DATA / "clips" / f"{clip}.roi.mp4", folder / "phim.roi.mp4")
    shutil.copy(DATA / "clips" / f"{clip}.roi.json", folder / "phim.roi.json")
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "quality_report.py"), str(folder), "--timing", str(timing)],
        capture_output=True, text=True, timeout=900, check=False,
    )
    assert result.returncode == 0, result.stderr[-1500:]
    return json.loads((folder / "quality_report.json").read_text(encoding="utf-8"))


def test_no_false_missed_lines_on_scene_without_subtitles(tmp_path: Path) -> None:
    # 260,9–286,3 s: cảnh không phụ đề (đã xem ảnh) — công cụ cũ báo một đoạn sót 123 mẫu.
    report = _run(tmp_path, "gap_254", 0)
    assert report["missed_pixels"] == []
    assert report["weak_pixel_lines"] == []          # câu một chữ "好" vẫn đủ nét chữ


def test_timing_unaffected_by_background_change_behind_text(tmp_path: Path) -> None:
    report = _run(tmp_path, "mon_2644", 2131)
    timing = report["timing"]
    assert timing["measured"] >= 2                   # "门没关" + "阿姨这是在"
    assert timing["end_within_1_frame"] == 1.0
    assert timing["outliers"] == []
