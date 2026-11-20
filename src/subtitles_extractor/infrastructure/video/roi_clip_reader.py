"""[v3.23.728] Đọc khung từ video vùng ROI (``<tên>.roi.mp4``) theo MỐC THỜI GIAN của dữ liệu OCR thô.

Trang Gỡ lỗi OCR dùng thay cho thư mục ảnh JPG cũ. Mốc trong clip = mốc seraw − ``source_start_sec`` (tệp mô tả
``.roi.json``). Giữ tệp mở và đọc tiếp khi đi tới (bấm "khung kế"), chỉ tua (seek) khi nhảy lùi hoặc nhảy xa.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Đi tới trong phạm vi này (giây) thì giải mã tiếp, xa hơn thì tua.
_FORWARD_DECODE_MAX_SEC = 3.0
#: Sai lệch tối đa giữa mốc cần và mốc khung tìm được (nửa khung ở 25 fps + dư).
_MATCH_TOLERANCE_SEC = 0.03

__all__ = ["RoiClipReader", "find_roi_clip"]


def find_roi_clip(raw_ocr_path: Path) -> tuple[Path, Path] | None:
    """``(video, mô tả)`` cạnh tệp OCR thô nếu có cả hai, ngược lại ``None``."""
    from subtitles_extractor.infrastructure.video.roi_clip_exporter import clip_paths_for

    clip, sidecar = clip_paths_for(raw_ocr_path)
    return (clip, sidecar) if clip.is_file() and sidecar.is_file() else None


class RoiClipReader:
    """Lấy khung RGB của clip ROI tại một mốc seraw."""

    def __init__(self, clip_path: Path, sidecar_path: Path) -> None:
        info = json.loads(sidecar_path.read_text(encoding="utf-8"))
        self.clip_path = clip_path
        self.crop_xywh: tuple[int, int, int, int] = tuple(int(v) for v in info["crop_xywh"])  # type: ignore[assignment]
        roi = info.get("roi_xywh")
        self.roi_xywh: tuple[int, int, int, int] | None = tuple(int(v) for v in roi) if roi else None  # type: ignore[assignment]
        self.source_start_sec = float(info.get("source_start_sec") or 0.0)
        self._container: Any = None
        self._stream: Any = None
        self._frames: Any = None
        self._last: tuple[float, np.ndarray] | None = None

    @property
    def box_offset(self) -> tuple[int, int]:
        """Hộp OCR tính theo ROI; clip cắt từ ``crop`` (có thể rộng hơn ROI 1 px để chẵn) → dời hộp bấy nhiêu."""
        if self.roi_xywh is None:
            return 0, 0
        return self.roi_xywh[0] - self.crop_xywh[0], self.roi_xywh[1] - self.crop_xywh[1]

    def _open(self) -> None:
        import av

        self._container = av.open(str(self.clip_path))
        self._stream = self._container.streams.video[0]
        self._frames = None
        self._last = None

    def _seek(self, clip_time: float) -> None:
        self._container.seek(int(max(0.0, clip_time - 2.5) / float(self._stream.time_base)), stream=self._stream)
        self._frames = self._container.decode(self._stream)
        self._last = None

    def frame_at(self, timestamp_sec: float) -> np.ndarray | None:
        """Khung RGB của clip tại mốc seraw ``timestamp_sec`` (``None`` nếu ngoài clip / lỗi đọc)."""
        try:
            if self._container is None:
                self._open()
            target = timestamp_sec - self.source_start_sec
            if self._last is not None and abs(self._last[0] - target) <= _MATCH_TOLERANCE_SEC:
                return self._last[1]
            if self._frames is None or self._last is None or not (
                self._last[0] < target <= self._last[0] + _FORWARD_DECODE_MAX_SEC
            ):
                self._seek(target)
            for frame in self._frames:
                if frame.pts is None:
                    continue
                t = float(frame.pts * self._stream.time_base)
                image = frame.to_ndarray(format="rgb24")
                self._last = (t, image)
                if t >= target - _MATCH_TOLERANCE_SEC:
                    return image if abs(t - target) <= max(_MATCH_TOLERANCE_SEC, 0.5) else None
            return None
        except Exception as exc:  # noqa: BLE001 — clip hỏng / thiếu: trang Gỡ lỗi vẫn hiện dữ liệu OCR
            logger.debug("Không đọc được khung %.3f s từ %s: %s", timestamp_sec, self.clip_path.name, exc)
            self.close()
            return None

    def close(self) -> None:
        if self._container is not None:
            try:
                self._container.close()
            except Exception:  # noqa: BLE001
                pass
        self._container = self._stream = self._frames = self._last = None
