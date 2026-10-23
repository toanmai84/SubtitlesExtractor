"""[v3.23.632] Điểm dừng (checkpoint) cho OCR dài: bị ngắt giữa chừng thì chạy TIẾP, không làm lại.

Nhật ký thật của phim 86 tập nối thành một video: một lượt OCR chạy 2 giờ 24 phút và CHỈ lưu
kết quả ở cuối. Máy sập/treo/đóng ứng dụng ở phút thứ 140 là mất sạch 2 giờ GPU. Module này ghi
kết quả OCR dần dần ra một tệp JSON-lines cạnh cache OCR thô:

* dòng đầu = "vân tay" của lượt chạy (video, vùng chữ, cấu hình lấy mẫu và OCR);
* mỗi dòng sau = một frame đã OCR xong, đúng thứ tự (một TIỀN TỐ liên tục của kết quả);
* ghi nối thêm + ``fsync`` → tổng chi phí ghi là O(n) chứ không phải O(n²) như ghi lại cả tệp;
* mất điện giữa lúc ghi chỉ làm hỏng dòng cuối → bị cắt bỏ khi nạp, phần trước vẫn dùng được.

Chỉ nạp lại được khi vân tay KHỚP hoàn toàn: đổi video, vùng chữ, bước lấy mẫu, mô hình, ngưỡng…
đều bắt đầu lại từ đầu (không bao giờ ghép kết quả của hai cấu hình khác nhau).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult
from subtitles_extractor.infrastructure.serializers.raw_ocr_serializer import (
    _deserialize_frame,
    _serialize_frame,
)

if TYPE_CHECKING:
    from subtitles_extractor.application.dtos.extract_subtitles_dto import (
        ExtractSubtitlesRequest,
    )
    from subtitles_extractor.domain.entities.video_metadata import VideoMetadata

logger = logging.getLogger(__name__)

#: Phiên bản định dạng tệp điểm dừng (đổi khi không tương thích).
CHECKPOINT_VERSION = 1
#: Hậu tố tệp điểm dừng, nối vào đường dẫn cache OCR thô.
CHECKPOINT_SUFFIX = ".ckpt"
#: Tham số KHÔNG làm đổi kết quả OCR (chỉ đổi tốc độ) → không vào vân tay.
_SPEED_ONLY_OCR_FIELDS = ("batch_size", "parallel_workers", "enable_mkldnn", "use_tensorrt")


def checkpoint_path_for(raw_ocr_path: Path) -> Path:
    """Đường dẫn tệp điểm dừng đi kèm cache OCR thô ``raw_ocr_path``."""
    return raw_ocr_path.with_name(raw_ocr_path.name + CHECKPOINT_SUFFIX)


def run_fingerprint(request: ExtractSubtitlesRequest, metadata: VideoMetadata) -> str:
    """Vân tay của lượt OCR: mọi thứ làm đổi kết quả từng frame.

    Gồm tệp video (tên, kích thước, thời điểm sửa, thời lượng), vùng chữ, TOÀN BỘ cấu hình lấy
    mẫu (kể cả ``skip_intro``/``skip_outro`` GỐC của người dùng) và cấu hình OCR trừ các tham số
    chỉ ảnh hưởng tốc độ.
    """
    try:
        stat = request.video_path.stat()
        stamp: tuple[int, int] = (stat.st_size, stat.st_mtime_ns)
    except OSError:
        stamp = (-1, -1)
    ocr = dataclasses.asdict(request.ocr)
    for name in _SPEED_ONLY_OCR_FIELDS:
        ocr.pop(name, None)
    roi = (
        [request.roi.x, request.roi.y, request.roi.width, request.roi.height]
        if request.roi is not None else None
    )
    payload: dict[str, Any] = {
        "v": CHECKPOINT_VERSION,
        "video": request.video_path.name,
        "stamp": stamp,
        "duration": round(metadata.duration_sec, 3),
        "roi": roi,
        "sampling": dataclasses.asdict(request.sampling),
        "ocr": ocr,
    }
    text = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class OcrCheckpoint:
    """Tệp điểm dừng của MỘT lượt OCR (không an toàn đa luồng: chỉ luồng OCR dùng)."""

    def __init__(self, path: Path, fingerprint: str) -> None:
        """Khởi tạo.

        Args:
            path: Tệp điểm dừng.
            fingerprint: Vân tay lượt chạy hiện tại (:func:`run_fingerprint`).
        """
        self._path = path
        self._fingerprint = fingerprint
        self._written = 0
        self._ready = False

    @property
    def path(self) -> Path:
        """Đường dẫn tệp điểm dừng."""
        return self._path

    @property
    def written(self) -> int:
        """Số frame đã nằm trong tệp."""
        return self._written

    # ── Nạp ──────────────────────────────────────────────────────────────────

    def resume(self) -> list[OcrFrameResult]:
        """Nạp phần đã OCR từ lần trước; ``[]`` nếu không có / không khớp / hỏng.

        Khớp vân tay → giữ tệp (cắt bỏ dòng cuối bị dở nếu có) để ghi nối tiếp. Không khớp hoặc
        hỏng → xoá tệp cũ để lượt này bắt đầu sạch.
        """
        if not self._path.is_file():
            return []
        try:
            frames, good_bytes, matched = self._read()
        except OSError as exc:
            logger.warning("Không đọc được điểm dừng OCR %s: %s.", self._path.name, exc)
            self.discard()
            return []
        if not matched or not frames:
            if self._path.is_file():
                logger.info("Điểm dừng OCR cũ không khớp lượt chạy này → bỏ, OCR từ đầu.")
            self.discard()
            return []
        try:
            if good_bytes < self._path.stat().st_size:
                with self._path.open("r+b") as handle:
                    handle.truncate(good_bytes)
                logger.info("Điểm dừng OCR: cắt bỏ phần cuối bị dở do ngắt đột ngột.")
        except OSError as exc:
            logger.warning("Không cắt được phần dở của điểm dừng OCR: %s.", exc)
            self.discard()
            return []
        self._written = len(frames)
        self._ready = True
        return frames

    def _read(self) -> tuple[list[OcrFrameResult], int, bool]:
        """Đọc tệp: ``(frame hợp lệ, số byte hợp lệ, vân tay khớp)``."""
        frames: list[OcrFrameResult] = []
        good = 0
        with self._path.open("rb") as handle:
            header_line = handle.readline()
            try:
                header = json.loads(header_line.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return [], 0, False
            if (
                not isinstance(header, dict)
                or header.get("version") != CHECKPOINT_VERSION
                or header.get("fingerprint") != self._fingerprint
            ):
                return [], 0, False
            good = len(header_line)
            for line in handle:
                if not line.endswith(b"\n"):
                    break                       # dòng cuối bị dở (mất điện giữa lúc ghi)
                try:
                    frames.append(_deserialize_frame(json.loads(line.decode("utf-8"))))
                except (ValueError, UnicodeDecodeError, TypeError, KeyError):
                    break                       # từ đây trở đi không tin được nữa
                good += len(line)
        return frames, good, True

    # ── Ghi ──────────────────────────────────────────────────────────────────

    def append(self, frames: list[OcrFrameResult]) -> None:
        """Ghi nối ``frames`` (phần kế tiếp của tiền tố). Lỗi ghi chỉ cảnh báo, không dừng OCR."""
        if not frames:
            return
        try:
            if not self._ready:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                header = {"version": CHECKPOINT_VERSION, "fingerprint": self._fingerprint}
                self._path.write_text(
                    json.dumps(header, separators=(",", ":")) + "\n", encoding="utf-8",
                )
                self._ready = True
            body = "".join(
                json.dumps(_serialize_frame(frame), ensure_ascii=False, separators=(",", ":")) + "\n"
                for frame in frames
            )
            with self._path.open("ab") as handle:
                handle.write(body.encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            self._written += len(frames)
        except OSError as exc:
            logger.warning("Không ghi được điểm dừng OCR %s: %s.", self._path.name, exc)

    def discard(self) -> None:
        """Xoá tệp điểm dừng (lượt OCR đã xong, hoặc điểm dừng cũ không dùng được)."""
        self._ready = False
        self._written = 0
        try:
            self._path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Không xoá được điểm dừng OCR %s: %s.", self._path.name, exc)


__all__ = [
    "CHECKPOINT_SUFFIX", "CHECKPOINT_VERSION", "OcrCheckpoint", "checkpoint_path_for",
    "run_fingerprint",
]
