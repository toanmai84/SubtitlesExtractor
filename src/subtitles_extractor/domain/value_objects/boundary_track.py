"""[v3.23.680] Dải ranh giới: dữ liệu nhẹ về MỌI khung hình (kể cả khung không OCR) để đặt mốc phụ đề đúng khung.

Bước lấy mẫu mặc định 0,12 s (25 fps → OCR 1/3 khung) làm mốc câu lệch tới 2-3 khung: đo trên đáp án tập 1–5
(428 câu), mốc bắt đầu lệch trung bình 45 ms, 210 câu lệch > 1 khung. Thay vì OCR thêm, bộ giải mã ghi cho mỗi khung
BỊ BỎ QUA bốn con số: thời điểm, số điểm chữ sáng, độ giống với khung mẫu TRƯỚC và khung mẫu SAU (so mặt nạ chữ
sáng của vùng phụ đề). Vài MB cho cả phim dài.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class BoundaryProbe:
    """Một khung KHÔNG OCR nằm giữa hai khung mẫu."""

    timestamp_sec: float
    #: Số ô có chữ sáng (mặt nạ thu nhỏ 4×4) trong vùng phụ đề.
    text_pixels: int
    #: Độ giống (giao/hợp mặt nạ) với khung mẫu liền TRƯỚC, 0..1.
    sim_prev: float
    #: Độ giống với khung mẫu liền SAU, 0..1.
    sim_next: float


@dataclass(frozen=True, slots=True)
class BoundaryTrack:
    """Toàn bộ dải ranh giới của một lượt trích xuất (sắp theo thời gian)."""

    sample_times: tuple[float, ...] = ()
    sample_pixels: tuple[int, ...] = ()
    probes: tuple[BoundaryProbe, ...] = field(default_factory=tuple)
    frame_period_sec: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.probes or self.frame_period_sec <= 0 or not self.sample_times

    def to_dict(self) -> dict[str, object]:
        """Dạng gọn để lưu kèm dữ liệu OCR thô (mảng cột, làm tròn)."""
        return {
            "period": round(self.frame_period_sec, 6),
            "st": [round(t, 4) for t in self.sample_times],
            "sp": list(self.sample_pixels),
            "pt": [round(p.timestamp_sec, 4) for p in self.probes],
            "pp": [p.text_pixels for p in self.probes],
            "ps": [round(p.sim_prev, 3) for p in self.probes],
            "pn": [round(p.sim_next, 3) for p in self.probes],
        }

    @classmethod
    def from_dict(cls, data: object) -> BoundaryTrack | None:
        """Ngược :meth:`to_dict`; dữ liệu hỏng/thiếu → ``None`` (dựng phụ đề như cũ)."""
        if not isinstance(data, dict):
            return None
        try:
            st = [float(x) for x in data["st"]]
            sp = [int(x) for x in data["sp"]]
            pt, pp, ps, pn = (data[k] for k in ("pt", "pp", "ps", "pn"))
            if len(st) != len(sp) or not (len(pt) == len(pp) == len(ps) == len(pn)):
                return None
            probes = tuple(
                BoundaryProbe(float(t), int(p), float(a), float(b)) for t, p, a, b in zip(pt, pp, ps, pn, strict=True)
            )
            return cls(tuple(st), tuple(sp), probes, float(data["period"]))
        except (KeyError, TypeError, ValueError):
            return None


__all__ = ["BoundaryProbe", "BoundaryTrack"]
