"""Đặc tả không gian tham số cho hiệu chuẩn.

Một :class:`SearchSpace` gồm nhiều :class:`ParameterSpec`; optimizer rời rạc hoá
mỗi spec thành lưới giá trị và tinh dần (đệ quy) quanh điểm tốt nhất.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tools.calibration.exceptions import InvalidParameterSpecError

ParameterKind = Literal["float", "int", "bool"]


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Đặc tả một tham số cần hiệu chuẩn.

    Attributes:
        name: Tên thuộc tính trên config (vd ``"merge_gap_sec"``).
        low: Cận dưới khả dĩ.
        high: Cận trên khả dĩ.
        kind: Kiểu dữ liệu — ``"float"``, ``"int"`` hoặc ``"bool"``.
        grid_points: Số điểm lưới khi rời rạc hoá ở mỗi cấp tinh chỉnh.
    """

    name: str
    low: float
    high: float
    kind: ParameterKind = "float"
    grid_points: int = 5

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise InvalidParameterSpecError(
                f"ParameterSpec '{self.name}': low ({self.low}) > high ({self.high})"
            )
        if self.grid_points < 2:
            raise InvalidParameterSpecError(
                f"ParameterSpec '{self.name}': grid_points phải >= 2"
            )

    def coerce(self, value: float) -> float | int | bool:
        """Ép một giá trị thực về đúng kiểu của tham số, có kẹp biên."""
        clamped = max(self.low, min(self.high, value))
        if self.kind == "bool":
            return clamped >= 0.5
        if self.kind == "int":
            return int(round(clamped))
        return float(clamped)

    def grid(self, center: float | None = None, span_ratio: float = 1.0) -> list[float]:
        """Sinh lưới giá trị quanh ``center`` với độ rộng ``span_ratio``.

        Args:
            center: Tâm lưới; mặc định là trung điểm khoảng.
            span_ratio: Tỷ lệ co độ rộng (1.0 = toàn khoảng; 0.25 = hẹp lại 4 lần)
                — đây là cơ chế *tinh chỉnh đệ quy*.

        Returns:
            Danh sách giá trị lưới đã loại trùng, theo thứ tự tăng dần.
        """
        if self.kind == "bool":
            return [0.0, 1.0]
        full_span = self.high - self.low
        if full_span == 0:
            return [self.low]
        midpoint = self.low if center is None else center
        half_window = (full_span * span_ratio) / 2.0
        window_low = max(self.low, midpoint - half_window)
        window_high = min(self.high, midpoint + half_window)
        if window_low == window_high:
            return [window_low]
        step = (window_high - window_low) / (self.grid_points - 1)
        values = [window_low + step * index for index in range(self.grid_points)]
        if self.kind == "int":
            values = sorted({float(int(round(value))) for value in values})
        return values


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """Tập các :class:`ParameterSpec` định nghĩa không gian tìm kiếm."""

    specs: tuple[ParameterSpec, ...]

    def __post_init__(self) -> None:
        if not self.specs:
            raise InvalidParameterSpecError("SearchSpace phải có ít nhất 1 tham số.")
        names = [spec.name for spec in self.specs]
        if len(names) != len(set(names)):
            raise InvalidParameterSpecError("SearchSpace có tên tham số trùng nhau.")

    def names(self) -> list[str]:
        """Danh sách tên tham số."""
        return [spec.name for spec in self.specs]

    def spec_for(self, name: str) -> ParameterSpec:
        """Lấy spec theo tên."""
        for spec in self.specs:
            if spec.name == name:
                return spec
        raise InvalidParameterSpecError(f"Không có tham số tên '{name}' trong SearchSpace.")

    def midpoint_assignment(self) -> dict[str, float]:
        """Bộ tham số khởi đầu = trung điểm mỗi khoảng."""
        return {spec.name: (spec.low + spec.high) / 2.0 for spec in self.specs}

    def coerce_assignment(self, assignment: dict[str, float]) -> dict[str, float | int | bool]:
        """Ép toàn bộ assignment về đúng kiểu từng tham số."""
        return {
            spec.name: spec.coerce(assignment[spec.name])
            for spec in self.specs
            if spec.name in assignment
        }
