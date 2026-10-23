"""[v3.23.645] Đo A/B tốc độ + chất lượng OCR ngay trong ứng dụng — phần THUẦN (kịch bản, so sánh, đề xuất).

Vì sao: các tối ưu OCR của v3.23.644 (bỏ phân loại hướng chữ, batch bộ dò chữ, lọc khung theo chữ sáng) đo được rõ trên CPU nhưng
KHÔNG đoán được trên GPU của từng máy; thay vì bắt người dùng mở cmd chạy script, trang Trích xuất tự chạy các kịch bản trên một
đoạn phim ngắn bằng đúng video, ROI và cài đặt hiện tại, so phụ đề dựng ra với cấu hình hiện tại và đề xuất cấu hình nhanh nhất mà
KHÔNG đổi kết quả.
"""

from __future__ import annotations

import dataclasses
import difflib
from collections.abc import Sequence
from dataclasses import dataclass, field

from subtitles_extractor.application.dtos.extract_subtitles_dto import ExtractSubtitlesRequest
from subtitles_extractor.application.services.ocr_probe_window import ProbeWindow

__all__ = [
    "DEFAULT_KEYS",
    "DEFAULT_SCENARIOS",
    "MAX_TIMING_DIFF_SEC",
    "MIN_SPEEDUP",
    "BenchmarkScenario",
    "Comparison",
    "EventSig",
    "Recommendation",
    "ScenarioResult",
    "apply_scenario",
    "compare_events",
    "recommend",
]

#: (start_sec, end_sec, text) — chữ ký tối thiểu của một câu phụ đề để so sánh.
EventSig = tuple[float, float, str]

#: Lệch mốc thời gian tối đa (giây) mà vẫn coi là "không đổi kết quả" (≈ 3 khung ở 25 fps).
MAX_TIMING_DIFF_SEC = 0.12
#: Chỉ đề xuất đổi khi nhanh hơn ít nhất chừng này lần.
MIN_SPEEDUP = 1.10


@dataclass(frozen=True, slots=True)
class BenchmarkScenario:
    """Một cấu hình đem đo. ``None`` = giữ giá trị của cấu hình hiện tại."""

    key: str
    label_vi: str
    orientation: bool | None = None
    det_batch: int | None = None
    gate: bool | None = None
    luma: int | None = None

    @property
    def lossless(self) -> bool:
        """Không dùng bộ lọc khung (không có rủi ro bỏ sót phụ đề)."""
        return not self.gate


DEFAULT_SCENARIOS: tuple[BenchmarkScenario, ...] = (
    BenchmarkScenario("current", "Cấu hình hiện tại"),
    BenchmarkScenario("no_orientation", "Tắt phân loại hướng chữ", orientation=False),
    BenchmarkScenario("det4", "Tắt hướng chữ + batch dò chữ 4", orientation=False, det_batch=4),
    BenchmarkScenario("det8", "Tắt hướng chữ + batch dò chữ 8", orientation=False, det_batch=8),
    BenchmarkScenario("gate200", "Lọc khung theo chữ sáng (ngưỡng 200)", orientation=False, gate=True, luma=200),
    BenchmarkScenario("gate220", "Lọc khung theo chữ sáng (ngưỡng 220)", orientation=False, gate=True, luma=220),
    BenchmarkScenario(
        "gate200_det8", "Lọc chữ sáng 200 + batch dò chữ 8", orientation=False, det_batch=8, gate=True, luma=200,
    ),
)

#: Các kịch bản chọn sẵn (vừa đủ để quyết định, ~5 lượt đo).
DEFAULT_KEYS: tuple[str, ...] = ("current", "no_orientation", "det8", "gate200", "gate200_det8")


@dataclass(frozen=True, slots=True)
class Comparison:
    """So phụ đề của một kịch bản với cấu hình hiện tại."""

    events: int
    same_text_ratio: float
    n_missing: int
    n_extra: int
    max_start_diff_s: float
    max_end_diff_s: float
    missing: tuple[str, ...] = ()
    extra: tuple[str, ...] = ()

    @property
    def identical(self) -> bool:
        """Cùng tập câu và mốc thời gian lệch không đáng kể."""
        return (
            self.n_missing == 0 and self.n_extra == 0
            and self.max_start_diff_s <= MAX_TIMING_DIFF_SEC and self.max_end_diff_s <= MAX_TIMING_DIFF_SEC
        )


@dataclass(slots=True)
class ScenarioResult:
    """Kết quả đo một kịch bản."""

    scenario: BenchmarkScenario
    wall_s: float = 0.0
    ocr_s: float = 0.0
    ocr_frames: int = 0
    batches: int = 0
    events: tuple[EventSig, ...] = ()
    error: str = ""
    comparison: Comparison | None = None

    @property
    def other_s(self) -> float:
        """Thời gian ngoài bộ OCR (giải mã, lọc khung, dựng phụ đề)."""
        return max(0.0, self.wall_s - self.ocr_s)

    @property
    def avg_batch(self) -> float:
        return self.ocr_frames / self.batches if self.batches else 0.0


@dataclass(frozen=True, slots=True)
class Recommendation:
    """Đề xuất sau khi đo."""

    key: str
    label_vi: str
    reason_vi: str
    settings_patch: dict[str, dict[str, object]] = field(default_factory=dict)
    speedup: float = 1.0
    warnings_vi: tuple[str, ...] = ()

    @property
    def changes_anything(self) -> bool:
        return bool(self.settings_patch)


def apply_scenario(
    base: ExtractSubtitlesRequest, scenario: BenchmarkScenario, window: ProbeWindow,
) -> ExtractSubtitlesRequest:
    """Yêu cầu trích xuất của kịch bản: cấu hình nền + ghi đè của kịch bản + giới hạn vào cửa sổ đo (không xuất tệp)."""
    ocr = base.ocr
    if scenario.orientation is not None:
        ocr = dataclasses.replace(ocr, use_textline_orientation=scenario.orientation)
    if scenario.det_batch is not None:
        ocr = dataclasses.replace(ocr, det_batch_size=scenario.det_batch)
    sampling = dataclasses.replace(
        base.sampling, skip_intro_sec=window.skip_intro_sec, skip_outro_sec=window.skip_outro_sec,
    )
    if scenario.gate is not None:
        sampling = dataclasses.replace(sampling, white_text_gate=scenario.gate)
    if scenario.luma is not None:
        sampling = dataclasses.replace(sampling, white_text_luma=scenario.luma)
    return dataclasses.replace(
        base, ocr=ocr, sampling=sampling, skip_export=True, use_frame_cache=False, raw_ocr_output_path=None,
        export_raw_images=False, save_debug_frames=False,
    )


def compare_events(reference: Sequence[EventSig], other: Sequence[EventSig]) -> Comparison:
    """So hai danh sách câu: chuỗi chữ giống nhau bao nhiêu, thiếu/thừa câu nào, mốc lệch tối đa."""
    ref_texts = [item[2] for item in reference]
    got_texts = [item[2] for item in other]
    matcher = difflib.SequenceMatcher(None, ref_texts, got_texts, autojunk=False)
    start_diffs: list[float] = []
    end_diffs: list[float] = []
    missing: list[str] = []
    extra: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                start_diffs.append(abs(reference[i1 + offset][0] - other[j1 + offset][0]))
                end_diffs.append(abs(reference[i1 + offset][1] - other[j1 + offset][1]))
        else:
            missing += ref_texts[i1:i2]
            extra += got_texts[j1:j2]
    return Comparison(
        events=len(other),
        same_text_ratio=round(matcher.ratio(), 4) if (ref_texts or got_texts) else 1.0,
        n_missing=len(missing), n_extra=len(extra),
        max_start_diff_s=round(max(start_diffs, default=0.0), 3),
        max_end_diff_s=round(max(end_diffs, default=0.0), 3),
        missing=tuple(missing[:8]), extra=tuple(extra[:8]),
    )


def current_settings_values(base: ExtractSubtitlesRequest) -> dict[str, dict[str, object]]:
    """Giá trị HIỆN TẠI của các cài đặt mà kịch bản có thể đổi, cùng dạng với ``Recommendation.settings_patch``."""
    return {
        "ocr": {"use_textline_orientation": base.ocr.use_textline_orientation},
        "hardware": {"det_batch_size": int(getattr(base.ocr, "det_batch_size", 1) or 1)},
        "frame": {
            "white_text_gate": bool(getattr(base.sampling, "white_text_gate", False)),
            "white_text_luma": int(getattr(base.sampling, "white_text_luma", 200) or 200),
        },
    }


def is_redundant_scenario(
    base: ExtractSubtitlesRequest, scenario: BenchmarkScenario, window: ProbeWindow,
) -> bool:
    """Kịch bản (không phải ``current``) mà cấu hình hiệu lực TRÙNG cấu hình hiện tại — đo cũng vô nghĩa."""
    if scenario.key == "current":
        return False
    mine = apply_scenario(base, scenario, window)
    ref = apply_scenario(base, BenchmarkScenario(key="current", label_vi=""), window)
    return mine.ocr == ref.ocr and mine.sampling == ref.sampling


def recommend(
    results: Sequence[ScenarioResult],
    base_values: dict[str, dict[str, object]] | None = None,
) -> Recommendation | None:
    """Chọn cấu hình NHANH NHẤT không đổi kết quả so với cấu hình hiện tại.

    Trả ``None`` khi không có mốc so sánh (kịch bản ``current`` lỗi hoặc đoạn đo không có phụ đề — khi đó mọi cấu hình
    đều "giống nhau" một cách vô nghĩa).
    """
    reference = next((r for r in results if r.scenario.key == "current" and not r.error), None)
    if reference is None or not reference.events:
        return None
    for result in results:
        if result is reference or result.error:
            continue
        result.comparison = compare_events(reference.events, result.events)
    reference.comparison = compare_events(reference.events, reference.events)

    eligible = [
        r for r in results
        if not r.error and r.comparison is not None and r.comparison.identical and r.wall_s > 0
    ]
    if not eligible:
        return None
    best = min(eligible, key=lambda r: (r.wall_s, not r.scenario.lossless))
    speedup = reference.wall_s / best.wall_s if best.wall_s > 0 else 1.0
    if best is reference or speedup < MIN_SPEEDUP:
        return Recommendation(
            key="current", label_vi=reference.scenario.label_vi,
            reason_vi=(
                "Cấu hình hiện tại đã gần như nhanh nhất trong các kịch bản đã đo "
                f"(chênh dưới {int((MIN_SPEEDUP - 1) * 100)}%) — không cần đổi."
            ),
            speedup=1.0,
        )
    scenario = best.scenario
    patch: dict[str, dict[str, object]] = {}
    if scenario.orientation is not None:
        patch.setdefault("ocr", {})["use_textline_orientation"] = scenario.orientation
    if scenario.det_batch is not None:
        patch.setdefault("hardware", {})["det_batch_size"] = scenario.det_batch
    if scenario.gate is not None:
        patch.setdefault("frame", {})["white_text_gate"] = scenario.gate
    if scenario.luma is not None:
        patch.setdefault("frame", {})["white_text_luma"] = scenario.luma
    # Chỉ ghi những khoá THỰC SỰ khác cài đặt hiện tại (kịch bản trùng cấu hình nền không phải thay đổi).
    if base_values is not None:
        patch = {
            group: changed
            for group, values in patch.items()
            if (changed := {k: v for k, v in values.items() if base_values.get(group, {}).get(k) != v})
        }
        if not patch:
            return Recommendation(
                key="current", label_vi=reference.scenario.label_vi,
                reason_vi="Cấu hình hiện tại đã là cấu hình nhanh nhất trong các kịch bản đã đo — không cần đổi.",
                speedup=1.0,
            )
    warnings: list[str] = []
    if scenario.gate:
        warnings.append(
            "Bộ lọc chữ sáng CHỈ đúng khi phụ đề trắng/sáng: phụ đề tối màu hoặc nền sáng lớn có thể bị bỏ sót. "
            "Đoạn đo ngắn có thể không đại diện — đo thêm ở một đoạn khác trước khi chạy cả phim."
        )
    return Recommendation(
        key=scenario.key, label_vi=scenario.label_vi,
        reason_vi=(
            f"Nhanh hơn {speedup:.1f}× so với hiện tại ({reference.wall_s:.0f}s → {best.wall_s:.0f}s) "
            "và cho cùng tập câu phụ đề."
        ),
        settings_patch=patch, speedup=round(speedup, 2), warnings_vi=tuple(warnings),
    )
