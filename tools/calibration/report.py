"""Sinh báo cáo hiệu chuẩn dạng Markdown gọn cho người đọc."""

from __future__ import annotations

from tools.calibration.metrics import SubtitleScore
from tools.calibration.recursive_calibrator import CalibrationOutcome


def render_subtitle_report(
    outcome: CalibrationOutcome,
    baseline_detail: SubtitleScore,
    tuned_detail: SubtitleScore,
) -> str:
    """Tạo báo cáo Markdown so sánh baseline ↔ sau hiệu chuẩn build phụ đề."""
    lines: list[str] = []
    lines.append("# Báo cáo hiệu chuẩn build phụ đề\n")
    delta = outcome.best_score - outcome.baseline_score
    lines.append(
        f"- Điểm tổng hợp: baseline **{outcome.baseline_score:.4f}** → "
        f"tốt nhất **{outcome.best_score:.4f}** (Δ {delta:+.4f})"
    )
    lines.append(f"- Số lần đánh giá: {outcome.evaluations}")
    lines.append(f"- Cải thiện so với phiên trước: {'CÓ' if outcome.improved else 'KHÔNG'}\n")

    lines.append("## Chỉ số chi tiết\n")
    lines.append("| Chỉ số | Baseline | Sau hiệu chuẩn |")
    lines.append("|--------|----------|----------------|")
    lines.append(
        f"| Exact-match | {baseline_detail.exact_match_rate:.1%} "
        f"| {tuned_detail.exact_match_rate:.1%} |"
    )
    lines.append(
        f"| Recall | {baseline_detail.recall:.1%} | {tuned_detail.recall:.1%} |"
    )
    lines.append(
        f"| CER trung bình | {baseline_detail.average_cer:.4f} "
        f"| {tuned_detail.average_cer:.4f} |"
    )
    lines.append(
        f"| Spurious | {baseline_detail.spurious_rate:.1%} "
        f"| {tuned_detail.spurious_rate:.1%} |"
    )

    lines.append("\n## Bộ tham số tốt nhất\n")
    for name, value in sorted(outcome.best_assignment.items()):
        lines.append(f"- `{name}` = {value}")
    return "\n".join(lines)


def render_roi_report(outcome: CalibrationOutcome, has_labels: bool) -> str:
    """Tạo báo cáo Markdown cho hiệu chuẩn ROI."""
    lines: list[str] = []
    lines.append("# Báo cáo hiệu chuẩn ROI\n")
    mode = "IoU so nhãn thật" if has_labels else "proxy coverage × compactness"
    lines.append(f"- Chế độ chấm điểm: **{mode}**")
    lines.append(
        f"- Điểm: baseline **{outcome.baseline_score:.4f}** → "
        f"tốt nhất **{outcome.best_score:.4f}** "
        f"(Δ {outcome.best_score - outcome.baseline_score:+.4f})"
    )
    lines.append(f"- Số lần đánh giá: {outcome.evaluations}\n")
    lines.append("## Bộ tham số tốt nhất\n")
    for name, value in sorted(outcome.best_assignment.items()):
        lines.append(f"- `{name}` = {value}")
    if not has_labels:
        lines.append(
            "\n> ⚠️ Chưa có nhãn dải phụ đề thật — điểm là *proxy*. Để hiệu chuẩn "
            "ROI chính xác (IoU), cần cung cấp toạ độ dải phụ đề thật cho mỗi video."
        )
    return "\n".join(lines)
