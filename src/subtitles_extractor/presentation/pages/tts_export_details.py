"""[v3.23.692] Xuất chi tiết lượt lồng tiếng (từng dòng) và tệp cấu hình gỡ lỗi kèm theo.

Tách nguyên văn khỏi ``tts_page.py`` (U6). Lớp trộn chỉ chứa phương thức; widget, tín hiệu và trạng thái vẫn thuộc ``TTSPage``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from subtitles_extractor.application.services.dubbing_cast import CAST_MODE_TWO_VOICES
from subtitles_extractor.presentation.pages.tts_constants import (
    _EDGE_ONLY_FIELDS,
    _ENGINE_EDGE,
)
from subtitles_extractor.presentation.utils import safe_dialogs as _safe_dialogs

logger = logging.getLogger("subtitles_extractor.presentation.pages.tts_page")


class TTSExportDetailsMixin:
    """Xuất chi tiết lượt lồng tiếng (từng dòng) và tệp cấu hình gỡ lỗi kèm theo."""

    def _on_export_details(self) -> None:
        """Xuất CSV chi tiết kết quả TTS + file cấu hình kèm theo để debug."""
        if not self._last_results:
            self._show_warning(self._translator.translate("tts.toast_nodata_t"), self._translator.translate("tts.toast_run_tts_first"))
            return
        import csv
        import statistics as _st

        # [v3.23.428] Tên nhất quán có timestamp → mỗi lần xuất một tệp riêng, không ghi đè.
        from subtitles_extractor.application.services.diagnostic_naming import (
            build_diagnostic_filename,
        )

        _tts_default = build_diagnostic_filename(None, "tts_debug", "csv")
        path = _safe_dialogs.save_file(
            self, "Xuất chi tiết TTS", _tts_default, "CSV (*.csv)"
        )
        if not path:
            return

        def _b(v: bool) -> str:
            return "x" if v else ""

        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "STT", "Văn bản", "Loại",
                    "Start gốc (s)", "End gốc (s)",
                    "Start điều chỉnh (s)", "End điều chỉnh (s)", "Dời (ms)",
                    "Khung gốc (s)", "Khung mở rộng (s)", "Khung an toàn (s)",
                    "Pass1 (s)", "Pass2 (s)", "Dùng Pass2",
                    "Tốc độ scheduler (×)", "Tốc độ Edge API (×)", "Tốc độ dùng (×)",
                    "Thời lượng audio (s)", "Lấn (s)",
                    "Nén (cách)", "Tỉ lệ nén (×)", "Cắt (ms)",
                    "Nghỉ (ms)", "Dìm câu trước (s)",
                    "Bị cắt", "Bị bỏ", "Lỗi", "Giọng",
                ])
                last_request = getattr(self._view_model, "last_request", None)
                for i, r in enumerate(self._last_results, 1):
                    adj_start = r.adjusted_start_sec if r.adjusted_start_sec >= 0 else r.start_sec
                    adj_end = r.adjusted_end_sec if r.adjusted_end_sec >= 0 else r.end_sec
                    drift_ms = (adj_start - r.start_sec) * 1000.0
                    writer.writerow([
                        i, r.text, "Hội thoại" if r.is_dialog else "Thường",
                        f"{r.start_sec:.3f}", f"{r.end_sec:.3f}",
                        f"{adj_start:.3f}", f"{adj_end:.3f}", f"{drift_ms:.0f}",
                        f"{r.window_strict_s:.3f}", f"{r.window_ext_s:.3f}", f"{r.safe_window_s:.3f}",
                        f"{r.pass1_dur_s:.3f}", f"{r.pass2_dur_s:.3f}", _b(r.used_pass2),
                        f"{r.scheduled_speed:.3f}", f"{r.api_speed:.3f}", f"{r.speed_used:.3f}",
                        f"{r.audio_duration_s:.3f}", f"{r.overlap_s:.3f}",
                        r.stretch_method or "", f"{r.stretch_ratio:.3f}", f"{r.cut_amount_ms:.0f}",
                        f"{r.pause_ms:.0f}", f"{r.ducked_prev_s:.3f}",
                        _b(r.was_truncated), _b(r.was_skipped), r.error_msg or "",
                        # [v3.23.593] Giọng thật đọc dòng này (lồng tiếng: theo vai).
                        last_request.voice_at(
                            int(r.event_index), last_request.speaker or "mặc định"
                        ) if last_request is not None else "",
                    ])

            self._write_debug_config(Path(path), _st)
            self._show_success(
                self._translator.translate("tts.toast_exported_t"),
                self._translator.translate("tts.toast_exported_b")
                .replace("{name}", Path(path).name)
                .replace("{config}", Path(path).with_suffix('.config.txt').name),
            )
        except OSError as exc:
            self._show_warning(self._translator.translate("tts.toast_write_err_t"), str(exc))

    def _write_debug_config(self, csv_path: Path, _st) -> None:
        """Ghi file .config.txt: toàn bộ cài đặt + tóm tắt thống kê để dễ debug."""
        from datetime import datetime

        rs = self._last_results
        req = getattr(self._view_model, "last_request", None)
        engine = getattr(self._view_model, "last_engine_name", "") or "?"
        cfg_path = csv_path.with_suffix(".config.txt")

        ok = [r for r in rs if not r.was_skipped]
        speeds = [r.speed_used for r in ok if r.speed_used > 0]
        drifts = [
            ((r.adjusted_start_sec if r.adjusted_start_sec >= 0 else r.start_sec) - r.start_sec) * 1000.0
            for r in ok
        ]
        # [v3.23.601] Chỉ đếm chồng tiếng THẬT (≥1 ms), bỏ phần dư làm tròn sau nén.
        overlaps = [r.overlap_s for r in ok if r.has_audible_overlap]
        stretched = [r for r in ok if r.stretch_method]
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("CẤU HÌNH & TÓM TẮT TTS (debug)")
        lines.append(f"Thời điểm xuất : {datetime.now():%Y-%m-%d %H:%M:%S}")
        # [v3.23.714] Phiên bản ứng dụng: tệp debug TTS trước đây không cho biết bản nào sinh ra nó.
        from subtitles_extractor import __version__ as app_version

        lines.append(f"Phiên bản      : {app_version}")
        lines.append(f"Engine         : {engine}")
        lines.append("=" * 60)
        lines.append("\n── TÓM TẮT KẾT QUẢ ──")
        lines.append(f"Tổng dòng              : {len(rs)}")
        lines.append(f"Đọc thành công         : {len(ok)}")
        lines.append(f"Bị bỏ (không đọc)      : {sum(1 for r in rs if r.was_skipped)}")
        lines.append(f"Bị cắt đuôi            : {sum(1 for r in rs if r.was_truncated)}")
        lines.append(f"Có lỗi                 : {sum(1 for r in rs if r.error_msg)}")
        lines.append(f"Cần nén thêm (stretch) : {len(stretched)}")
        lines.append(f"Có chồng tiếng (lấn)   : {len(overlaps)}")
        if speeds:
            lines.append(
                f"Tốc độ đọc (×)         : min {min(speeds):.2f} | "
                f"median {_st.median(speeds):.2f} | max {max(speeds):.2f}"
            )
        if drifts:
            lines.append(
                f"Dời mốc (ms)           : min {min(drifts):.0f} | "
                f"median {_st.median(drifts):.0f} | max {max(drifts):.0f}"
            )
        if overlaps:
            lines.append(
                f"Lấn (s)                : tổng {sum(overlaps):.2f} | max {max(overlaps):.2f}"
            )

        lines.append("\n── CÀI ĐẶT ĐÃ DÙNG ──")
        if req is not None:
            def _fmt(v: object) -> str:
                if isinstance(v, bool):
                    return "Bật" if v else "Tắt"
                if isinstance(v, float):
                    return f"{v:g}"
                return str(v)

            fields = [
                ("Ngôn ngữ", "language"), ("Giọng đọc", "speaker"),
                ("Thiết bị", "device"), ("Số luồng song song", "edge_concurrency"),
                ("Tốc độ cơ bản (×)", "base_speed"), ("Tốc độ tối đa (×)", "max_speed"),
                ("Chiến lược timing", "timing_strategy"),
                ("Elastic timing", "elastic_timing"), ("Double Pass", "double_pass"),
                (self._translator.translate("tts.opt_hq"), "high_quality"),
                ("Dung sai cộng dồn (s)", "max_drift_s"), ("Ăn gian đầu (s)", "lead_in_s"),
                ("Kéo dài câu cuối (s)", "last_line_max_extend_s"),
                ("Cắt lấn (ms)", "max_overlap_ms"), ("Bỏ qua lấn (ms)", "skip_overlap_ms"),
                (self._translator.translate("tts.opt_overlap"), "allow_audio_overlap"),
                ("Giãn tối đa (×base)", "min_stretch_ratio"),
                ("Nén êm (×base)", "comfort_speed_ratio"),
                ("Nghỉ tối thiểu (tỉ lệ)", "min_pause_ratio"),
                ("Cách câu tối đa (s)", "max_intra_gap_s"),
                ("Neo cụm (s)", "anchor_gap_s"), ("Cụm tối đa (s)", "max_segment_s"),
                ("Nghỉ hội thoại (ms)", "dialog_pause_ms"),
                ("Chuẩn hoá loudness", "normalize"), ("Loudness mục tiêu (LUFS)", "target_lufs"),
                ("Định dạng xuất", "output_format"), ("Bitrate (kbps)", "output_bitrate_kbps"),
                ("Độ sâu WAV/FLAC", "wav_subtype"),
                (self._translator.translate("tts.opt_clarify"), "voice_clarity"), (self._translator.translate("tts.opt_strip_tags"), "clean_tags"),
                ("Số lần thử lại", "retry_count"), ("Delay thử lại (s)", "retry_delay_s"),
                ("Bỏ đọc ( )", "skip_paren"), ("Bỏ đọc [ ]", "skip_square"),
                ("Bỏ đọc { }", "skip_curly"),
                ("Bỏ đọc ♪..♪", "skip_music_pair"), ("Bỏ đọc dòng ♪", "skip_music_line"),
            ]
            # [v3.23.214] Engine hiện tại có đọc trường này không? Trước đây in TẤT CẢ
            # -> người dùng (và cả việc phân tích debug) tưởng "Elastic timing: Bật" có
            # tác dụng khi chạy VieNeu, thực tế bị bỏ qua hoàn toàn.
            is_edge = (self._engine_combo.currentData() or _ENGINE_EDGE) == _ENGINE_EDGE
            line_voices = tuple(getattr(req, "line_voices", ()) or ())
            if line_voices:
                distinct = sorted({voice for voice in line_voices if voice})
                # [v3.23.597] Ghi kèm cách phân giọng để soi chẩn đoán.
                two_voices = self._dubbing_panel.cast_mode() == CAST_MODE_TWO_VOICES
                lines.append(
                    f"{'Lồng tiếng':<24}: {len(distinct)} giọng cho "
                    f"{sum(1 for voice in line_voices if voice)}/{len(line_voices)} dòng "
                    f"({', '.join(distinct)})"
                    + (" — chế độ chỉ 2 giọng nam/nữ" if two_voices else "")
                )
            for label, attr in fields:
                if hasattr(req, attr):
                    value = _fmt(getattr(req, attr))
                    if not is_edge and attr in _EDGE_ONLY_FIELDS:
                        value += "   (không áp dụng cho engine này)"
                    lines.append(f"{label:<24}: {value}")
        else:
            lines.append("(Không có snapshot cấu hình — chạy lại TTS để ghi nhận.)")

        cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["TTSExportDetailsMixin"]
