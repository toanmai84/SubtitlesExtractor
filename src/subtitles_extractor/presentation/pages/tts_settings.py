"""[v3.23.695] Lưu, khôi phục và đặt lại cài đặt trang Lồng tiếng.

Tách nguyên văn khỏi ``src/subtitles_extractor/presentation/pages/tts_page.py`` (U6). Lớp trộn chỉ chứa phương thức; widget, tín hiệu và trạng thái vẫn thuộc ``TTSPage``.
"""

from __future__ import annotations

import logging

from PySide6.QtWidgets import QMessageBox

from subtitles_extractor.presentation.pages.tts_constants import (
    _DEFAULT_MAX_SPEED,
    _ENGINE_EDGE,
)

logger = logging.getLogger("subtitles_extractor.presentation.pages.tts_page")



class TTSSettingsMixin:
    """Lưu, khôi phục và đặt lại cài đặt trang Lồng tiếng."""

    def _save_settings(self) -> None:
        s = self._settings
        # [v3.23.195] Lưu theo engine_id (bền vững) thay vì index — index lệch khi danh
        # sách engine thay đổi giữa các phiên bản (chính là bug "engine một đàng, cấu
        # hình một nẻo" sau khi gỡ/thêm engine).
        s.setValue("engine_id", self._engine_combo.currentData() or _ENGINE_EDGE)
        s.setValue("edge_lang", self._edge_lang.currentData())
        s.setValue("edge_voice", self._edge_voice.currentText())
        s.setValue("gemini_model", self._gemini_model.currentText())
        s.setValue("gemini_voice", self._gemini_voice.currentText())
        s.setValue("gemini_style", self._gemini_style.toPlainText())
        s.setValue("gemini_affective", self._gemini_affective.isChecked())
        s.setValue("gemini_line_emotion", self._gemini_line_emotion.isChecked())
        s.setValue("gemini_temperature", self._gemini_temperature.value())
        s.setValue("vieneu_mode", self._vieneu_mode.currentData() or "standard")
        s.setValue("vieneu_emotion", self._vieneu_emotion.currentData() or "natural")
        s.setValue("vieneu_emotion_cues", self._vieneu_emotion_cues.isChecked())
        s.setValue("vieneu_int8", self._vieneu_int8.isChecked())
        # Chỉ ghi đè giọng đã lưu khi danh sách giọng ĐÃ nạp (combo phản ánh lựa chọn
        # thật); chưa nạp -> giữ nguyên giá trị cũ, tránh mất giọng người dùng đã chọn.
        if getattr(self, "_vieneu_voices_loaded", False):
            s.setValue("vieneu_voice", self._vieneu_voice.currentData() or "")
        s.setValue("vieneu_ref_path", self._vieneu_ref_path.text())
        s.setValue("vieneu_force_cpu", self._vieneu_force_cpu.isChecked())
        s.setValue("speed", self._speed.value())
        s.setValue("max_speed", self._max_speed.value())
        s.setValue("retry", self._retry.value())
        s.setValue("retry_delay", self._retry_delay.value())
        s.setValue("normalize", self._normalize.isChecked())
        s.setValue("incremental", self._incremental.isChecked())
        s.setValue("voice_clarity", self._voice_clarity.isChecked())
        s.setValue("high_quality", self._high_quality.isChecked())
        s.setValue("target_lufs", self._lufs.currentData())
        s.setValue("output_format", self._fmt.currentData())
        s.setValue("output_bitrate", self._bitrate.currentData())
        s.setValue("wav_subtype", self._wav_subtype.currentData())
        s.setValue("device", self._device.currentText())
        s.setValue("clean_tags", self._clean_tags.isChecked())
        s.setValue("double_pass", self._double_pass.isChecked())
        s.setValue("dialog_pause_ms", self._dialog_pause.value())
        s.setValue("max_overlap_ms", self._max_overlap.value())
        s.setValue("skip_overlap_ms", self._skip_overlap.value())
        s.setValue("elastic_timing", self._elastic_timing.isChecked())
        s.setValue("max_drift_ms", self._max_drift.value())
        s.setValue("lead_in_ms", self._lead_in.value())
        s.setValue("anchor_gap_s", self._anchor_gap.value())
        s.setValue("max_segment_s", self._max_segment.value())
        s.setValue("comfort_ratio", self._comfort_ratio.value())
        s.setValue("min_pause_pct", self._min_pause.value())
        s.setValue("max_intra_gap_s", self._max_intra_gap.value())
        s.setValue("allow_overlap", self._allow_overlap.isChecked())
        s.setValue("timing_strategy", self._strategy.currentData())
        s.setValue("min_stretch", self._min_stretch.value())
        s.setValue("last_extend_ms", self._last_extend.value())
        s.setValue("skip_paren", self._skip_paren.isChecked())
        s.setValue("skip_square", self._skip_square.isChecked())
        s.setValue("skip_curly", self._skip_curly.isChecked())
        s.setValue("skip_music_pair", self._skip_music_pair.isChecked())
        s.setValue("skip_music_line", self._skip_music_line.isChecked())
        s.setValue("edge_concurrency", self._edge_concurrency.value())
        s.sync()

    def _restore_settings(self) -> None:
        s = self._settings
        # [v3.23.195] Restore theo engine_id (bền vững). Tương thích ngược: bản cũ chỉ
        # có "engine_idx" (index) — index có thể lệch sau khi danh sách engine thay đổi
        # nên KHÔNG dùng lại; thiếu engine_id -> mặc định Edge (an toàn).
        saved_engine = s.value("engine_id", _ENGINE_EDGE, type=str)
        engine_pos = self._engine_combo.findData(saved_engine)
        self._engine_combo.setCurrentIndex(engine_pos if engine_pos >= 0 else 0)
        # Edge
        lang = s.value("edge_lang", "vi-VN", type=str)
        for i in range(self._edge_lang.count()):
            if self._edge_lang.itemData(i) == lang:
                self._edge_lang.setCurrentIndex(i); break
        self._update_edge_speakers()
        if v := s.value("edge_voice", type=str):
            self._edge_voice.setCurrentText(v)
        # Gemini
        if m := s.value("gemini_model", type=str):
            idx = self._gemini_model.findText(m)
            if idx >= 0: self._gemini_model.setCurrentIndex(idx)
        if v := s.value("gemini_voice", type=str):
            idx = self._gemini_voice.findText(v)
            if idx >= 0: self._gemini_voice.setCurrentIndex(idx)
        if st := s.value("gemini_style", type=str):
            self._gemini_style.setPlainText(st)
        if s.contains("gemini_affective"):
            self._gemini_affective.setChecked(s.value("gemini_affective", type=bool))
        if s.contains("gemini_line_emotion"):
            self._gemini_line_emotion.setChecked(s.value("gemini_line_emotion", type=bool))
        if s.contains("gemini_temperature"):
            self._gemini_temperature.setValue(
                s.value("gemini_temperature", type=float)
            )
        self._on_gemini_model_changed()
        # VieNeu
        if m := s.value("vieneu_mode", type=str):
            mode_pos = self._vieneu_mode.findData(m)
            if mode_pos >= 0:
                self._vieneu_mode.setCurrentIndex(mode_pos)
        if e := s.value("vieneu_emotion", type=str):
            emo_pos = self._vieneu_emotion.findData(e)
            if emo_pos >= 0:
                self._vieneu_emotion.setCurrentIndex(emo_pos)
        if p := s.value("vieneu_ref_path", type=str):
            self._vieneu_ref_path.setText(p)
        if s.contains("vieneu_force_cpu"):
            self._vieneu_force_cpu.setChecked(s.value("vieneu_force_cpu", type=bool))
        if s.contains("vieneu_int8"):
            self._vieneu_int8.setChecked(s.value("vieneu_int8", type=bool))
        if s.contains("vieneu_emotion_cues"):
            self._vieneu_emotion_cues.setChecked(s.value("vieneu_emotion_cues", type=bool))
        # Giọng VieNeu: danh sách nạp lazy (khi mở panel) — lưu tạm để khôi phục sau nạp.
        self._vieneu_saved_voice = s.value("vieneu_voice", "", type=str)
        # Common
        for key, spin, default in [
            ("speed", self._speed, 1.0), ("max_speed", self._max_speed, 1.8),
            ("retry_delay", self._retry_delay, 1.0),
        ]:
            if s.contains(key): spin.setValue(s.value(key, type=float))
        # [v3.23.661] Nâng mặc định cũ 2,0× → 1,5× đúng một lần (chỉ khi còn giữ mặc định cũ).
        if not s.contains("max_speed_v661"):
            if s.contains("max_speed") and abs(s.value("max_speed", type=float) - 2.0) < 1e-6:
                self._max_speed.setValue(_DEFAULT_MAX_SPEED)
            s.setValue("max_speed_v661", True)
        if s.contains("retry"): self._retry.setValue(s.value("retry", type=int))
        if s.contains("normalize"): self._normalize.setChecked(s.value("normalize", type=bool))
        if s.contains("incremental"): self._incremental.setChecked(s.value("incremental", type=bool))
        if s.contains("voice_clarity"): self._voice_clarity.setChecked(s.value("voice_clarity", type=bool))
        if s.contains("high_quality"): self._high_quality.setChecked(s.value("high_quality", type=bool))
        if s.contains("target_lufs"):
            _li = self._lufs.findData(s.value("target_lufs", type=float))
            if _li >= 0: self._lufs.setCurrentIndex(_li)
        if fv := s.value("output_format", type=str):
            _fi = self._fmt.findData(fv)
            if _fi >= 0: self._fmt.setCurrentIndex(_fi)
        if (bv := s.value("output_bitrate", type=int)):
            _bi = self._bitrate.findData(bv)
            if _bi >= 0: self._bitrate.setCurrentIndex(_bi)
        if sv := s.value("wav_subtype", type=str):
            _si = self._wav_subtype.findData(sv)
            if _si >= 0: self._wav_subtype.setCurrentIndex(_si)
        if d := s.value("device", type=str):
            idx = self._device.findText(d)
            if idx >= 0: self._device.setCurrentIndex(idx)
        if o := s.value("output", type=str):
            self._output.setText(o)
        for key, spin, _default in [
            ("dialog_pause_ms", self._dialog_pause, 300),
            ("max_overlap_ms", self._max_overlap, 500),
            ("skip_overlap_ms", self._skip_overlap, 0),
        ]:
            if s.contains(key):
                val = s.value(key, type=int)
                # Migration: skip_overlap_ms default cũ là 2000 → đổi sang 0 (disabled)
                if key == "skip_overlap_ms" and val == 2000:
                    val = 0
                spin.setValue(val)
        for key, chk, _default in [
            ("clean_tags", self._clean_tags, True),
            ("double_pass", self._double_pass, True),
            ("elastic_timing", self._elastic_timing, True),
        ]:
            if s.contains(key): chk.setChecked(s.value(key, type=bool))
        if s.contains("max_drift_ms"):
            self._max_drift.setValue(s.value("max_drift_ms", type=int))
        if s.contains("lead_in_ms"):
            self._lead_in.setValue(s.value("lead_in_ms", type=int))
        if s.contains("anchor_gap_s"): self._anchor_gap.setValue(s.value("anchor_gap_s", type=float))
        if s.contains("max_segment_s"): self._max_segment.setValue(s.value("max_segment_s", type=float))
        if s.contains("comfort_ratio"): self._comfort_ratio.setValue(s.value("comfort_ratio", type=float))
        if s.contains("min_pause_pct"): self._min_pause.setValue(s.value("min_pause_pct", type=int))
        if s.contains("max_intra_gap_s"): self._max_intra_gap.setValue(s.value("max_intra_gap_s", type=float))
        if s.contains("allow_overlap"): self._allow_overlap.setChecked(s.value("allow_overlap", type=bool))
        if s.contains("timing_strategy"):
            i = self._strategy.findData(s.value("timing_strategy", type=str))
            if i >= 0: self._strategy.setCurrentIndex(i)
        if s.contains("min_stretch"): self._min_stretch.setValue(s.value("min_stretch", type=float))
        if s.contains("last_extend_ms"):
            self._last_extend.setValue(s.value("last_extend_ms", type=int))
        for _key, _cb in (("skip_paren", self._skip_paren), ("skip_square", self._skip_square),
                          ("skip_curly", self._skip_curly), ("skip_music_pair", self._skip_music_pair),
                          ("skip_music_line", self._skip_music_line)):
            if s.contains(_key):
                _cb.setChecked(s.value(_key, type=bool))
        if s.contains("edge_concurrency"):
            self._edge_concurrency.setValue(s.value("edge_concurrency", type=int))

    def _on_reset_defaults(self) -> None:
        """Đặt lại toàn bộ cấu hình TTS về mặc định khuyến nghị."""
        # [v3.23.112] Reset ghi đè mọi tinh chỉnh -> xác nhận tránh lỡ tay.
        confirm = QMessageBox.question(
            self, self._translator.translate("tts.dlg_reset_title"),
            self._translator.translate("tts.dlg_reset_body"),
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self._speed.setValue(1.1)
        self._max_speed.setValue(_DEFAULT_MAX_SPEED)
        self._retry.setValue(10)
        self._retry_delay.setValue(2.0)
        self._normalize.setChecked(True)
        self._voice_clarity.setChecked(True)
        self._high_quality.setChecked(True)
        _li = self._lufs.findData(-23.0)
        if _li >= 0:
            self._lufs.setCurrentIndex(_li)
        self._device.setCurrentText("auto")
        self._clean_tags.setChecked(True)
        self._double_pass.setChecked(True)
        self._dialog_pause.setValue(100)
        self._max_overlap.setValue(300)
        self._skip_overlap.setValue(0)
        self._elastic_timing.setChecked(True)
        self._max_drift.setValue(300)
        self._lead_in.setValue(300)
        self._anchor_gap.setValue(0.7)
        self._max_segment.setValue(10)
        self._comfort_ratio.setValue(1.25)
        self._min_pause.setValue(35)
        self._max_intra_gap.setValue(0.5)
        self._allow_overlap.setChecked(True)
        self._min_stretch.setValue(0.75)
        self._last_extend.setValue(300)
        self._edge_concurrency.setValue(16)
        for cb in (self._skip_paren, self._skip_square, self._skip_curly,
                   self._skip_music_pair, self._skip_music_line):
            cb.setChecked(True)
        self._show_success(self._translator.translate("tts.toast_restored_t"), self._translator.translate("tts.toast_restored_b"))


__all__ = ["TTSSettingsMixin"]
