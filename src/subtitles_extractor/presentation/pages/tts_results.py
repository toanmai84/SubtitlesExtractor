"""[v3.23.695] Bảng kết quả lồng tiếng: điền bảng, lọc dòng, chọn/sửa dòng, nghe thử từng câu.

Tách nguyên văn khỏi ``src/subtitles_extractor/presentation/pages/tts_page.py`` (U6). Lớp trộn chỉ chứa phương thức; widget, tín hiệu và trạng thái vẫn thuộc ``TTSPage``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt

from subtitles_extractor.domain.ports.subtitle_tts_port import TTSSegmentResult
from subtitles_extractor.presentation.pages.tts_constants import _OVERLAP_WARN_S
from subtitles_extractor.presentation.pages.tts_line_edits import (
    editable_text_item,
    format_row_time,
    read_only_item,
    row_head_item,
    selected_positions,
)

logger = logging.getLogger("subtitles_extractor.presentation.pages.tts_page")



class TTSResultsMixin:
    """Bảng kết quả lồng tiếng: điền bảng, lọc dòng, chọn/sửa dòng, nghe thử từng câu."""

    def _selected_event_indices(self) -> list[int]:
        """[v3.23.450] event_index của các dòng đang chọn (đọc từ UserRole cột 0)."""
        return selected_positions(self._table)

    def _result_by_event_index(self, event_index: int):
        for r in self._last_results:
            if int(r.event_index) == event_index:
                return r
        return None

    def _on_result_selection_changed(self) -> None:
        """Bật/tắt nút Nghe & TTS-lại theo dòng đang chọn."""
        if not hasattr(self, "_btn_play_sel"):
            return
        sel = self._selected_event_indices()
        busy = self._view_model.is_busy if hasattr(self._view_model, "is_busy") else False
        # Nghe: đúng 1 dòng có audio segment. TTS lại: ≥1 dòng, không bận.
        one = len(sel) == 1
        has_audio = False
        if one:
            r = self._result_by_event_index(sel[0])
            has_audio = bool(r and r.segment_audio_path)
        self._btn_play_sel.setEnabled(one and has_audio and not busy)
        # [v3.23.598] Chỉ khi lần TTS gần nhất giữ audio từng dòng ([v3.23.655] mọi engine). Bản trước bật
        # cả sau Edge/Gemini: tổng hợp xong các dòng chọn mới báo lỗi ghép.
        can_resynth = bool(getattr(self._view_model, "can_resynthesize", True))
        self._btn_resynth.setEnabled(bool(sel) and not busy and can_resynth)
        self._btn_resynth.setToolTip(self._translator.translate(
            "tts.tip_resynth" if can_resynth or not self._last_results
            else "tts.tip_resynth_unsupported"
        ))
        if hasattr(self, "_btn_clean_line_audio"):
            self._btn_clean_line_audio.setEnabled(self._line_audio_bytes > 0 and not busy)
        if hasattr(self, "_btn_check_pron"):
            self._btn_check_pron.setEnabled(
                bool(self._last_results) and not busy and not self._pronunciation.is_running)
        # [v3.23.459] Nút chọn dòng bất thường: bật khi có ≥1 dòng bất thường + không bận.
        if hasattr(self, "_btn_select_abnormal"):
            n_abnormal = len(self._abnormal_event_indices())
            self._btn_select_abnormal.setEnabled(n_abnormal > 0 and not busy)
            self._btn_select_abnormal.setText(
                f"⚠ Chọn dòng bất thường ({n_abnormal})" if n_abnormal
                else "⚠ Chọn dòng bất thường"
            )

    def _on_result_item_edited(self, item) -> None:
        """[v3.23.453] Ghi nội dung SỬA (cột 2) vào dòng nguồn để TTS lại đọc ĐÚNG.

        LƯU Ý: ``TTSSegmentResult`` là dataclass FROZEN → KHÔNG gán ``r.text=`` (ném
        FrozenInstanceError, khiến hàm dừng giữa chừng và bỏ qua cập nhật source — chính là
        bug 'TTS lại với nội dung cũ'). Cập nhật source TRƯỚC, rồi thay result bằng ``replace``.
        """
        if item.column() != 2:
            return
        row0 = self._table.item(item.row(), 0)
        if row0 is None:
            return
        ev_idx = row0.data(Qt.ItemDataRole.UserRole)
        if ev_idx is None:
            return
        ev_idx = int(ev_idx)
        new_text = item.text().strip()
        # 1. QUAN TRỌNG: cập nhật dòng nguồn để lần TTS lại đọc nội dung mới.
        self._view_model.update_event_text(ev_idx, new_text)
        # 2. Đồng bộ hiển thị trong _last_results (frozen → thay bằng bản replace).
        from dataclasses import replace as _dc_replace

        for i, r in enumerate(self._last_results):
            if int(r.event_index) == ev_idx:
                try:
                    self._last_results[i] = _dc_replace(r, text=new_text)
                except (TypeError, ValueError):
                    pass
                break

    def _on_play_selected(self) -> None:
        """Phát audio segment của dòng đang chọn."""
        sel = self._selected_event_indices()
        if len(sel) != 1:
            return
        r = self._result_by_event_index(sel[0])
        if not r or not r.segment_audio_path:
            self._show_warning("Chưa có audio", "Dòng này chưa có audio để nghe.")
            return
        path = Path(r.segment_audio_path)
        if not path.exists():
            self._show_warning("Thiếu file", f"Không tìm thấy: {path.name}")
            return
        self._play_audio_file(path)

    def _play_audio_file(self, path: Path) -> None:
        """[v3.23.455] Nghe thử một dòng ĐÁNG TIN CẬY.

        Hai lớp chống lỗi:
        1. ``setSource()`` bất đồng bộ → phát trong ``_on_media_status`` khi ĐÃ nạp xong +
           ``setPosition(0)`` (luôn phát từ đầu, sửa "chỉ nghe vài chữ cuối" khi phát lại).
        2. Thiết bị âm thanh có độ trễ khởi động → tạo WAV xem trước = [im lặng ~0.8s][segment]
           để thiết bị "nóng máy" xong mới tới tiếng nói (tránh nuốt đầu câu / rè "tẹt tẹt").
        """
        try:
            from PySide6.QtCore import QUrl
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
        except ImportError:
            self._show_warning("Thiếu QtMultimedia", "Không phát được audio trên máy này.")
            return
        # Ưu tiên phát bản có đệm im lặng; nếu tạo lỗi thì phát thẳng file gốc.
        play_path = self._make_preview_wav(path) or path
        if self._audio_player is None:
            self._audio_out = QAudioOutput()
            self._audio_out.setVolume(1.0)
            self._audio_player = QMediaPlayer()
            self._audio_player.setAudioOutput(self._audio_out)
            self._audio_player.mediaStatusChanged.connect(self._on_media_status)
            self._audio_player.errorOccurred.connect(self._on_media_error)
        self._audio_player.stop()
        self._pending_play = True
        # Xoá source cũ trước → buộc phát lại được cùng một file (không bị cache "đã nạp").
        self._audio_player.setSource(QUrl())
        self._audio_player.setSource(QUrl.fromLocalFile(str(play_path)))

    def _make_preview_wav(self, path: Path) -> Path | None:
        """Tạo WAV xem trước = im lặng + segment. Luân phiên 2 tên file tạm để tránh khoá file
        khi phát lại nhanh trên Windows. Trả None nếu lỗi (gọi phát thẳng file gốc)."""
        try:
            import tempfile

            import soundfile as sf

            from subtitles_extractor.infrastructure.tts.audio_preview import (
                prepend_silence,
            )

            audio, sr = sf.read(str(path), dtype="float32")
            padded = prepend_silence(audio, sr, self._PREVIEW_SILENCE_S)
            self._preview_idx = (getattr(self, "_preview_idx", 0) + 1) % 2
            tmp = Path(tempfile.gettempdir()) / f"subext_tts_preview_{self._preview_idx}.wav"
            sf.write(str(tmp), padded, sr, subtype="PCM_16")
            return tmp
        except (OSError, RuntimeError, ImportError, ValueError) as exc:
            logger.debug("Không tạo được WAV nghe thử có đệm im lặng: %s", exc)
            return None

    def _on_media_status(self, status) -> None:
        """Phát ngay khi media ĐÃ NẠP XONG (chống race của setSource bất đồng bộ)."""
        from PySide6.QtMultimedia import QMediaPlayer

        loaded = (
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
        )
        if self._pending_play and status in loaded:
            self._pending_play = False
            if self._audio_player is not None:
                self._audio_player.setPosition(0)  # luôn phát từ ĐẦU
                self._audio_player.play()

    def _on_media_error(self, error, error_string: str = "") -> None:
        """Báo lỗi phát (thay vì im lặng)."""
        self._pending_play = False
        logger.warning("Lỗi phát audio dòng: %s", error_string or error)

    def _commit_active_edit(self) -> None:
        """[v3.23.453] Buộc COMMIT ô đang sửa (nếu có) TRƯỚC khi đọc trạng thái.

        Lỗi Qt kinh điển: bấm nút trong lúc còn con trỏ trong ô đang sửa → Qt xử lý click
        TRƯỚC khi editor commit → nội dung sửa CHƯA vào model → TTS lại đọc nội dung CŨ. Gọi
        hàm này để đẩy giá trị editor vào model (kích hoạt itemChanged → update_event_text)
        trước khi tiếp tục.
        """
        try:
            from PySide6.QtWidgets import (
                QAbstractItemDelegate,
                QAbstractItemView,
                QApplication,
            )
            if self._table.state() != QAbstractItemView.State.EditingState:
                return
            editor = QApplication.focusWidget()
            if editor is not None:
                try:
                    self._table.commitData(editor)
                    self._table.closeEditor(
                        editor, QAbstractItemDelegate.EndEditHint.NoHint
                    )
                except (RuntimeError, TypeError, AttributeError):
                    pass
            # Fallback: nếu VẪN đang sửa, rời ô hiện hành sang cột 0 (không editable) → Qt tự
            # commit editor cột 2 (đổi ô hiện hành luôn kích hoạt commit).
            if self._table.state() == QAbstractItemView.State.EditingState:
                row = self._table.currentRow()
                if row >= 0:
                    self._table.setCurrentCell(row, 0)
        except (ImportError, RuntimeError, TypeError, AttributeError) as exc:
            logger.debug("Không commit được editor đang mở: %s", exc)

    def _populate_table(self, results: list[TTSSegmentResult]) -> None:
        try:
            from subtitles_extractor.infrastructure.tts.edge_tts_adapter import (
                _EDGE_API_SPEED_MAX,
            )
        except ImportError:
            _EDGE_API_SPEED_MAX = 1.5

        # [2.1] Khoá sắp xếp tự động + tín hiệu nội bộ + vẽ màn hình trong lúc nạp
        # hàng nghìn ô. Nếu để bật, mỗi setItem sẽ kích hoạt re-sort và phát tín hiệu
        # itemChanged → chậm theo cấp số nhân. Bật lại toàn bộ ở finally.
        self._table.setSortingEnabled(False)
        self._table.blockSignals(True)
        self._table.setUpdatesEnabled(False)
        try:
            # [SỬA HIỆU NĂNG QUAN TRỌNG] Xoá sạch nội dung bảng ở tầng C++ bằng một
            # lệnh duy nhất TRƯỚC khi đổ dữ liệu mới. Nếu không, việc gọi setItem đè
            # lên hàng nghìn ô cũ buộc Qt huỷ/tạo lại từng ô — gây treo UI khi chạy
            # TTS lần thứ hai trở đi (bảng đã đầy dữ liệu từ lần trước).
            self._table.clearContents()
            self._table.setRowCount(0)

            self._table.setRowCount(len(results))
            # [v3.23.598] Ô chỉ đọc (trước đây mọi ô đều nhấp đúp sửa được, kể cả "#",
            # "Thời gian", "Trạng thái"); chỉ cột Nội dung sửa được.
            _it = read_only_item
            for row, r in enumerate(results):
                ts = format_row_time(r.start_sec)
                dur = f"{r.audio_duration_s:.2f}s" if not r.was_skipped else "—"
                spd = f"{r.speed_used:.2f}×" if not r.was_skipped else "—"

                # Cột "Loại": phân biệt cách tốc độ được tạo ra
                if r.was_skipped:
                    kind = "—"
                elif r.speed_used <= 1.02:
                    kind = self._translator.translate("tts.kind_base")           # tốc độ cơ bản, không tăng
                elif r.speed_used <= _EDGE_API_SPEED_MAX + 0.02:
                    kind = self._translator.translate("tts.kind_api")             # tăng qua Edge TTS rate param
                elif r.speed_used < 2.5:
                    kind = self._translator.translate("tts.kind_apits")          # API + time-stretch vừa
                else:
                    kind = self._translator.translate("tts.kind_tshigh")          # time-stretch nặng (≥2.5×, chất lượng giảm)

                if r.was_skipped:
                    st = self._translator.translate("tts.st_skip") + (f": {r.error_msg[:22]}" if r.error_msg else "")
                elif r.was_truncated:
                    st = self._translator.translate("tts.st_cut")
                elif r.overlap_s > _OVERLAP_WARN_S:
                    # [v3.23.206] Chồng đáng kể sang câu sau — gợi ý rút gọn bản dịch.
                    st = self._translator.translate("tts.st_overlap").replace("{s}", f"{r.overlap_s:.1f}")
                else:
                    st = self._translator.translate("tts.st_ok")
                # [v3.23.459] Đánh dấu audio DÀI BẤT THƯỜNG (model ngân dài) — thêm cảnh báo vào
                # trạng thái để người dùng lọc/nghe lại/TTS lại nhanh (không phải nghe dò).
                is_abnormal = getattr(r, "abnormal_length", False) and not r.was_skipped
                if is_abnormal:
                    st = f"⚠ {st} · dài bất thường"

                # [v3.23.450] Lưu event_index vào row để map khi nghe lại / TTS lại dòng chọn.
                item0 = row_head_item(int(r.event_index))
                item0.setText(str(row + 1))
                self._table.setItem(row, 0, item0)
                self._table.setItem(row, 1, _it(ts))
                # [v3.23.450] Cột Nội dung CHO PHÉP SỬA (double-click) để chỉnh rồi TTS lại.
                self._table.setItem(row, 2, editable_text_item(r.text))
                self._table.setItem(row, 3, _it(dur))
                self._table.setItem(row, 4, _it(spd))
                self._table.setItem(row, 5, _it(kind))
                st_item = _it(st)
                self._table.setItem(row, 6, st_item)
                # Tô nền vàng nhạt cho ô trạng thái dòng bất thường (nổi bật, dễ thấy).
                if is_abnormal:
                    from PySide6.QtGui import QColor
                    st_item.setBackground(QColor(255, 244, 200))
                    st_item.setForeground(QColor(140, 90, 0))
        finally:
            # Luôn bật lại updates dù có lỗi giữa chừng — tránh bảng "đóng băng".
            self._table.setUpdatesEnabled(True)
            self._table.blockSignals(False)
            # Giữ KHÔNG cho phép sort lại bằng click header để bảo toàn thứ tự dòng
            # phụ đề theo thời gian (sort theo cột sẽ phá vỡ trật tự timeline).
            self._table.setSortingEnabled(False)
        if len(results) <= 1500:
            self._table.resizeRowsToContents()
        self._apply_row_filter()
        # [v3.23.598] Cột "Người nói · Giọng" + tô dòng đã sửa chưa TTS lại.
        self._line_edits.refresh()

    def _apply_row_filter(self) -> None:
        """Ẩn/hiện hàng theo bộ lọc đang chọn (dùng setRowHidden — không tạo lại ô)."""
        mode = self._row_filter.currentData() if hasattr(self, "_row_filter") else "all"
        rs = self._last_results
        if not rs:
            if hasattr(self, "_lbl_filter_count"):
                self._lbl_filter_count.setText("")
            return
        shown = 0
        self._table.setUpdatesEnabled(False)
        try:
            for row, r in enumerate(rs):
                if mode == "skipped":
                    visible = r.was_skipped
                elif mode == "truncated":
                    visible = r.was_truncated
                elif mode == "fast":
                    visible = (not r.was_skipped) and r.speed_used >= 1.8
                elif mode == "issues":
                    visible = (
                        r.was_skipped or r.was_truncated
                        or ((not r.was_skipped) and r.speed_used >= 1.8)
                        or ((not r.was_skipped) and r.overlap_s > _OVERLAP_WARN_S)
                    )
                else:  # "all"
                    visible = True
                self._table.setRowHidden(row, not visible)
                if visible:
                    shown += 1
        finally:
            self._table.setUpdatesEnabled(True)
        self._lbl_filter_count.setText(
            "" if mode == "all" else self._translator.translate("tts.res_showing").replace("{shown}", str(shown)).replace("{total}", str(len(rs)))
        )


__all__ = ["TTSResultsMixin"]
