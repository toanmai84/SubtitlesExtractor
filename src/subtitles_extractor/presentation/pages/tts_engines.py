"""[v3.23.695] Chọn engine TTS trên trang Lồng tiếng: đổi engine, nạp giọng VieNeu ở nền, chế độ/độ chính xác VieNeu, model Gemini, giọng Edge, gợi ý cài đặt.

Tách nguyên văn khỏi ``src/subtitles_extractor/presentation/pages/tts_page.py`` (U6). Lớp trộn chỉ chứa phương thức; widget, tín hiệu và trạng thái vẫn thuộc ``TTSPage``.
"""

from __future__ import annotations

import logging

from subtitles_extractor.infrastructure.tts.edge_tts_adapter import _EDGE_VOICE_MAP
from subtitles_extractor.presentation.pages.tts_constants import (
    _ENGINE_EDGE,
    _ENGINE_GEMINI,
    _ENGINE_VIENEU,
)
from subtitles_extractor.presentation.pages.tts_voice_loader import _VieNeuVoiceLoader
from subtitles_extractor.presentation.qt_compat import is_deleted
from subtitles_extractor.presentation.theme.styles import caption_style
from subtitles_extractor.presentation.utils import safe_dialogs as _safe_dialogs

logger = logging.getLogger("subtitles_extractor.presentation.pages.tts_page")



class TTSEnginesMixin:
    """Chọn engine TTS trên trang Lồng tiếng: đổi engine, nạp giọng VieNeu ở nền, chế độ/độ chính xác VieNeu, model Gemini, giọng Edge, gợi ý cài đặt."""

    def _on_vieneu_catalog_loaded(self, mode: str, catalog: list) -> None:
        """Giọng VieNeu (kèm giới tính) về từ luồng nền → cập nhật bảng phân vai."""
        self._vieneu_catalog[mode] = list(catalog)
        if (self._engine_combo.currentData() or _ENGINE_EDGE) == _ENGINE_VIENEU:
            self._dubbing.refresh_catalog()

    def _on_engine_changed(self, idx: int) -> None:
        engine_id = self._engine_combo.itemData(idx) or _ENGINE_EDGE
        stack_map = {_ENGINE_EDGE: 0, _ENGINE_GEMINI: 1, _ENGINE_VIENEU: 2}
        self._engine_stack.setCurrentIndex(stack_map.get(engine_id, 0))
        # max_speed dùng cho tất cả engines (time-stretch post-processing)
        self._max_speed.setVisible(True)
        self._sync_edge_only_controls(engine_id)
        # [v3.23.593] Mỗi engine một bộ giọng → bảng phân vai đổi theo (thẻ lồng tiếng
        # dựng SAU bộ chọn engine nên có thể chưa tồn tại khi tín hiệu phát lúc dựng
        # giao diện).
        if hasattr(self, "_dubbing"):
            self._dubbing.refresh_catalog()
        # [v3.23.195] Nạp giọng VieNeu CHỈ khi trang đang hiển thị (người dùng tương tác
        # thật). Nạp lúc khởi động (restore engine đã lưu) sẽ tải model ~9s CHẶN app mở
        # (log thực tế: model nạp trước cả MainWindow). Trang chưa hiện -> hoãn tới
        # showEvent.
        if (
            engine_id == _ENGINE_VIENEU
            and not getattr(self, "_vieneu_voices_loaded", False)
            and self.isVisible()
        ):
            self._load_vieneu_voices()
        else:
            # [v3.23.252] Rời VieNeu (hoặc giọng đã nạp xong) -> cập nhật lại trạng thái
            # nút Tổng hợp. Nếu trước đó nút bị khoá vì VieNeu đang nạp mà nay chuyển sang
            # Edge/Gemini (không cần giọng VieNeu), nút phải bật lại.
            self._update_action_states()

    def _sync_edge_only_controls(self, engine_id: str) -> None:
        """[v3.23.214] Bật/tắt các tuỳ chọn CHỈ Edge đọc theo engine đang chọn.

        VieNeu/Gemini bỏ qua hoàn toàn nhóm này; để chúng bật gây hiểu nhầm là có tác
        dụng (người dùng chỉnh vô ích, debug config in ra sai lệch).

        Args:
            engine_id: Engine đang chọn (``_ENGINE_EDGE`` / ``_ENGINE_GEMINI`` /
                ``_ENGINE_VIENEU``).
        """
        is_edge = engine_id == _ENGINE_EDGE
        # [v3.23.596] "Elastic timing" và "Nén êm" nay áp cho cả VieNeu/Gemini.
        for widget in (
            self._strategy, self._high_quality, self._double_pass, self._min_pause,
        ):
            widget.setEnabled(is_edge)
            edge_hint = self._translator.translate("tts.edge_only_hint")
            tip = widget.toolTip().split(edge_hint)[0]
            widget.setToolTip(tip if is_edge else tip + edge_hint)

    def _load_vieneu_voices(self) -> None:
        """Nạp danh sách giọng preset VieNeu (BACKGROUND thread) theo chế độ đang chọn.

        [v3.23.194] Mỗi chế độ (standard/v3turbo) có bộ giọng KHÁC NHAU (7 vs 10 giọng,
        tên khác) — nạp lại khi đổi chế độ, tránh "Voice not found".
        [v3.23.200] Chạy ở thread NỀN: nạp model ~9s trên UI thread làm app đứng hình.
        Trong lúc nạp combo hiển thị self._translator.translate("tts.voice_loading") và tạm khoá; không bao giờ có
        2 loader song song (đổi chế độ giữa chừng -> ghi nhận, nạp lại khi xong).
        """
        mode = self._vieneu_mode.currentData() or "standard"
        if self._voice_loader_running():
            # Đang nạp: handler khi xong sẽ tự so chế độ hiện tại và nạp lại nếu khác.
            return
        self._vieneu_voice.blockSignals(True)
        self._vieneu_voice.clear()
        self._vieneu_voice.addItem(self._translator.translate("tts.voice_loading"), userData="")
        self._vieneu_voice.setEnabled(False)
        self._vieneu_voice.blockSignals(False)
        # [v3.23.252] Khoá nút Tổng hợp trong lúc nạp giọng — tránh bấm chạy khi
        # giọng chưa sẵn sàng (sẽ lỗi). Chỉ khoá khi engine hiện tại là VieNeu.
        # Nút bật lại khi nạp xong/lỗi.
        if (self._engine_combo.currentData() or _ENGINE_EDGE) == _ENGINE_VIENEU:
            self._btn_gen.setEnabled(False)
        loader = _VieNeuVoiceLoader(
            self._view_model._container, mode, parent=self,
            precision=self._vieneu_precision(),
        )
        loader.voices_loaded.connect(self._on_vieneu_voices_loaded)
        loader.catalog_loaded.connect(self._on_vieneu_catalog_loaded)
        loader.load_failed.connect(self._on_vieneu_voices_failed)
        loader.finished.connect(loader.deleteLater)
        self._vieneu_voice_loader = loader
        loader.start()

    def _voice_loader_running(self) -> bool:
        """[v3.23.650] Loader giọng còn chạy? An toàn khi đối tượng C++ đã bị ``deleteLater`` (finished → deleteLater
        trong lúc Python còn giữ wrapper làm ``isRunning()`` ném "Internal C++ object already deleted")."""
        loader = getattr(self, "_vieneu_voice_loader", None)
        if loader is None or is_deleted(loader):
            return False
        return bool(loader.isRunning())

    def _on_vieneu_voices_loaded(self, mode: str, voice_ids: list) -> None:
        """Nhận danh sách giọng từ thread nền; bỏ qua kết quả STALE nếu đã đổi chế độ."""
        self._vieneu_voice_loader = None
        current_mode = self._vieneu_mode.currentData() or "standard"
        if mode != current_mode:
            self._load_vieneu_voices()  # kết quả cũ — nạp lại theo chế độ hiện tại
            return
        self._vieneu_voice.setEnabled(True)
        if not voice_ids:
            self._reset_vieneu_voice_combo()
            return
        # Ưu tiên: giọng đã lưu QSettings (combo đang là placeholder) -> mặc định.
        current = getattr(self, "_vieneu_saved_voice", "")
        self._vieneu_voice.blockSignals(True)
        self._vieneu_voice.clear()
        self._vieneu_voice.addItem(self._translator.translate("tts.voice_default"), userData="")
        for voice_id in voice_ids:
            self._vieneu_voice.addItem(voice_id, userData=voice_id)
        restore = self._vieneu_voice.findData(current)
        if restore < 0 and current:
            # [v3.23.626] Giọng lưu dưới tên cũ (VieNeu 3.8 đổi tên: "Minh Quân" → "Hải Đăng").
            from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
                resolve_voice_alias,
            )

            restore = self._vieneu_voice.findData(resolve_voice_alias(current))
        if restore >= 0:
            self._vieneu_voice.setCurrentIndex(restore)
        else:
            self._vieneu_voice.setCurrentIndex(0)  # giọng cũ không có -> về mặc định
        self._vieneu_voice.blockSignals(False)
        self._vieneu_voices_loaded = True
        # [v3.23.252] Nạp xong -> khôi phục trạng thái nút Tổng hợp theo điều
        # kiện chuẩn (có phụ đề + không bận), bù cho việc khoá nút lúc bắt đầu.
        self._update_action_states()

    def _on_vieneu_voices_failed(self, mode: str) -> None:
        """Nạp giọng thất bại (engine chưa cài/lỗi): trả combo về trạng thái mặc định."""
        self._vieneu_voice_loader = None
        self._reset_vieneu_voice_combo()

    def _reset_vieneu_voice_combo(self) -> None:
        """Trả combo giọng về self._translator.translate("tts.voice_default") và mở khoá (dùng khi nạp lỗi/rỗng)."""
        self._vieneu_voice.blockSignals(True)
        self._vieneu_voice.clear()
        self._vieneu_voice.addItem(self._translator.translate("tts.voice_default"), userData="")
        self._vieneu_voice.setEnabled(True)
        self._vieneu_voice.blockSignals(False)
        # [v3.23.252] Nạp lỗi/rỗng cũng phải bật lại nút Tổng hợp (đã khoá lúc
        # bắt đầu nạp). Dùng giọng mặc định vẫn chạy được, không nên kẹt nút.
        self._update_action_states()

    def _vieneu_int8_tooltip(self) -> str:
        """Chú thích ô int8 kèm kết quả dò VNNI của máy này."""
        from subtitles_extractor.infrastructure.tts.cpu_features import cpu_vnni_support

        support = cpu_vnni_support()
        key = {
            True: "tts.vnni_yes", False: "tts.vnni_no", None: "tts.vnni_unknown",
        }[support]
        return (
            f"{self._translator.translate('tts.tip_vieneu_int8')}\n"
            f"{self._translator.translate(key)}"
        )

    def _vieneu_precision(self) -> str:
        """``"int8"`` khi người dùng bật CPU nhanh cho v3 Turbo; rỗng = mặc định SDK."""
        mode = self._vieneu_mode.currentData() or "standard"
        return "int8" if (mode == "v3turbo" and self._vieneu_int8.isChecked()) else ""

    def _sync_vieneu_mode_controls(self) -> None:
        """[v3.23.595] Bật/tắt các tuỳ chọn theo chế độ VieNeu.

        * Sắc thái giọng: VieNeu 3.8 bỏ ``style`` ở v3 Turbo (nhận nhưng bỏ qua — phong cách
          đọc nằm trong giọng mẫu), nên ô này không có tác dụng ở v3 Turbo.
        * CPU nhanh int8: chỉ v3 Turbo có graph int8.
        """
        is_v3turbo = (self._vieneu_mode.currentData() or "standard") == "v3turbo"
        self._vieneu_emotion.setEnabled(not is_v3turbo)
        self._vieneu_emotion.setToolTip(
            self._translator.translate(
                "tts.tip_emotion_v3turbo" if is_v3turbo else "tts.tip_emotion"
            )
        )
        if hasattr(self, "_vieneu_int8"):
            self._vieneu_int8.setEnabled(is_v3turbo)

    def _on_vieneu_mode_changed(self, _idx: int = 0) -> None:
        """Đổi chế độ VieNeu -> nạp lại danh sách giọng của chế độ mới (thread nền)."""
        if getattr(self, "_vieneu_voices_loaded", False) or self._voice_loader_running():
            self._vieneu_voices_loaded = False  # danh sách cũ không còn đúng chế độ
            self._load_vieneu_voices()

    def _on_gemini_model_changed(self, _idx: int = 0) -> None:
        """Ẩn/hiện Native Audio config dựa theo model đang chọn."""
        from subtitles_extractor.infrastructure.tts.gemini_tts_adapter import (
            _NATIVE_AUDIO_MODELS,
        )
        is_native = self._gemini_model.currentText() in _NATIVE_AUDIO_MODELS
        self._gemini_native_group.setVisible(is_native)

    def _update_edge_speakers(self) -> None:
        lang = self._edge_lang.currentData() or "vi-VN"
        speakers = _EDGE_VOICE_MAP.get(lang, [])
        curr = self._edge_voice.currentText()
        self._edge_voice.blockSignals(True)
        self._edge_voice.clear(); self._edge_voice.addItems(speakers)
        idx = self._edge_voice.findText(curr)
        if idx >= 0:
            self._edge_voice.setCurrentIndex(idx)
        self._edge_voice.blockSignals(False)
        if hasattr(self, "_dubbing"):
            self._dubbing.refresh_catalog()

    def _on_browse_vieneu_ref(self) -> None:
        """Chọn file audio nhân bản giọng cho VieNeu-TTS (tuỳ chọn)."""
        path = _safe_dialogs.open_file(
            self, "Chọn audio nhân bản giọng (3–5 giây)", "",
            "Audio (*.wav *.mp3 *.flac *.m4a);;Tất cả (*)"
        )
        if path:
            self._vieneu_ref_path.setText(path)

    def _refresh_install_hint(self) -> None:
        hint = getattr(self, "_install_hint", None)
        if hint is None:
            return
        avail = getattr(self, "_engine_avail", {})
        missing = [(name, cmd) for engine_id, name, cmd in self._INSTALL_COMMANDS
                   if avail.get(engine_id) is False]
        hint.setText(" &nbsp;|&nbsp; ".join(
            f"💡 {name}: <code>{cmd}</code>" for name, cmd in missing))
        hint.setVisible(bool(missing))

    def _on_engine_available(self, name: str, available: bool) -> None:
        icon = "✅" if available else "❌"
        color = "#27ae60" if available else "#c0392b"
        note = "" if available else self._translator.translate("tts.res_not_installed")
        style = caption_style(color)
        n = name.lower()
        # [v3.23.115] Ghi nhớ tình trạng theo engine_id để kiểm tra trước khi chạy.
        if not hasattr(self, "_engine_avail"):
            self._engine_avail: dict[str, bool] = {}
        if "edge" in n:
            self._engine_avail[_ENGINE_EDGE] = available
            self._lbl_edge_st.setText(f"{icon} Edge TTS{note}")
            self._lbl_edge_st.setStyleSheet(style)
        elif "gemini" in n:
            self._engine_avail[_ENGINE_GEMINI] = available
            self._lbl_gemini_st.setText(f"{icon} Gemini TTS{note}")
            self._lbl_gemini_st.setStyleSheet(style)
        elif "vieneu" in n:
            self._engine_avail[_ENGINE_VIENEU] = available
            self._lbl_vieneu_st.setText(f"{icon} VieNeu-TTS{note}")
            self._lbl_vieneu_st.setStyleSheet(style)
        self._refresh_install_hint()

        # [v3.23.291] Khi da kiem du 3 engine: neu engine dang chon CHUA san sang nhung co
        # engine khac san sang -> tu chuyen sang engine do (tranh nguoi dung bi ket o engine
        # chua cai nhu VieNeu/Edge, trong khi Gemini da san sang). Chi tu chuyen 1 lan.
        if len(self._engine_avail) >= 3 and not getattr(self, "_auto_switched_engine", False):
            current_id = self._engine_combo.currentData()
            if not self._engine_avail.get(current_id, False):
                # [v3.23.292] VieNeu bundle mac dinh (offline, khong can API key) -> uu tien
                # truoc; roi Gemini (can key); Edge cuoi (GPL, cai ngoai).
                for engine_id in (_ENGINE_VIENEU, _ENGINE_GEMINI, _ENGINE_EDGE):
                    if self._engine_avail.get(engine_id, False):
                        pos = self._engine_combo.findData(engine_id)
                        if pos >= 0:
                            self._auto_switched_engine = True
                            self._engine_combo.setCurrentIndex(pos)
                            break


__all__ = ["TTSEnginesMixin"]
