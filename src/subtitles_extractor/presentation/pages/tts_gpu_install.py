"""[v3.23.692] VieNeu chạy GPU: cài gói, theo dõi tiến độ, dò khả năng GPU (chạy nền) và ghi chú trên trang.

Tách nguyên văn khỏi ``tts_page.py`` (U6). Lớp trộn chỉ chứa phương thức; widget, tín hiệu và trạng thái vẫn thuộc ``TTSPage``.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import QMessageBox

logger = logging.getLogger("subtitles_extractor.presentation.pages.tts_page")


class TTSGpuInstallMixin:
    """VieNeu chạy GPU: cài gói, theo dõi tiến độ, dò khả năng GPU (chạy nền) và ghi chú trên trang."""

    def _on_install_vieneu_gpu_clicked(self) -> None:
        """Cài các gói còn thiếu để VieNeu chạy GPU, chạy nền có báo tiến độ."""
        from PySide6.QtWidgets import QMessageBox

        from subtitles_extractor.presentation.workers.install_vieneu_gpu_worker import (
            InstallVieneuGpuWorker,
        )

        if getattr(self, "_gpu_install_thread", None) is not None:
            return

        # [v3.23.652] Dùng kết quả dò nền (``pip list`` tới 120 s không được chạy trên luồng giao diện).
        plan = getattr(self, "_gpu_plan_cache", None)
        if plan is None:
            if not getattr(self, "_gpu_plan_probe_running", False):
                self._refresh_vieneu_gpu_note()
                if self._vieneu_force_cpu.isChecked() and not self._gpu_plan_probe_running:
                    self._gpu_plan_probe_running = True
                    import threading

                    threading.Thread(target=self._probe_gpu_plan, name="vieneu-gpu-plan", daemon=True).start()
            self._vieneu_gpu_note.setText("Đang kiểm tra môi trường GPU… bấm lại nút sau vài giây.")
            return
        if plan.is_ready:
            QMessageBox.information(
                self, self._translator.translate("tts.dlg_ready_title"),
                self._translator.translate("tts.dlg_ready_body"),
            )
            self._refresh_vieneu_gpu_note(force=True)
            return
        if plan.python_exe is None:
            QMessageBox.warning(
                self, self._translator.translate("tts.dlg_no_env_title"),
                self._translator.translate("tts.dlg_no_env_body"),
            )
            return

        answer = QMessageBox.question(
            self, self._translator.translate("tts.dlg_install_title"),
            self._translator.translate("tts.dlg_install_body")
            .replace("{packages}", ", ".join(plan.missing_packages))
            .replace("{mb}", f"{plan.download_estimate_gb * 1024:.0f}"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self._btn_install_gpu.setEnabled(False)
        self._gpu_install_progress.setValue(0)
        self._gpu_install_progress.setVisible(True)

        self._gpu_install_thread = QThread(self)
        self._gpu_install_worker = InstallVieneuGpuWorker()
        self._gpu_install_worker.moveToThread(self._gpu_install_thread)

        # QueuedConnection: worker ở luồng khác, tín hiệu phải xếp hàng về luồng UI.
        self._gpu_install_thread.started.connect(self._gpu_install_worker.run)
        self._gpu_install_worker.progress.connect(
            self._on_gpu_install_progress, Qt.ConnectionType.QueuedConnection
        )
        self._gpu_install_worker.finished.connect(
            self._on_gpu_install_finished, Qt.ConnectionType.QueuedConnection
        )
        self._gpu_install_worker.failed.connect(
            self._on_gpu_install_failed, Qt.ConnectionType.QueuedConnection
        )
        self._gpu_install_worker.done.connect(
            self._cleanup_gpu_install_thread, Qt.ConnectionType.QueuedConnection
        )
        self._gpu_install_thread.start()

    def _on_gpu_install_progress(self, percent: int, label: str) -> None:
        self._gpu_install_progress.setValue(percent)
        self._vieneu_gpu_note.setText(label)

    def _on_gpu_install_finished(self, python_exe: str) -> None:
        """Cài xong: bỏ tick ép CPU giúp người dùng và làm mới ghi chú."""

        self._vieneu_force_cpu.setChecked(False)   # kích hoạt đường GPU
        self._refresh_vieneu_gpu_note(force=True)
        QMessageBox.information(
            self, self._translator.translate("tts.dlg_installed_title"),
            self._translator.translate("tts.dlg_installed_body").replace("{env}", str(python_exe))
        )

    def _on_gpu_install_failed(self, message: str) -> None:

        logger.warning("Cài VieNeu GPU thất bại: %s", message.replace("\n", " | "))
        self._refresh_vieneu_gpu_note(force=True)
        QMessageBox.warning(self, self._translator.translate("tts.dlg_install_failed_title"), message)

    def _cleanup_gpu_install_thread(self) -> None:
        """Dừng và xoá luồng cài (kể cả khi lỗi)."""
        thread = getattr(self, "_gpu_install_thread", None)
        if thread is not None:
            thread.quit()
            thread.wait(5000)
            thread.deleteLater()
        self._gpu_install_thread = None
        self._gpu_install_worker = None
        self._btn_install_gpu.setEnabled(True)
        self._gpu_install_progress.setVisible(False)

    def _refresh_vieneu_gpu_note(self, *, force: bool = False) -> None:
        """Hiện tình trạng THẬT của khả năng chạy VieNeu trên GPU.

        VieNeu chỉ dùng GPU qua đường PyTorch — engine ONNX ghi cứng
        ``CPUExecutionProvider``. Mà torch bị loại khỏi bản đóng gói (nó xung đột CUDA
        với paddle), nên bỏ tick "Ép chạy CPU" một mình là KHÔNG đủ.

        [v3.23.652] Lập kế hoạch chạy ``pip list`` của môi trường riêng (tới 120 s khi env hỏng) — trước đây chạy
        đồng bộ trên luồng giao diện mỗi lần mở trang / tick "Ép CPU" → treo app. Nay: đang ép CPU thì không dò;
        còn lại dò ở luồng nền một lần rồi nhớ kết quả (``force`` = dò lại sau khi cài).
        """
        if not hasattr(self, "_gpu_plan_cache"):
            self._gpu_plan_cache = None
            self._gpu_plan_probe_running = False
            self._gpu_plan_ready.connect(self._on_gpu_plan_ready)
        if force:
            self._gpu_plan_cache = None
            self._gpu_plan_stale = self._gpu_plan_probe_running  # lượt dò đang chạy đã cũ → dò lại khi xong
        if self._vieneu_force_cpu.isChecked():
            self._apply_vieneu_gpu_note(None)
            return
        if self._gpu_plan_cache is not None:
            self._apply_vieneu_gpu_note(self._gpu_plan_cache)
            return
        if self._gpu_plan_probe_running:
            return
        self._gpu_plan_probe_running = True
        self._vieneu_gpu_note.setText("Đang kiểm tra khả năng chạy GPU…")
        import threading

        threading.Thread(target=self._probe_gpu_plan, name="vieneu-gpu-plan", daemon=True).start()

    def _probe_gpu_plan(self) -> None:
        """[v3.23.652] Chạy ở luồng nền: lập kế hoạch GPU rồi phát tín hiệu về luồng giao diện."""
        plan = None
        try:
            from subtitles_extractor.infrastructure.tts.vieneu_gpu_plan import (
                build_gpu_tts_plan,
            )

            plan = build_gpu_tts_plan()
        except Exception as exc:  # noqa: BLE001 — luồng nền không được ném
            logger.debug("Không dò được khả năng GPU cho TTS: %s", exc)
        try:
            self._gpu_plan_ready.emit(plan)
        except RuntimeError:  # trang đã bị huỷ khi đang dò
            pass

    def _on_gpu_plan_ready(self, plan) -> None:
        self._gpu_plan_probe_running = False
        if getattr(self, "_gpu_plan_stale", False):
            self._gpu_plan_stale = False
            self._refresh_vieneu_gpu_note()
            return
        if plan is None:
            if not self._vieneu_force_cpu.isChecked():
                self._vieneu_gpu_note.setText("")
            return
        self._gpu_plan_cache = plan
        self._apply_vieneu_gpu_note(plan)

    def _apply_vieneu_gpu_note(self, plan) -> None:
        """[v3.23.652] Hiện ghi chú GPU theo kế hoạch đã có (``None`` = đang ép CPU)."""
        if self._vieneu_force_cpu.isChecked():
            self._vieneu_gpu_note.setText(
                "Đang chạy ONNX/CPU — ổn định và đủ nhanh (khoảng 0,5 giây mỗi câu)."
            )
            self._btn_install_gpu.setVisible(False)
            return

        if plan is None:
            return
        if plan.is_ready:
            self._vieneu_gpu_note.setText(
                "✓ Sẽ thử dùng GPU qua môi trường riêng. Đường GPU còn gộp lô nên "
                "nhanh hơn rõ rệt với tập nhiều câu."
            )
            self._btn_install_gpu.setVisible(False)
            return
        self._btn_install_gpu.setVisible(True)

        # Đây là điểm quan trọng: nói RÕ rằng bỏ tick không đủ.
        self._vieneu_gpu_note.setText(
            "⚠️ Bỏ tick MỘT MÌNH chưa bật được GPU: VieNeu chỉ dùng GPU qua PyTorch, "
            f"mà torch không nằm trong bản đóng gói. {plan.note} "
            "Trong lúc chờ, ONNX/CPU vẫn chạy bình thường."
        )


__all__ = ["TTSGpuInstallMixin"]
