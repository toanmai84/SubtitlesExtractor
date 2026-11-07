"""[v3.23.679] Trang Lồng tiếng: "Kiểm tra phát âm (AI)" — AI nghe lại từng câu, TTS lại câu đọc sai (tối đa 2 lượt).

Luồng:
1. Lấy các câu đã có audio riêng (``segment_audio_path``), đủ ≥ 3 âm tiết → Gemini chép lời theo lô 10 câu.
2. Câu có tỉ lệ sai chữ > 0,15 → đánh dấu ở cột Trạng thái ("⚠ Đọc sai", rê chuột xem lời AI nghe được).
3. Nếu engine giữ audio từng dòng: tự TTS lại các câu sai rồi chấm lại riêng chúng; tối đa ``MAX_RETRY_ROUNDS`` lượt.
4. Kết thúc: chọn sẵn các câu vẫn sai để người dùng nghe/sửa lời.

Mô hình đọc (VieNeu) có ngẫu nhiên nên đọc lại thường hết lỗi; câu sai do CHỮ viết (từ lạ, tên riêng) thì đọc lại vẫn
sai — vì vậy chỉ thử lại có giới hạn rồi để người dùng xử lý.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QTableWidget

from subtitles_extractor.application.services.pronunciation_check import (
    PronunciationVerdict,
    lines_to_check,
    summarize,
)

logger = logging.getLogger(__name__)

#: Số lượt "TTS lại câu sai rồi chấm lại" tối đa sau lượt chấm đầu.
MAX_RETRY_ROUNDS = 2
_WRONG_BG = QColor(255, 224, 178)


class PronunciationController(QObject):
    """Điều phối kiểm tra phát âm trên trang TTS (không giữ widget riêng ngoài nút do trang tạo)."""

    def __init__(
        self,
        page: object,
        translate: Callable[..., str],
        *,
        judge_factory: Callable[[list[str]], object] | None = None,
        worker_factory: Callable[[object, list], object] | None = None,
    ) -> None:
        super().__init__(page)  # type: ignore[arg-type]
        self._page = page
        self._tr = translate
        self._judge_factory = judge_factory or self._default_judge
        self._worker_factory = worker_factory or self._default_worker
        self._worker: object | None = None
        #: Worker đã xong việc nhưng luồng chưa thoát hẳn — giữ tham chiếu tới tín hiệu ``finished`` của QThread.
        #: Bỏ tham chiếu ngay trong handler (đang chạy trong lúc worker phát tín hiệu) làm Python thu gom đối tượng
        #: giữa chừng → crash (bộ test toàn phần: segfault mã 139; QThread thật: "Destroyed while thread is running").
        self._retired: list[object] = []
        self._judge: object | None = None
        self._round = 0
        self._awaiting_resynth: set[int] = set()
        self._verdicts: dict[int, PronunciationVerdict] = {}
        self._first_wrong = 0
        self._checked = 0
        self._active = False
        self._stop_reason = ""

    # ── Mặc định (thật) ──
    @staticmethod
    def _default_judge(keys: list[str]) -> object:
        from subtitles_extractor.infrastructure.tts.pronunciation_judge import GeminiPronunciationJudge

        return GeminiPronunciationJudge(keys)

    @staticmethod
    def _default_worker(judge: object, lines: list) -> object:
        from subtitles_extractor.presentation.workers.pronunciation_worker import PronunciationWorker

        return PronunciationWorker(judge, lines)  # type: ignore[arg-type]

    # ── Trạng thái ──
    @property
    def is_running(self) -> bool:
        return self._active

    @property
    def verdicts(self) -> dict[int, PronunciationVerdict]:
        return dict(self._verdicts)

    def wrong_indices(self) -> set[int]:
        return {index for index, verdict in self._verdicts.items() if verdict.wrong}

    def _status(self, text: str) -> None:
        self._page._stage_lbl.setText(text)  # type: ignore[attr-defined]

    # ── Bắt đầu / chấm ──
    def start(self) -> bool:
        """Chấm mọi câu đủ điều kiện. False nếu không bắt đầu được (đã báo lý do ở dòng trạng thái)."""
        if self._active or self._page._view_model.is_busy:  # type: ignore[attr-defined]
            return False
        lines = lines_to_check(self._page._last_results)  # type: ignore[attr-defined]
        if not lines:
            self._status(self._tr("tts.pron_nothing"))
            return False
        keys = self._page._ai_store.gemini_keys()  # type: ignore[attr-defined]
        if not keys:
            self._status(self._tr("tts.pron_no_key"))
            return False
        self._verdicts.clear()
        self._round = 0
        self._first_wrong = 0
        self._checked = 0
        self._stop_reason = ""
        self._active = True
        self._refresh_buttons()
        return self._run(lines, keys)

    def _run(self, lines: list, keys: list[str]) -> bool:
        try:
            judge = self._judge_factory(keys)
        except RuntimeError as exc:          # PronunciationJudgeError là RuntimeError
            self._finish_error(str(exc))
            return False
        self._judge = judge
        worker = self._worker_factory(judge, lines)
        worker.progress.connect(self._on_progress)          # type: ignore[attr-defined]
        worker.finished_ok.connect(self._on_checked)        # type: ignore[attr-defined]
        worker.failed.connect(self._finish_error)           # type: ignore[attr-defined]
        self._worker = worker
        finished = getattr(worker, "finished", None)
        if finished is not None and hasattr(finished, "connect"):
            finished.connect(lambda w=worker: self._release(w))
        self._status(self._tr("tts.pron_running", done=0, total=len(lines)))
        worker.start()                                      # type: ignore[attr-defined]
        return True

    def _retire(self) -> None:
        """Worker hiện tại xong việc: chuyển sang danh sách chờ luồng thoát (không bỏ tham chiếu ngay)."""
        worker, self._worker = self._worker, None
        if worker is None:
            return
        self._retired.append(worker)
        if not hasattr(getattr(worker, "finished", None), "connect"):
            # Không phải QThread: thả ở vòng sự kiện kế (sau khi tín hiệu đang phát đã xong).
            QTimer.singleShot(0, lambda w=worker: self._release(w))

    def _release(self, worker: object) -> None:
        if worker is self._worker:            # QThread thoát trước khi handler kết quả chạy
            self._worker = None
            self._retired.append(worker)
        # Chỉ bỏ tham chiếu (luồng đã thoát / tín hiệu đã phát xong) — KHÔNG deleteLater: worker do Python sở hữu, gọi
        # deleteLater rồi thả tham chiếu làm đối tượng bị xoá hai lần (gdb: ~QObject trong DeferredDelete, segfault).
        if worker in self._retired:
            self._retired.remove(worker)

    def cancel(self) -> None:
        if self._worker is not None:
            self._worker.request_cancel()  # type: ignore[attr-defined]
        self._active = False
        self._awaiting_resynth.clear()

    def _on_progress(self, done: int, total: int) -> None:
        self._status(self._tr("tts.pron_running", done=done, total=total))

    def _on_checked(self, verdicts: list[PronunciationVerdict]) -> None:
        self._retire()
        if not self._active:
            return
        for verdict in verdicts:
            self._verdicts[verdict.event_index] = verdict
        # Hết hạn mức giữa chừng: giữ phần đã chấm, không TTS lại thêm (lượt chấm lại cũng sẽ hết hạn mức).
        self._stop_reason = str(getattr(self._judge, "stop_reason", "") or "")
        if self._round == 0:
            checked, wrong, _cer = summarize(verdicts)
            self._checked, self._first_wrong = checked, wrong
        wrong = sorted(index for index in self.wrong_indices() if index in {v.event_index for v in verdicts})
        self.apply_marks()
        vm = self._page._view_model  # type: ignore[attr-defined]
        if wrong and self._round < MAX_RETRY_ROUNDS and vm.can_resynthesize and not self._stop_reason:
            self._round += 1
            self._awaiting_resynth = set(wrong)
            self._status(self._tr("tts.pron_retrying", count=len(wrong), round=self._round, rounds=MAX_RETRY_ROUNDS))
            voices = self._page._resynthesis_line_voices()  # type: ignore[attr-defined]
            if vm.resynthesize_lines(set(wrong), line_voices=voices):
                return
            self._awaiting_resynth.clear()
        self._finish()

    def on_result_ready(self, _results: object) -> None:
        """Kẻ lại dấu; nếu đang chờ TTS lại thì chấm lại các câu đó.

        Hoãn sang vòng sự kiện kế: tín hiệu này được nối TRƯỚC handler của trang (lúc dựng nút), khi chạy ngay thì bảng
        và ``_last_results`` vẫn là của lần trước.
        """
        QTimer.singleShot(0, self._after_result)

    def _after_result(self) -> None:
        self.apply_marks()
        if not self._active or not self._awaiting_resynth:
            return
        pending, self._awaiting_resynth = self._awaiting_resynth, set()
        lines = lines_to_check(self._page._last_results, only=pending)  # type: ignore[attr-defined]
        for index in pending:
            self._verdicts.pop(index, None)
        if not lines:
            self._finish()
            return
        self._run(lines, self._page._ai_store.gemini_keys())  # type: ignore[attr-defined]

    def on_run_failed(self, _message: str = "") -> None:
        """TTS lại lỗi / bị huỷ: dừng vòng lặp, giữ dấu đã có."""
        if self._awaiting_resynth:
            self._awaiting_resynth.clear()
            self._finish()

    def _refresh_buttons(self) -> None:
        refresh = getattr(self._page, "_on_result_selection_changed", None)
        if callable(refresh):
            refresh()

    def _finish(self) -> None:
        self._active = False
        self._refresh_buttons()
        still = sorted(self.wrong_indices())
        self.apply_marks(select=True)
        text = self._tr(
            "tts.pron_done", checked=self._checked, wrong=self._first_wrong, rounds=self._round, left=len(still),
        )
        if self._stop_reason:
            text += " " + self._tr("tts.pron_partial", reason=self._stop_reason)
        self._status(text)
        logger.info("Kiểm tra phát âm: %d câu, sai %d → TTS lại %d lượt → còn %d.",
                    self._checked, self._first_wrong, self._round, len(still))

    def _finish_error(self, message: str) -> None:
        self._retire()
        self._active = False
        self._refresh_buttons()
        self._awaiting_resynth.clear()
        self._status(self._tr("tts.pron_failed", error=message))

    # ── Bảng ──
    def apply_marks(self, select: bool = False) -> None:
        """Cột Trạng thái: "⚠ Đọc sai" + lời AI nghe được (tooltip); ``select`` chọn sẵn các câu còn sai."""
        from subtitles_extractor.presentation.pages.tts_line_edits import COLUMN_STATUS
        from subtitles_extractor.presentation.utils.table_selection import select_rows_bulk

        table: QTableWidget = self._page._table  # type: ignore[attr-defined]
        wrong = self.wrong_indices()
        rows: list[int] = []
        table.blockSignals(True)
        try:
            for row in range(table.rowCount()):
                head = table.item(row, 0)
                status = table.item(row, COLUMN_STATUS)
                if head is None or status is None:
                    continue
                value = head.data(Qt.ItemDataRole.UserRole)
                if value is None or int(value) not in wrong:
                    continue
                verdict = self._verdicts[int(value)]
                status.setText(self._tr("tts.pron_mark"))
                status.setBackground(_WRONG_BG)
                status.setToolTip(self._tr("tts.pron_mark_tip", heard=verdict.heard, cer=f"{verdict.cer:.2f}"))
                rows.append(row)
        finally:
            table.blockSignals(False)
        if select and rows:
            select_rows_bulk(table, rows)
            table.scrollToItem(table.item(rows[0], 0))
        schedule = getattr(self._page, "_schedule_review_count", None)   # [v3.23.682] số việc cần duyệt
        if callable(schedule):
            schedule()
