"""Trang Xuất bản — tổng hợp video gốc + phụ đề đã dịch + giọng thuyết minh.

Đây là **bước cuối** của quy trình: sau khi đã trích phụ đề (trang Trích xuất), dịch
(trang Dịch) và tổng hợp giọng đọc (trang TTS), trang này ghép tất cả thành một tệp
video hoàn chỉnh để giao.

Bốn kiểu xuất
-------------
* **Phụ đề rời** — nhúng phụ đề dạng track, không mã hoá lại. Rất nhanh, người xem
  bật/tắt phụ đề được.
* **Phụ đề cháy** — vẽ phụ đề vào khung hình. Xem được ở mọi thiết bị, nhưng phải mã
  hoá lại nên chậm.
* **Thuyết minh (trộn)** — trộn giọng Việt LÊN TRÊN tiếng gốc, tự động hạ tiếng gốc khi
  thuyết minh nói. Nhạc và tiếng động nền vẫn còn.
* **Tiếng Việt (track riêng)** — thêm giọng Việt thành track riêng, người xem tự chọn.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from subtitles_extractor.infrastructure.video.video_render_command import (
    AudioMode,
    DuckLevel,
    RenderMode,
    RenderRequest,
    SubtitleMode,
    subtitle_styling_warning,
)
from subtitles_extractor.presentation.fluent_compat import (
    BodyLabel,
    CaptionLabel,
    ComboBox,
    HeaderCardWidget,
    InfoBar,
    InfoBarPosition,
    LineEdit,
    PrimaryPushButton,
    ProgressBar,
    PushButton,
    ScrollArea,
    StrongBodyLabel,
)
from subtitles_extractor.presentation.workers.render_video_worker import RenderVideoWorker

logger = logging.getLogger(__name__)

# Nhãn hiển thị cho từng kiểu xuất, kèm gợi ý ngắn về tốc độ.
# [v3.23.326] Phụ đề và âm thanh là HAI LỰA CHỌN ĐỘC LẬP, nên kết hợp được mọi cặp —
# đặc biệt "phụ đề + thuyết minh" trong cùng một tệp, thứ mà 4 chế độ gộp cũ không
# làm được (chúng loại trừ nhau).
# [v3.23.368] Hằng số giữ KHOÁ i18n (không phải nhãn) — nhãn được dịch lúc dựng combo.
# Ánh xạ combo→mode dùng currentIndex() nên dịch nhãn KHÔNG ảnh hưởng logic chọn.
_SUBTITLE_CHOICES: tuple[tuple[str, SubtitleMode], ...] = (
    ("publish.sub_none", SubtitleMode.NONE),
    ("publish.sub_soft", SubtitleMode.SOFT),
    ("publish.sub_burned", SubtitleMode.BURNED),
)

_AUDIO_CHOICES: tuple[tuple[str, AudioMode], ...] = (
    ("publish.audio_original", AudioMode.ORIGINAL),
    ("publish.audio_voiceover", AudioMode.VOICE_OVER),
    ("publish.audio_replace", AudioMode.REPLACE_TRACK),
)

_DUCK_CHOICES: tuple[tuple[str, DuckLevel], ...] = (
    ("publish.duck_gentle", DuckLevel.GENTLE),
    ("publish.duck_medium", DuckLevel.MEDIUM),
    ("publish.duck_strong", DuckLevel.STRONG),
)

_ENCODER_CHOICES: tuple[tuple[str, str], ...] = (
    ("publish.enc_auto", "h264_nvenc"),
    ("publish.enc_h264", "libx264"),
    ("publish.enc_h265", "libx265"),
)


class PublishPage(QWidget):
    """Trang tổng hợp và xuất bản phim hoàn chỉnh.

    Signals:
        publish_completed: Phát ra đường dẫn tệp đã xuất khi thành công.
    """

    publish_completed = Signal(object)

    def __init__(self, container: object, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("publishPage")
        self._container = container
        # [v3.23.366] Translator để externalize chuỗi UI (đa ngôn ngữ).
        self._translator = container.translator

        # [v3.23.326] Video mà tên tệp đích đang gợi ý cho — dùng để biết tên hiện tại
        # là tự sinh (thay được) hay do người dùng tự gõ (phải giữ).
        self._output_source: str = ""
        # [v3.23.329] Hàng đợi xuất bản hàng loạt.
        self._batch_items: list = []
        self._batch_index: int = 0
        self._batch_failures: list[str] = []
        self._batch_cancelled: bool = False
        self._thread: QThread | None = None
        self._worker: RenderVideoWorker | None = None
        # Hàm do MainWindow gắn vào để lấy dữ liệu từ các trang khác.
        self._subtitle_provider: object | None = None
        self._audio_provider: object | None = None
        self._video_provider: object | None = None

        self._build_ui()
        self._on_mode_changed()

    # ── Liên thông với các trang khác ────────────────────────────────────────
    def set_subtitle_provider(self, provider: object) -> None:
        """Gắn hàm trả về đường dẫn phụ đề đã dịch (từ trang Dịch/Biên tập)."""
        self._subtitle_provider = provider

    def set_audio_provider(self, provider: object) -> None:
        """Gắn hàm trả về đường dẫn tệp giọng đọc (từ trang TTS)."""
        self._audio_provider = provider

    def set_video_provider(self, provider: object) -> None:
        """Gắn hàm trả về đường dẫn video đang mở (từ trang Biên tập)."""
        self._video_provider = provider

    # ── Dựng giao diện ──────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = ScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 20, 24, 24)
        layout.setSpacing(16)

        title = StrongBodyLabel(self._translator.translate("publish.header"))
        subtitle = CaptionLabel(
            "Ghép video gốc với phụ đề đã dịch và giọng thuyết minh thành một tệp."
        )
        layout.addWidget(title)
        layout.addWidget(subtitle)

        layout.addWidget(self._build_source_card())
        layout.addWidget(self._build_options_card())
        layout.addWidget(self._build_output_card())
        layout.addStretch(1)

        scroll.setWidget(content)
        outer.addWidget(scroll)

    def _build_source_card(self) -> HeaderCardWidget:
        """Thẻ chọn ba nguồn dữ liệu: video, phụ đề, giọng đọc."""
        card = HeaderCardWidget(self)
        card.setTitle(self._translator.translate("publish.card_source"))
        form = QFormLayout()
        form.setSpacing(10)

        self._video_edit, video_row = self._file_row(
            self._translator.translate("publish.choose_video"),
            self._translator.translate("common.filter_video"),
            self._fill_video_from_pages,
        )
        self._subtitle_edit, subtitle_row = self._file_row(
            self._translator.translate("publish.choose_subtitle"),
            self._translator.translate("common.filter_subtitle"),
            self._fill_subtitle_from_pages,
        )
        self._audio_edit, audio_row = self._file_row(
            self._translator.translate("publish.choose_audio"),
            self._translator.translate("common.filter_audio"),
            self._fill_audio_from_pages,
        )

        form.addRow(BodyLabel(self._translator.translate("publish.lbl_video")), video_row)
        form.addRow(BodyLabel(self._translator.translate("publish.lbl_subtitle")), subtitle_row)
        form.addRow(BodyLabel(self._translator.translate("publish.lbl_audio")), audio_row)

        holder = QWidget()
        holder.setLayout(form)
        card.viewLayout.addWidget(holder)
        return card

    def _file_row(
        self, placeholder: str, file_filter: str, autofill
    ) -> tuple[LineEdit, QWidget]:
        """Dựng một hàng: ô nhập đường dẫn + nút Chọn + nút Lấy từ trang khác."""
        edit = LineEdit()
        edit.setPlaceholderText(placeholder)
        edit.setClearButtonEnabled(True)

        browse = PushButton(self._translator.translate("common.choose"))
        browse.clicked.connect(
            lambda: self._browse_into(edit, placeholder, file_filter)
        )
        auto = PushButton(self._translator.translate("publish.btn_from_app"))
        auto.setToolTip(self._translator.translate("publish.tip_from_app"))
        auto.clicked.connect(autofill)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        row_layout.addWidget(edit, 1)
        row_layout.addWidget(browse)
        row_layout.addWidget(auto)
        return edit, row

    def _build_options_card(self) -> HeaderCardWidget:
        """Thẻ tuỳ chọn: kiểu xuất, mức hạ tiếng gốc, bộ mã hoá."""
        card = HeaderCardWidget(self)
        card.setTitle(self._translator.translate("publish.card_mode"))
        form = QFormLayout()
        form.setSpacing(10)

        self._subtitle_combo = ComboBox()
        for key, _sm in _SUBTITLE_CHOICES:
            self._subtitle_combo.addItem(self._translator.translate(key))
        self._subtitle_combo.setCurrentIndex(1)  # mặc định: phụ đề rời
        self._subtitle_combo.currentIndexChanged.connect(self._on_mode_changed)

        self._audio_combo = ComboBox()
        for key, _am in _AUDIO_CHOICES:
            self._audio_combo.addItem(self._translator.translate(key))
        self._audio_combo.currentIndexChanged.connect(self._on_mode_changed)

        self._duck_combo = ComboBox()
        for key, _level in _DUCK_CHOICES:
            db = f"{abs(_level.approx_reduction_db):.0f}"
            self._duck_combo.addItem(self._translator.translate(key).replace("{db}", db))
        self._duck_combo.setCurrentIndex(1)  # Vừa

        self._encoder_combo = ComboBox()
        for key, _enc in _ENCODER_CHOICES:
            self._encoder_combo.addItem(self._translator.translate(key))

        self._duck_label = BodyLabel(self._translator.translate("publish.lbl_duck"))
        self._encoder_label = BodyLabel(self._translator.translate("publish.lbl_encoder"))

        form.addRow(BodyLabel(self._translator.translate("publish.lbl_sub_mode")), self._subtitle_combo)
        form.addRow(BodyLabel(self._translator.translate("publish.lbl_audio_mode")), self._audio_combo)
        form.addRow(self._duck_label, self._duck_combo)
        form.addRow(self._encoder_label, self._encoder_combo)

        self._mode_hint = CaptionLabel("")
        self._mode_hint.setWordWrap(True)
        form.addRow(BodyLabel(""), self._mode_hint)

        holder = QWidget()
        holder.setLayout(form)
        card.viewLayout.addWidget(holder)
        return card

    def _build_output_card(self) -> HeaderCardWidget:
        """Thẻ đích xuất + tiến độ + nút hành động."""
        card = HeaderCardWidget(self)
        card.setTitle(self._translator.translate("publish.card_output"))
        layout = QVBoxLayout()
        layout.setSpacing(12)

        self._output_edit = LineEdit()
        self._output_edit.setPlaceholderText(self._translator.translate("publish.output_placeholder"))
        output_browse = PushButton(self._translator.translate("common.choose"))
        output_browse.clicked.connect(self._browse_output)

        output_row = QHBoxLayout()
        output_row.setSpacing(8)
        output_row.addWidget(self._output_edit, 1)
        output_row.addWidget(output_browse)
        layout.addLayout(output_row)

        self._progress = ProgressBar()
        self._progress.setValue(0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._status = CaptionLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        # [v3.23.329] Xuất bản hàng loạt cho phim bộ — trước đây chỉ khâu Trích xuất có
        # xử lý hàng loạt, còn 4 khâu sau phải làm thủ công từng tập.
        self._batch_button = PushButton(self._translator.translate("publish.btn_batch"))
        self._batch_button.setToolTip(self._translator.translate("publish.tip_batch"))
        self._batch_button.clicked.connect(self._on_batch_publish_clicked)

        # [v3.23.366] Nối tất cả các tập đã xuất bản thành MỘT video trọn bộ.
        self._concat_button = PushButton(self._translator.translate("publish.btn_concat"))
        self._concat_button.setToolTip(self._translator.translate("publish.tip_concat"))
        self._concat_button.clicked.connect(self._on_concat_series_clicked)

        self._start_button = PrimaryPushButton(self._translator.translate("publish.btn_start"))
        self._start_button.clicked.connect(self._start_render)
        self._cancel_button = PushButton(self._translator.translate("common.cancel"))
        self._cancel_button.clicked.connect(self._cancel_render)
        self._cancel_button.setEnabled(False)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self._concat_button)
        buttons.addWidget(self._batch_button)
        buttons.addWidget(self._cancel_button)
        buttons.addWidget(self._start_button)
        layout.addLayout(buttons)

        holder = QWidget()
        holder.setLayout(layout)
        card.viewLayout.addWidget(holder)
        return card

    # ── Trạng thái giao diện ────────────────────────────────────────────────
    @property
    def _selected_subtitle_mode(self) -> SubtitleMode:
        return _SUBTITLE_CHOICES[self._subtitle_combo.currentIndex()][1]

    @property
    def _selected_audio_mode(self) -> AudioMode:
        return _AUDIO_CHOICES[self._audio_combo.currentIndex()][1]

    def _on_mode_changed(self) -> None:
        """Bật/tắt các ô không liên quan và cập nhật gợi ý theo cặp chế độ đang chọn."""
        subtitle_mode = self._selected_subtitle_mode
        audio_mode = self._selected_audio_mode
        needs_subtitle = subtitle_mode is not SubtitleMode.NONE
        needs_audio = audio_mode is not AudioMode.ORIGINAL
        needs_encoder = subtitle_mode is SubtitleMode.BURNED
        needs_duck = audio_mode is AudioMode.VOICE_OVER

        self._subtitle_edit.setEnabled(needs_subtitle)
        self._audio_edit.setEnabled(needs_audio)
        for widget in (self._duck_label, self._duck_combo):
            widget.setVisible(needs_duck)
        for widget in (self._encoder_label, self._encoder_combo):
            widget.setVisible(needs_encoder)

        self._mode_hint.setText(self._describe_selection(subtitle_mode, audio_mode))
        # [v3.23.326] Tên tệp đích phải ĐỔI THEO chế độ. Trước đây chỉ điền khi ô trống
        # nên đổi sang "thuyết minh" vẫn giữ tên "_phude" — gây nhầm lẫn khi xem lại.
        self._refresh_output_suggestion()
        self._update_readiness()

    def _describe_selection(
        self, subtitle_mode: SubtitleMode, audio_mode: AudioMode
    ) -> str:
        """Mô tả ngắn hệ quả của lựa chọn hiện tại (tốc độ, chất lượng)."""
        parts: list[str] = []
        if subtitle_mode is SubtitleMode.SOFT:
            parts.append(self._translator.translate("publish.hint_sub_soft"))
        elif subtitle_mode is SubtitleMode.BURNED:
            parts.append(self._translator.translate("publish.hint_sub_burned"))
        if audio_mode is AudioMode.VOICE_OVER:
            parts.append(self._translator.translate("publish.hint_audio_voiceover"))
        elif audio_mode is AudioMode.REPLACE_TRACK:
            parts.append(self._translator.translate("publish.hint_audio_replace"))
        if not parts:
            parts.append(self._translator.translate("publish.hint_default"))
        return " ".join(parts)

    def _set_busy(self, busy: bool) -> None:
        self._start_button.setEnabled(not busy)
        self._cancel_button.setEnabled(busy)
        self._progress.setVisible(busy)
        self._batch_button.setEnabled(not busy)
        self._subtitle_combo.setEnabled(not busy)
        self._audio_combo.setEnabled(not busy)

    # ── Chọn tệp ────────────────────────────────────────────────────────────
    def autofill_from_app(self) -> None:
        """Tự điền cả 3 nguồn từ các trang trước — gọi khi người dùng VÀO trang này.

        Chỉ điền vào ô đang TRỐNG, không ghi đè thứ người dùng đã tự chọn. Im lặng khi
        chưa có dữ liệu (khác nút "Lấy từ app" — bấm tay thì phải báo rõ vì sao trống).
        """
        for provider, edit in (
            (self._video_provider, self._video_edit),
            (self._subtitle_provider, self._subtitle_edit),
            (self._audio_provider, self._audio_edit),
        ):
            if provider is None or edit.text().strip():
                continue
            try:
                value = provider()  # type: ignore[operator]
            except Exception as exc:  # noqa: BLE001 — tự điền hỏng không được phá trang
                logger.debug("Bỏ qua tự điền: %s", exc)
                continue
            if value:
                edit.setText(str(value))
        self._refresh_output_suggestion()
        self._update_readiness()
        self._update_readiness()

    def _update_readiness(self) -> None:
        """Cập nhật dòng trạng thái cho biết còn thiếu gì để xuất được."""
        subtitle_mode = self._selected_subtitle_mode
        audio_mode = self._selected_audio_mode
        missing: list[str] = []
        if not self._video_edit.text().strip():
            missing.append(self._translator.translate("publish.missing_video"))
        if subtitle_mode is not SubtitleMode.NONE and not self._subtitle_edit.text().strip():
            missing.append(self._translator.translate("publish.missing_subtitle"))
        if audio_mode is not AudioMode.ORIGINAL and not self._audio_edit.text().strip():
            missing.append(self._translator.translate("publish.missing_audio"))

        if missing:
            self._status.setText(
                self._translator.translate("publish.status_missing").replace(
                    "{items}", ", ".join(missing)
                )
            )
        else:
            self._status.setText(
                self._translator.translate("publish.status_ready") + self._sync_note()
            )

    def _sync_note(self) -> str:
        """Ghi chú cho biết phụ đề đang dùng có khớp giọng đọc hay không.

        [v3.23.318] Khâu TTS chỉnh lại mốc thời gian phụ đề cho khớp lời thoại đã tổng
        hợp và ghi ra tệp ``*.tts.*.srt`` riêng. Dùng nhầm bản dịch gốc sẽ khiến phụ đề
        LỆCH so với giọng nói — nên phải nói rõ đang dùng bản nào.
        """
        subtitle_text = self._subtitle_edit.text().strip()
        if not subtitle_text:
            return ""
        is_synced = ".tts." in Path(subtitle_text).name.lower()
        has_audio = bool(self._audio_edit.text().strip())

        # [v3.23.327] Cảnh báo nếu định dạng đích làm mất màu/kiểu chữ của phụ đề .ass.
        styling = subtitle_styling_warning(
            Path(self._output_edit.text().strip() or "x.mkv"), Path(subtitle_text)
        )
        prefix = f"  ⚠ {styling}" if styling else ""

        if is_synced:
            return f"{prefix}  {self._translator.translate('publish.sync_matched')}"
        if has_audio:
            return f"{prefix}  {self._translator.translate('publish.sync_original')}"
        return prefix

    def _browse_into(self, edit: LineEdit, caption: str, file_filter: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, caption, "", file_filter)
        if path:
            edit.setText(path)
            self._refresh_output_suggestion()
            self._update_readiness()

    def _browse_output(self) -> None:
        suggested = self._output_edit.text() or self._default_output_path()
        path, _ = QFileDialog.getSaveFileName(
            self, self._translator.translate("publish.save_video_caption"), suggested,
            self._translator.translate("common.filter_video_out"),
        )
        if path:
            self._output_edit.setText(path)

    def _default_output_path(self) -> str:
        """Gợi ý tên tệp đích, phản ánh ĐÚNG cặp chế độ đang chọn."""
        source = self._video_edit.text().strip()
        if not source:
            return ""
        video = Path(source)
        subtitle_tag = {
            SubtitleMode.NONE: "",
            SubtitleMode.SOFT: "_phude",
            SubtitleMode.BURNED: "_phudechay",
        }[self._selected_subtitle_mode]
        audio_tag = {
            AudioMode.ORIGINAL: "",
            AudioMode.VOICE_OVER: "_thuyetminh",
            AudioMode.REPLACE_TRACK: "_tiengviet",
        }[self._selected_audio_mode]
        suffix = f"{subtitle_tag}{audio_tag}" or "_xuatban"
        return str(video.with_name(f"{video.stem}{suffix}.mkv"))

    def _refresh_output_suggestion(self) -> None:
        """Cập nhật tên tệp đích cho khớp video nguồn VÀ cặp chế độ đang chọn.

        [v3.23.326] Trước đây chỉ điền khi ô trống, gây hai lỗi:

        * Đổi chế độ (vd sang thuyết minh) mà tên vẫn giữ ``_phude`` — đúng thứ xảy ra
          trong log thực tế: xuất ``voice_over`` ra tệp tên ``第19集_phude.mkv``.
        * **Đổi video gốc mà tên đích không đổi** — xuất phim B vào tệp mang tên A,
          và nếu bấm xuất lần nữa sẽ GHI ĐÈ kết quả của phim A.

        Nay tự cập nhật, nhưng KHÔNG bao giờ ghi đè tên do người dùng tự gõ. Nhận biết
        bằng cách so tên hiện tại với mọi gợi ý khả dĩ của **video mà gợi ý đó thuộc về**
        (không phải video mới) — nếu khớp thì đó là tên tự sinh, thay được.
        """
        current = self._output_edit.text().strip()
        if current and current not in self._all_suggested_paths(self._output_source):
            return  # người dùng tự đặt tên -> tôn trọng

        suggested = self._default_output_path()
        self._output_edit.setText(suggested)
        # Ghi nhớ tên gợi ý này thuộc về video nào, để lần sau biết có được thay không.
        self._output_source = self._video_edit.text().strip()

    @staticmethod
    def _all_suggested_paths(video_text: str) -> set[str]:
        """Mọi tên tệp mà hệ thống có thể đã gợi ý cho một video.

        Args:
            video_text: Đường dẫn video nguồn (rỗng = chưa có).

        Returns:
            Tập tên tệp tự sinh; dùng để phân biệt với tên người dùng tự gõ.
        """
        if not video_text:
            return set()
        video = Path(video_text)
        subtitle_tags = ("", "_phude", "_phudechay")
        audio_tags = ("", "_thuyetminh", "_tiengviet")
        results = {
            str(video.with_name(f"{video.stem}{subtitle}{audio}.mkv"))
            for subtitle in subtitle_tags
            for audio in audio_tags
        }
        results.add(str(video.with_name(f"{video.stem}_xuatban.mkv")))
        # Tên gợi ý của bản trước v3.23.326 cũng coi là tự sinh.
        for legacy in ("_phude_chay",):
            results.add(str(video.with_name(f"{video.stem}{legacy}.mkv")))
        return results

    # ── Lấy dữ liệu từ trang khác ───────────────────────────────────────────
    def _fill_from_provider(self, provider: object, edit: LineEdit, what: str) -> None:
        """Gọi provider và điền kết quả vào ô nhập; báo rõ khi chưa có dữ liệu."""
        if provider is None:
            self._warn(self._translator.translate("publish.warn_no_data").replace("{what}", what))
            return
        try:
            value = provider()  # type: ignore[operator]
        except Exception as exc:  # noqa: BLE001 — lỗi ở trang khác không được làm sập
            logger.warning("Không lấy được %s: %s", what, exc)
            self._warn(self._translator.translate("publish.warn_fetch_failed").replace("{what}", what).replace("{exc}", str(exc)))
            return
        if not value:
            self._warn(self._translator.translate("publish.warn_no_data_hint").replace("{what}", what))
            return
        edit.setText(str(value))
        self._refresh_output_suggestion()
        self._update_readiness()

    def _fill_video_from_pages(self) -> None:
        self._fill_from_provider(self._video_provider, self._video_edit, self._translator.translate("publish.what_video"))

    def _fill_subtitle_from_pages(self) -> None:
        self._fill_from_provider(
            self._subtitle_provider, self._subtitle_edit, self._translator.translate("publish.what_subtitle")
        )

    def _fill_audio_from_pages(self) -> None:
        self._fill_from_provider(
            self._audio_provider, self._audio_edit, self._translator.translate("publish.what_audio")
        )

    # ── Xuất video ──────────────────────────────────────────────────────────
    def _build_request(self) -> RenderRequest | None:
        """Dựng yêu cầu xuất từ giao diện; ``None`` nếu thiếu dữ liệu."""
        video_text = self._video_edit.text().strip()
        output_text = self._output_edit.text().strip()
        if not video_text:
            self._warn(self._translator.translate("publish.warn_need_video"))
            return None
        if not output_text:
            self._warn(self._translator.translate("publish.warn_need_output"))
            return None

        subtitle_mode = self._selected_subtitle_mode
        audio_mode = self._selected_audio_mode
        subtitle_text = self._subtitle_edit.text().strip()
        audio_text = self._audio_edit.text().strip()

        if subtitle_mode is not SubtitleMode.NONE and not subtitle_text:
            self._warn(self._translator.translate("publish.warn_need_subtitle"))
            return None
        if audio_mode is not AudioMode.ORIGINAL and not audio_text:
            self._warn(self._translator.translate("publish.warn_need_audio"))
            return None
        if subtitle_mode is SubtitleMode.NONE and audio_mode is AudioMode.ORIGINAL:
            self._warn(self._translator.translate("publish.warn_nothing_selected"))
            return None

        return RenderRequest(
            video_path=Path(video_text),
            output_path=Path(output_text),
            # `mode` giữ lại cho tương thích; hành vi thật do hai trường dưới quyết định.
            mode=RenderMode.SOFT_SUB,
            subtitle_mode=subtitle_mode,
            audio_mode=audio_mode,
            subtitle_path=Path(subtitle_text) if subtitle_text else None,
            audio_path=Path(audio_text) if audio_text else None,
            video_encoder=_ENCODER_CHOICES[self._encoder_combo.currentIndex()][1],
            duck_level=_DUCK_CHOICES[self._duck_combo.currentIndex()][1],
        )

    def _probe_duration_sec(self, video_path: Path) -> float:
        """Đọc thời lượng video để quy đổi phần trăm tiến độ (0 nếu không đọc được)."""
        try:
            import av

            with av.open(str(video_path)) as container:
                if container.duration is not None:
                    return float(container.duration) / av.time_base
        except Exception as exc:  # noqa: BLE001 — không đọc được thì chỉ mất % tiến độ
            logger.debug("Không đọc được thời lượng %s: %s", video_path.name, exc)
        return 0.0

    # ── Xuất bản hàng loạt ──────────────────────────────────────────────────
    def _on_concat_series_clicked(self) -> None:
        """[v3.23.366] Nối tất cả các tập trong một thư mục thành 1 video trọn bộ."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        from subtitles_extractor.application.services.concat_plan import (
            default_concat_output,
            find_concat_videos,
        )

        start_dir = ""
        current_video = self._video_edit.text().strip()
        if current_video:
            start_dir = str(Path(current_video).parent)
        folder = QFileDialog.getExistingDirectory(
            self, "Chọn thư mục chứa các tập cần nối", start_dir
        )
        if not folder:
            return
        folder_path = Path(folder)

        # Ưu tiên các bản ĐÃ xuất bản (có hậu tố của khâu Xuất bản); nếu không có thì
        # hỏi người dùng có muốn nối TẤT CẢ video trong thư mục không.
        suffix = self._current_output_suffix()
        videos = find_concat_videos(folder_path, name_filter=suffix)
        used_published = bool(videos)
        if not videos:
            videos = find_concat_videos(folder_path)
        if len(videos) < 2:
            QMessageBox.information(
                self, self._translator.translate("publish.concat_none_title"),
                self._translator.translate("publish.concat_none_body"),
            )
            return

        output_path = default_concat_output(folder_path, videos)
        preview = "\n".join(f"  {i + 1}. {v.name}" for i, v in enumerate(videos[:8]))
        if len(videos) > 8:
            preview += "\n  " + self._translator.translate("publish.concat_more").replace("{count}", str(len(videos) - 8))
        note = (
            self._translator.translate("publish.concat_note_published")
            if used_published
            else self._translator.translate("publish.concat_note_raw")
        )
        answer = QMessageBox.question(
            self, self._translator.translate("publish.concat_confirm_title"),
            self._translator.translate("publish.concat_confirm_body")
            .replace("{n}", str(len(videos))).replace("{preview}", preview)
            .replace("{note}", note).replace("{name}", output_path.name),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self._start_concat(videos, output_path)

    def _start_concat(
        self, videos: list[Path], output_path: Path, *, reencode: bool = False
    ) -> None:
        """Khởi chạy worker nối video trên QThread riêng."""
        if getattr(self, "_concat_thread", None) is not None:
            return
        from subtitles_extractor.presentation.workers.concat_video_worker import (
            ConcatVideoWorker,
        )

        self._concat_videos = videos
        self._concat_output = output_path
        self._status.setText(self._translator.translate("publish.concat_running").replace("{n}", str(len(videos))))
        self._concat_button.setEnabled(False)
        self._batch_button.setEnabled(False)
        self._start_button.setEnabled(False)

        thread = QThread(self)
        worker = ConcatVideoWorker(videos, output_path, 0.0, reencode=reencode)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_concat_progress)
        worker.finished.connect(self._on_concat_finished)
        worker.failed.connect(self._on_concat_failed)
        worker.done.connect(self._cleanup_concat_thread, Qt.ConnectionType.QueuedConnection)
        self._concat_thread = thread
        self._concat_worker = worker
        thread.start()

    def _on_concat_progress(self, percent: int) -> None:
        self._status.setText(self._translator.translate("publish.concat_progress").replace("{p}", str(percent)))

    def _on_concat_finished(self, output: object) -> None:
        from PySide6.QtWidgets import QMessageBox

        out_path = Path(str(output))
        self._status.setText(self._translator.translate("publish.concat_done").replace("{name}", out_path.name))
        QMessageBox.information(
            self, self._translator.translate("publish.concat_success_title"),
            self._translator.translate("publish.concat_success_body").replace("{path}", str(out_path)),
        )

    def _on_concat_failed(self, message: str) -> None:
        from PySide6.QtWidgets import QMessageBox

        # Nếu sao chép luồng lỗi do lệch thông số, mời người dùng NÉN LẠI (chắc chắn hơn).
        if not getattr(self, "_concat_reencode_offered", False) and (
            "copy" in message.lower() or "codec" in message.lower()
            or "Invalid" in message or "not match" in message.lower()
        ):
            self._concat_reencode_offered = True
            answer = QMessageBox.question(
                self, self._translator.translate("publish.concat_reencode_title"),
                self._translator.translate("publish.concat_reencode_body"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.Yes:
                videos = getattr(self, "_concat_videos", [])
                output = getattr(self, "_concat_output", None)
                if videos and output is not None:
                    self._start_concat(videos, output, reencode=True)
                    return
        self._status.setText(self._translator.translate("publish.concat_failed"))
        QMessageBox.warning(self, self._translator.translate("publish.concat_failed_title"), message)

    def _cleanup_concat_thread(self) -> None:
        """Dọn luồng nối video sau khi xong/lỗi."""
        thread = getattr(self, "_concat_thread", None)
        if thread is not None:
            thread.quit()
            thread.wait(3000)
        self._concat_thread = None
        self._concat_worker = None
        self._concat_button.setEnabled(True)
        self._batch_button.setEnabled(True)
        self._start_button.setEnabled(True)

    def _on_batch_publish_clicked(self) -> None:
        """Quét thư mục phim bộ rồi xuất lần lượt các tập đã sẵn sàng."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        from subtitles_extractor.application.services.batch_publish_plan import (
            build_publish_plan,
            find_episode_videos,
            summarise_publish_plan,
        )

        start_dir = ""
        current_video = self._video_edit.text().strip()
        if current_video:
            start_dir = str(Path(current_video).parent)
        folder = QFileDialog.getExistingDirectory(
            self, self._translator.translate("publish.batch_folder_caption"), start_dir
        )
        if not folder:
            return

        videos = find_episode_videos(Path(folder))
        if not videos:
            QMessageBox.information(
                self, self._translator.translate("publish.batch_none_title"),
                self._translator.translate("publish.batch_none_body"),
            )
            return

        subtitle_mode = self._selected_subtitle_mode
        audio_mode = self._selected_audio_mode
        plan = build_publish_plan(
            videos,
            output_suffix=self._current_output_suffix(),
            needs_subtitle=subtitle_mode is not SubtitleMode.NONE,
            needs_audio=audio_mode is not AudioMode.ORIGINAL,
            skip_existing=True,
        )
        runnable = [item for item in plan if item.will_run]
        summary = summarise_publish_plan(plan)

        if not runnable:
            QMessageBox.information(
                self, self._translator.translate("publish.batch_nothing_title"),
                self._translator.translate("publish.batch_nothing_body").replace("{summary}", summary),
            )
            return

        answer = QMessageBox.question(
            self, self._translator.translate("publish.batch_confirm_title"),
            self._translator.translate("publish.batch_confirm_body")
            .replace("{n}", str(len(videos))).replace("{summary}", summary)
            .replace("{runnable}", str(len(runnable))),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self._batch_items = runnable
        self._batch_index = 0
        self._batch_failures = []
        self._batch_cancelled = False
        self._run_next_publish()

    def _current_output_suffix(self) -> str:
        """Hậu tố tên tệp đích ứng với cặp chế độ đang chọn."""
        subtitle_tag = {
            SubtitleMode.NONE: "",
            SubtitleMode.SOFT: "_phude",
            SubtitleMode.BURNED: "_phudechay",
        }[self._selected_subtitle_mode]
        audio_tag = {
            AudioMode.ORIGINAL: "",
            AudioMode.VOICE_OVER: "_thuyetminh",
            AudioMode.REPLACE_TRACK: "_tiengviet",
        }[self._selected_audio_mode]
        return f"{subtitle_tag}{audio_tag}" or "_xuatban"

    def _run_next_publish(self) -> None:
        """Xuất tập kế tiếp trong hàng đợi; báo tổng kết khi hết."""
        if not getattr(self, "_batch_items", None):
            return
        if self._batch_cancelled or self._batch_index >= len(self._batch_items):
            self._finish_batch_publish()
            return

        item = self._batch_items[self._batch_index]
        total = len(self._batch_items)
        self._status.setText(
            f"Hàng loạt {self._batch_index + 1}/{total}: {item.video_path.name}"
        )
        # Điền vào các ô để lần xuất này dùng đúng dữ liệu của tập đang tới lượt.
        self._video_edit.setText(str(item.video_path))
        self._subtitle_edit.setText(str(item.subtitle_path or ""))
        self._audio_edit.setText(str(item.audio_path or ""))
        self._output_edit.setText(str(item.output_path))
        self._output_source = str(item.video_path)
        self._start_render()

    def _advance_batch_publish(self, *, failed: str | None = None) -> bool:
        """Chuyển sang tập kế. Trả ``True`` nếu đang chạy hàng loạt."""
        if not getattr(self, "_batch_items", None):
            return False
        if failed:
            self._batch_failures.append(failed)
        if self._batch_cancelled:
            self._finish_batch_publish()
            return True
        self._batch_index += 1
        # Trả quyền về vòng lặp sự kiện trước khi chạy tập kế — tránh đệ quy sâu.
        QTimer.singleShot(0, self._run_next_publish)
        return True

    def _finish_batch_publish(self) -> None:
        """Báo tổng kết và dọn trạng thái hàng đợi."""
        from PySide6.QtWidgets import QMessageBox

        total = len(self._batch_items)
        failures = list(self._batch_failures)
        cancelled = self._batch_cancelled
        processed = self._batch_index
        done = max(0, processed - len(failures))

        self._batch_items = []
        self._batch_index = 0
        self._batch_failures = []
        self._batch_cancelled = False

        if cancelled:
            message = (self._translator.translate("publish.batch_done_cancelled")
                       .replace("{done}", str(done)).replace("{total}", str(total))
                       .replace("{skipped}", str(max(0, total - processed))))
        else:
            message = self._translator.translate("publish.batch_done_ok").replace("{done}", str(done)).replace("{total}", str(total))
        if failures:
            message += "\n\n" + self._translator.translate("publish.batch_done_failures") + "\n• " + "\n• ".join(failures[:10])
            if len(failures) > 10:
                message += "\n" + self._translator.translate("publish.batch_done_more").replace("{count}", str(len(failures) - 10))

        self._status.setText(message.splitlines()[0])
        QMessageBox.information(self, self._translator.translate("publish.batch_complete_title"), message)

    def _start_render(self) -> None:
        """Bắt đầu xuất trên luồng riêng."""
        if self._thread is not None:
            return
        request = self._build_request()
        if request is None:
            return

        self._set_busy(True)
        self._progress.setValue(0)
        self._status.setText(self._translator.translate("publish.status_running"))

        duration = self._probe_duration_sec(request.video_path)
        self._thread = QThread(self)
        self._worker = RenderVideoWorker(request, duration)
        self._worker.moveToThread(self._thread)

        # QueuedConnection: worker chạy ở luồng khác, tín hiệu phải xếp hàng về luồng UI.
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(
            self._progress.setValue, Qt.ConnectionType.QueuedConnection
        )
        self._worker.finished.connect(
            self._on_render_finished, Qt.ConnectionType.QueuedConnection
        )
        self._worker.failed.connect(
            self._on_render_failed, Qt.ConnectionType.QueuedConnection
        )
        self._worker.done.connect(
            self._cleanup_thread, Qt.ConnectionType.QueuedConnection
        )
        self._thread.start()

    def _cancel_render(self) -> None:
        if getattr(self, "_batch_items", None):
            self._batch_cancelled = True
            remaining = len(self._batch_items) - self._batch_index - 1
            self._status.setText(
                self._translator.translate("publish.batch_cancelling").replace(
                    "{n}", str(max(0, remaining))
                )
            )
        if self._worker is not None:
            self._worker.cancel()
            self._status.setText(self._translator.translate("publish.status_cancelling"))

    def _cleanup_thread(self) -> None:
        """Dừng và xoá luồng sau khi worker xong (kể cả khi lỗi)."""
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread.deleteLater()
        self._thread = None
        self._worker = None
        self._set_busy(False)

    def _on_render_finished(self, output_path: object) -> None:
        # [v3.23.329] Đang chạy hàng loạt -> sang tập kế, không hiện hộp thoại từng tập.
        if getattr(self, "_batch_items", None):
            self.publish_completed.emit(output_path)
            self._advance_batch_publish()
            return
        self._progress.setValue(100)
        self._status.setText(f"Xong: {output_path}")
        InfoBar.success(
            title="Xuất bản thành công",
            content=str(output_path),
            parent=self,
            position=InfoBarPosition.TOP,
            duration=6000,
        )
        self.publish_completed.emit(output_path)

    def _on_render_failed(self, message: str) -> None:
        if getattr(self, "_batch_items", None):
            index = min(self._batch_index, len(self._batch_items) - 1)
            failed = self._batch_items[index].video_path.name
            logger.warning("Hàng loạt: tập %s lỗi — %s", failed, message)
            self._advance_batch_publish(failed=failed)
            return
        self._status.setText(message)
        InfoBar.error(
            title="Xuất bản thất bại",
            content=message,
            parent=self,
            position=InfoBarPosition.TOP,
            duration=8000,
        )

    def _warn(self, message: str) -> None:
        self._status.setText(message)
        InfoBar.warning(
            title="Thiếu thông tin",
            content=message,
            parent=self,
            position=InfoBarPosition.TOP,
            duration=5000,
        )


__all__ = ["PublishPage"]
