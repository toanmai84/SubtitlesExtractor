"""[v3.23.593] Bảng phân vai LỒNG TIẾNG trên trang TTS: mỗi nhân vật một giọng.

Widget chỉ lo trình bày (SRP): hiện danh sách vai, cho người dùng đổi giọng từng vai, phát
tín hiệu khi cần nạp vai / phân vai lại. Việc tìm người nói và chọn giọng nằm ở
``application.services.dubbing_cast`` (hàm thuần); trang TTS nối hai bên.

[v3.23.597] Thêm chế độ CHỈ HAI GIỌNG: một giọng nam cho mọi vai nam, một giọng nữ cho mọi
vai nữ; cột "Giọng" của từng vai khi đó chỉ chọn giữa hai giọng đó.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from subtitles_extractor.application.services.dubbing_cast import (
    AGE_CHILD,
    AGE_ELDER,
    CAST_MODE_PER_ROLE,
    CAST_MODE_TWO_VOICES,
    CastMember,
)
from subtitles_extractor.domain.value_objects.voice_profile import (
    VOICE_FEMALE,
    VOICE_MALE,
    VoiceProfile,
)
from subtitles_extractor.presentation.fluent_compat import PushButton
from subtitles_extractor.presentation.theme import metrics as _m
from subtitles_extractor.presentation.theme.styles import caption_style
from subtitles_extractor.presentation.utils.accessibility import set_accessible_name
from subtitles_extractor.presentation.utils.wheel_guard import protect_scroll_widgets
from subtitles_extractor.presentation.widgets.section_card import SectionCard

_COLUMN_NAME = 0
_COLUMN_TRAITS = 1
_COLUMN_LINES = 2
_COLUMN_VOICE = 3
_SIDES = (VOICE_MALE, VOICE_FEMALE)


class DubbingCastPanel(QWidget):
    """Thẻ "Lồng tiếng theo nhân vật": bật/tắt, nạp vai, phân vai tự động, bảng giọng.

    Signals:
        load_requested: Người dùng bấm nạp vai từ trang Dịch.
        auto_cast_requested: Người dùng bấm phân vai tự động lại.
        casting_changed: Người dùng đổi giọng một vai (để trang lưu lựa chọn).
        enabled_changed: Bật/tắt lồng tiếng (bool).
        mode_changed: [v3.23.597] Đổi cách phân giọng (``CAST_MODE_*``).
        gender_voices_changed: [v3.23.597] Đổi giọng nam/giọng nữ hoặc giới cho vai chưa
            rõ giới tính (chế độ hai giọng).
    """

    load_requested = Signal()
    auto_cast_requested = Signal()
    casting_changed = Signal()
    enabled_changed = Signal(bool)
    mode_changed = Signal(str)
    gender_voices_changed = Signal()

    def __init__(self, translator: object, parent: QWidget | None = None) -> None:
        """Dựng giao diện.

        Args:
            translator: Bộ dịch chuỗi giao diện (``translate(key) -> str``).
            parent: Widget cha.
        """
        super().__init__(parent)
        self._translator = translator
        self._members: list[CastMember] = []
        self._catalog: list[VoiceProfile] = []
        self._build_ui()

    # ── Public API ────────────────────────────────────────────────────────────

    def is_dubbing_enabled(self) -> bool:
        """Người dùng có bật lồng tiếng nhiều giọng không."""
        return self._enable_check.isChecked()

    def set_dubbing_enabled(self, enabled: bool) -> None:
        """Đặt trạng thái bật/tắt (khôi phục cài đặt) mà không phát tín hiệu."""
        self._enable_check.blockSignals(True)
        self._enable_check.setChecked(enabled)
        self._enable_check.blockSignals(False)
        self._sync_enabled_state()

    def cast_mode(self) -> str:
        """[v3.23.597] Cách phân giọng đang chọn (``CAST_MODE_*``)."""
        return str(self._mode_combo.currentData() or CAST_MODE_PER_ROLE)

    def set_cast_mode(self, mode: str) -> None:
        """[v3.23.597] Đặt cách phân giọng (khôi phục cài đặt) mà không phát tín hiệu."""
        index = self._mode_combo.findData(mode)
        self._mode_combo.blockSignals(True)
        self._mode_combo.setCurrentIndex(index if index >= 0 else 0)
        self._mode_combo.blockSignals(False)
        self._sync_mode_widgets()

    def has_cast(self) -> bool:
        """Đã có danh sách vai chưa."""
        return bool(self._members)

    def set_status(self, text: str) -> None:
        """Dòng trạng thái dưới nút bấm."""
        self._status.setText(text)

    def status(self) -> str:
        """Dòng trạng thái đang hiện."""
        return self._status.text()

    def set_cast(
        self,
        members: list[CastMember],
        catalog: list[VoiceProfile],
        casting: dict[str, str],
    ) -> None:
        """Hiện danh sách vai cùng giọng đang giao (chế độ mỗi vai một giọng).

        Args:
            members: Vai theo thứ tự hiển thị.
            catalog: Giọng của engine đang chọn.
            casting: ``{tên vai: voice_id}``.
        """
        self._fill_table(members, catalog, lambda member: self._voice_combo(
            member, casting.get(member.name, "")
        ))

    def set_catalog(self, catalog: list[VoiceProfile], casting: dict[str, str]) -> None:
        """Đổi bộ giọng (đổi engine) và giao lại giọng cho các vai đang có."""
        self.set_cast(self._members, catalog, casting)

    def set_two_voice_cast(
        self,
        members: list[CastMember],
        catalog: list[VoiceProfile],
        voices: tuple[str, str],
        sides: dict[str, str],
        unknown_side: str,
    ) -> None:
        """[v3.23.597] Hiện danh sách vai ở chế độ CHỈ HAI GIỌNG.

        Args:
            members: Vai theo thứ tự hiển thị.
            catalog: Giọng của engine đang chọn.
            voices: ``(giọng nam, giọng nữ)``.
            sides: ``{tên vai: VOICE_MALE | VOICE_FEMALE}``.
            unknown_side: Giới dùng cho vai chưa rõ giới tính.
        """
        self._catalog = list(catalog)
        self._fill_gender_voice_combos(voices)
        self._unknown_combo.blockSignals(True)
        index = self._unknown_combo.findData(unknown_side)
        self._unknown_combo.setCurrentIndex(index if index >= 0 else 0)
        self._unknown_combo.blockSignals(False)
        self._fill_table(members, catalog, lambda member: self._side_combo(
            member, sides.get(member.name, member.gender or unknown_side)
        ))

    def casting(self) -> dict[str, str]:
        """``{tên vai: voice_id}`` người dùng đang chọn (rỗng = giọng chung)."""
        if self.cast_mode() == CAST_MODE_TWO_VOICES:
            voices = dict(zip(_SIDES, self.gender_voices(), strict=True))
            return {name: voices.get(side, "") for name, side in self.sides().items()}
        result: dict[str, str] = {}
        for row, member in enumerate(self._members):
            combo = self._table.cellWidget(row, _COLUMN_VOICE)
            if isinstance(combo, QComboBox):
                result[member.name] = str(combo.currentData() or "")
        return result

    def sides(self) -> dict[str, str]:
        """[v3.23.597] ``{tên vai: giới}`` ở chế độ hai giọng (rỗng ở chế độ kia)."""
        if self.cast_mode() != CAST_MODE_TWO_VOICES:
            return {}
        result: dict[str, str] = {}
        for row, member in enumerate(self._members):
            combo = self._table.cellWidget(row, _COLUMN_VOICE)
            if isinstance(combo, QComboBox) and combo.currentData() in _SIDES:
                result[member.name] = str(combo.currentData())
        return result

    def gender_voices(self) -> tuple[str, str]:
        """[v3.23.597] ``(giọng nam, giọng nữ)`` đang chọn."""
        return (
            str(self._male_combo.currentData() or ""),
            str(self._female_combo.currentData() or ""),
        )

    def unknown_side(self) -> str:
        """[v3.23.597] Giới dùng cho vai chưa rõ giới tính."""
        return str(self._unknown_combo.currentData() or VOICE_MALE)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _tr(self, key: str) -> str:
        return self._translator.translate(f"tts.{key}")  # type: ignore[attr-defined]

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        card = SectionCard(self._tr("dub_title"))
        column = QVBoxLayout()
        column.setSpacing(_m.SPACING_SM)

        self._enable_check = QCheckBox(self._tr("dub_enable"))
        self._enable_check.setToolTip(self._tr("dub_enable_tip"))
        self._enable_check.toggled.connect(self._on_enable_toggled)
        column.addWidget(self._enable_check)

        hint = QLabel(self._tr("dub_hint"))
        hint.setWordWrap(True)
        hint.setStyleSheet(caption_style())
        column.addWidget(hint)

        column.addLayout(self._build_mode_row())
        self._two_voice_box = self._build_two_voice_row()
        column.addWidget(self._two_voice_box)

        buttons = QHBoxLayout()
        buttons.setSpacing(_m.SPACING_SM)
        self._load_button = PushButton(self._tr("dub_btn_load"))
        self._load_button.setToolTip(self._tr("dub_btn_load_tip"))
        self._load_button.clicked.connect(self.load_requested.emit)
        self._auto_button = PushButton(self._tr("dub_btn_auto"))
        self._auto_button.setToolTip(self._tr("dub_btn_auto_tip"))
        self._auto_button.clicked.connect(self.auto_cast_requested.emit)
        buttons.addWidget(self._load_button, 1)
        buttons.addWidget(self._auto_button, 1)
        column.addLayout(buttons)

        self._status = QLabel(self._tr("dub_status_empty"))
        self._status.setWordWrap(True)
        column.addWidget(self._status)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels([
            self._tr("dub_col_character"), self._tr("dub_col_traits"),
            self._tr("dub_col_lines"), self._tr("dub_col_voice"),
        ])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._table.setMinimumHeight(260)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(_COLUMN_NAME, QHeaderView.ResizeMode.Stretch)
        for column_index in (_COLUMN_TRAITS, _COLUMN_LINES):
            header.setSectionResizeMode(
                column_index, QHeaderView.ResizeMode.ResizeToContents
            )
        header.setSectionResizeMode(_COLUMN_VOICE, QHeaderView.ResizeMode.Stretch)
        set_accessible_name(self._table, self._tr("dub_title"))
        column.addWidget(self._table)

        card.add_layout(column)
        root.addWidget(card)
        self._sync_mode_widgets()
        self._sync_enabled_state()

    def _build_mode_row(self) -> QHBoxLayout:
        """[v3.23.597] Hàng chọn cách phân giọng."""
        row = QHBoxLayout()
        row.setSpacing(_m.SPACING_SM)
        label = QLabel(self._tr("dub_mode"))
        self._mode_combo = QComboBox()
        for key, mode in (
            ("dub_mode_per_role", CAST_MODE_PER_ROLE),
            ("dub_mode_two", CAST_MODE_TWO_VOICES),
        ):
            self._mode_combo.addItem(self._tr(key), userData=mode)
        self._mode_combo.setToolTip(self._tr("dub_mode_tip"))
        label.setBuddy(self._mode_combo)
        set_accessible_name(self._mode_combo, self._tr("dub_mode"))
        protect_scroll_widgets(self._mode_combo)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        row.addWidget(label)
        row.addWidget(self._mode_combo, 1)
        return row

    def _build_two_voice_row(self) -> QWidget:
        """[v3.23.597] Hàng chọn giọng nam / giọng nữ / giới cho vai chưa rõ giới tính."""
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(_m.SPACING_SM)
        self._male_combo = QComboBox()
        self._female_combo = QComboBox()
        self._unknown_combo = QComboBox()
        self._unknown_combo.addItem(self._tr("dub_gender_male"), userData=VOICE_MALE)
        self._unknown_combo.addItem(self._tr("dub_gender_female"), userData=VOICE_FEMALE)
        self._unknown_combo.setToolTip(self._tr("dub_unknown_tip"))
        for key, combo in (
            ("dub_voice_male", self._male_combo),
            ("dub_voice_female", self._female_combo),
            ("dub_unknown", self._unknown_combo),
        ):
            label = QLabel(self._tr(key))
            label.setBuddy(combo)
            set_accessible_name(combo, self._tr(key))
            protect_scroll_widgets(combo)
            combo.currentIndexChanged.connect(self._on_gender_voices_changed)
            row.addWidget(label)
            row.addWidget(combo, 1)
        return box

    def _fill_gender_voice_combos(self, voices: tuple[str, str]) -> None:
        """Nạp giọng engine vào hai ô giọng nam/nữ (giọng cùng giới lên trước)."""
        for combo, gender, voice_id in zip(
            (self._male_combo, self._female_combo), _SIDES, voices, strict=True
        ):
            combo.blockSignals(True)
            combo.clear()
            matching = [p for p in self._catalog if p.gender == gender]
            others = [p for p in self._catalog if p.gender != gender]
            for profile in matching + others:
                combo.addItem(profile.display, userData=profile.voice_id)
            index = combo.findData(voice_id)
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(False)

    def _fill_table(
        self,
        members: list[CastMember],
        catalog: list[VoiceProfile],
        make_combo: Callable[[CastMember], QComboBox],
    ) -> None:
        self._members = list(members)
        self._catalog = list(catalog)
        self._table.setRowCount(len(self._members))
        for row, member in enumerate(self._members):
            self._table.setItem(row, _COLUMN_NAME, self._read_only_item(
                member.display_name, tooltip=member.name,
            ))
            self._table.setItem(
                row, _COLUMN_TRAITS, self._read_only_item(self._traits_text(member))
            )
            self._table.setItem(
                row, _COLUMN_LINES, self._read_only_item(str(member.line_count))
            )
            self._table.setCellWidget(row, _COLUMN_VOICE, make_combo(member))
        self._sync_enabled_state()

    def _voice_combo(self, member: CastMember, voice_id: str) -> QComboBox:
        """Ô chọn giọng của một vai: giọng cùng giới tính lên trước."""
        combo = QComboBox()
        combo.addItem(self._tr("dub_voice_shared"), userData="")
        matching = [
            p for p in self._catalog if member.gender and p.gender == member.gender
        ]
        others = [p for p in self._catalog if p not in matching]
        for profile in matching + others:
            combo.addItem(profile.display, userData=profile.voice_id)
        index = combo.findData(voice_id)
        combo.setCurrentIndex(index if index >= 0 else 0)
        return self._finish_row_combo(combo, member)

    def _side_combo(self, member: CastMember, side: str) -> QComboBox:
        """[v3.23.597] Ô chọn giọng nam/giọng nữ của một vai (chế độ hai giọng)."""
        combo = QComboBox()
        for gender in _SIDES:
            combo.addItem(self._side_label(gender), userData=gender)
        index = combo.findData(side)
        combo.setCurrentIndex(index if index >= 0 else 0)
        return self._finish_row_combo(combo, member)

    def _finish_row_combo(self, combo: QComboBox, member: CastMember) -> QComboBox:
        set_accessible_name(combo, f"{self._tr('dub_col_voice')}: {member.display_name}")
        # [v3.23.594] Ô chọn tạo SAU khi trang đã gắn chống cuộn
        # (protect_scroll_widgets chạy một lần lúc dựng trang): cuộn chuột lướt qua
        # bảng từng đổi giọng vai và lưu luôn.
        protect_scroll_widgets(combo)
        combo.currentIndexChanged.connect(lambda _index: self.casting_changed.emit())
        return combo

    def _side_label(self, gender: str) -> str:
        """"Giọng nam · Minh Đức" — nhãn ô giới của một vai."""
        combo = self._male_combo if gender == VOICE_MALE else self._female_combo
        key = "dub_voice_male" if gender == VOICE_MALE else "dub_voice_female"
        voice = combo.currentText()
        return f"{self._tr(key)} · {voice}" if voice else self._tr(key)

    def _relabel_side_combos(self) -> None:
        for row in range(self._table.rowCount()):
            combo = self._table.cellWidget(row, _COLUMN_VOICE)
            if not isinstance(combo, QComboBox):
                continue
            for item in range(combo.count()):
                if combo.itemData(item) in _SIDES:
                    combo.setItemText(item, self._side_label(combo.itemData(item)))

    @staticmethod
    def _read_only_item(text: str, *, tooltip: str = "") -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        if tooltip:
            item.setToolTip(tooltip)
        return item

    def _traits_text(self, member: CastMember) -> str:
        gender = {
            VOICE_MALE: self._tr("dub_gender_male"),
            VOICE_FEMALE: self._tr("dub_gender_female"),
        }.get(member.gender, "?")
        age = {
            AGE_CHILD: self._tr("dub_age_child"),
            AGE_ELDER: self._tr("dub_age_elder"),
        }.get(member.age_group, "")
        return f"{gender} · {age}" if age else gender

    def _on_enable_toggled(self, enabled: bool) -> None:
        self._sync_enabled_state()
        self.enabled_changed.emit(enabled)

    def _on_mode_changed(self, _index: int) -> None:
        self._sync_mode_widgets()
        self.mode_changed.emit(self.cast_mode())

    def _on_gender_voices_changed(self, _index: int) -> None:
        self._relabel_side_combos()
        self.gender_voices_changed.emit()

    def _sync_mode_widgets(self) -> None:
        self._two_voice_box.setVisible(self.cast_mode() == CAST_MODE_TWO_VOICES)

    def _sync_enabled_state(self) -> None:
        enabled = self._enable_check.isChecked()
        for widget in (
            self._load_button, self._auto_button, self._table, self._mode_combo,
            self._two_voice_box,
        ):
            widget.setEnabled(enabled)
        self._auto_button.setEnabled(enabled and bool(self._members))


__all__ = ["DubbingCastPanel"]
