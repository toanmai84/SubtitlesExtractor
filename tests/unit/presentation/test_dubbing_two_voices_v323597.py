"""[v3.23.597] Lồng tiếng CHỈ HAI GIỌNG: vai nam một giọng nam, vai nữ một giọng nữ.

Yêu cầu người dùng 10/10/2026. Lượt lồng tiếng thật hôm đó đã tự làm tay đúng việc này
(Minh Đức cho mọi vai nam và vai chưa rõ giới, Ngọc Huyền cho mọi vai nữ — 19 vai) bằng
cách chọn từng ô một. Edge tiếng Việt cũng chỉ có hai giọng gốc (HoaiMy nữ, NamMinh nam).

Dữ liệu: phụ đề + Visual Cues thật phim "灭门惨案II" (``test_dubbing_cast_v323593``),
giọng VieNeu 3.8.3 thật (nhãn trong ``voices_v3_turbo.json``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QComboBox

from subtitles_extractor.application.services.dubbing_cast import (
    CAST_MODE_PER_ROLE,
    CAST_MODE_TWO_VOICES,
    CastMember,
    default_gender_voices,
    two_voice_casting,
    two_voice_sides,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.domain.value_objects.voice_profile import (
    VOICE_FEMALE,
    VOICE_MALE,
    VoiceProfile,
    gender_from_voice_label,
)
from subtitles_extractor.presentation.pages.tts_dubbing import (
    SETTINGS_GENDER_VOICES_PREFIX,
    SETTINGS_ROLE_SIDES_KEY,
)
from tests.unit.presentation.test_dubbing_bugfixes_v323594 import (
    ENGINE_KEY,
    _coordinator,
    _film_events,
)

#: Nhãn giọng VieNeu 3.8.3 thật, đúng thứ tự engine trả về (giọng nổi bật lên đầu).
VIENEU_LABELS = [
    ("⭐ Adam bựa — Nam · Bắc · Phong cách tự nhiên", "Adam bựa"),
    ("⭐ Trúc Ly — Nữ · Bắc · Phong cách tự nhiên", "Trúc Ly"),
    ("⭐ Thiện Minh — Nam · Bắc · Phong cách kể chuyện", "Thiện Minh"),
    ("⭐ Ngọc Huyền — Nữ · Bắc · Giọng đọc tự nhiên", "Ngọc Huyền"),
    ("Minh Đức — Nam · Bắc · Phong cách tin tức", "Minh Đức"),
]
VIENEU = [
    VoiceProfile(voice_id=voice, label=label, gender=gender_from_voice_label(label))
    for label, voice in VIENEU_LABELS
]
MALE, FEMALE = "vi-VN-NamMinhNeural", "vi-VN-HoaiMyNeural"


def _member(name: str, gender: str) -> CastMember:
    return CastMember(name, name, gender, "", 1, True)


class TestPureFunctions:
    def test_main_voice_kept_for_its_gender(self) -> None:
        assert default_gender_voices(VIENEU, "Minh Đức") == ("Minh Đức", "Trúc Ly")
        assert default_gender_voices(VIENEU, "Ngọc Huyền") == ("Adam bựa", "Ngọc Huyền")

    def test_defaults_without_main_voice(self) -> None:
        assert default_gender_voices(VIENEU, "") == ("Adam bựa", "Trúc Ly")
        unlabelled = [VoiceProfile("a", "A"), VoiceProfile("b", "B")]
        assert default_gender_voices(unlabelled) == ("a", "b")
        assert default_gender_voices([]) == ("", "")

    def test_sides(self) -> None:
        cast = [
            _member("龍傳人", VOICE_MALE), _member("秦可秀", VOICE_FEMALE),
            _member("dân làng", ""), _member("Người bán hàng", ""),
        ]
        sides = two_voice_sides(cast, VOICE_MALE, {"Người bán hàng": VOICE_FEMALE})
        assert sides == {
            "龍傳人": VOICE_MALE, "秦可秀": VOICE_FEMALE,
            "dân làng": VOICE_MALE, "Người bán hàng": VOICE_FEMALE,
        }
        assert two_voice_sides(cast, VOICE_FEMALE)["dân làng"] == VOICE_FEMALE
        # Lựa chọn hỏng trong cài đặt không được làm mất vai.
        broken = two_voice_sides(cast, VOICE_MALE, {"秦可秀": "x"})
        assert broken["秦可秀"] == VOICE_FEMALE

    def test_casting(self) -> None:
        assert two_voice_casting(
            {"a": VOICE_MALE, "b": VOICE_FEMALE}, "Minh Đức", "Ngọc Huyền"
        ) == {"a": "Minh Đức", "b": "Ngọc Huyền"}


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path: Path) -> QSettings:
    return QSettings(str(tmp_path / "tts.ini"), QSettings.Format.IniFormat)


def _two_voice_coordinator(
    settings: QSettings, events: list[SubtitleEvent] | None = None
):
    coordinator, panel, view_model = _coordinator(settings, events)
    panel.set_dubbing_enabled(True)
    assert coordinator.load_cast()
    panel._mode_combo.setCurrentIndex(panel._mode_combo.findData(CAST_MODE_TWO_VOICES))
    return coordinator, panel, view_model


def _role_combo(panel, name: str) -> QComboBox:
    row = [member.name for member in panel._members].index(name)
    combo = panel._table.cellWidget(row, 3)
    assert isinstance(combo, QComboBox)
    return combo


class TestTwoVoiceMode:
    def test_every_line_uses_one_of_two_voices(
        self, app: QApplication, settings: QSettings
    ) -> None:
        coordinator, panel, view_model = _two_voice_coordinator(settings)
        cast = {member.name: member.gender for member in view_model.dubbing_cast}
        casting = panel.casting()
        assert set(casting.values()) == {MALE, FEMALE}
        for name, voice in casting.items():
            assert voice == (FEMALE if cast[name] == VOICE_FEMALE else MALE)
        assert set(coordinator.line_voices()) - {""} == {MALE, FEMALE}
        assert "chỉ 2 giọng: 5 vai giọng nam, 1 vai giọng nữ" in panel.status()

    def test_per_role_mode_is_unchanged(
        self, app: QApplication, settings: QSettings
    ) -> None:
        coordinator, panel, _view_model = _coordinator(settings)
        panel.set_dubbing_enabled(True)
        assert coordinator.load_cast()
        assert panel.cast_mode() == CAST_MODE_PER_ROLE
        assert not panel._two_voice_box.isVisibleTo(panel)
        assert len(set(panel.casting().values())) > 2  # biến thể cao độ của Edge

    def test_choices_survive_reload(
        self, app: QApplication, settings: QSettings
    ) -> None:
        _coordinator_one, panel, _vm = _two_voice_coordinator(settings)
        _role_combo(panel, "阿積").setCurrentIndex(1)  # vai nam → giọng nữ
        panel._male_combo.setCurrentIndex(panel._male_combo.findData(f"{MALE}@-12Hz"))
        sides = json.loads(settings.value(SETTINGS_ROLE_SIDES_KEY))
        assert sides == {"阿積": VOICE_FEMALE}
        voices = settings.value(f"{SETTINGS_GENDER_VOICES_PREFIX}{ENGINE_KEY}")
        assert json.loads(voices) == {VOICE_MALE: f"{MALE}@-12Hz", VOICE_FEMALE: FEMALE}

        _coordinator_two, reloaded, _vm2 = _coordinator(settings)
        assert reloaded.cast_mode() == CAST_MODE_TWO_VOICES
        assert _coordinator_two.load_cast()
        casting = reloaded.casting()
        assert casting["阿積"] == FEMALE
        assert casting["龍傳人"] == f"{MALE}@-12Hz"

    def test_auto_cast_puts_roles_back_on_their_gender(
        self, app: QApplication, settings: QSettings
    ) -> None:
        coordinator, panel, _vm = _two_voice_coordinator(settings)
        _role_combo(panel, "阿積").setCurrentIndex(1)
        coordinator.auto_cast()
        assert panel.casting()["阿積"] == MALE
        assert json.loads(settings.value(SETTINGS_ROLE_SIDES_KEY)) == {}

    def test_unknown_gender_roles_follow_the_setting(
        self, app: QApplication, settings: QSettings
    ) -> None:
        events = [*_film_events(), SubtitleEvent(
            index=999, text="[Người bán hàng:] Xem cà chua này tươi ngon chưa kìa.",
            interval=TimeInterval(9000.0, 9002.0),
        )]
        _coordinator_one, panel, _vm = _two_voice_coordinator(settings, events)
        assert panel.casting()["Người bán hàng"] == MALE
        panel._unknown_combo.setCurrentIndex(panel._unknown_combo.findData(VOICE_FEMALE))
        assert panel.casting()["Người bán hàng"] == FEMALE
        assert panel.casting()["龍傳人"] == MALE  # vai rõ giới không đổi

    def test_side_labels_show_the_chosen_voices(
        self, app: QApplication, settings: QSettings
    ) -> None:
        _coordinator_one, panel, _vm = _two_voice_coordinator(settings)
        combo = _role_combo(panel, "余小蓮")
        assert combo.count() == 2
        assert combo.currentText() == "Giọng nữ · HoaiMy (nữ)"
        panel._female_combo.setCurrentIndex(
            panel._female_combo.findData(f"{FEMALE}@+12Hz")
        )
        relabelled = _role_combo(panel, "余小蓮").currentText()
        assert relabelled == "Giọng nữ · HoaiMy · cao (nữ)"


def test_tts_page_keeps_the_main_voice_as_the_male_voice(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import (
        JsonFileRepository,
    )
    from subtitles_extractor.presentation.pages import tts_page as tts_page_module
    from tests.unit.presentation.test_dubbing_bugfixes_v323594 import _source

    monkeypatch.setattr(
        tts_page_module, "QSettings",
        lambda *_args: QSettings(str(tmp_path / "page.ini"), QSettings.Format.IniFormat),
    )
    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp_path / "config.json"),
        i18n_data_dir=Path(tts_page_module.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp_path,
    )
    page = tts_page_module.TTSPage(container)
    page._engine_combo.setCurrentIndex(
        page._engine_combo.findData(tts_page_module._ENGINE_VIENEU)
    )
    page._vieneu_mode.setCurrentIndex(page._vieneu_mode.findData("v3turbo"))
    page._on_vieneu_catalog_loaded("v3turbo", VIENEU)
    page._on_vieneu_voices_loaded("v3turbo", [voice for _label, voice in VIENEU_LABELS])
    page._vieneu_voice.setCurrentIndex(page._vieneu_voice.findData("Minh Đức"))
    page._view_model.set_source_events(_film_events())
    page._dubbing.set_source_provider(_source)
    panel = page._dubbing_panel
    panel.set_dubbing_enabled(True)
    assert page._dubbing.load_cast()
    panel._mode_combo.setCurrentIndex(panel._mode_combo.findData(CAST_MODE_TWO_VOICES))
    assert panel.gender_voices() == ("Minh Đức", "Trúc Ly")
    assert set(page._dubbing.line_voices()) - {""} == {"Minh Đức", "Trúc Ly"}
