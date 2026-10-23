"""[v3.23.598] Sửa tay TỪNG DÒNG khi lồng tiếng: người nói (phân vai sai) và lời thoại.

Yêu cầu người dùng 10/10/2026: "phân vai sai ở khâu dịch thì người dùng có thể sửa lại và
có thể sửa cả lời thoại". Trước bản này: không chọn được người nói từng dòng (chỉ gõ tag
``[Tên:]`` ở trang Dịch rồi nạp lại), lời thoại chỉ sửa được SAU khi TTS cả phim, chỉnh
sửa mất khi nạp lại phụ đề.

Dữ liệu: dòng 490-499 nguyên văn của lượt dịch thật 10/10/2026 (phim "灭门惨案II",
``44887efa``, v3.23.595) — mốc câu gốc, mốc bản dịch, nhãn người nói/người nghe Visual
Cues, câu gốc, bản dịch; hai dòng bảng nhân vật thật. Dòng 495 "阿秀，你幫我" (gọi
A Tú nhờ giúp — người nói là chồng cô, 13/19 lượt phân tích gán 甄劍聲) mang nhãn
秦可秀; câu đáp 496 "好，我幫你" mang nhãn 甄劍聲 nên bản dịch xưng hô ngược: "Được,
anh giúp em."
"""  # noqa: RUF002 — dấu phẩy chữ Hán trích nguyên văn

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication

from subtitles_extractor.application.services.dubbing_cast import (
    DubbingSource,
    build_cast,
    line_speakers,
    role_choices,
)
from subtitles_extractor.application.services.dubbing_line_edits import (
    LineEdits,
    TextEdit,
    apply_speaker_overrides,
    decode_line_edits,
    describe_line_speakers,
    encode_line_edits,
    remember_line_edits,
    restorable_text_edits,
    subtitle_fingerprint,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_tts_port import (
    TTSRequest,
    TTSSegmentResult,
)
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from tests.unit.presentation.test_dubbing_bugfixes_v323594 import (
    _coordinator,
    _film_events,
)

#: Hai dòng bảng nhân vật thật của lượt ``44887efa``.
ROSTER = "\n".join([
    "秦可秀 (Tần Khả Tú) [nữ, ~25] — Người vợ đáng thương chịu số phận cực kỳ bi thảm. "
    "Cách xưng hô: với 甄劍聲 tự xưng 'em' gọi 'anh Kiếm'; với 醫生 tự xưng 'tôi' gọi "
    "'bác sĩ'; với 龍傳人 tự xưng 'tôi' gọi 'đồng chí công an'.",
    "甄劍聲 (Chân Kiếm Thanh) [nam, ~30] — Chồng của Tần Khả Tú: Cựu chiến binh bị bất "
    "lực do vết thương chiến tranh. Cách xưng hô: với 秦可秀 tự xưng 'anh' gọi 'Khả Tú' "
    "(hoặc 'A Tú'); với 朱正軍 tự xưng 'tôi' gọi 'cậu' (hoặc 'A Quân'); với 洪震 tự xưng "
    "'tôi' gọi 'ông'.",
])
#: (dòng, start gốc, end gốc, end bản dịch, người nói, người nghe, câu gốc, bản dịch)
LINES = [
    (490, 1897.6, 1898.533, 1898.533, "甄劍聲", "秦可秀", "我今天晚上很高興",
     "Đêm nay anh rất vui."),
    (491, 1899.2, 1900.067, 1900.067, "秦可秀", "甄劍聲", "我知道", "Em biết."),
    (492, 2024.067, 2025.2, 2025.733, "甄劍聲", "秦可秀", "為什麼都是不行的",
     "Tại sao cái gì cũng không được chứ?"),
    (493, 2026.733, 2028.667, 2028.667, "甄劍聲", "秦可秀", "為什麼？為什麼",  # noqa: RUF001
     "Tại sao? Tại sao chứ?"),
    (494, 2028.733, 2032.067, 2032.067, "秦可秀", "甄劍聲", "劍哥不要這樣嘛！",  # noqa: RUF001
     "Anh Kiếm, đừng thế này mà!"),
    (495, 2033.4, 2037.4, 2037.4, "秦可秀", "甄劍聲", "阿秀，你幫我，幫幫我..",  # noqa: RUF001
     "A Tú, em giúp anh, giúp anh với..."),
    (496, 2038.733, 2040.267, 2040.267, "甄劍聲", "秦可秀", "好，我幫你",  # noqa: RUF001
     "Được, anh giúp em."),
    (497, 2050.2, 2052.267, 2052.267, "秦可秀", "甄劍聲", "快點，快點.",  # noqa: RUF001
     "Nhanh lên, nhanh lên..."),
    (498, 2059.867, 2062.933, 2062.933, "甄劍聲", "秦可秀", "不行不行呀.",
     "Không được, không được rồi."),
    (499, 2063.467, 2064.267, 2064.8, "甄劍聲", "秦可秀", "根本就沒用",
     "Hoàn toàn không có tác dụng."),
]
P495, P496 = 5, 6
WIFE, HUSBAND = "秦可秀", "甄劍聲"
MALE, FEMALE = "vi-VN-NamMinhNeural", "vi-VN-HoaiMyNeural"
FIXED_496 = "Được, em giúp anh."


def _source() -> DubbingSource:
    cues = [
        {"id": i, "spk": row[4], "to": row[5], "cue": ""}
        for i, row in enumerate(LINES, 1)
    ]
    return DubbingSource(
        characters=ROSTER,
        visual_cues=json.dumps(cues, ensure_ascii=False),
        source_spans=tuple((row[1], row[2]) for row in LINES),
    )


def _events(column: int = 7) -> list[SubtitleEvent]:
    """Bản dịch (``column=7``, mốc kết thúc của bản dịch) hoặc câu gốc (``column=6``)."""
    end = 3 if column == 7 else 2
    return [
        SubtitleEvent(index=i, text=row[column], interval=TimeInterval(row[1], row[end]))
        for i, row in enumerate(LINES, 1)
    ]


# ── Hàm thuần ────────────────────────────────────────────────────────────────


class TestPureFunctions:
    def test_real_wrong_labels_are_what_the_app_finds(self) -> None:
        speakers = line_speakers(_events(), _source())
        assert speakers[P495] == WIFE  # sai: câu gọi "A Tú" mà nhãn là chính A Tú
        assert speakers[P496] == HUSBAND

    def test_overrides_swap_the_two_lines(self) -> None:
        auto = line_speakers(_events(), _source())
        fixed = apply_speaker_overrides(auto, {P495: HUSBAND, P496: WIFE, 99: "x"})
        assert fixed[P495] == HUSBAND
        assert fixed[P496] == WIFE
        assert fixed[:P495] == auto[:P495]
        assert len(fixed) == len(auto)

    def test_describe_lines(self) -> None:
        auto = line_speakers(_events(), _source())
        lines = describe_line_speakers(
            auto, {P495: HUSBAND, 9: ""}, {HUSBAND: MALE, WIFE: FEMALE}
        )
        assert (lines[P495].speaker, lines[P495].voice) == (HUSBAND, MALE)
        assert lines[P495].auto_speaker == WIFE
        assert lines[P495].overridden
        assert (lines[9].speaker, lines[9].voice, lines[9].overridden) == ("", "", True)
        assert (lines[P496].voice, lines[P496].overridden) == (MALE, False)

    def test_fingerprint_survives_translation_but_not_other_subtitles(self) -> None:
        source, translated = _events(6), _events(7)
        # Khâu dịch đổi mốc KẾT THÚC dòng 492, 499 nhưng không đổi mốc bắt đầu.
        assert [e.end_sec for e in source] != [e.end_sec for e in translated]
        assert subtitle_fingerprint(source) == subtitle_fingerprint(translated)
        assert subtitle_fingerprint(translated[:-1]) != subtitle_fingerprint(translated)
        shifted = [
            SubtitleEvent(index=e.index, text=e.text, interval=TimeInterval(
                e.start_sec + 1.0, e.end_sec + 1.0
            ))
            for e in translated
        ]
        assert subtitle_fingerprint(shifted) != subtitle_fingerprint(translated)
        assert subtitle_fingerprint([]) == ""

    def test_text_edits_restore_only_on_the_same_text(self) -> None:
        events = _events()
        edits = {
            P496: TextEdit(LINES[P496][7], FIXED_496),
            P495: TextEdit("bản dịch lượt trước", "khác"),
            40: TextEdit("a", "b"),
        }
        assert restorable_text_edits(events, edits) == {P496: edits[P496]}

    def test_encode_decode_round_trip(self) -> None:
        edits = LineEdits(
            speakers={P495: HUSBAND, P496: WIFE, 9: ""},
            texts={P496: TextEdit(LINES[P496][7], FIXED_496)},
        )
        restored = decode_line_edits(json.loads(json.dumps(encode_line_edits(edits))))
        assert restored == edits

    def test_decode_skips_broken_parts(self) -> None:
        raw = {
            "speakers": {"5": HUSBAND, "x": "a", "-1": "b", "6": 3},
            "texts": {"6": ["a", "b"], "7": ["chỉ một"], "8": "sai"},
        }
        assert decode_line_edits(raw) == LineEdits(
            speakers={5: HUSBAND}, texts={6: TextEdit("a", "b")}
        )
        assert decode_line_edits("hỏng") == LineEdits()

    def test_store_keeps_latest_subtitles(self) -> None:
        one = LineEdits(speakers={1: "a"})
        store = remember_line_edits({}, "phim1", one, max_subtitles=2)
        store = remember_line_edits(store, "phim2", one, max_subtitles=2)
        store = remember_line_edits(store, "phim1", one, max_subtitles=2)
        assert list(store) == ["phim2", "phim1"]
        store = remember_line_edits(store, "phim3", one, max_subtitles=2)
        assert list(store) == ["phim1", "phim3"]
        emptied = remember_line_edits(store, "phim3", LineEdits(), 2)
        assert emptied == {"phim1": store["phim1"]}

    def test_role_choices_add_roster_characters_without_lines(self) -> None:
        cast = build_cast([WIFE, WIFE], ROSTER)
        choices = role_choices(cast, ROSTER)
        assert [(m.name, m.line_count) for m in choices] == [(WIFE, 2), (HUSBAND, 0)]
        assert choices[1].display_name == "Chân Kiếm Thanh"
        assert choices[1].gender == "male"


# ── View model ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path: Path) -> QSettings:
    return QSettings(str(tmp_path / "tts.ini"), QSettings.Format.IniFormat)


def _scene_coordinator(settings: QSettings):
    coordinator, panel, view_model = _coordinator(settings, _events())
    coordinator.set_source_provider(_source)
    panel.set_dubbing_enabled(True)
    assert coordinator.load_cast()
    return coordinator, panel, view_model


class TestViewModel:
    def test_override_changes_cast_and_line_voices(
        self, app: QApplication, settings: QSettings
    ) -> None:
        coordinator, panel, view_model = _scene_coordinator(settings)
        before = {m.name: m.line_count for m in view_model.dubbing_cast}
        assert coordinator.line_voices()[P495] == FEMALE
        assert coordinator.set_line_speaker([P495], HUSBAND) == 1
        after = {m.name: m.line_count for m in view_model.dubbing_cast}
        assert after[HUSBAND] == before[HUSBAND] + 1
        assert after[WIFE] == before[WIFE] - 1
        assert coordinator.line_voices()[P495] == MALE
        assert "✎ 1 dòng chọn tay người nói." in panel.status()

    def test_text_edit_is_remembered_and_reverted(self, app: QApplication) -> None:
        from subtitles_extractor.presentation.view_models.tts_page_view_model import (
            TTSPageViewModel,
        )

        view_model = TTSPageViewModel(SimpleNamespace())
        view_model.set_source_events(_events())
        view_model.update_event_text(P496, LINES[P496][7] + "  ")  # chỉ khoảng trắng
        assert not view_model.has_line_edits()
        view_model.update_event_text(P496, FIXED_496)
        assert view_model.source_events[P496].text == FIXED_496
        assert view_model.line_edits().texts == {
            P496: TextEdit(LINES[P496][7], FIXED_496)
        }
        assert view_model.edited_positions == {P496}
        view_model.update_event_text(P496, "Được, em giúp anh nhé.")
        assert view_model.line_edits().texts[P496].original == LINES[P496][7]
        assert view_model.revert_line_edits([P496, 99]) == [P496]
        assert view_model.source_events[P496].text == LINES[P496][7]
        assert not view_model.has_line_edits()

    def test_tag_typed_into_the_text_changes_the_speaker(
        self, app: QApplication, settings: QSettings
    ) -> None:
        _coordinator_one, _panel, view_model = _scene_coordinator(settings)
        assert view_model.effective_line_speakers()[P495] == WIFE
        view_model.update_event_text(P495, f"[{HUSBAND}:] {LINES[P495][7]}")
        assert view_model.effective_line_speakers()[P495] == HUSBAND

    def test_new_subtitle_drops_edits_and_old_results(self, app: QApplication) -> None:
        from subtitles_extractor.presentation.view_models.tts_page_view_model import (
            TTSPageViewModel,
        )

        view_model = TTSPageViewModel(SimpleNamespace())
        view_model.set_source_events(_events())
        view_model.set_speaker_override([P495], HUSBAND)
        view_model._last_results = [_result(0, "seg.wav")]
        view_model._last_adapter = object()
        view_model.set_source_events(_film_events())
        assert not view_model.has_line_edits()
        assert not view_model.has_result
        assert not view_model.can_resynthesize


def _result(position: int, segment: str = "") -> TTSSegmentResult:
    row = LINES[position]
    return TTSSegmentResult(
        event_index=position, start_sec=row[1], end_sec=row[3], text=row[7],
        audio_duration_s=0.5, segment_audio_path=segment,
    )


class TestResynthesisBookkeeping:
    def _view_model(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, segment: str):
        from subtitles_extractor.presentation.view_models.tts_page_view_model import (
            TTSPageViewModel,
        )
        from subtitles_extractor.presentation.workers import resynthesize_worker

        started: list[dict] = []

        class _Worker:
            def __init__(self, _use_case: object, **kwargs: object) -> None:
                started.append(kwargs)
                signal = SimpleNamespace(connect=lambda *_a: None)
                self.progress_changed = self.finished_ok = signal
                self.failed = self.finished = self.cancelled = signal

            def start(self) -> None:
                return None

            def isRunning(self) -> bool:  # noqa: N802 — API Qt
                return False

        monkeypatch.setattr(resynthesize_worker, "ResynthesizeWorker", _Worker)
        output = tmp_path / "out.wav"
        sf.write(output, np.zeros(2400, dtype=np.float32), 24000)
        view_model = TTSPageViewModel(
            SimpleNamespace(make_resynthesize_use_case=lambda _adapter: object())
        )
        view_model.set_source_events(_events())
        view_model._last_adapter = object()
        view_model._last_output_path = output
        view_model._last_results = [_result(i, segment) for i in range(len(LINES))]
        view_model._last_request = TTSRequest(events=_events())
        return view_model, started

    def test_only_resynthesized_lines_stop_being_edited(
        self, app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        view_model, started = self._view_model(tmp_path, monkeypatch, "seg.wav")
        view_model.set_speaker_override([P495], HUSBAND)
        view_model.update_event_text(P496, FIXED_496)
        assert view_model.edited_positions == {P495, P496}
        assert view_model.resynthesize_lines({P496})
        # Luồng nền nhận BẢN SAO danh sách dòng (sửa tay giữa chừng không lọt vào).
        request = started[-1]["full_request"]
        assert request.events is not view_model.source_events
        assert request.events[P496].text == FIXED_496
        view_model._on_worker_finished(view_model.last_results)
        assert view_model.edited_positions == {P495}

    def test_edit_during_the_run_still_needs_resynthesis(
        self, app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        view_model, _started = self._view_model(tmp_path, monkeypatch, "seg.wav")
        view_model.update_event_text(P496, FIXED_496)
        assert view_model.resynthesize_lines({P496})
        view_model.update_event_text(P496, "Được, em giúp anh nhé.")
        view_model._on_worker_finished(view_model.last_results)
        assert view_model.edited_positions == {P496}

    def test_engine_without_line_audio_cannot_resynthesize(
        self, app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bản trước chạy tổng hợp các dòng chọn rồi mới báo lỗi ghép (Edge/Gemini)."""
        view_model, started = self._view_model(tmp_path, monkeypatch, "")
        assert not view_model.can_resynthesize
        assert not view_model.resynthesize_lines({P496})
        assert started == []


# ── Bộ điều phối lồng tiếng ──────────────────────────────────────────────────


class TestCoordinator:
    def test_vietnamese_name_maps_to_the_roster_role(
        self, app: QApplication, settings: QSettings
    ) -> None:
        coordinator, _panel, view_model = _scene_coordinator(settings)
        coordinator.set_line_speaker([P495], "Chân Kiếm Thanh")
        assert view_model.speaker_overrides == {P495: HUSBAND}

    def test_fixing_a_speaker_turns_dubbing_on(
        self, app: QApplication, settings: QSettings
    ) -> None:
        coordinator, panel, _view_model = _coordinator(settings, _events())
        coordinator.set_source_provider(_source)
        assert not panel.is_dubbing_enabled()
        coordinator.set_line_speaker([P495], HUSBAND)
        assert panel.is_dubbing_enabled()
        assert coordinator.line_voices()[P495] == MALE

    def test_other_roles_keep_their_voices(
        self, app: QApplication, settings: QSettings
    ) -> None:
        """Đo trên 19 dòng thật của phim: gán dòng 1 cho "Cậu bé trong thôn" làm phân vai
        tự động đổi giọng vai 阿積 (+12Hz → -12Hz) và "Lão nhân trong thôn" (-12Hz →
        gốc) — hai vai không liên quan mà audio đã tổng hợp vẫn là giọng cũ."""
        coordinator, panel, _view_model = _coordinator(settings)
        panel.set_dubbing_enabled(True)
        assert coordinator.load_cast()
        before = panel.casting()
        coordinator.set_line_speaker([0], "Cậu bé trong thôn")
        after = panel.casting()
        assert {name: after[name] for name in before} == before

    def test_cells_show_role_voice_and_edit_mark(
        self, app: QApplication, settings: QSettings
    ) -> None:
        coordinator, _panel, _view_model = _scene_coordinator(settings)
        coordinator.set_line_speaker([P495], HUSBAND)
        coordinator.set_line_speaker([P497], "")
        cells = coordinator.line_speaker_cells()
        assert cells[P494].text == "Tần Khả Tú · HoaiMy (nữ)"
        assert cells[P495].text == "✎ Chân Kiếm Thanh · NamMinh (nam)"
        assert "Tần Khả Tú" in cells[P495].tooltip
        assert cells[P497].text == "✎ (giọng chung)"
        assert not cells[P494].overridden

    def test_other_films_roster_is_not_offered(
        self, app: QApplication, settings: QSettings
    ) -> None:
        """Phụ đề TTS là phim khác (phim "he thong sung the", lượt thật ``fed6ba08``) với
        bảng nhân vật "灭门惨案II" trên trang Dịch: không gợi ý nhân vật phim kia."""
        from tests.unit.presentation.test_dubbing_bugfixes_v323594 import (
            FILM_WINDOW,
            _other_events,
        )

        same_film, _panel, _vm = _coordinator(settings, _film_events(FILM_WINDOW))
        names = [member.display_name for member in same_film.role_choices()]
        assert names[len(_vm.dubbing_cast):] == [
            "Cậu bé trong thôn", "Lão nhân trong thôn"
        ]
        other_film, _panel2, _vm2 = _coordinator(settings, _other_events())
        assert other_film.role_choices() == []
        other_film.set_line_speaker([0], "Long Truyền Nhân")
        assert _vm2.speaker_overrides == {0: "Long Truyền Nhân"}

    def test_revert_returns_the_auto_speaker(
        self, app: QApplication, settings: QSettings
    ) -> None:
        coordinator, _panel, view_model = _scene_coordinator(settings)
        coordinator.set_line_speaker([P495, P496], HUSBAND)
        coordinator.revert_lines([P495])
        assert view_model.speaker_overrides == {P496: HUSBAND}
        assert coordinator.line_voices()[P495] == FEMALE


P494, P497 = 4, 7


# ── Trang TTS ────────────────────────────────────────────────────────────────


@pytest.fixture
def page(app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import (
        JsonFileRepository,
    )
    from subtitles_extractor.presentation.pages import tts_page as tts_page_module

    ini = tmp_path / "page.ini"
    monkeypatch.setattr(
        tts_page_module, "QSettings",
        lambda *_args: QSettings(str(ini), QSettings.Format.IniFormat),
    )
    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp_path / "config.json"),
        i18n_data_dir=Path(tts_page_module.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp_path,
    )
    tts = tts_page_module.TTSPage(container)
    tts.set_dubbing_source_provider(_source)
    return tts


def _row_of(tts, position: int) -> int:
    for row in range(tts._table.rowCount()):
        if tts._table.item(row, 0).data(Qt.ItemDataRole.UserRole) == position:
            return row
    raise AssertionError(position)


def _trigger(menu, text: str) -> None:
    for action in menu.actions():
        if action.text().startswith(text):
            action.trigger()
            return
        if action.menu() is not None and action.text() != "":
            for sub in action.menu().actions():
                if sub.text().startswith(text):
                    sub.trigger()
                    return
    raise AssertionError(f"không có mục {text!r}: {[a.text() for a in menu.actions()]}")


class TestTTSPage:
    def test_lines_show_before_synthesis_and_only_text_is_editable(self, page) -> None:
        page._view_model.set_source_events(_events())
        table = page._table
        assert table.rowCount() == len(LINES)
        assert table.item(P496, 2).text() == LINES[P496][7]
        assert table.item(P496, 6).text() == "⏳ Chưa tổng hợp"
        editable = Qt.ItemFlag.ItemIsEditable
        assert table.item(P496, 2).flags() & editable
        for column in (0, 1, 6):
            assert not table.item(P496, column).flags() & editable

    def test_result_rows_only_edit_the_text(self, page) -> None:
        """Bản trước: nhấp đúp sửa được cả "#", "Thời gian", "Trạng thái"."""
        page._view_model.set_source_events(_events())
        page._on_result_ready([_result(i) for i in range(len(LINES))])
        editable = Qt.ItemFlag.ItemIsEditable
        assert page._table.item(P496, 2).flags() & editable
        for column in (0, 1, 3, 4, 5, 6):
            assert not page._table.item(P496, column).flags() & editable

    def test_fix_the_real_swapped_pair_from_the_table(self, page) -> None:
        page._dubbing_panel.set_dubbing_enabled(True)
        page._view_model.set_source_events(_events())
        controller = page._line_edits
        table = page._table
        assert not table.isColumnHidden(7)
        assert table.item(_row_of(page, P495), 7).text().startswith("Tần Khả Tú · ")

        _trigger(controller.build_speaker_menu([P495]), "Chân Kiếm Thanh")
        _trigger(controller.build_speaker_menu([P496]), "Tần Khả Tú")
        table.item(_row_of(page, P496), 2).setText(FIXED_496)
        controller.refresh()

        assert table.item(_row_of(page, P495), 7).text().startswith("✎ Chân Kiếm Thanh")
        assert table.item(_row_of(page, P496), 7).text().startswith("✎ Tần Khả Tú")
        voices = page._dubbing.line_voices()
        assert voices[P495] != voices[P496]
        assert page._dubbing.line_speakers()[P495].speaker == HUSBAND
        assert page._view_model.source_events[P496].text == FIXED_496

    def test_edits_come_back_when_the_subtitle_is_loaded_again(self, page) -> None:
        page._view_model.set_source_events(_events())
        page._line_edits.apply_speaker([P495], HUSBAND)
        page._table.item(_row_of(page, P496), 2).setText(FIXED_496)

        page._view_model.set_source_events(_film_events())  # phim khác: không áp
        assert not page._view_model.has_line_edits()

        page._view_model.set_source_events(_events())  # nạp lại đúng phụ đề này
        assert page._view_model.speaker_overrides == {P495: HUSBAND}
        assert page._view_model.source_events[P496].text == FIXED_496
        assert page._table.item(_row_of(page, P496), 2).text() == FIXED_496
        assert "1 dòng người nói, 1 dòng lời thoại" in page._stage_lbl.text()

        retranslated = _events()
        retranslated[P496] = SubtitleEvent(
            index=P496 + 1, text="Ừ, em giúp anh.", interval=retranslated[P496].interval
        )
        page._view_model.set_source_events(retranslated)
        assert page._view_model.source_events[P496].text == "Ừ, em giúp anh."
        assert page._view_model.speaker_overrides == {P495: HUSBAND}

    def test_revert_puts_back_the_loaded_text(self, page) -> None:
        page._view_model.set_source_events(_events())
        page._table.item(_row_of(page, P496), 2).setText(FIXED_496)
        page._line_edits.apply_speaker([P496], WIFE)
        assert page._line_edits.revert([P496]) == 1
        assert page._table.item(_row_of(page, P496), 2).text() == LINES[P496][7]
        assert not page._view_model.has_line_edits()

    def test_other_role_dialog(self, page, monkeypatch: pytest.MonkeyPatch) -> None:
        from subtitles_extractor.presentation.pages import tts_line_edits

        page._view_model.set_source_events(_events())
        monkeypatch.setattr(
            tts_line_edits, "ask_text", lambda *_args: ("Chân Kiếm Thanh", True)
        )
        _trigger(page._line_edits.build_speaker_menu([P495]), "Vai khác")
        assert page._view_model.speaker_overrides == {P495: HUSBAND}

    def test_select_edited_lines_after_synthesis(self, page) -> None:
        page._view_model.set_source_events(_events())
        page._view_model._on_worker_finished([_result(i) for i in range(len(LINES))])
        page._line_edits.apply_speaker([P495], HUSBAND)
        page._table.item(_row_of(page, P496), 2).setText(FIXED_496)
        page._line_edits.refresh()
        assert page._line_edits._btn_select_edited.text().endswith("(2)")
        assert page._table.item(_row_of(page, P496), 2).background().color().isValid()
        assert page._line_edits.select_edited() == 2
        assert page._selected_event_indices() == [P495, P496]

    def test_resynthesis_button_needs_line_audio(self, page) -> None:
        page._view_model.set_source_events(_events())
        page._view_model._last_adapter = object()
        page._view_model._on_worker_finished([_result(i) for i in range(len(LINES))])
        page._table.selectRow(P496)
        assert not page._btn_resynth.isEnabled()
        page._view_model._last_results = [
            _result(i, "seg.wav") for i in range(len(LINES))
        ]
        page._on_result_selection_changed()
        assert page._btn_resynth.isEnabled()
