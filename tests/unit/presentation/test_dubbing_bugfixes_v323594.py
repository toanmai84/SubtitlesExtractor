"""[v3.23.594] Hồi quy các lỗi của tính năng lồng tiếng v3.23.593 (rà lỗi sau phát hành).

1. Bảng vai không gắn với phụ đề đã nạp: nạp phụ đề phim KHÁC (hay "Tổng hợp cả
   bộ" các tập khác) thì dòng nào cũng "trùng thời gian" một câu gốc nào đó của phim
   trên trang Dịch và nhận giọng vai của phim đó. Thử thật: phụ đề "he thong sung
   the" (5540 dòng) với bảng vai "灭门惨案II" → 1488 dòng nhận giọng vai phim kia.
2. Mở lại ứng dụng: ô lồng tiếng vẫn bật nhưng chưa nạp vai → âm thầm đọc một giọng.
3. Chỉnh giọng vai phim B ghi đè lựa chọn đã lưu của phim A.
4. Cuộn chuột lướt qua bảng vai đổi giọng vai và lưu luôn (ô chọn tạo sau khi trang gắn
   chống cuộn).
5. TTS lại vài dòng sau khi đổi giọng một vai vẫn ra giọng cũ.
6. Worker GPU: vai thiếu giọng mượn giọng preset ĐẦU TIÊN (trùng vai khác) thay vì
   giọng chung.

Phụ đề phim thứ hai chép nguyên văn từ lượt dịch thật ``fed6ba08`` ("he thong sung the",
dòng 124-143) — cùng khoảng thời gian với các dòng 43-56 của phim "灭门惨案II".
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication, QComboBox

from subtitles_extractor.application.services.dubbing_cast import (
    DubbingSource,
    align_to_source,
    line_speakers,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_tts_port import TTSRequest
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.i18n.json_translator import JsonTranslator
from subtitles_extractor.infrastructure.tts import vieneu_gpu_subprocess as gpu_worker
from subtitles_extractor.infrastructure.tts.edge_voice_variants import edge_voice_catalog
from subtitles_extractor.presentation.pages.tts_dubbing import (
    SETTINGS_CASTING_PREFIX,
    TTSDubbingCoordinator,
)
from subtitles_extractor.presentation.view_models.tts_page_view_model import (
    TTSPageViewModel,
)
from subtitles_extractor.presentation.widgets.dubbing_cast_panel import DubbingCastPanel
from tests.unit.application.test_dubbing_cast_v323593 import ROSTER, ROWS, _source

DATA_DIR = Path(__file__).resolve().parents[3] / "src" / "subtitles_extractor" / "data"
EDGE = edge_voice_catalog(["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"])
ENGINE_KEY = "edge_vi-VN"
#: (dòng, start, end, bản dịch) — phim "he thong sung the", lượt dịch thật ``fed6ba08``.
OTHER_FILM = [
    (124, 254.533, 256.0, "kích hoạt hoàn tiền gấp mười lần."),
    (125, 256.067, 256.733, "Số tiền hoàn"),
    (126, 256.8, 258.0, "20 nghìn tệ đã về tài khoản."),
    (127, 258.333, 260.267, "Ôi chao, được thật đấy chứ!"),
    (128, 260.733, 261.733, "Chị dâu,"),
    (129, 264.467, 266.067, "em ở chỗ chị"),
    (130, 266.133, 266.933, "lâu thế này rồi,"),
    (131, 267.0, 268.667, "mà vẫn chưa trả tiền phòng cho chị."),
    (132, 268.8, 270.4, "Em nghĩ giờ trong tay có tiền rồi,"),
    (133, 270.467, 271.3, "nên chuyển cho chị ít,"),
    (134, 271.4, 271.933, "để gánh vác"),
    (135, 272.0, 272.667, "cùng chị một chút mà."),
    (136, 272.733, 274.667, "Cậu nhóc thối này nói cái gì thế hả?"),
    (137, 275.6, 276.6, "Đại gia của cậu đi rồi,"),
    (138, 276.667, 277.667, "cậu chính là người thân của chị."),
    (139, 277.733, 279.733, "Sao chị có thể nhận tiền của cậu được chứ?"),
    (140, 284.467, 285.333, "Tiền của cậu nè,"),
    (141, 285.4, 286.567, "chị trả lại cho cậu rồi đấy."),
    (142, 286.667, 288.477, "Chị chuyển thêm cho cậu hai nghìn nữa."),
    (143, 288.667, 289.8, "Không phải, em..."),
]
#: Các dòng 43-56 của "灭门惨案II" (cùng khoảng 254-336 giây).
FILM_WINDOW = [row for row in ROWS if 43 <= row[0] <= 56]


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path: Path) -> QSettings:
    return QSettings(str(tmp_path / "tts.ini"), QSettings.Format.IniFormat)


def _film_events(rows: list[tuple] = ROWS) -> list[SubtitleEvent]:
    return [
        SubtitleEvent(index=i, text=row[6], interval=TimeInterval(row[1], row[3]))
        for i, row in enumerate(rows, 1)
    ]


def _other_events() -> list[SubtitleEvent]:
    return [
        SubtitleEvent(index=line, text=text, interval=TimeInterval(start, end))
        for line, start, end, text in OTHER_FILM
    ]


def _coordinator(
    settings: QSettings, events: list[SubtitleEvent] | None = None
) -> tuple[TTSDubbingCoordinator, DubbingCastPanel, TTSPageViewModel]:
    translator = JsonTranslator(data_dir=DATA_DIR, default_locale="vi")
    panel = DubbingCastPanel(translator)
    view_model = TTSPageViewModel(SimpleNamespace())
    view_model.set_source_events(events if events is not None else _film_events())
    coordinator = TTSDubbingCoordinator(
        panel, view_model, settings, translator.translate, lambda: (ENGINE_KEY, EDGE),
    )
    coordinator.set_source_provider(_source)
    return coordinator, panel, view_model


class TestOtherFilmGetsNoRoleVoices:
    def test_real_case_other_film_lines_get_no_speaker(self) -> None:
        source = _source(FILM_WINDOW)
        assert line_speakers(_other_events(), source) == [""] * len(OTHER_FILM)

    def test_same_film_still_aligns(self) -> None:
        assert line_speakers(_film_events(FILM_WINDOW), _source(FILM_WINDOW)) == [
            row[4] for row in FILM_WINDOW
        ]

    def test_whole_film_shifted_keeps_its_speakers(self) -> None:
        """Phụ đề bị dời giờ cả bộ 1,5 giây (Biên tập → dời tất cả) vẫn là phim đó."""
        shifted = [
            SubtitleEvent(
                index=i, text=row[6], interval=TimeInterval(row[1] + 1.5, row[3] + 1.5)
            )
            for i, row in enumerate(ROWS, 1)
        ]
        assert line_speakers(shifted, _source()) == [row[4] for row in ROWS]

    def test_film_with_a_deleted_line_still_aligns(self) -> None:
        kept = ROWS[:5] + ROWS[6:]
        positions = align_to_source(_film_events(kept), _source().source_spans)
        assert positions == [0, 1, 2, 3, 4, *range(6, len(ROWS))]

    def test_coordinator_reloads_and_drops_the_old_cast(self, app, settings) -> None:
        coordinator, panel, view_model = _coordinator(settings)
        panel._enable_check.setChecked(True)
        coordinator.load_cast()
        assert coordinator.line_voices()
        view_model.set_source_events(_other_events())  # nạp phụ đề phim khác
        assert coordinator.line_voices() == ()
        assert "không cùng phim" in panel.status()

    def test_speaker_tags_still_work_on_another_film(self, app, settings) -> None:
        events = _other_events()
        events[4] = SubtitleEvent(
            index=128, text="[Dư Tiểu Liên:] Chị dâu,", interval=events[4].interval
        )
        coordinator, panel, _vm = _coordinator(settings, events)
        panel._enable_check.setChecked(True)
        voices = coordinator.line_voices()
        assert voices[4].startswith("vi-VN-HoaiMy")
        assert set(voices[:4] + voices[5:]) == {""}
        assert "chỉ dùng tag" in panel.status()


class TestEnabledWithoutLoading:
    def test_restart_with_dubbing_on_still_dubs(self, app, settings) -> None:
        """Ô lồng tiếng bật từ phiên trước, chưa bấm "Nạp vai" → vẫn lồng tiếng."""
        settings.setValue("dubbing_enabled", True)
        coordinator, panel, _vm = _coordinator(settings)
        assert panel.is_dubbing_enabled() and not panel.has_cast()
        voices = coordinator.line_voices()
        assert len(voices) == len(ROWS)
        assert all(voices)


class TestSavedChoices:
    def test_choice_of_another_film_is_kept(self, app, settings) -> None:
        settings.setValue(
            f"{SETTINGS_CASTING_PREFIX}{ENGINE_KEY}",
            json.dumps({"Chu Chính Quân": "vi-VN-NamMinhNeural@+12Hz"}),
        )
        coordinator, panel, _vm = _coordinator(settings)
        panel._enable_check.setChecked(True)
        coordinator.load_cast()
        row = [member.name for member in panel._members].index("老莫")
        combo = panel._table.cellWidget(row, 3)
        assert isinstance(combo, QComboBox)
        chosen = next(
            p.voice_id for p in EDGE if p.voice_id != combo.currentData()
        )
        combo.setCurrentIndex(combo.findData(chosen))
        saved = json.loads(settings.value(f"{SETTINGS_CASTING_PREFIX}{ENGINE_KEY}"))
        assert saved["Chu Chính Quân"] == "vi-VN-NamMinhNeural@+12Hz"
        assert saved["老莫"] == chosen


class TestWheelDoesNotChangeVoices:
    def test_wheel_over_an_unfocused_voice_box(self, app, settings) -> None:
        coordinator, panel, _vm = _coordinator(settings)
        panel._enable_check.setChecked(True)
        coordinator.load_cast()
        combo = panel._table.cellWidget(0, 3)
        assert isinstance(combo, QComboBox)
        before = combo.currentData()
        event = QWheelEvent(
            QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, -120),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase, False,
        )
        QApplication.sendEvent(combo, event)
        assert combo.currentData() == before
        assert combo.focusPolicy() == Qt.FocusPolicy.StrongFocus


class TestResynthesisUsesCurrentVoices:
    def test_new_role_voice_reaches_the_resynthesis_request(
        self, app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list[TTSRequest] = []

        class _Worker:
            def __init__(self, _use_case: Any, **kwargs: Any) -> None:
                captured.append(kwargs["full_request"])
                signal = SimpleNamespace(connect=lambda *_a: None)
                self.progress_changed = self.finished_ok = signal
                self.failed = self.finished = self.cancelled = signal

            def start(self) -> None:
                return None

            def isRunning(self) -> bool:  # noqa: N802 — API Qt
                return False

        import subtitles_extractor.presentation.workers.resynthesize_worker as module

        monkeypatch.setattr(module, "ResynthesizeWorker", _Worker)
        output = tmp_path / "out.wav"
        sf.write(output, np.zeros(2400, dtype=np.float32), 24000)
        view_model = TTSPageViewModel(
            SimpleNamespace(make_resynthesize_use_case=lambda _adapter: object())
        )
        view_model.set_source_events(_film_events())
        view_model._last_adapter = object()
        view_model._last_output_path = output
        # [v3.23.598] TTS lại cần audio từng dòng như kết quả VieNeu thật (Edge/Gemini
        # không giữ → nút bị tắt thay vì chạy rồi báo lỗi ghép).
        view_model._last_results = [
            SimpleNamespace(event_index=0, segment_audio_path="seg_000000.wav")
        ]
        view_model._last_request = TTSRequest(
            events=_film_events(), line_voices=("cũ",) * len(ROWS)
        )
        new_voices = ("mới",) * len(ROWS)
        assert view_model.resynthesize_lines({0}, line_voices=new_voices)
        assert captured[-1].line_voices == new_voices
        view_model._worker = None
        view_model._is_busy = False
        assert view_model.resynthesize_lines({0})
        assert captured[-1].line_voices == ("cũ",) * len(ROWS)


class TestGpuWorkerMissingRoleVoice:
    def test_missing_role_voice_falls_back_to_the_shared_voice(self) -> None:
        def get_preset_voice(name: str) -> dict[str, str]:
            if name == "Giọng thiếu":
                raise ValueError(f"Voice '{name}' not found.")
            return {"id": name}

        engine = SimpleNamespace(
            get_preset_voice=get_preset_voice,
            list_preset_voices=lambda: [("Thiện Minh — Nam", "Thiện Minh")],
        )
        voice = gpu_worker._voice_for_item(
            engine, {"voice_preset": "Giọng thiếu"}, "giọng chung", {}
        )
        assert voice == "giọng chung"


def test_roster_reference_is_real() -> None:
    """Dòng bảng nhân vật dùng chung với test lồng tiếng v3.23.593 (đã đối chiếu)."""
    assert "龍傳人 (Long Truyền Nhân)" in ROSTER
    assert isinstance(DubbingSource(characters=ROSTER).characters, str)
