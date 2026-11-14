"""[v3.23.724] "Đọc liền cụm câu" gộp mọi cụm trong MỘT lượt — trước đây từng cụm một làm đơ trang TTS.

Đo 3664 dòng / 916 cụm trên trang thật: mã cũ 222,7 s (916 lần vẽ lại bảng), mã mới 0,29 s (1 lần).
"""

from __future__ import annotations

import random
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from tests.unit.presentation.test_fragment_merge_v323705 import _REAL, _ev, _vm


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _film(blocks: int) -> list[SimpleNamespace]:
    events: list[SimpleNamespace] = []
    for block in range(blocks):
        events += [_ev(e.start_sec + block * 10.0, e.end_sec + block * 10.0, e.text) for e in _REAL]
    return events


def test_whole_film_merge_rebuilds_table_once(app: QApplication) -> None:
    vm = _vm(_film(916))
    structure: list[int] = []
    edits: list[int] = []
    vm.structure_changed.connect(lambda: structure.append(1))
    vm.line_edits_changed.connect(lambda: edits.append(1))
    started = time.perf_counter()
    merged = vm.merge_fast_fragments()
    elapsed = time.perf_counter() - started
    assert len(merged) == 916 and len(vm.source_events) == 916 * 3
    assert len(structure) == 1, "mỗi cụm một lần vẽ lại bảng = đơ ứng dụng"
    assert len(edits) == 1
    assert elapsed < 3.0
    assert vm.undo() and len(vm.source_events) == 916 * 4


def _loop_merge(vm) -> list[int]:  # noqa: ANN001 — cách làm CŨ: từng cụm một, từ cuối lên
    done = []
    for first, last in reversed(vm.fragment_merge_plan()):
        if vm.merge_lines(range(first, last + 1)) is not None:
            done.append((first, last))
    merged, removed = [], 0
    for first, last in sorted(done):
        merged.append(first - removed)
        removed += last - first
    return merged


@pytest.mark.parametrize("seed", range(6))
def test_batch_equals_one_by_one(app: QApplication, seed: int) -> None:
    rng = random.Random(seed)
    words = ["một", "hai,", "ba", "bốn năm", "sáu bảy tám", "chín?", "mười.", "đi", "về"]
    events, clock = [], 0.0
    for _ in range(120):
        length = rng.uniform(0.25, 2.0)
        text = " ".join(rng.choice(words) for _ in range(rng.randint(1, 5)))
        events.append(_ev(clock, clock + length, text))
        clock += length + rng.choice([0.0, 0.04, 0.1, 0.6])
    old, new = _vm(events), _vm(events)
    for vm in (old, new):
        vm.set_speaker_override(range(0, 120, 7), "B")
    assert old.fragment_merge_plan() == new.fragment_merge_plan()
    assert len(new.fragment_merge_plan()) >= 3
    expected = _loop_merge(old)
    assert new.merge_fast_fragments() == expected
    assert [(e.start_sec, e.end_sec, e.text) for e in new.source_events] == [
        (e.start_sec, e.end_sec, e.text) for e in old.source_events
    ]
    assert new.edited_positions == old.edited_positions
    assert new.effective_line_speakers() == old.effective_line_speakers()


def test_page_button_merges_film_without_freezing(app: QApplication, tmp_path) -> None:  # noqa: ANN001
    from pathlib import Path

    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
    from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
    from subtitles_extractor.infrastructure.settings.json_file_repository import JsonFileRepository
    from subtitles_extractor.presentation.pages import translate_page
    from subtitles_extractor.presentation.pages.tts_page import TTSPage

    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp_path / "c.json"),
        i18n_data_dir=Path(translate_page.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp_path,
    )
    page = TTSPage(container)
    try:
        page._view_model.set_source_events([
            SubtitleEvent(index=i + 1, interval=TimeInterval(e.start_sec, e.end_sec), text=e.text)
            for i, e in enumerate(_film(300))
        ])
        app.processEvents()
        started = time.perf_counter()
        assert page._line_edits.merge_fast_fragments() == 300
        app.processEvents()
        assert time.perf_counter() - started < 5.0      # mã cũ: ~70 s cho 300 cụm
        assert len(page._view_model.source_events) == 900
    finally:
        page.close()
