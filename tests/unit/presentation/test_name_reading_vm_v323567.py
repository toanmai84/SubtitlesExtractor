"""[v3.23.567] Hồi quy tầng giao diện: phiên đã lưu tên sai được sửa khi nạp lại/khi dịch.

Người dùng đã phân tích phim bằng bản v3.23.566 (bảng nhân vật ghi "Liễu Mộc Vân" cho
柳慕云). Không bắt phân tích lại 22 request: nạp lại video hoặc bấm dịch là tên được sửa,
lưu lại vào phiên, và bộ nhớ dịch (TM) còn tên sai cũng không lọt vào prompt.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from subtitles_extractor.application.services.name_reading_check import NameReadings
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.database.sqlite_translation_session_store import (
    SqliteTranslationSessionStore,
)

READINGS = NameReadings.from_encoded({"柳": "liễu", "慕": "mộ", "云": "vân"})
ROSTER = "柳慕云 (Liễu Mộc Vân) [nữ, ~25] — tổng giám đốc. Xưng hô: gọi 'chị Mộc Vân'"


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _events() -> list[SubtitleEvent]:
    return [
        SubtitleEvent(index=1, text="我叫柳慕云", interval=TimeInterval(0.0, 1.0)),
        SubtitleEvent(index=2, text="慕云姐", interval=TimeInterval(1.0, 2.0)),
    ]


def _vm(container: object):
    from subtitles_extractor.presentation.view_models.translate_page_view_model import (
        TranslatePageViewModel,
    )

    vm = TranslatePageViewModel(container)  # type: ignore[arg-type]
    vm.set_source_events(_events())
    return vm


def _container(tmp_path: Path) -> MagicMock:
    container = MagicMock()
    container.name_readings = READINGS
    container.translation_session_store = SqliteTranslationSessionStore(
        tmp_path / "s.db"
    )
    return container


def test_restored_session_is_corrected_and_saved(app, tmp_path: Path) -> None:
    from subtitles_extractor.application.services.translation_session_hashing import (
        hash_analysis_input,
    )

    container = _container(tmp_path)
    store = container.translation_session_store
    input_hash = hash_analysis_input([e.text for e in _events()], "Vietnamese")
    store.save_analysis(
        "video.mp4",
        characters=ROSTER,
        overview="Liễu Mộc Vân là nữ chính.",
        source_lang="zh",
        target_lang="Vietnamese",
        input_hash=input_hash,
    )
    restored = _vm(container).restore_analysis_for_video("video.mp4", "Vietnamese")
    assert restored is not None
    assert "Liễu Mộ Vân" in restored.characters
    assert "chị Mộ Vân" in restored.characters
    session = store.get("video.mp4")
    assert session is not None
    assert session.analysis_overview == "Liễu Mộ Vân là nữ chính."
    assert session.has_valid_analysis(input_hash)


def test_translation_request_and_memory_use_the_fixed_name(
    app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from subtitles_extractor.presentation.view_models import (
        translate_page_view_model as module,
    )

    captured: dict[str, object] = {}

    def _fake_worker(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(module, "TranslateWorker", _fake_worker)
    vm = _vm(_container(tmp_path))
    monkeypatch.setattr(
        vm,
        "retrieve_memory_for_lines",
        lambda *_a: "BỘ NHỚ DỊCH:\n我是柳慕云 => Tôi là Liễu Mộc Vân",
    )
    notified: list[object] = []
    vm.context_names_corrected.connect(notified.append)
    context = TranslationContext(
        target_lang="Vietnamese", characters=ROSTER, overview="Liễu Mộc Vân."
    )
    stage = TranslationStageConfig(
        kind=TranslationStageKind.LITERAL, model_name="gemini-3.6-flash"
    )
    assert vm.start_translation("key", 1, [stage], context, video_path="video.mp4")
    request = captured["request"]
    assert "Mộc Vân" not in request.context.characters  # type: ignore[attr-defined]
    assert "Mộc Vân" not in request.context.overview  # type: ignore[attr-defined]
    assert "Tôi là Liễu Mộ Vân" in request.context.overview  # type: ignore[attr-defined]
    assert notified and "Liễu Mộ Vân" in notified[0].characters  # type: ignore[attr-defined]


def test_container_without_table_changes_nothing(app, tmp_path: Path) -> None:
    container = _container(tmp_path)
    container.name_readings = MagicMock()  # không phải NameReadings → bỏ qua
    context = TranslationContext(target_lang="Vietnamese", characters=ROSTER)
    corrected, corrections = _vm(container)._correct_context_names(context, "video.mp4")
    assert corrected is context
    assert corrections == {}


def test_saved_translation_is_corrected_on_restore(app, tmp_path: Path) -> None:
    """Bản dịch v3.23.566 đã lưu (44 dòng "Mộc Vân") được sửa mà không phải dịch lại."""
    import json

    from subtitles_extractor.application.services.translation_session_hashing import (
        hash_analysis_input,
    )
    from subtitles_extractor.infrastructure.database import (
        sqlite_translation_session_store as session_store,
    )

    container = _container(tmp_path)
    store = container.translation_session_store
    input_hash = hash_analysis_input([e.text for e in _events()], "Vietnamese")
    store.save_analysis(
        "video.mp4",
        characters=ROSTER,
        overview="",
        source_lang="zh",
        target_lang="Vietnamese",
        input_hash=input_hash,
    )
    lines = [
        {"index": 1, "text": "Tôi tên là Liễu Mộc Vân."},
        {"index": 2, "text": "Chị Mộc Vân, gỗ mộc đây."},
    ]
    store.save_stage(
        "video.mp4",
        session_store.StageResult(
            stage_id="final",
            input_hash="",
            lines_json=json.dumps(lines, ensure_ascii=False),
        ),
    )
    vm = _vm(container)
    assert vm.try_restore_saved_analysis("video.mp4", "Vietnamese")
    texts = [event.text for event in vm.translated_events]
    assert texts == ["Tôi tên là Liễu Mộ Vân.", "Chị Mộ Vân, gỗ mộc đây."]
    saved = store.get("video.mp4")
    assert saved is not None
    final = next(stage for stage in saved.stages if stage.stage_id == "final")
    assert "Mộc Vân" not in final.lines_json
