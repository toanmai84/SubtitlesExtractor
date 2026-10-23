"""[v3.23.589] Tầng giao diện: phiên ĐÃ LƯU có chức danh lệch bảng thuật ngữ được sửa.

Người dùng đã phân tích phim "灭门惨案II" (17 request của gemini-3.8-flash, hạn mức 20
request/ngày/key) và dùng lại phân tích đó qua nhiều lượt dịch. Bảng nhân vật của phân
tích ghi "gọi 'Sở trưởng'" trong khi bảng thuật ngữ ghi ``所長 = Trưởng đồn công an``.
Không bắt phân tích lại: nạp lại video hoặc bấm dịch là bảng nhân vật được đổi theo bảng
thuật ngữ, lưu vào phiên, và ô "Nhân vật" trên giao diện hiện đúng bản đã đổi.
"""

# ruff: noqa: RUF001 — dấu câu fullwidth CJK thuộc câu gốc chép từ phim

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from subtitles_extractor.application.services.translation_session_hashing import (
    hash_analysis_input,
)
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
from subtitles_extractor.infrastructure.translation.name_readings_loader import (
    HANVIET_READINGS_FILENAME,
    load_name_readings,
)

DATA_DIR = Path(__file__).resolve().parents[3] / "src" / "subtitles_extractor" / "data"

# Dòng 愛國 và mục thuật ngữ chép nguyên văn từ phân tích thật; dòng 龍傳人 rút gọn phần
# mô tả và các quy tắc xưng hô.
ROSTER = "\n".join([
    "龍傳人 (Long Truyền Nhân) [nam, ~35] — Trưởng đồn công an thôn Đông Long, tính tình "
    "khoe khoang, háo danh; xưng hô: với Ái Quốc tự xưng 'tôi' gọi 'cậu'",
    "愛國 (Ái Quốc) [nam, ~20] — công an viên đồn Đông Long, dáng người gầy, hay nịnh "
    "nọt cấp trên Long Truyền Nhân; xưng hô: với Long Truyền Nhân tự xưng 'tôi' gọi 'Sở "
    "trưởng'",
])
GLOSSARY = "所長 = Trưởng đồn công an"
FIXED_RULE = "với Long Truyền Nhân tự xưng 'tôi' gọi 'Trưởng đồn công an'"


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _events() -> list[SubtitleEvent]:
    return [
        SubtitleEvent(index=1, text="所長", interval=TimeInterval(0.0, 1.0)),
        SubtitleEvent(index=2, text="所長，我來", interval=TimeInterval(1.0, 2.0)),
    ]


def _container(tmp_path: Path) -> MagicMock:
    container = MagicMock()
    container.name_readings = load_name_readings(DATA_DIR / HANVIET_READINGS_FILENAME)
    container.translation_session_store = SqliteTranslationSessionStore(
        tmp_path / "s.db"
    )
    return container


def _vm(container: object) -> object:
    from subtitles_extractor.presentation.view_models.translate_page_view_model import (
        TranslatePageViewModel,
    )

    vm = TranslatePageViewModel(container)  # type: ignore[arg-type]
    vm.set_source_events(_events())
    return vm


def _context(glossary: str = GLOSSARY) -> TranslationContext:
    return TranslationContext(
        target_lang="Vietnamese", characters=ROSTER, glossary=glossary
    )


def test_restored_session_is_reconciled_and_saved(
    app: QApplication, tmp_path: Path
) -> None:
    container = _container(tmp_path)
    store = container.translation_session_store
    input_hash = hash_analysis_input([e.text for e in _events()], "Vietnamese")
    store.save_analysis(
        "video.mp4",
        characters=ROSTER,
        overview="Tóm tắt.",
        source_lang="zh",
        target_lang="Vietnamese",
        input_hash=input_hash,
        glossary=GLOSSARY,
    )
    restored = _vm(container).restore_analysis_for_video("video.mp4", "Vietnamese")
    assert restored is not None
    assert FIXED_RULE in restored.characters
    session = store.get("video.mp4")
    assert session is not None
    assert FIXED_RULE in session.analysis_characters
    assert session.analysis_glossary == GLOSSARY
    assert session.has_valid_analysis(input_hash)  # không phải phân tích lại


def test_translation_request_uses_the_glossary_title(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    monkeypatch.setattr(vm, "retrieve_memory_for_lines", lambda *_a: "")
    shown: list[object] = []
    messages: list[str] = []
    vm.context_names_corrected.connect(shown.append)  # type: ignore[attr-defined]
    vm.status_message.connect(messages.append)  # type: ignore[attr-defined]
    stage = TranslationStageConfig(
        kind=TranslationStageKind.LITERAL, model_name="gemini-3.5-flash-lite"
    )
    assert vm.start_translation(  # type: ignore[attr-defined]
        "key", 1, [stage], _context(), video_path="video.mp4"
    )
    request = captured["request"]
    assert "Sở trưởng" not in request.context.characters  # type: ignore[attr-defined]
    assert FIXED_RULE in request.context.characters  # type: ignore[attr-defined]
    # Ô "Nhân vật" trên giao diện phải hiện đúng bảng mà khâu dịch dùng.
    assert shown and FIXED_RULE in shown[0].characters  # type: ignore[attr-defined]
    assert any("所長: 'Sở trưởng' → 'Trưởng đồn công an' (1 chỗ)" in m for m in messages)


def test_title_change_alone_reports_no_name_corrections(
    app: QApplication, tmp_path: Path
) -> None:
    """Ánh xạ sửa TÊN rỗng: bản dịch đã lưu và bộ nhớ dịch không bị thay chữ nào."""
    context = _context()
    corrected, corrections = _vm(_container(tmp_path))._correct_context_names(  # type: ignore[attr-defined]
        context, "video.mp4"
    )
    assert corrected is not context
    assert corrections == {}


def test_agreeing_tables_change_nothing(app: QApplication, tmp_path: Path) -> None:
    context = _context(glossary="所長 = sở trưởng (trưởng đồn)")
    corrected, corrections = _vm(_container(tmp_path))._correct_context_names(  # type: ignore[attr-defined]
        context, "video.mp4"
    )
    assert corrected is context
    assert corrections == {}
