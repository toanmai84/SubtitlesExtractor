"""[v3.23.631] Không dò lại file video đã bị 403 với key hiện tại (nhật ký thật: 93 lần 403)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)

FOREIGN_403 = (
    "403 PERMISSION_DENIED. {'error': {'code': 403, 'message': 'You do not have permission "
    "to access the File files/abc or it may not exist.', 'status': 'PERMISSION_DENIED'}}"
)


class _Files:
    def __init__(self) -> None:
        self.gets: list[str] = []

    def get(self, name: str):
        self.gets.append(name)
        if name == "files/foreign":
            raise RuntimeError(FOREIGN_403)
        if name == "files/flaky":
            raise RuntimeError("500 INTERNAL")
        return SimpleNamespace(name=name)


@pytest.fixture
def translator(monkeypatch: pytest.MonkeyPatch):
    adapter = GeminiSubtitleTranslator(api_key="KEY_A\nKEY_B")
    files = _Files()
    monkeypatch.setattr(adapter, "_get_client", lambda: SimpleNamespace(files=files))
    return adapter, files


def _ref(name: str):
    return SimpleNamespace(remote_name=name)


def test_forbidden_file_is_probed_once_per_key(translator) -> None:
    adapter, files = translator
    refs = [_ref("files/foreign"), _ref("files/mine")]
    first = adapter._resolve_video_handles(refs)
    assert [h.name for h in first] == ["files/mine"]
    adapter._video_handle_cache.clear()
    second = adapter._resolve_video_handles(refs)
    assert [h.name for h in second] == ["files/mine"]
    assert files.gets.count("files/foreign") == 1            # lần 2 không dò lại


def test_another_key_probes_again(translator) -> None:
    adapter, files = translator
    adapter._resolve_video_handles([_ref("files/foreign")])
    adapter._switch_to_key(1)
    adapter._resolve_video_handles([_ref("files/foreign")])
    assert files.gets.count("files/foreign") == 2


def test_transient_errors_are_not_remembered(translator) -> None:
    adapter, files = translator
    adapter._resolve_video_handles([_ref("files/flaky")])
    adapter._resolve_video_handles([_ref("files/flaky")])
    assert files.gets.count("files/flaky") == 2
