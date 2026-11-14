"""[v3.23.698] Rà soát quản lý API key Gemini ↔ nhà cung cấp tuỳ chỉnh (mỗi ca thất bại trên mã 3.23.697)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, QStandardPaths


@pytest.fixture()
def store(tmp_path: Path) -> Iterator[object]:
    QSettings.setPath(QSettings.Format.NativeFormat, QSettings.Scope.UserScope, str(tmp_path))
    from subtitles_extractor.presentation.ai_access_store import AiAccessStore

    yield AiAccessStore()
    QSettings.setPath(
        QSettings.Format.NativeFormat, QSettings.Scope.UserScope,
        QStandardPaths.writableLocation(QStandardPaths.StandardLocation.GenericConfigLocation),
    )


def _raw(key: str, value: str) -> None:
    from subtitles_extractor.presentation.ai_access_store import SETTINGS_APP, SETTINGS_ORG

    s = QSettings(SETTINGS_ORG, SETTINGS_APP)
    s.setValue(key, value)
    s.sync()


# ── 1. Nhiều key dán vào ô key chính / cấu hình cũ lưu danh sách phẩy ──────────────────────────────────────────


def test_comma_list_in_main_key_is_several_keys(store) -> None:
    _raw("api_key", "AIzaFAKE1, AIzaFAKE2")
    _raw("api_keys_extra", "AIzaFAKE3\nAIzaFAKE1")
    assert store.gemini_keys() == ["AIzaFAKE1", "AIzaFAKE2", "AIzaFAKE3"]
    store.set_gemini_keys(["AIzaFAKE4,AIzaFAKE5"])
    assert store.gemini_keys() == ["AIzaFAKE4", "AIzaFAKE5"]


# ── 2. Đèn "sẵn sàng" của nhà cung cấp tuỳ chỉnh dùng đúng phép kiểm của bộ dịch ───────────────────────────────


@pytest.mark.parametrize(("url", "key", "ready"), [
    ("https://oxionai.com/v1", "sk-ok", True),
    ("oxionai.com/v1", "sk-ok", False),                 # thiếu http(s)
    ("https://oxionai.com/v1", "sk-​ok", False),   # ký tự vô hình (không đặt được vào header)
    ("https://oxionai.com:abc/v1", "sk-ok", False),     # cổng sai
])
def test_custom_complete_matches_translator_check(store, url, key, ready) -> None:
    store.set_custom(url, key)
    assert store.custom_complete() is ready


# ── 3. TTS lại dòng dùng key HIỆN TẠI ─────────────────────────────────────────────────────────────────────────


def test_gemini_tts_adapter_takes_new_keys() -> None:
    from subtitles_extractor.infrastructure.tts.gemini_tts_adapter import GeminiTTSAdapter

    adapter = GeminiTTSAdapter("AIzaOLD1\nAIzaOLD2")
    adapter._daily_exhausted = {"AIzaOLD1", "AIzaOLD2"}
    adapter.set_api_keys("AIzaOLD2, AIzaNEW")
    assert adapter._api_keys == ["AIzaOLD2", "AIzaNEW"]
    assert adapter._daily_exhausted == {"AIzaOLD2"}          # key đã gỡ không còn; key mới chưa bị đánh dấu cạn
    assert adapter._api_key == "AIzaOLD2"


def test_resynthesis_refreshes_reused_adapter_keys() -> None:
    from subtitles_extractor.presentation.view_models.tts_page_view_model import TTSPageViewModel

    got: list[str] = []
    vm = SimpleNamespace(_last_adapter=SimpleNamespace(set_api_keys=got.append),
                         _gemini_key_provider=lambda: "AIzaNEW1\nAIzaNEW2")
    TTSPageViewModel._refresh_adapter_keys(vm)
    assert got == ["AIzaNEW1\nAIzaNEW2"]
    vm = SimpleNamespace(_last_adapter=object(), _gemini_key_provider=lambda: "x")   # Edge/VieNeu: không đụng
    TTSPageViewModel._refresh_adapter_keys(vm)


# ── 4–5. Lựa chọn model dự phòng không lẫn giữa hai nhà cung cấp ─────────────────────────────────────────────


def _vm(custom_active: bool):
    from subtitles_extractor.infrastructure.translation.model_health import ModelHealth

    container = SimpleNamespace(
        gemini_model_health=ModelHealth(seed_models=("gemini-3.8-flash", "gemini-3.1-flash-lite")),
        custom_model_health=ModelHealth(same_family_only=True),
    )
    return SimpleNamespace(_container=container, custom_provider_active=lambda: custom_active)


def test_fallback_chosen_on_custom_does_not_reach_gemini() -> None:
    from subtitles_extractor.presentation.view_models.translate_page_view_model import TranslatePageViewModel

    vm = _vm(custom_active=True)
    TranslatePageViewModel.set_fallback_model(vm, "gpt-4o-mini")
    assert vm._container.gemini_model_health.preferred_fallback == ""      # Gemini không bị ghi tên lạ
    # Model tự chọn được dùng ngay cả khi chưa bấm "Tải danh sách model" (khởi động lại / đổi key).
    assert vm._container.custom_model_health.fallbacks("gpt-4o") == ["gpt-4o-mini"]
    vm.custom_provider_active = lambda: False                                # quay về Gemini
    TranslatePageViewModel.set_fallback_model(vm, "gpt-4o-mini")
    assert vm._container.gemini_model_health.preferred_fallback == ""
    assert "gpt-4o-mini" not in vm._container.gemini_model_health.fallbacks("gemini-3.8-flash")
    TranslatePageViewModel.set_fallback_model(vm, "gemini-3.1-flash-lite")
    assert vm._container.gemini_model_health.preferred_fallback == "gemini-3.1-flash-lite"


def test_custom_model_list_survives_key_change_and_provider_toggle() -> None:
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.translation.openai_compat_client import CustomProviderConfig

    c = ApplicationContainer.__new__(ApplicationContainer)
    c.set_custom_text_provider(CustomProviderConfig("https://oxionai.com/v1", "sk-1"))
    health = c.custom_model_health
    health.set_known_models(["gpt-4o", "gpt-4o-mini"])
    c.set_custom_text_provider(CustomProviderConfig("https://oxionai.com/v1", "sk-2"))   # đổi key
    c.set_custom_text_provider(None)                                                     # sang Gemini
    c.set_custom_text_provider(CustomProviderConfig("https://oxionai.com/v1/", "sk-2"))  # quay lại
    assert c.custom_model_health is health and "gpt-4o-mini" in health.known_models
    c.set_custom_text_provider(CustomProviderConfig("https://other.example/v1", "sk-2"))   # đổi địa chỉ → quên
    assert c.custom_model_health is not health


# ── 6. Danh sách model đến muộn của nhà cung cấp đã đổi bị bỏ ─────────────────────────────────────────────────


def test_late_model_list_from_previous_provider_is_dropped() -> None:
    from subtitles_extractor.presentation.pages.translate_models import TranslateModelsMixin

    filled: list = []
    page = SimpleNamespace(
        _btn_refresh_models=SimpleNamespace(setEnabled=lambda *_: None, setText=lambda *_: None),
        _translator=SimpleNamespace(translate=lambda k: k),
        _models_request_custom=False,                     # bấm "Tải" lúc đang dùng Gemini…
        _custom_mode=lambda: True,                        # …rồi đổi sang tuỳ chỉnh trước khi xong
        _stage_preprocess=SimpleNamespace(set_model_items=filled.append),
    )
    TranslateModelsMixin._on_models_ready(page, ["gemini-3.8-flash"])
    assert filled == []


# ── 7. "Tải danh sách model" Gemini thử mọi key ───────────────────────────────────────────────────────────────


def test_model_list_tries_every_gemini_key() -> None:
    from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import GeminiSubtitleTranslator

    adapter = GeminiSubtitleTranslator(api_key="AIzaBAD\nAIzaGOOD")
    adapter.is_available = lambda: True                    # type: ignore[method-assign]
    tried: list[str] = []

    def once() -> list[str] | None:
        tried.append(adapter._api_key)
        return None if adapter._api_key == "AIzaBAD" else ["gemini-3.8-flash"]

    adapter._list_gemini_models_once = once                # type: ignore[method-assign]
    assert adapter.list_available_models() == ["gemini-3.8-flash"]
    assert tried == ["AIzaBAD", "AIzaGOOD"]


def test_not_ready_message_names_the_active_provider() -> None:
    from subtitles_extractor.presentation.pages.translate_constants import _BATCH_STOP_MARKERS
    from subtitles_extractor.presentation.view_models.translate_page_view_model import TranslatePageViewModel

    custom = TranslatePageViewModel._not_ready_message(SimpleNamespace(custom_provider_active=lambda: True))
    gemini = TranslatePageViewModel._not_ready_message(SimpleNamespace(custom_provider_active=lambda: False))
    assert "tuỳ chỉnh" in custom and "Gemini" not in custom
    assert "Gemini" in gemini
    for message in (custom, gemini):                    # "Dịch cả bộ" vẫn dừng hẳn (lỗi chung mọi tập)
        assert any(marker in message.lower() for marker in _BATCH_STOP_MARKERS)
