"""[v3.23.666] Nhà cung cấp AI tuỳ chỉnh kiểu OpenAI (vd OxionAI ``https://oxionai.com/v1``).

Chạy bộ dịch THẬT (vòng thử lại, sửa JSON, kiểm lô) trên một máy chủ HTTP giả cục bộ nói đúng
giao thức ``/v1/chat/completions`` + ``/v1/models`` của OpenAI — không cần key thật.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleTranslationError,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)
from subtitles_extractor.infrastructure.translation.model_health import ModelHealth
from subtitles_extractor.infrastructure.translation.openai_compat_client import (
    DEFAULT_MAX_TOKENS,
    CustomProviderConfig,
    CustomProviderError,
    UnsupportedContentError,
    build_chat_payload,
    normalize_base_url,
    parse_chat_response,
)

MODEL = "gemini-3.7-flash"


# ── Máy chủ giả kiểu OpenAI ───────────────────────────────────────────────


class RawBody:
    """Thân trả lời nguyên văn (vd trang HTML khi người dùng quên ``/v1``)."""

    def __init__(self, data: bytes, content_type: str) -> None:
        self.data = data
        self.content_type = content_type


class FakeOpenAIServer:
    """Máy chủ ``/v1`` giả: mỗi POST lấy một bước kịch bản (``(mã HTTP, thân JSON)``)."""

    def __init__(self, script: list[tuple[int, Any]] | None = None,
                 models: list[dict[str, Any]] | None = None) -> None:
        self.script = list(script or [])
        self.models = models or []
        self.requests: list[dict[str, Any]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:
                pass

            def _reply(self, status: int, body: Any) -> None:
                if isinstance(body, RawBody):
                    data, ctype = body.data, body.content_type
                else:
                    data, ctype = json.dumps(body).encode("utf-8"), "application/json"
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                server.requests.append({
                    "method": "GET", "path": self.path,
                    "auth": self.headers.get("Authorization"),
                })
                if self.path != "/v1/models":
                    self._reply(404, {"error": {"message": "not found"}})
                    return
                self._reply(200, {"object": "list", "data": server.models})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                server.requests.append({
                    "method": "POST", "path": self.path,
                    "auth": self.headers.get("Authorization"), "body": body,
                })
                index = min(len(server.requests_post()) - 1, len(server.script) - 1)
                status, reply = server.script[index]
                self._reply(status, reply)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def requests_post(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["method"] == "POST"]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"

    def __enter__(self) -> FakeOpenAIServer:
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


def chat_ok(content: str, finish: str = "stop", usage: dict[str, Any] | None = None) -> tuple[int, Any]:
    return 200, {
        "id": "chatcmpl-1", "object": "chat.completion", "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": finish}],
        "usage": usage or {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
    }


def chat_error(status: int, message: str, code: str = "error") -> tuple[int, Any]:
    return status, {"error": {"code": code, "message": message, "type": "api_error"}}


def make_translator(base_url: str, *, retry_count: int = 3,
                    model_health: ModelHealth | None = None) -> GeminiSubtitleTranslator:
    translator = GeminiSubtitleTranslator(
        api_key="GEMINI_KEY_KHONG_DUNG",
        retry_count=retry_count,
        request_timeout_s=10,
        quota_manager=None,
        model_health=model_health,
        custom_provider=CustomProviderConfig(base_url=base_url, api_key="sk-test-123"),
    )
    translator._interruptible_sleep = lambda _s, _cb: False  # type: ignore[method-assign]
    return translator


def json_config(translator: GeminiSubtitleTranslator) -> Any:
    schema = {
        "type": "OBJECT",
        "properties": {"subtitles": {"type": "ARRAY", "items": {"type": "STRING"}}},
        "required": ["subtitles"],
    }
    return translator._build_config(
        temperature=0.3, response_schema=schema, system_instruction="Bạn là người dịch phim.",
        enable_thinking=True, model_name=MODEL, thinking_level="low",
    )


# ── Hàm thuần: dựng request / đọc phản hồi ─────────────────────────────────


def test_normalize_base_url_bo_duoi_thua() -> None:
    assert normalize_base_url(" https://oxionai.com/v1/ ") == "https://oxionai.com/v1"
    assert normalize_base_url("https://oxionai.com/v1/chat/completions") == "https://oxionai.com/v1"
    assert normalize_base_url("https://oxionai.com/v1/models") == "https://oxionai.com/v1"
    assert not CustomProviderConfig("oxionai.com/v1", "k").is_complete
    assert not CustomProviderConfig("https://oxionai.com/v1", " ").is_complete
    assert CustomProviderConfig("https://oxionai.com/v1", "k").is_complete


def test_payload_chuyen_system_schema_json_va_tham_so() -> None:
    config = SimpleNamespace(
        system_instruction="Quy tắc dịch", response_mime_type="application/json",
        response_schema={"type": "OBJECT", "properties": {"a": {"type": "STRING"}}},
        temperature=0.4, top_p=None, thinking_config=SimpleNamespace(thinking_level="low"),
    )
    payload = build_chat_payload(MODEL, ["<lines>你好</lines>"], config)
    system, user = payload["messages"]
    assert system["role"] == "system" and system["content"].startswith("Quy tắc dịch")
    assert '"properties":{"a":{"type":"STRING"}}' in system["content"]
    assert user == {"role": "user", "content": "<lines>你好</lines>"}
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["temperature"] == 0.4 and "top_p" not in payload
    assert payload["max_tokens"] == DEFAULT_MAX_TOKENS
    assert "thinking_config" not in json.dumps(payload)


def test_payload_anh_thanh_data_url_va_tu_choi_video() -> None:
    image = SimpleNamespace(inline_data=SimpleNamespace(mime_type="image/png", data=b"\x89PNG"),
                            file_data=None, text=None)
    payload = build_chat_payload(MODEL, ["mô tả", image], None)
    parts = payload["messages"][-1]["content"]
    assert parts[0] == {"type": "text", "text": "mô tả"}
    assert parts[1]["image_url"]["url"] == "data:image/png;base64,iVBORw=="
    video = SimpleNamespace(file_data=SimpleNamespace(file_uri="https://x/files/1"))
    with pytest.raises(UnsupportedContentError):
        build_chat_payload(MODEL, [video, "prompt"], None)


def test_doc_phan_hoi_finish_reason_usage_va_think() -> None:
    _, body = chat_ok("<think>suy nghĩ</think>\n{\"a\": 1}", finish="length", usage={
        "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
        "completion_tokens_details": {"reasoning_tokens": 20},
        "prompt_tokens_details": {"cached_tokens": 40},
    })
    response = parse_chat_response(body)
    assert response.text == '{"a": 1}'
    assert response.candidates[0].finish_reason == "MAX_TOKENS"
    usage = response.usage_metadata
    assert (usage.prompt_token_count, usage.candidates_token_count, usage.thoughts_token_count,
            usage.cached_content_token_count, usage.total_token_count) == (100, 30, 20, 40, 150)
    assert parse_chat_response(chat_ok("   ")[1]).text is None


def test_loi_co_ma_http_va_co_thu_lai() -> None:
    assert CustomProviderError(429, "slow down").retryable
    assert CustomProviderError(503, "busy").retryable
    assert not CustomProviderError(402, "insufficient_quota").retryable
    assert str(CustomProviderError(429, "x")).startswith("429 RESOURCE_EXHAUSTED")


# ── Bộ dịch thật trên máy chủ giả ─────────────────────────────────────────


def test_dich_qua_nha_cung_cap_tuy_chinh() -> None:
    with FakeOpenAIServer([chat_ok('{"subtitles": [{"line_no": 1, "text": "Xin chào"}]}')]) as server:
        translator = make_translator(server.base_url)
        assert translator.uses_custom_provider and translator.is_available()
        data = translator._call_gemini(MODEL, "<lines>你好</lines>", json_config(translator),
                                       lambda _d: None)
    assert data["subtitles"] == [{"line_no": 1, "text": "Xin chào"}]
    (request,) = server.requests_post()
    assert request["path"] == "/v1/chat/completions"
    assert request["auth"] == "Bearer sk-test-123"
    body = request["body"]
    assert body["model"] == MODEL and body["stream"] is False
    assert body["response_format"] == {"type": "json_object"}
    assert "Bạn là người dịch phim." in body["messages"][0]["content"]
    assert body["messages"][1]["content"] == "<lines>你好</lines>"


def test_json_hong_duoc_sua_nhu_gemini() -> None:
    # Model ngoài Gemini hay bọc JSON trong ```json … ``` — lớp sửa JSON sẵn có xử lý.
    with FakeOpenAIServer([chat_ok('```json\n{"subtitles": [{"line_no": 1, "text": "A"},]}\n```')]) as server:
        translator = make_translator(server.base_url)
        data = translator._call_gemini(MODEL, "p", json_config(translator), lambda _d: None)
    assert data["subtitles"] == [{"line_no": 1, "text": "A"}]


def test_may_chu_che_response_format_thi_bo_va_nho() -> None:
    script = [
        chat_error(400, "response_format json_object is not supported for this model"),
        chat_ok('{"subtitles": []}'),
        chat_ok('{"subtitles": []}'),
    ]
    with FakeOpenAIServer(script) as server:
        translator = make_translator(server.base_url)
        translator._call_gemini(MODEL, "p", json_config(translator), lambda _d: None)
        translator._call_gemini(MODEL, "p", json_config(translator), lambda _d: None)
    bodies = [r["body"] for r in server.requests_post()]
    assert "response_format" in bodies[0]
    assert "response_format" not in bodies[1] and "response_format" not in bodies[2]


def test_may_chu_che_max_tokens_thi_ha_dung_gioi_han_neu() -> None:
    script = [chat_error(400, "max_tokens is too large: 32768 > 16384"), chat_ok('{"subtitles": []}'),
              chat_ok('{"subtitles": []}')]
    with FakeOpenAIServer(script) as server:
        translator = make_translator(server.base_url)
        translator._call_gemini(MODEL, "p", json_config(translator), lambda _d: None)
        translator._call_gemini(MODEL, "p", json_config(translator), lambda _d: None)
    sent = [r["body"].get("max_tokens") for r in server.requests_post()]
    assert sent == [DEFAULT_MAX_TOKENS, 16384, 16384]


def test_max_tokens_khong_neu_gioi_han_thi_bo_truong() -> None:
    script = [chat_error(400, "Unsupported parameter: max_tokens"), chat_ok('{"subtitles": []}')]
    with FakeOpenAIServer(script) as server:
        translator = make_translator(server.base_url)
        translator._call_gemini(MODEL, "p", json_config(translator), lambda _d: None)
    first, second = (r["body"] for r in server.requests_post())
    assert first["max_tokens"] == DEFAULT_MAX_TOKENS and "max_tokens" not in second


def test_prompt_qua_dai_khong_lam_mat_max_tokens_ve_sau() -> None:
    """Lỗi "maximum context length" là do prompt — không được bỏ max_tokens cho mọi lô sau."""
    script = [
        chat_error(400, "This model's maximum context length is 128000 tokens; request too large"),
        chat_ok('{"subtitles": []}'),
    ]
    with FakeOpenAIServer(script) as server:
        translator = make_translator(server.base_url)
        with pytest.raises(SubtitleTranslationError):
            translator._call_gemini(MODEL, "p", json_config(translator), lambda _d: None)
        translator._call_gemini(MODEL, "p", json_config(translator), lambda _d: None)
    sent = [r["body"].get("max_tokens") for r in server.requests_post()]
    assert sent == [DEFAULT_MAX_TOKENS, DEFAULT_MAX_TOKENS]


@pytest.mark.parametrize("reply", [
    (200, {"hello": "world"}),
    (200, RawBody(b"<!doctype html><html>OxionAI</html>", "text/html; charset=utf-8")),
    (200, RawBody(b"not json at all", "text/plain")),
])
def test_sai_dia_chi_api_bao_ngay_khong_cho_qua_tai(reply: tuple[int, Any]) -> None:
    """Quên ``/v1`` → trang HTML / JSON lạ: trước đây 11 lượt thử, ~664 giây chờ "quá tải"."""
    with FakeOpenAIServer([reply]) as server:
        translator = make_translator(server.base_url, retry_count=3)
        slept: list[float] = []
        translator._interruptible_sleep = (  # type: ignore[method-assign]
            lambda seconds, _cb: slept.append(seconds) or False
        )
        with pytest.raises(SubtitleTranslationError) as info:
            translator._call_gemini(MODEL, "p", json_config(translator), lambda _d: None)
    assert len(server.requests_post()) == 1 and slept == []
    assert "/v1" in str(info.value)


def test_429_va_503_duoc_thu_lai() -> None:
    script = [chat_error(429, "Rate limit exceeded"), chat_error(503, "upstream overloaded"),
              chat_ok('{"subtitles": [{"line_no": 1, "text": "ok"}]}')]
    with FakeOpenAIServer(script) as server:
        translator = make_translator(server.base_url, retry_count=4)
        data = translator._call_gemini(MODEL, "p", json_config(translator), lambda _d: None)
    assert data["subtitles"][0]["text"] == "ok" and len(server.requests_post()) == 3


@pytest.mark.parametrize("status,message", [
    (401, "Invalid API key."),
    (402, "insufficient_quota: số dư không đủ"),
    (404, "model 'gemini-9' not found"),
])
def test_loi_vinh_vien_dung_ngay_khong_thu_lai(status: int, message: str) -> None:
    with FakeOpenAIServer([chat_error(status, message)]) as server:
        translator = make_translator(server.base_url, retry_count=5)
        with pytest.raises(SubtitleTranslationError):
            translator._call_gemini(MODEL, "p", json_config(translator), lambda _d: None)
    assert len(server.requests_post()) == 1


def test_video_bi_bo_thay_vi_hong_loi_goi() -> None:
    video = SimpleNamespace(file_data=SimpleNamespace(file_uri="https://x/files/1"),
                            uri="https://x/files/1", start_sec=0.0, end_sec=60.0)
    with FakeOpenAIServer([chat_ok('{"subtitles": []}')]) as server:
        translator = make_translator(server.base_url)
        translator._call_gemini(MODEL, "p", json_config(translator), lambda _d: None,
                                video_files=[video])
        assert translator._resolve_video_handles([SimpleNamespace(remote_name="files/1")]) == []
    (request,) = server.requests_post()
    assert request["body"]["messages"][1]["content"] == "p"


def test_het_thoi_han_la_loi_thu_lai_duoc(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    from subtitles_extractor.infrastructure.translation import openai_compat_client as occ

    def _timeout(*_a: Any, **_k: Any) -> Any:
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", _timeout)
    client = occ.OpenAICompatModule("https://oxionai.com/v1").Client(
        api_key="k", http_options={"timeout": 30_000})
    with pytest.raises(CustomProviderError) as info:
        client.models.generate_content(model=MODEL, contents="p", config=None)
    assert info.value.retryable and "timed out" in str(info.value)


def test_thoi_han_moi_loi_goi_theo_http_options(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    from subtitles_extractor.infrastructure.translation import openai_compat_client as occ

    seen: list[float] = []

    def _capture(_req: Any, timeout: float) -> Any:
        seen.append(timeout)
        raise OSError("connection reset")

    monkeypatch.setattr(urllib.request, "urlopen", _capture)
    client = occ.OpenAICompatModule("https://oxionai.com/v1").Client(
        api_key="k", http_options={"timeout": 30_000})
    for config in (None, SimpleNamespace(http_options=SimpleNamespace(timeout=450_000))):
        with pytest.raises(CustomProviderError):
            client.models.generate_content(model=MODEL, contents="p", config=config)
    assert seen == [30.0, 450.0]


def test_danh_sach_model_lay_ca_ban_pro_bo_model_khong_sinh_chu() -> None:
    models = [
        {"id": "gemini-3.7-flash"}, {"id": "gemini-3.1-pro"}, {"id": "gpt-5.5"},
        {"id": "claude-opus-5.5"}, {"id": "text-embedding-3-large"}, {"id": "gpt-4o-mini-tts"},
    ]
    health = ModelHealth(same_family_only=True)
    with FakeOpenAIServer(models=models) as server:
        translator = make_translator(server.base_url, model_health=health)
        names = translator.list_available_models()
    assert names == ["claude-opus-5.5", "gemini-3.1-pro", "gemini-3.7-flash", "gpt-5.5"]
    assert server.requests[0]["auth"] == "Bearer sk-test-123"
    assert set(health.known_models) == set(names)


def test_danh_sach_model_loi_thi_bao_ly_do() -> None:
    """Nút "Tải danh sách model" là cách người dùng thử địa chỉ/key: phải nêu lý do, không im lặng."""
    with FakeOpenAIServer() as server:
        translator = make_translator(server.base_url.replace("/v1", "/sai"))
        with pytest.raises(SubtitleTranslationError, match="404"):
            translator.list_available_models()


def test_khong_tach_giong_khi_dung_nha_cung_cap_tuy_chinh() -> None:
    with FakeOpenAIServer() as server:
        translator = make_translator(server.base_url)
        assert translator._diarize_with_keys(object(), "zh", None) == []
    assert server.requests == []


def test_du_phong_tu_dong_khong_nhay_len_ban_dat_hon() -> None:
    health = ModelHealth(same_family_only=True)
    health.set_known_models(["gemini-3.7-flash", "gemini-3.8-pro", "gemini-3.6-flash", "gpt-5.6",
                             "gpt-5.6-pro", "gemini-3.5-flash-lite"])
    assert health.fallbacks("gemini-3.7-flash") == ["gemini-3.6-flash"]
    assert health.fallbacks("gpt-5.6") == []
    # Lựa chọn dự phòng kiểu Gemini cũ mà nhà cung cấp không bán → bỏ qua (tránh 404 hỏng lời gọi).
    health.set_preferred_fallback("gemini-3.1-flash-lite")
    assert health.fallbacks("gemini-3.7-flash") == ["gemini-3.6-flash"]


def test_du_phong_tu_dong_chi_trong_cung_ho_model() -> None:
    health = ModelHealth(same_family_only=True)
    health.set_known_models(["gemini-3.7-flash", "gemini-3.6-flash", "claude-opus-5.5", "gpt-5.6-pro"])
    assert health.fallbacks("gemini-3.7-flash") == ["gemini-3.6-flash"]
    health.set_preferred_fallback("claude-opus-5.5")  # người dùng tự chọn (có trong danh sách) thì vẫn dùng
    assert health.fallbacks("gemini-3.7-flash")[0] == "claude-opus-5.5"
    loose = ModelHealth()
    loose.set_known_models(["gemini-3.7-flash", "gpt-5.6-pro"])
    assert "gpt-5.6-pro" in loose.fallbacks("gemini-3.7-flash")


def test_container_bat_tat_nha_cung_cap(tmp_path: Path) -> None:
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import (
        JsonFileRepository,
    )

    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp_path / "c.json"),
        i18n_data_dir=Path(__file__).resolve().parents[3] / "src" / "subtitles_extractor" / "data",
        user_data_dir=tmp_path,
    )
    gemini = container.make_subtitle_translator("GEMINI_KEY")
    assert not gemini.uses_custom_provider
    assert gemini._quota_manager is container.gemini_quota_manager
    container.set_custom_text_provider(CustomProviderConfig("https://oxionai.com/v1/", "sk-x"))
    custom = container.make_subtitle_translator("GEMINI_KEY")
    assert custom.uses_custom_provider and custom.is_available()
    assert custom._api_keys == ["sk-x"] and custom._quota_manager is None
    assert custom._model_health is container.custom_model_health
    assert custom._model_health is not container.gemini_model_health
    # Thiếu key / sai địa chỉ → quay về Gemini thay vì gọi một nhà cung cấp hỏng.
    container.set_custom_text_provider(CustomProviderConfig("oxionai.com", "sk-x"))
    assert not container.make_subtitle_translator("GEMINI_KEY").uses_custom_provider
    container.set_custom_text_provider(None)
    assert not container.make_subtitle_translator("GEMINI_KEY").uses_custom_provider


def test_chay_duoc_khi_may_khong_co_google_genai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nhà cung cấp tuỳ chỉnh không cần google-genai: dùng bộ types tối thiểu."""
    import builtins

    from subtitles_extractor.infrastructure.translation.openai_compat_client import (
        FALLBACK_TYPES,
    )

    real_import = builtins.__import__

    def _no_genai(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("google"):
            raise ImportError("No module named 'google.genai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_genai)
    with FakeOpenAIServer([chat_ok('{"subtitles": [{"line_no": 1, "text": "A"}]}')]) as server:
        translator = make_translator(server.base_url)
        assert translator._types_module is FALLBACK_TYPES
        config = json_config(translator)
        lowered = translator._strip_thinking_from_config(config)
        data = translator._call_gemini(MODEL, "p", lowered, lambda _d: None)
    assert data["subtitles"][0]["text"] == "A"
    body = server.requests_post()[0]["body"]
    assert body["response_format"] == {"type": "json_object"}
    assert "Bạn là người dịch phim." in body["messages"][0]["content"]


@pytest.mark.parametrize("url,key", [
    ("https://oxionai.com:abc/v1", "sk-1"),
    ("https://oxi onai.com/v1", "sk-1"),
    ("https:///v1", "sk-1"),
    ("https://oxionai.com/v1", "sk-\u200b1"),
    ("https://oxionai.com/v1", "sk 1"),
])
def test_cau_hinh_hong_khong_duoc_coi_la_du(url: str, key: str) -> None:
    assert not CustomProviderConfig(url, key).is_complete


def test_than_bi_cat_giua_chung_la_loi_mang_thu_lai_duoc(monkeypatch: pytest.MonkeyPatch) -> None:
    import http.client
    import urllib.request

    from subtitles_extractor.infrastructure.translation import openai_compat_client as occ

    class _Resp:
        headers: dict[str, str] = {"Content-Type": "application/json"}

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_a: Any) -> None:
            return None

        def read(self) -> bytes:
            raise http.client.IncompleteRead(b'{"choi', 100)

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: _Resp())
    client = occ.OpenAICompatModule("https://oxionai.com/v1").Client(api_key="k")
    with pytest.raises(CustomProviderError) as info:
        client.models.generate_content(model=MODEL, contents="p", config=None)
    assert info.value.retryable and info.value.code == 0


def test_key_nha_cung_cap_khong_bao_gio_gui_sang_google(tmp_path: Path) -> None:
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import (
        JsonFileRepository,
    )

    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp_path / "c.json"),
        i18n_data_dir=Path(__file__).resolve().parents[3] / "src" / "subtitles_extractor" / "data",
        user_data_dir=tmp_path,
    )
    container.set_custom_text_provider(CustomProviderConfig("https://oxionai.com/v1", "sk-secret"))
    provider = container.make_video_context_provider("sk-secret")
    assert "sk-secret" not in provider._known_api_keys
    provider = container.make_video_context_provider("G1\nsk-secret,G2")
    assert provider._known_api_keys == ["G1", "G2"]
