"""[v3.23.622] MAX_TOKENS (suy nghĩ ăn hết ngân sách đầu ra) không còn làm mất video / vá JSON cụt.

thinking.md: khi ``max_output_tokens`` cạn, đầu ra cụt hoặc rỗng nhưng vẫn tính tiền token suy
nghĩ; cách chữa là hạ ``thinking_level``. Trước đây: (1) phản hồi rỗng + MAX_TOKENS bị coi là
"rỗng không rõ lý do", lần thứ hai ứng dụng BỎ VIDEO (mất căn cứ gán người nói của đoạn); (2)
JSON cụt của khâu phân tích được json_repair vá lặng lẽ thành kết quả "đầy đủ".
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from subtitles_extractor.infrastructure.translation.empty_response_policy import (
    EmptyResponse,
    should_drop_video,
)
from tests.unit.infrastructure.gemini_call_harness import build_translator, install_fake_clock


def _response(text: str | None, finish: str, *, thoughts: int = 0, served: str = "") -> Any:
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=finish)],
        prompt_feedback=None,
        model_version=served,
        usage_metadata=SimpleNamespace(
            prompt_token_count=10, candidates_token_count=0, thoughts_token_count=thoughts,
            cached_content_token_count=0, total_token_count=10 + thoughts,
        ),
    )


class _Server:
    def __init__(self, replies: list[Any]) -> None:
        self.replies = replies
        self.configs: list[Any] = []
        self.contents: list[list] = []

    def generate_content(self, model: str, contents: list, config: Any) -> Any:
        self.configs.append(config)
        self.contents.append(list(contents))
        return self.replies[min(len(self.configs) - 1, len(self.replies) - 1)]


class _Config(SimpleNamespace):
    def model_copy(self, update: dict) -> "_Config":
        return _Config(**{**self.__dict__, **update})


def _config(level: str) -> _Config:
    return _Config(
        temperature=None, response_mime_type="application/json", response_schema={},
        system_instruction="x", thinking_config=SimpleNamespace(thinking_level=level),
    )


def _translator(monkeypatch: pytest.MonkeyPatch, server: _Server):
    clock = install_fake_clock(monkeypatch)
    translator, _ = build_translator(monkeypatch, clock, None, server)
    translator._types_module = SimpleNamespace(
        ThinkingConfig=lambda thinking_level=None, **_k: SimpleNamespace(
            thinking_level=thinking_level),
    )
    return translator


def test_policy_never_drops_video_for_truncation() -> None:
    truncated = EmptyResponse(finish_reason="MAX_TOKENS")
    assert truncated.truncated and not truncated.blocked
    assert not should_drop_video(truncated, unexplained_empties=5)
    assert should_drop_video(EmptyResponse(finish_reason="SAFETY"), 0)


def test_empty_max_tokens_lowers_thinking_and_keeps_video(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _Server([_response(None, "MAX_TOKENS", thoughts=60000),
                      _response('{"ok": 1}', "STOP", thoughts=900)])
    translator = _translator(monkeypatch, server)
    video = SimpleNamespace(uri="files/abc")
    out = translator._call_gemini(
        "gemini-3.6-flash", "prompt", _config("high"), lambda d: None,
        video_files=[video], reject_truncated=True,
    )
    assert out == {"ok": 1}
    assert [c.thinking_config.thinking_level for c in server.configs] == ["high", "medium"]
    assert all(video in contents for contents in server.contents)  # video còn nguyên


def test_truncated_json_in_analysis_is_retried_not_repaired(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _Server([_response('{"characters": [{"name": "A"', "MAX_TOKENS"),
                      _response('{"characters": []}', "STOP")])
    translator = _translator(monkeypatch, server)
    out = translator._call_gemini(
        "gemini-3.6-flash", "p", _config("medium"), lambda d: None, reject_truncated=True)
    assert out == {"characters": []}
    assert len(server.configs) == 2
    assert server.configs[1].thinking_config.thinking_level == "low"


def test_translation_batches_still_repair_truncated_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Khâu dịch có validator đếm dòng + vá phần thiếu: giữ hành vi cũ (không reject)."""
    server = _Server([_response('{"subtitles": [{"line_no": 1, "text": "a"}', "MAX_TOKENS")])
    translator = _translator(monkeypatch, server)
    out = translator._call_gemini("gemini-3.6-flash", "p", _config("low"), lambda d: None)
    assert out["subtitles"][0]["text"] == "a" and len(server.configs) == 1


@pytest.mark.parametrize(("model", "level", "expected"), [
    ("gemini-3.6-flash", "high", "medium"),
    ("gemini-3.6-flash", "medium", "low"),
    ("gemini-3.6-flash", "low", "minimal"),
    ("gemini-3.8-flash", "low", None),        # 3.8 không nhận minimal
    ("gemini-4-argon", "low", None),
    ("gemini-3.1-flash-lite", "minimal", None),
])
def test_lower_thinking_level_steps(model: str, level: str, expected: str | None,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    translator = _translator(monkeypatch, _Server([]))
    lowered = translator._lower_thinking_level(model, _config(level))
    got = None if lowered is None else lowered.thinking_config.thinking_level
    assert got == expected


def test_call_summary_names_served_model_and_thoughts(caplog: pytest.LogCaptureFixture) -> None:
    from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
        GeminiSubtitleTranslator,
    )
    caplog.set_level("INFO")
    GeminiSubtitleTranslator._log_call_summary(
        "gemini-3.5-flash", _config("low"),
        _response("{}", "STOP", thoughts=0, served="gemini-3.6-flash"))
    assert "phục vụ bởi gemini-3.6-flash" in caplog.text and "thực=0" in caplog.text
