"""[v3.23.688] Cảm xúc từng câu (nhãn cue của phân tích) → chỉ dẫn cách đọc riêng của Gemini TTS 3.8.

Nhãn thật từ phân tích tập 2 phim mẫu: "愤怒" (phẫn nộ), "自言自语" (tự nói một mình), "đáp", "nghĩ thầm".
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from subtitles_extractor.application.services.dubbing_cast import DubbingSource
from subtitles_extractor.application.services.line_delivery_style import delivery_style, line_delivery_styles
from subtitles_extractor.application.services.tts_line_cache import line_signatures
from subtitles_extractor.domain.ports.subtitle_tts_port import TTSRequest
from subtitles_extractor.infrastructure.tts import gemini_tts_adapter
from subtitles_extractor.infrastructure.tts.gemini_tts_adapter import _verbatim_contents


def _events() -> list[SimpleNamespace]:
    texts = ["Đồ đệ vô liêm sỉ!", "Sư thúc.", "Cũng tại ta năm đó", "Vâng."]
    return [SimpleNamespace(index=i + 1, text=t, start_sec=2.0 * i, end_sec=2.0 * i + 1.5) for i, t in enumerate(texts)]


def _source() -> DubbingSource:
    cues = [
        {"id": 1, "spk": "南宫梦", "cue": "愤怒"},
        {"id": 2, "spk": "叶然", "cue": "đáp"},
        {"id": 3, "spk": "南宫梦", "cue": "自言自语"},
        {"id": 4, "spk": "叶然"},
    ]
    spans = tuple((2.0 * i, 2.0 * i + 1.5) for i in range(4))
    return DubbingSource(characters="", visual_cues=json.dumps(cues, ensure_ascii=False), source_spans=spans)


def test_cue_labels_become_line_styles_and_neutral_ones_are_dropped() -> None:
    assert line_delivery_styles(_events(), _source()) == ("愤怒", "", "自言自语", "")
    assert delivery_style("  nghĩ   thầm ") == "nghĩ thầm"
    assert delivery_style("Đáp.") == "" and delivery_style("") == ""
    assert delivery_style("x" * 41) == ""                                   # mô tả cảnh dài, không phải giọng đọc
    no_cues = DubbingSource(visual_cues=json.dumps([{"id": 1, "spk": "叶然"}]), source_spans=((0.0, 1.5),))
    assert line_delivery_styles(_events()[:1], no_cues) == ()
    # Phụ đề phim khác (không khớp mốc) → không gán nhãn của phim này.
    other = [SimpleNamespace(text="x", start_sec=100.0 + i, end_sec=100.5 + i) for i in range(4)]
    assert line_delivery_styles(other, _source()) == ()


def test_style_goes_to_speech_metadata_never_into_spoken_text() -> None:
    assert _verbatim_contents("Ăn nói hàm hồ!", "") == "Ăn nói hàm hồ!"
    [content] = _verbatim_contents("Ăn nói hàm hồ!", "愤怒")
    [part] = content.parts
    assert part.text == "Ăn nói hàm hồ!" and part.speech_metadata.style == "愤怒"


def test_request_styles_by_position_and_for_subsets() -> None:
    request = TTSRequest(events=_events(), line_styles=("愤怒", "", "自言自语"))
    assert [request.style_at(i) for i in range(5)] == ["愤怒", "", "自言自语", "", ""]
    assert request.line_styles_for([2, 0]) == ("自言自语", "愤怒")
    assert TTSRequest(events=_events()).line_styles_for([0]) == ()


def test_line_cache_signature_changes_only_for_styled_lines() -> None:
    base = TTSRequest(events=_events(), speaker="Kore")
    styled = replace(base, line_styles=("愤怒", "", "", ""))
    before, after = line_signatures(base), line_signatures(styled)
    assert before[0] != after[0]                                            # đổi cách đọc → tổng hợp lại
    assert before[1:] == after[1:]                                          # dòng khác giữ cache (cả cache cũ)


def test_gemini_standard_sends_each_line_its_own_style(tmp_path: Path, monkeypatch) -> None:
    adapter = gemini_tts_adapter.GeminiTTSAdapter(api_key="k")
    seen: dict[str, str] = {}

    def fake_retry(client, model, text, config, request, cancel_cb, available_s=0.0, style=""):
        seen[text] = style
        return np.full(int(0.5 * 24000), 0.1, dtype=np.float32)

    monkeypatch.setattr(adapter, "is_available", lambda: True)
    monkeypatch.setattr(adapter, "_make_client", lambda **_: object())
    monkeypatch.setattr(adapter, "_standard_speech_config", lambda *a, **k: None)
    monkeypatch.setattr(adapter, "_standard_with_retry", fake_retry)
    request = TTSRequest(
        events=_events(), speaker="Kore", language="gemini-3.8-flash-lite-tts",
        line_styles=("愤怒", "", "自言自语", ""), normalize=False, double_pass=False, media_duration_s=10.0,
    )
    adapter.generate(request, tmp_path / "ep.wav")
    assert seen == {"Đồ đệ vô liêm sỉ!": "愤怒", "Sư thúc.": "", "Cũng tại ta năm đó": "自言自语", "Vâng.": ""}
