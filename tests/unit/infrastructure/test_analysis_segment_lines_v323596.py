"""[v3.23.596] Mỗi đoạn phân tích nhận danh sách dòng của CHÍNH đoạn đó, kèm mốc giờ.

Lượt thật 10/10/2026 (phim "灭门惨案II", 17 đoạn ~344 s): khâu phân tích trả nhãn "ai
nói với ai" cho 446/1245 dòng (36%) vì prompt chỉ có ``số. chữ`` của cả phim — model phải
tự dò dòng nào thuộc đoạn và hiện lúc nào. 800 dòng còn lại phải quét bổ sung bằng model
nhỏ, mà nhãn quét bổ sung sai người nghe 29% so với 7% của khâu phân tích (đo ở
v3.23.581).

Dữ liệu: dòng 56-64 thật (mốc giây), ranh giới hai đoạn video đầu thật (0-344,18 s,
344,18-688,35 s).
"""

from __future__ import annotations

import json
import re
from unittest.mock import MagicMock, patch

from subtitles_extractor.application.services.visual_cue_batching import lines_in_clip
from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    _segment_cue_lines_directive,
)
from tests.unit.infrastructure.gemini_call_harness import bare_analysis_adapter

SEGMENTS = [(0.0, 344.1764705882353), (344.1764705882353, 688.3529411764706)]
LINES = [
    TranslationLine(index=index, start_ms=int(start * 1000), end_ms=int(end * 1000),
                    text=text)
    for index, start, end, text in [
        (56, 334.933, 335.8, "所長"),
        (57, 336.533, 338.6, "現場沒有發現被搜掠過"),
        (58, 339.8, 341.867, "不過在女死者的下體"),
        (59, 342.067, 343.6, "就發現了有一些精液"),
        (60, 345.933, 347.667, "叫所有男村民明天去派出所"),
        (61, 348.067, 349.867, "告訴他們太公派豬肉"),
        (62, 351.133, 352.667, "派豬肉"),
        (63, 363.867, 365.067, "正嗎"),
        (64, 365.8, 368.467, "簡直是英雄本色，靈氣迫人呀"),  # noqa: RUF001 — chữ thật
    ]
]


def _listed(prompt: str) -> list[dict]:
    match = re.search(r"<segment_lines>\n(.*)\n</segment_lines>", prompt, re.S)
    assert match, "prompt thiếu danh sách dòng của đoạn"
    return json.loads(match.group(1))


class TestDirective:
    def test_lines_belong_to_the_clip_by_start_time(self) -> None:
        assert [line.index for line in lines_in_clip(LINES, *SEGMENTS[0])] == [
            56, 57, 58, 59,
        ]
        assert [line.index for line in lines_in_clip(LINES, *SEGMENTS[1])] == [
            60, 61, 62, 63, 64,
        ]

    def test_times_are_relative_to_the_clip(self) -> None:
        items = _listed(_segment_cue_lines_directive(LINES, *SEGMENTS[1]))
        assert [(item["id"], item["t"]) for item in items] == [
            (60, "00:01"), (61, "00:03"), (62, "00:06"), (63, "00:19"), (64, "00:21"),
        ]
        assert items[0]["text"] == "叫所有男村民明天去派出所"

    def test_empty_clip_adds_nothing(self) -> None:
        assert _segment_cue_lines_directive(LINES, 700.0, 900.0) == ""


def _ref(index: int) -> MagicMock:
    ref = MagicMock()
    ref.chunk_index = index
    ref.remote_name = f"files/f{index}"
    ref.start_sec, ref.end_sec = SEGMENTS[index]
    return ref


def _analyse(with_cues: bool) -> list[str]:
    adapter = bare_analysis_adapter()
    prompts: list[str] = []

    def fake_call(_model, prompt, _config, _validator, video_refs=None, **_kw):
        prompts.append(prompt)
        return {"source_lang": "zh", "characters": [], "overview": "x", "cues": []}

    with (
        patch.object(adapter, "_resolve_video_handles", lambda refs: ["h0", "h1"]),
        patch.object(adapter, "_build_config", lambda *a, **k: MagicMock()),
        patch.object(adapter, "_call_gemini", fake_call),
        patch.object(adapter, "_ensure_available", lambda: None),
    ):
        adapter.analyze_global_context(
            LINES, "Vietnamese", video_refs=[_ref(0), _ref(1)],
            with_visual_cues=with_cues,
        )
    return prompts


def test_each_segment_prompt_lists_its_own_lines() -> None:
    first, second = _analyse(with_cues=True)
    assert [item["id"] for item in _listed(first)] == [56, 57, 58, 59]
    assert [item["id"] for item in _listed(second)] == [60, 61, 62, 63, 64]


def test_no_list_when_cues_are_off() -> None:
    assert all("<segment_lines>" not in prompt for prompt in _analyse(with_cues=False))
