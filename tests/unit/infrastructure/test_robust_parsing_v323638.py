"""[v3.23.638] Phản hồi của model sai kiểu không được làm sập lượt gán người nói / phân tích.

Fuzz 4000 mẫu: ``"cues": null|số|true`` làm ``_parse_visual_cue_items`` ném TypeError
('NoneType' object is not iterable) — cả lượt gán người nói của lô bị bỏ.
"""
from __future__ import annotations

import pytest

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)

parse = GeminiSubtitleTranslator._parse_visual_cue_items


@pytest.mark.parametrize("junk", [None, 0, 5, 2.5, True, "abc", {"id": 1}])
def test_non_list_items_give_no_cues(junk) -> None:
    assert parse(junk) == []                                   # type: ignore[arg-type]


def test_valid_items_still_parse_and_junk_items_are_skipped() -> None:
    cues = parse([{"id": 2, "spk": "叶然", "to": "南宫梦", "cue": "giận"}, None, 3, "x", {"id": "z"}], 10)
    assert [(c.line_no, c.speaker, c.addressee) for c in cues] == [(2, "叶然", "南宫梦")]
