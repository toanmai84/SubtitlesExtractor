"""[v3.23.143] Test lớp điều phối sửa JSON: json_repair (chính) -> bộ vá nội bộ -> retry.

Ưu tiên chất lượng+hiệu năng: JSON hỏng (kể cả thiếu phẩy GIỮA cấu trúc như lỗi S51E05)
được sửa NGAY bằng json_repair; chỉ khi bó tay (rỗng) mới trả None để tầng trên retry.
"""

from __future__ import annotations

import json

from subtitles_extractor.infrastructure.translation import (
    gemini_translation_adapter as adapter,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    _repair_json_text,
)


def test_repairs_missing_comma_mid_structure() -> None:
    # Đúng ca S51E05: "Expecting ',' delimiter" — bộ vá cắt cụt cũ KHÔNG sửa được,
    # json_repair khôi phục ĐẦY ĐỦ (giữ cả "b": 2).
    result = _repair_json_text('{"a": 1 "b": 2}')
    assert result is not None
    data, method = result
    assert data == {"a": 1, "b": 2}
    assert method == "json_repair"


def test_repairs_truncated_json() -> None:
    result = _repair_json_text(
        '{"characters": "A", "cues": [{"id": 1, "cue": "xin chào"}, {"id": 2'
    )
    assert result is not None
    data, _ = result
    assert data["characters"] == "A"
    assert data["cues"][0] == {"id": 1, "cue": "xin chào"}


def test_repairs_single_quotes_and_bare_keys() -> None:
    result = _repair_json_text("{cues:[{'id':1,'cue':'hi'}]}")
    assert result is not None
    data, _ = result
    assert data == {"cues": [{"id": 1, "cue": "hi"}]}


def test_empty_garbage_returns_none_for_retry() -> None:
    # Rác không có dữ liệu -> None để tầng trên RETRY thay vì nuốt rỗng.
    assert _repair_json_text("") is None
    assert _repair_json_text("   ") is None


def test_fallback_when_library_absent(monkeypatch) -> None:
    # Giả lập chưa cài json_repair -> phải tự lùi về bộ vá nội bộ cho JSON cắt cụt.
    monkeypatch.setattr(adapter, "_json_repair", None)
    result = _repair_json_text('{"cues": [{"id": 1}, {"id": 2},')
    assert result is not None
    data, method = result
    assert data == {"cues": [{"id": 1}, {"id": 2}]}
    assert method == "bộ vá nội bộ"
    # JSON hợp lệ vẫn parse được qua nhánh nội bộ.
    assert json.loads('{"ok": true}') == {"ok": True}
