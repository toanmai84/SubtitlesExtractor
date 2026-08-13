"""[v3.23.142] Test lớp CỨU JSON cắt cụt + JSON hỏng được coi là RETRYABLE.

Log A.I.-Revolution: đoạn 2 phân tích lỗi "Expecting ',' delimiter" mà KHÔNG được thử lại
-> mất trắng đóng góp đoạn. Nguyên nhân: JSONDecodeError không nằm trong tập retryable.
Nay: (1) JSON hỏng -> retryable (gen lại, vì lỗi model thường nhất thời); (2) JSON CẮT CỤT
(hết token output) -> cứu một phần bằng cách vá ngoặc, không mất trắng cả đoạn.
"""

from __future__ import annotations

import json

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
    _repair_truncated_json,
)


class TestRepairTruncatedJson:
    def test_recover_cut_mid_string(self) -> None:
        truncated = (
            '{"characters": "A", "cues": [{"id": 1, "cue": "xin chào"}, '
            '{"id": 2, "cue": "tạm bi'
        )
        repaired = _repair_truncated_json(truncated)
        assert repaired is not None
        data = json.loads(repaired)
        assert data["characters"] == "A"
        assert data["cues"][0] == {"id": 1, "cue": "xin chào"}  # phần tử trọn được giữ

    def test_recover_trailing_comma(self) -> None:
        repaired = _repair_truncated_json('{"cues": [{"id": 1}, {"id": 2},')
        assert repaired is not None
        assert json.loads(repaired)["cues"] == [{"id": 1}, {"id": 2}]

    def test_recover_cut_mid_number(self) -> None:
        repaired = _repair_truncated_json(
            '{"cues": [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.'
        )
        assert repaired is not None
        assert json.loads(repaired)["cues"][0] == {"id": 1, "score": 0.9}

    def test_recover_dangling_key(self) -> None:
        repaired = _repair_truncated_json('{"a": 1, "cues": [{"id": 1}], "summary": ')
        assert repaired is not None
        data = json.loads(repaired)
        assert data["a"] == 1
        assert data["cues"] == [{"id": 1}]

    def test_empty_returns_none(self) -> None:
        assert _repair_truncated_json("") is None
        assert _repair_truncated_json("   ") is None


class TestJsonErrorRetryable:
    def test_json_decode_error_is_retryable(self) -> None:
        err = json.JSONDecodeError("Expecting ',' delimiter", '{"a": 1 "b": 2}', 8)
        assert GeminiSubtitleTranslator._is_retryable(err) is True

    def test_plain_value_error_not_retryable(self) -> None:
        # Lỗi thường (không phải JSON, không phải HTTP tạm thời) không được retry.
        assert GeminiSubtitleTranslator._is_retryable(ValueError("sai tham số")) is False
