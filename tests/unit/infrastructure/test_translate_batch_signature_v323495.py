"""[v3.23.495] Chống hồi quy TypeError: chữ ký `_translate_single_batch` phải khớp lời gọi.

v3.23.483 thêm `video_refs=` vào lời gọi trong `translate_stage` nhưng hàm nhận là
`_translate_single_batch` (không phải `_call_gemini`) và hàm này chưa có tham số đó →
`TypeError: ... got an unexpected keyword argument 'video_refs'` làm HỎNG toàn bộ khâu dịch.
"""
from __future__ import annotations

import inspect
import re

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)


def test_batch_accepts_video_refs() -> None:
    params = inspect.signature(GeminiSubtitleTranslator._translate_single_batch).parameters
    assert "video_refs" in params


def test_call_gemini_accepts_video_refs() -> None:
    params = inspect.signature(GeminiSubtitleTranslator._call_gemini).parameters
    assert "video_refs" in params


def test_batch_forwards_video_refs_to_call_gemini() -> None:
    # Không chuyển tiếp thì heal 403 (tải lại đoạn video khi xoay key) sẽ không chạy.
    source = inspect.getsource(GeminiSubtitleTranslator._translate_single_batch)
    assert source.count("video_refs=video_refs") == source.count("video_files=video_files")


def test_translate_stage_kwargs_all_exist_in_batch_signature() -> None:
    """Mọi tham số `translate_stage` truyền phải tồn tại trong chữ ký hàm nhận."""
    stage_source = inspect.getsource(GeminiSubtitleTranslator.translate_stage)
    call_match = re.search(
        r"self\._translate_single_batch\((.*?)\n\s*\)", stage_source, re.DOTALL
    )
    assert call_match is not None, "không tìm thấy lời gọi _translate_single_batch"
    passed = set(re.findall(r"(\w+)\s*=", call_match.group(1)))
    accepted = set(
        inspect.signature(GeminiSubtitleTranslator._translate_single_batch).parameters
    )
    assert passed <= accepted, f"tham số không được nhận: {sorted(passed - accepted)}"
