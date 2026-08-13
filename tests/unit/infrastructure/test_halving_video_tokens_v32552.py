"""[v3.23.152] Test halving GIỮ ước lượng token VIDEO (quota không "mù" khi chia đôi).

Bug: hai lời gọi đệ quy trong halving (và đường vá cửa sổ) truyền tiếp ``video_files``
nhưng KHÔNG truyền ``video_token_estimate`` -> mặc định 0 -> khi chia đôi batch có đính
video, ``est_tokens`` gửi quota manager chỉ còn phần text -> acquire cho qua -> server
429 vì input vượt TPM. Đây chính là bug đã vá ở v3.23.19 nhưng tái xuất ở đường đệ quy.
"""

from __future__ import annotations

from typing import Any

from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
    _BatchCountMismatchError,
)

_VIDEO_TOKENS = 50_000


def _batch(n: int) -> list[TranslationLine]:
    return [
        TranslationLine(
            index=i + 1, start_ms=i * 1000, end_ms=i * 1000 + 900, text=f"你好 {i + 1}"
        )
        for i in range(n)
    ]


def test_halving_preserves_video_token_estimate() -> None:
    adapter = GeminiSubtitleTranslator(api_key="KEY_A")
    recorded_est: list[int] = []
    call_count = {"n": 0}

    def fake_call_gemini(
        model_name: str, prompt: str, config: Any, validator: Any,
        *, cancel_cb: Any = None, video_files: Any = None, est_tokens: int = 0,
    ) -> dict[str, Any]:
        recorded_est.append(est_tokens)
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Lần đầu (batch 4 dòng): ép kích hoạt halving.
            raise _BatchCountMismatchError("giả lập sai số dòng")
        # Các lần sau: trả đúng số dòng của <current_batch> trong prompt.
        import re

        line_nos = [int(m) for m in re.findall(r'"line_no":\s*(\d+)', prompt)]
        # prompt chứa cả ngữ cảnh; current_batch là khối liên tục — lấy theo validator:
        # đơn giản hoá bằng cách thử giảm dần kích thước tới khi validator chấp nhận.
        for size in range(len(line_nos), 0, -1):
            for start in range(0, len(line_nos) - size + 1):
                subset = line_nos[start:start + size]
                payload = {
                    "subtitles": [
                        {"line_no": no, "text": f"xin chào {no}"} for no in subset
                    ]
                }
                try:
                    validator(payload)
                except Exception:
                    continue
                return payload
        raise AssertionError("Không dựng được payload hợp lệ cho validator")

    adapter._call_gemini = fake_call_gemini  # type: ignore[method-assign]

    result = adapter._translate_single_batch(
        batch=_batch(4),
        source_before=[], source_after=[], history_before=[],
        start_idx=0, is_preprocess=False, is_literal=False,
        config=None, model_name="gemini-3.1-flash-lite", cancel_cb=None,
        video_files=["fake_video_handle"], dual_payload=False,
        video_token_estimate=_VIDEO_TOKENS,
    )

    assert len(result) == 4
    assert call_count["n"] >= 3  # 1 lần fail + >= 2 lời gọi cho hai nửa
    # BẤT BIẾN cốt lõi: MỌI lời gọi (kể cả các nửa sau halving) đều cộng token video.
    assert all(est >= _VIDEO_TOKENS for est in recorded_est), recorded_est


def test_no_video_estimate_stays_text_only() -> None:
    adapter = GeminiSubtitleTranslator(api_key="KEY_A")
    recorded_est: list[int] = []

    def fake_call_gemini(
        model_name: str, prompt: str, config: Any, validator: Any,
        *, cancel_cb: Any = None, video_files: Any = None, est_tokens: int = 0,
    ) -> dict[str, Any]:
        recorded_est.append(est_tokens)
        return {
            "subtitles": [
                {"line_no": i + 1, "text": f"xin chào {i + 1}"} for i in range(2)
            ]
        }

    adapter._call_gemini = fake_call_gemini  # type: ignore[method-assign]
    adapter._translate_single_batch(
        batch=_batch(2),
        source_before=[], source_after=[], history_before=[],
        start_idx=0, is_preprocess=False, is_literal=False,
        config=None, model_name="gemini-3.1-flash-lite", cancel_cb=None,
        video_files=None, dual_payload=False, video_token_estimate=0,
    )
    # Không video -> ước lượng thuần text (nhỏ hơn hẳn ngưỡng video).
    assert recorded_est and all(est < _VIDEO_TOKENS for est in recorded_est)
