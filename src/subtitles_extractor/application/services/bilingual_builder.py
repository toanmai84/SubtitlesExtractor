"""Ghép phụ đề GỐC và bản DỊCH thành phụ đề SONG NGỮ (hai dòng mỗi câu).

[v3.23.116] Phục vụ người xem/người học muốn thấy cả nguyên văn (CJK) lẫn bản dịch tiếng
Việt trên cùng một câu. Đây là HÀM THUẦN (không phụ thuộc GUI/IO) nên dễ test: nhận hai
danh sách :class:`SubtitleEvent` đã căn theo ``index`` và trả về event mới với
``text`` gồm hai dòng, giữ nguyên thời gian/uid của bản dịch.
"""

from __future__ import annotations

from dataclasses import replace

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent

__all__ = ["build_bilingual_events"]


def build_bilingual_events(
    source_events: list[SubtitleEvent],
    translated_events: list[SubtitleEvent],
    *,
    translation_on_top: bool = False,
) -> list[SubtitleEvent]:
    """Tạo danh sách phụ đề song ngữ từ phụ đề gốc và bản dịch.

    Mỗi câu trong kết quả lấy thời gian/uid từ bản dịch, phần ``text`` là hai dòng:
    nguyên văn và bản dịch (thứ tự tuỳ ``translation_on_top``). Căn cặp gốc - dịch theo
    ``index``; nếu một câu dịch không tìm được câu gốc tương ứng thì chỉ giữ bản dịch.

    Args:
        source_events: Phụ đề gốc (nguyên văn).
        translated_events: Phụ đề đã dịch (quyết định thời gian & số câu của kết quả).
        translation_on_top: Đặt bản dịch lên dòng trên (mặc định gốc ở trên, dịch ở dưới).

    Returns:
        Danh sách :class:`SubtitleEvent` song ngữ mới (không sửa danh sách đầu vào).
    """
    source_by_index: dict[int, SubtitleEvent] = {ev.index: ev for ev in source_events}
    result: list[SubtitleEvent] = []
    for translated in translated_events:
        original = source_by_index.get(translated.index)
        original_text = (original.text if original else "").strip()
        translated_text = (translated.text or "").strip()
        if original_text and translated_text and original_text != translated_text:
            ordered = (
                [translated_text, original_text]
                if translation_on_top
                else [original_text, translated_text]
            )
            merged_text = "\n".join(ordered)
        else:
            # Thiếu một vế (hoặc trùng nhau) -> giữ phần có nội dung, tránh dòng trống.
            merged_text = translated_text or original_text
        result.append(replace(translated, text=merged_text))
    return result
