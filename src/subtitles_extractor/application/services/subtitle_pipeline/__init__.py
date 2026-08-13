"""Package pipeline xây dựng phụ đề từ kết quả OCR.

Cấu trúc:

* :mod:`constants` — regex pattern, lookup table.
* :mod:`frame_group` — dataclass :class:`FrameGroup` + utilities không gian.
* :mod:`text_correction` — sửa lỗi OCR cấp ký tự, restore ký tự bị drop.
* :mod:`box_filters` — lọc rác cấp box (Latin gibberish, single-char...).
* :mod:`spatial_filters` — lọc theo ROI alignment + Y-axis outlier.
* :mod:`voting` — ROVER voting + similarity helpers.
* :mod:`grouping` — gộp frame thành :class:`FrameGroup` (greedy/Viterbi).
* :mod:`event_filters` — hậu xử lý cấp event (echo trail, merge, filter rác).
* :mod:`orchestrator` — lớp facade :class:`SubtitleBuilder`.

Public API:
    >>> from subtitles_extractor.application.services.subtitle_pipeline import (
    ...     SubtitleBuilder, FrameGroup, text_similarity_cached,
    ... )
"""

from subtitles_extractor.application.services.subtitle_pipeline.frame_group import (
    FrameGroup,
)
from subtitles_extractor.application.services.subtitle_pipeline.orchestrator import (
    SubtitleBuilder,
    text_similarity_cached,
)

__all__ = ["FrameGroup", "SubtitleBuilder", "text_similarity_cached"]
