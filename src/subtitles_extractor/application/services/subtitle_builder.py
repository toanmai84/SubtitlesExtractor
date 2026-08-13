"""Backward-compat shim cho module ``subtitle_builder``.

.. deprecated:: 3.0
    Toàn bộ logic đã được tách ra package
    :mod:`subtitles_extractor.application.services.subtitle_pipeline`.
    Module này chỉ còn re-export các symbol cũ để các import path hiện hữu
    (``from subtitles_extractor.application.services.subtitle_builder
    import SubtitleBuilder``) vẫn hoạt động.

    Mã nguồn mới nên import trực tiếp từ package:

    .. code-block:: python

        from subtitles_extractor.application.services.subtitle_pipeline \\
            import SubtitleBuilder

Trước refactor (v2.36):
    1 file, 1901 dòng, 1 class với 27 method — vi phạm SRP nghiêm trọng,
    khó test, khó maintain.

Sau refactor (v3.0):
    1 package 9 module, mỗi module 1 trách nhiệm. Orchestrator chỉ ~120
    dòng, các function thuần (pure) dễ unit test. F1 tương đương baseline.

Các private symbol (``_is_latin_gibberish``, ``_DIGIT_ONLY_TOKEN_REGEX``...)
cũng được re-export từ vị trí mới để giữ backward compat cho unit test
hiện hữu — đây là **bridge tạm thời**, test mới nên import trực tiếp từ
các module nội bộ trong :mod:`subtitle_pipeline`.
"""

from __future__ import annotations

from subtitles_extractor.application.services.subtitle_pipeline.box_filters import (
    is_latin_gibberish as _is_latin_gibberish,
)
from subtitles_extractor.application.services.subtitle_pipeline.constants import (
    DIGIT_ONLY_TOKEN_REGEX as _DIGIT_ONLY_TOKEN_REGEX,
)
from subtitles_extractor.application.services.subtitle_pipeline.constants import (
    LATIN_REPETITIVE_REGEX as _LATIN_REPETITIVE_REGEX,
)
from subtitles_extractor.application.services.subtitle_pipeline.frame_group import (
    FrameGroup,
)
from subtitles_extractor.application.services.subtitle_pipeline.orchestrator import (
    SubtitleBuilder,
    text_similarity_cached,
)
from subtitles_extractor.application.services.subtitle_pipeline.text_correction import (
    accumulate_confidence_bucket as _accumulate_confidence_bucket,
)
from subtitles_extractor.application.services.subtitle_pipeline.text_correction import (
    is_single_cjk_char as _is_single_cjk_char,
)
from subtitles_extractor.application.services.text_similarity import text_similarity

__all__ = [
    "_DIGIT_ONLY_TOKEN_REGEX",
    "_LATIN_REPETITIVE_REGEX",
    "FrameGroup",
    "SubtitleBuilder",
    "_accumulate_confidence_bucket",
    "_is_latin_gibberish",
    "_is_single_cjk_char",
    "text_similarity",
    "text_similarity_cached",
]
