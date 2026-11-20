"""[v3.23.757] Phim 大乾悍刀行 (lượt 3.23.754): bảng cụm đọc nhầm sửa đúng "剔冲动" → "别冲动", rồi bước soát câu
bằng PaddleOCR-VL đọc lại ['別冲动', '剔冲动', '剔冲动'] (đúng cặp chữ dễ nhầm đó) và GHI ĐÈ ngược thành "剔冲动".
Bản đọc lại phải qua cùng bước sửa chữ (phồn → giản, bảng cụm đọc nhầm) trước khi so / bỏ phiếu."""

from __future__ import annotations

import inspect

from subtitles_extractor.application.services.subtitle_pipeline.geometry_check import (
    accept_vl_reread,
)
from subtitles_extractor.application.services.subtitle_pipeline.text_correction import (
    correct_hallucination_typos,
    normalize_traditional_to_simplified,
)
from subtitles_extractor.application.use_cases.extract_subtitles import (
    ExtractSubtitlesUseCase,
)

RAW = ["別冲动", "剔冲动", "剔冲动"]                                   # bản đọc lại thật (seraw, câu 2103)


def test_raw_rereads_overwrite_the_corrected_line() -> None:
    assert accept_vl_reread("别冲动", RAW, []) == "剔冲动"              # lỗi tái hiện trên mã cũ


def test_normalized_rereads_keep_the_corrected_line() -> None:
    normalized = [correct_hallucination_typos(normalize_traditional_to_simplified(text)) for text in RAW]
    assert normalized == ["别冲动"] * 3 and accept_vl_reread("别冲动", normalized, []) is None


def test_verify_loop_normalizes_before_voting() -> None:
    source = inspect.getsource(ExtractSubtitlesUseCase._verify_lines_loop)
    normalize = source.index("reread = [correct_hallucination_typos(normalize_traditional_to_simplified(text))")
    assert normalize < source.index('entry["reread"] = list(reread)') < source.index("accept_vl_reread(")
