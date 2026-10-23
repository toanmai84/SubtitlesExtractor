"""[v3.23.482] Khâu DỊCH truyền callback tải-lại-video xuống adapter → heal 403 khi xoay key."""
from unittest.mock import MagicMock
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesUseCase,
)


def _uc(translator):
    uc = TranslateSubtitlesUseCase.__new__(TranslateSubtitlesUseCase)
    uc._translator = translator
    return uc


def test_passthrough_to_adapter():
    tr = MagicMock()
    uc = _uc(tr)
    cb = lambda k: []
    uc.set_video_reupload_callback(cb)
    tr.set_video_reupload_callback.assert_called_once_with(cb)


def test_adapter_without_method_is_safe():
    tr = MagicMock(spec=[])  # không có set_video_reupload_callback
    uc = _uc(tr)
    uc.set_video_reupload_callback(lambda k: [])  # không raise
