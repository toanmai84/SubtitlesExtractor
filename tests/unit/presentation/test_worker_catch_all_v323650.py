"""[v3.23.650] Worker nền phải luôn phát ``failed`` khi gặp ngoại lệ lạ — nếu không thread không dừng, trạng thái bận kẹt mãi."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from subtitles_extractor.presentation.workers.detection_workers import (  # noqa: E402
    DetectAutoRoiWorker,
    DetectHardsubWorker,
)
from subtitles_extractor.presentation.workers.embedded_extract_worker import (  # noqa: E402
    ExtractEmbeddedWorker,
    ListEmbeddedTracksWorker,
)
from subtitles_extractor.presentation.workers.seek_worker import SeekWorker  # noqa: E402
from subtitles_extractor.presentation.workers.transcribe_worker import TranscribeSpeechWorker  # noqa: E402

_app = QApplication.instance() or QApplication([])


class _Boom:
    """Use case ném ValueError (không thuộc SubtitlesExtractorError/OSError/RuntimeError)."""

    def __getattr__(self, name: str):
        def raiser(*_a, **_k):
            raise ValueError(f"boom {name}")
        return raiser


def _failures(worker) -> list[str]:
    messages: list[str] = []
    worker.failed.connect(messages.append)
    worker.run()
    return messages


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DetectHardsubWorker(_Boom(), metadata=None),
        lambda: ListEmbeddedTracksWorker(_Boom(), Path("x.mkv")),
        lambda: ExtractEmbeddedWorker(_Boom(), Path("x.mkv"), track=None),
        lambda: TranscribeSpeechWorker(_Boom(), Path("x.wav"), config=None),
    ],
    ids=["detect_hardsub", "list_embedded", "extract_embedded", "transcribe"],
)
def test_unexpected_exception_emits_failed_v323650(factory) -> None:
    messages = _failures(factory())
    assert len(messages) == 1 and "boom" in messages[0]


def test_auto_roi_unexpected_exception_emits_failed_and_done_v323650() -> None:
    worker = DetectAutoRoiWorker(_Boom(), metadata=None)
    done: list[int] = []
    worker.done.connect(lambda: done.append(1))
    messages = _failures(worker)
    assert len(messages) == 1 and done == [1]


def test_seek_worker_unexpected_exception_emits_failed_v323650() -> None:
    class _Reader:
        def seek(self, _t):
            raise OverflowError("cannot convert float infinity to integer")

    messages = _failures(SeekWorker(Path("x.mp4"), 1.0, reader=_Reader()))
    assert len(messages) == 1
