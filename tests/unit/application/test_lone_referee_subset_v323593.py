# ruff: noqa: RUF001 — dấu câu fullwidth CJK thuộc câu gốc chép từ phim
"""[v3.23.593] Hồi quy: lưới dòng lẻ (v3.23.592) chạy với ADAPTER THẬT không lùi dòng nào.

Lượt dịch thật v3.23.592 (phim "灭门惨案II", ``0c0c7c9a``), giai đoạn địa hoá:

    Giai đoạn 4: Bản địa hoá: 7 dòng lẻ bị viết lại khác hẳn bản trước — trọng tài
    không đứng về bản trước ở dòng nào → giữ.

nhưng bản trọng tài thật ra trống ở 12/21 dòng và lệch ở các dòng còn lại (xem
``test_subset_reindex_v323593``), nên dòng 10 "Cười chết được." (câu gốc "搔你胳肢窩",
lời đám đông nghe được) và dòng 186 "phải tùy tùng theo sát." (câu gốc "隨傳隨到") còn
nguyên ở bản cuối. Test bộ v3.23.592 dùng bộ dịch giả trả ĐÚNG dòng nên không thấy lỗi.

Ở đây chạy cả đường thật: use case → ``GeminiSubtitleTranslator.translate_stage``
(kiểm lô, ghép, đánh số lại) → lưới dòng lẻ; chỉ lời gọi API được thay bằng model giả
dịch theo bản thô của chính lượt đó. Cột: (dòng, câu gốc, bản thô, bản phong cách, bản
địa hoá).
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from subtitles_extractor.application.services.translation_process_log import (
    get_process_log,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
    _BatchPartialError,
)

Row = tuple[int, str, str, str, str]

ROWS: list[Row] = [
    (
        5, "本年度十大優秀公安獎", "Giải mười Công an ưu tú năm nay, tao nhổ bọt!",
        "Giải Mười Công an ưu tú năm nay, tao nhổ bọt!",
        "Giải Mười Công an ưu tú năm nay, tao nhổ bọt!",
    ),
    (6, "我呸！", "Pha! [cười]", "Phả!", "Ta nhổ!"),
    (
        7, "老表，别那麼神氣", "Người anh em, đừng hách dịch thế.",
        "Biểu đệ, đừng có hách dịch thế.", "Biểu đệ, đừng có hách dịch thế.",
    ),
    (
        9, "一下子就騎在你的頭上", "lập tức ngoi lên đầu mày.",
        "lập tức trèo lên đầu mày ngồi.", "lập tức trèo lên đầu mày ngồi.",
    ),
    (10, "搔你胳肢窩", "Cù nách mày.", "Cù nách mày.", "Cười chết được."),
    (
        11, "可惜我龍游淺水遭蝦欺", "Chỉ tiếc rồng lạc nước cạn bị tôm bắt nạt,",
        "Chỉ tiếc rồng lạc nước cạn bị tôm bắt nạt,",
        "Chỉ tiếc là rồng bơi nước cạn bị tôm bắt nạt,",
    ),
    (17, "來，跟我來", "Nào, theo tôi.", "Nào, theo tôi.", "Đến đây, theo tao."),
    (18, "甚麼事呀", "Chuyện gì vậy?", "Chuyện gì vậy?", "Sao cơ?"),
    (19, "幹甚麼", "Làm cái gì thế?", "Làm cái gì thế?", "Làm cái gì thế?"),
    (
        185, "不許離開本村", "không được rời khỏi bản thôn,",
        "không được rời khỏi bản thôn,", "không được rời khỏi bản thôn,",
    ),
    (186, "隨傳隨到", "gọi đâu có đó.", "gọi đâu có đó.", "phải tùy tùng theo sát."),
    (
        187, "阿積有結果沒有", "A Tích, có kết quả chưa?", "A Tích, có kết quả chưa?",
        "A Tích, có kết quả chưa?",
    ),
    (
        243, "不要那麼粗魯呀", "Đừng thô bạo thế chứ.", "Đừng thô bạo thế chứ.",
        "Đừng thô bạo thế chứ.",
    ),
    (244, "這裡", "Chỗ này.", "Chỗ này.", "Ở đây."),
    (245, "用力點", "Mạnh tay chút.", "Mạnh tay chút.", "Mạnh tay chút."),
    (501, "不要這樣嘛", "Đừng thế mà.", "Đừng thế chứ.", "Đừng thế chứ."),
    (
        502, "你這麼沒用", "Em vô dụng thế.", "Em còn giữ lại làm gì...",
        "Cậu vô dụng thế đấy.",
    ),
    (
        503, "為什麼還要留在這裡？為什麼？", "Tại sao còn ở lại đây? Tại sao chứ?",
        "Tại sao vẫn ở lại đây? Tại sao chứ?", "Tại sao vẫn ở lại đây? Tại sao chứ?",
    ),
    (
        848, "做了這麼多壞事", "Làm nhiều chuyện xấu...", "Làm nhiều điều ác như vậy,",
        "Làm bao nhiêu chuyện ác,",
    ),
    (
        849, "我一個字都不認識", "Tôi một chữ cũng không biết.",
        "tôi một chữ bẻ đôi cũng không biết,", "mình lại chẳng biết gì,",
    ),
    (
        850, "衹好嫁雞隨雞", "Đành phải gả gà theo gà.", "đành phải gả gà theo gà.",
        "đành phải gả gà theo gà.",
    ),
]


class _LiteralModel:
    """Model giả thay ``_call_gemini``: dịch thô từng mục theo bản thô của lượt thật."""

    def __init__(self) -> None:
        self.answers = {row[1]: row[2] for row in ROWS}
        self.calls = 0

    def __call__(
        self, model: str, prompt: str, config: Any, validator: Any, **_kw: Any
    ) -> dict[str, Any]:
        self.calls += 1
        block = re.search(r"<current_batch>\n(.*?)\n</current_batch>", prompt, re.S)
        assert block is not None
        items = json.loads(block.group(1))
        payload: dict[str, Any] = {
            "subtitles": [
                {"line_no": item["line_no"], "text": self.answers[item["text"]]}
                for item in items
            ]
        }
        try:
            validator(payload)
        except _BatchPartialError as partial:
            payload["_missing_line_nos"] = partial.missing_line_nos
        return payload


def _adapter(monkeypatch: pytest.MonkeyPatch, model: _LiteralModel) -> Any:
    adapter = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    monkeypatch.setattr(adapter, "_ensure_available", lambda: None, raising=False)
    monkeypatch.setattr(
        adapter, "_response_schema_for", lambda *a, **k: {}, raising=False
    )
    monkeypatch.setattr(adapter, "_system_instruction", lambda *a, **k: "", raising=False)
    monkeypatch.setattr(adapter, "_build_config", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(adapter, "_call_gemini", model, raising=False)
    return adapter


def _lines(column: int) -> list[TranslationLine]:
    return [
        TranslationLine(
            index=row[0], start_ms=row[0] * 1000, end_ms=row[0] * 1000 + 900,
            text=row[column], original_text=row[1],
        )
        for row in ROWS
    ]


def _request() -> TranslateSubtitlesRequest:
    return TranslateSubtitlesRequest(
        events=[],
        stages=[
            TranslationStageConfig(kind=kind, model_name="gemini-3.5-flash-lite")
            for kind in (
                TranslationStageKind.LITERAL,
                TranslationStageKind.STYLE,
                TranslationStageKind.LOCALIZE,
            )
        ],
        context=TranslationContext(target_lang="Vietnamese", source_lang="zh"),
    )


@pytest.fixture(autouse=True)
def _fresh_log() -> None:
    get_process_log().clear()


def _run(monkeypatch: pytest.MonkeyPatch, model: _LiteralModel) -> dict[int, str]:
    use_case = TranslateSubtitlesUseCase(translator=_adapter(monkeypatch, model))
    result = use_case._revert_lone_rewrites(
        _lines(4), _lines(3), _lines(1), _request(), None, "Giai đoạn 4: Bản địa hoá"
    )
    return {line.index: line.text for line in result}


class TestLoneGuardThroughTheRealAdapter:
    def test_real_case_heard_line_and_calque_are_reverted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        final = _run(monkeypatch, _LiteralModel())
        assert final[10] == "Cù nách mày."
        assert final[186] == "gọi đâu có đó."

    def test_real_case_fix_by_localize_is_kept(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dòng 502: giai đoạn phong cách viết hỏng, địa hoá sửa lại đúng nghĩa."""
        final = _run(monkeypatch, _LiteralModel())
        assert final[502] == "Cậu vô dụng thế đấy."
        assert final[6] == "Ta nhổ!"

    def test_real_case_judgement_is_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _run(monkeypatch, _LiteralModel())
        events = [
            event for event in get_process_log().snapshot()["events"]
            if event["event"] == "lone_rewrites_judged"
        ]
        assert events[-1]["candidates"] == [6, 10, 18, 186, 244, 502, 849]
        assert events[-1]["reverted"] == [10, 18, 186, 244, 849]

    def test_real_case_single_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lượt thật tốn ba request (hai lần bị coi là lệch dòng) mà vẫn hỏng."""
        model = _LiteralModel()
        _run(monkeypatch, model)
        assert model.calls == 1
