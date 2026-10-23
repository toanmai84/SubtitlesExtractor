# ruff: noqa: RUF001 — dấu câu fullwidth CJK thuộc câu gốc chép từ phim
"""[v3.23.593] Hồi quy: dịch lại một TẬP CON dòng không liên tục thì dòng nhận nhầm câu.

Lượt dịch thật v3.23.592 (phim "灭门惨案II"): lưới dòng lẻ gửi dịch trọng tài các dòng
5, 6, 7, 723, 724, 725 trong một lượt. Nhật ký:

    Model trả đủ 6 dòng, line_no lệch (từ 1) — ghép theo VỊ TRÍ.
    AI trả rỗng cho dòng 7 / 723 / 724 / 725 — giữ nguyên văn bản gốc (chống nuốt chữ).

Lõi kiểm lô coi số dòng hợp lệ là 1..6 (vị trí), nên đánh số lại các mục model trả về
thành 1..6; khâu ghép lại tra theo số dòng THẬT: dòng 5 và 6 nhận bản dịch của mục thứ 5
và 6 (tức dòng 724, 725), bốn dòng kia trống. Lượt địa hoá (21 dòng) còn bị coi là lệch
dòng, dịch lại hai lần. Kết quả: lưới dòng lẻ không phân xử được dòng nào ở cả hai giai
đoạn — dòng 10 "Cười chết được." và 186 "phải tùy tùng theo sát." còn nguyên ở bản cuối.

Từ v3.23.133 adapter đã đánh số lại liên tục 1..N rồi ánh xạ ngược — nhưng chỉ khi có
dòng nhiễu bị lọc. Nay áp cho mọi tập dòng không liên tục từ 1.

Dòng dưới đây chép nguyên văn từ lượt chạy đó (``0c0c7c9a``): (dòng, câu gốc, bản thô).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import replace
from typing import Any

import pytest

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.infrastructure.translation import (
    gemini_translation_adapter as adapter_module,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
    _BatchPartialError,
)

#: Lượt trọng tài của giai đoạn phong cách (ứng viên 6 và 724, kèm một dòng kề mỗi bên).
STYLE_REFEREE: list[tuple[int, str, str]] = [
    (5, "本年度十大優秀公安獎", "Giải mười Công an ưu tú năm nay, tao nhổ bọt!"),
    (6, "我呸！", "Pha! [cười]"),
    (7, "老表，别那麼神氣", "Người anh em, đừng hách dịch thế."),
    (723, "拿來", "Đưa đây."),
    (724, "去", "Đưa."),
    (725, "偷完腥還想搶菲林", "Lén lút ăn vụng mà còn muốn cướp cuộn phim à?"),
]
#: Lượt trọng tài của giai đoạn địa hoá (ứng viên 6, 10, 18, 186, 244, 502, 849).
LOCALIZE_REFEREE: list[tuple[int, str, str]] = [
    *STYLE_REFEREE[:3],
    (9, "一下子就騎在你的頭上", "lập tức ngoi lên đầu mày."),
    (10, "搔你胳肢窩", "Cù nách mày."),
    (11, "可惜我龍游淺水遭蝦欺", "Chỉ tiếc rồng lạc nước cạn bị tôm bắt nạt,"),
    (17, "來，跟我來", "Nào, theo tôi."),
    (18, "甚麼事呀", "Chuyện gì vậy?"),
    (19, "幹甚麼", "Làm cái gì thế?"),
    (185, "不許離開本村", "không được rời khỏi bản thôn,"),
    (186, "隨傳隨到", "gọi đâu có đó."),
    (187, "阿積有結果沒有", "A Tích, có kết quả chưa?"),
    (243, "不要那麼粗魯呀", "Đừng thô bạo thế chứ."),
    (244, "這裡", "Chỗ này."),
    (245, "用力點", "Mạnh tay chút."),
    (501, "不要這樣嘛", "Đừng thế mà."),
    (502, "你這麼沒用", "Em vô dụng thế."),
    (503, "為什麼還要留在這裡？為什麼？", "Tại sao còn ở lại đây? Tại sao chứ?"),
    (848, "做了這麼多壞事", "Làm nhiều chuyện xấu..."),
    (849, "我一個字都不認識", "Tôi một chữ cũng không biết."),
    (850, "衹好嫁雞隨雞", "Đành phải gả gà theo gà."),
]
#: Đầu phim, liên tục nhưng không bắt đầu từ 1 (dải lưới dải viết lại có thể gửi).
OPENING_SOURCE = [
    "老表，别那麼神氣", "一下子就騎在你的頭上", "搔你胳肢窩", "可惜我龍游淺水遭蝦欺",
]


class _Ctx:
    source_lang = "zh"
    target_lang = "Vietnamese"
    enable_tags = False
    include_desc = False

    def __getattr__(self, _name: str) -> Any:
        return None


class _Model:
    """Model giả "ngoan": dịch từng mục của <current_batch>, trả đúng line_no nhận được.

    Args:
        answers: ``câu gốc → bản dịch``.
        drop_once: Số thứ tự (0-based) của mục bị bỏ sót ở lần gọi ĐẦU.
    """

    def __init__(self, answers: dict[str, str], drop_once: int | None = None) -> None:
        self.answers = answers
        self.drop_once = drop_once
        self.seen_line_nos: list[list[int]] = []
        self.flags: list[dict[str, Any]] = []

    def __call__(
        self, model: str, prompt: str, config: Any, validator: Any, **_kw: Any
    ) -> dict[str, Any]:
        block = re.search(r"<current_batch>\n(.*?)\n</current_batch>", prompt, re.S)
        assert block is not None
        items = json.loads(block.group(1))
        self.seen_line_nos.append([item["line_no"] for item in items])
        self.flags.extend(item for item in items if item.get("needs_expansion"))
        reply = [
            {
                "line_no": item["line_no"],
                "text": self.answers[item.get("original") or item["text"]],
            }
            for item in items
        ]
        if self.drop_once is not None and len(self.seen_line_nos) == 1:
            reply.pop(self.drop_once)
        payload: dict[str, Any] = {"subtitles": reply}
        try:
            validator(payload)
        except _BatchPartialError as partial:
            # Đúng như ``_call_gemini`` thật: thiếu ít dòng thì trả phần đã có kèm danh
            # sách dòng thiếu để tầng trên vá riêng.
            payload["_missing_line_nos"] = partial.missing_line_nos
        return payload


def _adapter(monkeypatch: pytest.MonkeyPatch, model: _Model) -> GeminiSubtitleTranslator:
    adapter = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    monkeypatch.setattr(adapter, "_ensure_available", lambda: None, raising=False)
    monkeypatch.setattr(
        adapter, "_response_schema_for", lambda *a, **k: {}, raising=False
    )
    monkeypatch.setattr(adapter, "_system_instruction", lambda *a, **k: "", raising=False)
    monkeypatch.setattr(adapter, "_build_config", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(adapter, "_call_gemini", model, raising=False)
    return adapter


def _subset(rows: list[tuple[int, str, str]]) -> list[TranslationLine]:
    return [
        TranslationLine(
            index=index, start_ms=index * 1000, end_ms=index * 1000 + 900, text=source
        )
        for index, source, _literal in rows
    ]


def _literal(
    adapter: GeminiSubtitleTranslator, lines: list[TranslationLine]
) -> dict[int, str]:
    out = adapter.translate_stage(
        stage=TranslationStageConfig(
            kind=TranslationStageKind.LITERAL, model_name="m", batch_size=60,
            context_size=0,
        ),
        context=_Ctx(),
        source_lines=lines,
        input_lines=lines,
        has_prior_translation=False,
    )
    assert [line.index for line in out] == [line.index for line in lines]
    return {line.index: line.text for line in out}


class TestRefereeSubsets:
    @pytest.mark.parametrize("rows", [STYLE_REFEREE, LOCALIZE_REFEREE])
    def test_real_case_every_line_gets_its_own_translation(
        self, monkeypatch: pytest.MonkeyPatch, rows: list[tuple[int, str, str]]
    ) -> None:
        model = _Model({source: literal for _i, source, literal in rows})
        result = _literal(_adapter(monkeypatch, model), _subset(rows))
        assert result == {index: literal for index, _source, literal in rows}

    def test_real_case_one_request_without_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Lượt thật: 21 dòng bị coi là lệch, gửi ba lần."""
        model = _Model({source: literal for _i, source, literal in LOCALIZE_REFEREE})
        _literal(_adapter(monkeypatch, model), _subset(LOCALIZE_REFEREE))
        assert model.seen_line_nos == [list(range(1, 22))]

    def test_missing_line_is_patched_onto_the_right_line(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Model bỏ sót mục thứ 11 (dòng 186): vá bằng một cửa sổ quanh dòng đó, bản vá
        phải về đúng dòng 186."""
        model = _Model(
            {source: literal for _i, source, literal in LOCALIZE_REFEREE}, drop_once=10
        )
        result = _literal(_adapter(monkeypatch, model), _subset(LOCALIZE_REFEREE))
        assert result[186] == "gọi đâu có đó."
        assert result == {index: literal for index, _s, literal in LOCALIZE_REFEREE}
        assert len(model.seen_line_nos) == 2
        assert 11 in model.seen_line_nos[1]

    def test_contiguous_run_not_starting_at_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [
            (index, source, f"VI {source}")
            for index, source in enumerate(OPENING_SOURCE, start=7)
        ]
        model = _Model({source: text for _i, source, text in rows})
        result = _literal(_adapter(monkeypatch, model), _subset(rows))
        assert result == {index: text for index, _source, text in rows}
        assert model.seen_line_nos == [[1, 2, 3, 4]]


class TestWholeStageUnchanged:
    def test_lines_numbered_from_one_are_sent_as_they_are(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [(i, source, f"VI {i}") for i, (_n, source, _l) in enumerate(
            STYLE_REFEREE, start=1
        )]
        model = _Model({source: text for _i, source, text in rows})
        result = _literal(_adapter(monkeypatch, model), _subset(rows))
        assert model.seen_line_nos == [[1, 2, 3, 4, 5, 6]]
        assert result == {index: text for index, _source, text in rows}


class TestReindexMap:
    def test_numbered_from_one_needs_no_map(self) -> None:
        lines = _subset([(1, "一", ""), (2, "二", "")])
        assert GeminiSubtitleTranslator._contiguous_reindex_map(lines, lines) == {}

    def test_sparse_lines_map_back_to_their_numbers(self) -> None:
        lines = _subset(STYLE_REFEREE)
        assert GeminiSubtitleTranslator._contiguous_reindex_map(lines, lines) == {
            1: 5, 2: 6, 3: 7, 4: 723, 5: 724, 6: 725,
        }

    def test_lists_that_are_not_parallel_are_left_alone(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        lines = _subset(STYLE_REFEREE)
        with caplog.at_level(logging.WARNING):
            assert GeminiSubtitleTranslator._contiguous_reindex_map(
                lines, lines[:-1]
            ) == {}
        assert "không cùng tập số dòng" in caplog.text


class TestExpansionFlagFollowsItsLine:
    def test_flag_is_sent_on_the_line_that_needs_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dòng bị nghi dịch thiếu được đánh cờ theo số dòng GỐC; sau khi đánh số lại,
        cờ phải nằm đúng ở dòng đó."""
        rows = STYLE_REFEREE
        model = _Model({source: literal for _i, source, literal in rows})
        adapter = _adapter(monkeypatch, model)
        monkeypatch.setattr(
            adapter_module, "find_under_translated",
            lambda triples: [
                type("S", (), {"index": index})() for index, *_rest in triples
                if index == 723
            ],
        )
        lines = [
            replace(line, original_text=line.text, text=f"bản trước {line.index}")
            for line in _subset(rows)
        ]
        adapter.translate_stage(
            stage=TranslationStageConfig(
                kind=TranslationStageKind.STYLE, model_name="m", batch_size=60,
                context_size=0,
            ),
            context=_Ctx(),
            source_lines=lines,
            input_lines=lines,
            has_prior_translation=True,
        )
        assert [flag["original"] for flag in model.flags] == ["拿來"]

    def test_flag_survives_noise_filtering(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dòng nhiễu OCR (không có chữ Hán) bị lọc → các dòng sau được đánh số lại.
        Trước bản này cờ vẫn tra theo số cũ nên rơi vào dòng ĐỨNG SAU dòng cần bổ sung."""
        sources = ["我呸！", "akas", "老表，别那麼神氣", "拿來", "去"]
        model = _Model({source: f"VI {source}" for source in sources})
        adapter = _adapter(monkeypatch, model)
        monkeypatch.setattr(
            adapter_module, "find_under_translated",
            lambda triples: [
                type("S", (), {"index": index})() for index, *_rest in triples
                if index == 4
            ],
        )
        lines = [
            TranslationLine(
                index=pos, start_ms=pos * 1000, end_ms=pos * 1000 + 900,
                text=f"bản trước {pos}", original_text=source,
            )
            for pos, source in enumerate(sources, start=1)
        ]
        source_lines = [replace(line, text=line.original_text) for line in lines]
        out = adapter.translate_stage(
            stage=TranslationStageConfig(
                kind=TranslationStageKind.STYLE, model_name="m", batch_size=60,
                context_size=0,
            ),
            context=_Ctx(),
            source_lines=source_lines,
            input_lines=lines,
            has_prior_translation=True,
        )
        assert [flag["original"] for flag in model.flags] == ["拿來"]
        assert [line.text for line in out] == [
            "VI 我呸！", "bản trước 2", "VI 老表，别那麼神氣", "VI 拿來", "VI 去",
        ]
