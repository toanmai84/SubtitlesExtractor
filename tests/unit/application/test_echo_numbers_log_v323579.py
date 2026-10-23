"""[v3.23.579] Lô dịch thô trả nguyên văn, số đọc sai, nhật ký gắn nhầm phim.

Chuẩn đoán thật v3.23.578 (ba phim, cùng một phiên làm việc):

* Phim "傻子落水…": lô dịch thô 3421-3480 trả NGUYÊN VĂN chữ Hán ở 55 dòng liền mà vẫn
  đủ số dòng — không phép kiểm nào bắt (chỉ bắt khi 100% giống bản gốc).
* Cùng phim: khâu bản địa hoá ghi "một mươi triệu NDT" ở 4 dòng (3377, 3378, 3380, 3392).
  Đúng giá trị nên lưới số bỏ qua; người xem nghe "một mươi triệu".
* Phim "he thong sung the": "三十四点五万元" (34,5 vạn) bị lưới số đọc thành "五万"
  nên báo sai giá trị ở hai dòng dịch ĐÚNG — và có thể "sửa" thành 50.000.
* File chuẩn đoán của phim "傻子落水…" (3902 dòng) mang nhật ký diễn biến của phim khác
  (``stage_completed`` 5540 dòng) vì nhật ký là bộ ghi dùng chung trong bộ nhớ.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from subtitles_extractor.application.services.number_consistency_guard import (
    chinese_number_value,
    find_number_findings,
    find_number_fixes,
    is_malformed_reading,
    vietnamese_number_value,
)
from subtitles_extractor.application.services.source_echo_guard import (
    is_echo_batch,
    source_echo_lines,
)
from subtitles_extractor.application.services.translation_diagnostics import (
    FOREIGN_PROCESS_LOG_NOTE,
    build_diagnostics_bundle,
)
from subtitles_extractor.application.services.translation_process_log import (
    get_process_log,
    record_event,
    run_fingerprint,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine

# (dòng, câu gốc, bản dịch thô thật v3.23.578, bản tinh chỉnh thật)
ECHO_ROWS = [
    (3421, "你毁了他们的婚礼", "Em đã phá nát đám cưới của bọn chúng.",
     "Em đã phá nát đám cưới của bọn chúng."),
    (3422, "小钱", "Tống Tiền,", "Tống Tiền,"),
    (3423, "你可真是越来越厉害了", "em quả thật càng lúc càng lợi hại rồi đấy.",
     "em quả thật càng lúc càng lợi hại rồi đấy."),
    (3424, "嗨", "Hừm.", "Hừm,"),
    (3425, "也就还好吧", "cũng bình thường thôi chị Đào Hoa.",
     "cũng bình thường thôi chị Đào Hoa."),
    (3426, "既然事情都结束了", "既然事情都结束了", "Đã thế mọi chuyện cũng xong rồi,"),
    (3427, "走走走", "走走走", "đi thôi đi thôi."),
    (3428, "你快开着你的大奔", "你快开着你的大奔", "Anh mau lái chiếc xe Mẹc của anh,"),
    (3429, "带我兜一圈", "带我兜一圈", "đưa em lượn một vòng đi."),
    (3430, "我还从来没坐过", "我还从来没坐过", "Em chưa từng được ngồi"),
    (3431, "这么好的车呢", "这么好的车呢", "trên chiếc xe xịn thế bao giờ."),
    (3432, "没问题", "没问题", "Được thôi."),
    (3433, "小瑶", "小瑶", "Tiểu Dao,"),
    (3434, "桃花姐", "桃花姐", "chị Đào Hoa,"),
    (3435, "一起呀", "一起呀", "lên xe cùng đi."),
    (3436, "我就不去了", "我就不去了", "Em không đi đâu."),
    (3437, "我对车不感兴趣", "我对车不感兴趣", "Em không có hứng thú với xe cộ,"),
    (3438, "我对钱比较感兴趣", "我对钱比较感兴趣", "em có hứng thú với tiền hơn."),
    (3439, "我得去银行看看", "我得去银行看看", "Em phải ra ngân hàng xem thử,"),
    (3440, "一千万长什么样", "一千万长什么样", "mười triệu NDT trông như thế nào."),
]
ECHO_SOURCES = {row[0]: row[1] for row in ECHO_ROWS}
ECHO_LITERAL = {row[0]: row[2] for row in ECHO_ROWS}


class TestSourceEchoGuard:
    def test_real_block_is_an_echo_batch(self) -> None:
        echoed = source_echo_lines(ECHO_SOURCES, ECHO_LITERAL)
        assert echoed == list(range(3426, 3441))
        assert is_echo_batch(len(echoed), len(ECHO_ROWS))

    def test_real_full_batch_threshold(self) -> None:
        # Lô thật 60 dòng, 55 dòng nguyên văn.
        assert is_echo_batch(55, 60)

    def test_single_empty_line_kept_as_source_is_not_a_batch_error(self) -> None:
        # Dòng model trả rỗng (ứng dụng giữ câu gốc) — mức bình thường của 37 lần chạy.
        assert not is_echo_batch(1, 60)
        assert not is_echo_batch(5, 60)

    def test_latin_or_numeric_source_may_stay_unchanged(self) -> None:
        assert source_echo_lines({1: "OK", 2: "666"}, {1: "OK", 2: "666"}) == []


def _literal_batch() -> list[TranslationLine]:
    return [
        TranslationLine(
            index=line_no, start_ms=position * 1000, end_ms=position * 1000 + 900,
            text=source, original_text=source,
        )
        for position, (line_no, source, _, _) in enumerate(ECHO_ROWS)
    ]


def _payload(texts: dict[int, str]) -> dict:
    return {"subtitles": [{"line_no": n, "text": texts[n]} for n, *_ in ECHO_ROWS]}


class TestAdapterRetriesEchoBatch:
    def _translator(self, payloads: list[dict], calls: list[dict]):
        from subtitles_extractor.infrastructure.translation import (
            gemini_translation_adapter as adapter,
        )

        translator = adapter.GeminiSubtitleTranslator(api_key="dummy")
        remaining = list(payloads)

        def fake_call(model_name, prompt, config, validator, **_kwargs):
            # Mô phỏng vòng thử-lại của ``_call_gemini``: lỗi kiểm lô → gọi lại.
            while remaining:
                payload = remaining.pop(0)
                calls.append(payload)
                try:
                    validator(payload)
                except adapter._BatchValidationError:
                    continue
                return payload
            raise adapter._BatchCountMismatchError("hết lượt thử")

        translator._call_gemini = fake_call  # type: ignore[assignment]
        return translator

    def _run(self, translator, is_literal: bool) -> list[TranslationLine]:
        return translator._translate_single_batch(
            batch=_literal_batch(), source_before=[], source_after=[],
            history_before=[], start_idx=ECHO_ROWS[0][0] - 1, is_preprocess=False,
            is_literal=is_literal, config=None, model_name="m", cancel_cb=None,
        )

    def test_literal_echo_is_translated_again(self) -> None:
        get_process_log().clear()
        calls: list[dict] = []
        translated = {row[0]: row[3] for row in ECHO_ROWS}
        translator = self._translator(
            [_payload(ECHO_LITERAL), _payload(translated)], calls
        )
        result = self._run(translator, is_literal=True)
        assert len(calls) == 2
        assert {line.index: line.text for line in result} == translated
        events = [
            e for e in get_process_log().snapshot()["events"]
            if e["event"] == "batch_source_echo"
        ]
        assert events and events[0]["count"] == 15
        assert events[0]["lines"] == [3426, 3440]

    def test_style_stage_may_keep_text(self) -> None:
        calls: list[dict] = []
        translator = self._translator([_payload(ECHO_LITERAL)], calls)
        self._run(translator, is_literal=False)
        assert len(calls) == 1

    def test_retries_are_capped(self) -> None:
        calls: list[dict] = []
        translator = self._translator([_payload(ECHO_LITERAL)] * 5, calls)
        result = self._run(translator, is_literal=True)
        # 2 lần ép dịch lại rồi NHẬN kết quả (khâu sau vẫn dịch lại được từ bản gốc).
        assert len(calls) == 3
        assert len(result) == len(ECHO_ROWS)


class TestChineseDecimals:
    @pytest.mark.parametrize(
        ("expression", "value"),
        [
            ("三十四点五万", 345_000),
            ("3.5万", 35_000),
            ("一点五亿", 150_000_000),
            ("三千四百三十万", 34_300_000),
            ("一千万", 10_000_000),
        ],
    )
    def test_values(self, expression: str, value: int) -> None:
        assert chinese_number_value(expression) == value

    def test_real_lines_are_no_longer_mismatches(self) -> None:
        sources = {656: "三十四点五万元", 663: "消费三十四点五万元"}
        translations = {
            656: "là ba mươi bốn vạn năm nghìn tệ.",
            663: "ba mươi bốn vạn rưỡi tệ,",
        }
        assert [f for f in find_number_findings(sources, translations)
                if f.kind == "mismatch"] == []

    def test_decimal_value_is_never_fixed_to_the_digit_after_the_point(self) -> None:
        # Phim có "năm vạn" (五万) ở dòng khác: trước đây 656 bị coi là 50.000 và có thể
        # bị "sửa" thành "năm vạn".
        sources = {10: "五万块", 11: "五万", 656: "三十四点五万元"}
        translations = {
            10: "năm mươi nghìn tệ", 11: "Năm mươi nghìn",
            656: "là ba mươi bốn vạn năm nghìn tệ.",
        }
        # [v3.23.590] Phim đọc lối hiện đại nên cụm trộn "vạn … nghìn" được đọc lại theo
        # lối đó — vẫn đúng 345.000, không bao giờ thành "năm vạn"/"năm mươi nghìn".
        [fix] = find_number_fixes(sources, translations)
        assert (fix.index, fix.reason) == (656, "lệch lối đọc")
        assert fix.text_after == "là ba trăm bốn mươi lăm nghìn tệ."

    def test_numbers_with_three_units_are_checked(self) -> None:
        # Trước đây mọi cụm ≥ 3 đơn vị (千, 百, 十, 万) bị bỏ như bài đếm hàng, nên các
        # khoản tiền "三千四百三十万" của phim chưa từng được đối chiếu.
        sources = {1145: "三千四百三十万元"}
        assert find_number_findings(sources, {1145: "34,3 triệu Tệ"}) == []
        slipped = find_number_findings(
            sources, {1145: "ba triệu bốn trăm ba mươi nghìn tệ"}
        )
        assert [(f.index, f.kind, f.expected_value) for f in slipped] == [
            (1145, "mismatch", 34_300_000)
        ]

    def test_counting_chant_is_still_not_a_number(self) -> None:
        assert find_number_findings({1626: "个十百千万"}, {1626: "mười vạn"}) == []


class TestVietnameseReadings:
    @pytest.mark.parametrize(
        ("phrase", "value"),
        [
            ("34,5 vạn", 345_000),
            ("ba mươi bốn vạn rưỡi", 345_000),
            ("một triệu rưỡi", 1_500_000),
            ("1,5 triệu", 1_500_000),
            ("34,3 triệu", 34_300_000),
            ("một chục triệu", 10_000_000),
            ("3.450 nghìn", 3_450_000),
            ("hai mươi lăm nghìn", 25_000),
        ],
    )
    def test_values(self, phrase: str, value: int) -> None:
        assert vietnamese_number_value(phrase) == value

    def test_malformed_tens(self) -> None:
        assert is_malformed_reading("một mươi triệu")
        assert not is_malformed_reading("mười triệu")
        assert not is_malformed_reading("hai mươi triệu")


# (dòng, câu gốc, bản tinh chỉnh thật, bản địa hoá thật v3.23.578)
TEN_MILLION_ROWS = [
    (2338, "这张卡里有一千万", "Trong thẻ này có mười triệu.",
     "Trong thẻ này có mười triệu."),
    (3377, "给了我一千万", "họ còn cho anh mười triệu NDT.",
     "họ còn cho anh một mươi triệu NDT."),
    (3378, "一千万?", "Mười triệu NDT?", "Một mươi triệu NDT?"),
    (3380, "这卡里居然有一千万", "Trong thẻ này lại có mười triệu NDT,",
     "Trong thẻ này lại có một mươi triệu NDT,"),
    (3392, "又多了一千万", "lại có thêm mười triệu NDT nữa.",
     "lại có thêm một mươi triệu NDT nữa."),
]


class TestMalformedReadingIsFixed:
    def test_real_localize_lines(self) -> None:
        sources = {row[0]: row[1] for row in TEN_MILLION_ROWS}
        style = {row[0]: row[2] for row in TEN_MILLION_ROWS}
        localize = {row[0]: row[3] for row in TEN_MILLION_ROWS}
        fixes = {
            fix.index: fix.text_after
            for fix in find_number_fixes(sources, localize, prior_translations=style)
        }
        assert fixes == {
            3377: "họ còn cho anh mười triệu NDT.",
            3378: "Mười triệu NDT?",
            3380: "Trong thẻ này lại có mười triệu NDT,",
            3392: "lại có thêm mười triệu NDT nữa.",
        }

    def test_fixed_without_any_reference_reading(self) -> None:
        fixes = find_number_fixes({1: "给了我一千万"}, {1: "cho anh một mươi triệu."})
        assert [fix.text_after for fix in fixes] == ["cho anh mười triệu."]
        assert fixes[0].reason == "sai cách đọc"

    def test_diagnostics_report_malformed(self) -> None:
        findings = find_number_findings({3377: "给了我一千万"},
                                        {3377: "họ còn cho anh một mươi triệu NDT."})
        assert [(f.index, f.kind) for f in findings] == [(3377, "malformed")]


def _events(texts: tuple[str, ...]) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(index=i, text=text, start_sec=i, end_sec=i + 0.9, confidence=1.0)
        for i, text in enumerate(texts, start=1)
    ]


def _bundle(texts: tuple[str, ...]) -> dict:
    return build_diagnostics_bundle(
        app_version="3.23.579", exported_at="2026-09-30T00:00:00Z",
        video_path="/x/phim.mp4", source_events=_events(texts),
        translated_events=_events(texts), session=None,
    )


class TestProcessLogOwnership:
    FILM_A = ("既然事情都结束了", "走走走", "你快开着你的大奔")
    FILM_B = ("三十四点五万元", "消费三十四点五万元")

    def test_fingerprint_depends_on_every_line(self) -> None:
        assert run_fingerprint(self.FILM_A) == run_fingerprint(iter(self.FILM_A))
        assert run_fingerprint(self.FILM_A) != run_fingerprint(self.FILM_A[:2])
        assert run_fingerprint(self.FILM_A) != run_fingerprint(self.FILM_B)

    def test_log_of_the_same_film_is_attached(self) -> None:
        get_process_log().start_run(run_fingerprint(self.FILM_A))
        record_event("batch_retry", attempt=1)
        log = _bundle(self.FILM_A)["process_log"]
        assert log["counts"] == {"batch_retry": 1}

    def test_log_of_another_film_is_not_attached(self) -> None:
        get_process_log().start_run(run_fingerprint(self.FILM_B))
        record_event("stage_completed", lines=5540)
        log = _bundle(self.FILM_A)["process_log"]
        assert log["events"] == []
        assert log["foreign_run"] is True
        assert log["note"] == FOREIGN_PROCESS_LOG_NOTE

    def test_translation_run_binds_the_log(self) -> None:
        from subtitles_extractor.application.use_cases.translate_subtitles import (
            TranslateSubtitlesRequest,
            TranslateSubtitlesUseCase,
        )
        from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
        from subtitles_extractor.domain.ports.subtitle_translator_port import (
            TranslationContext,
            TranslationStageConfig,
            TranslationStageKind,
        )
        from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

        class _Translator:
            def translate_stage(self, *, input_lines, **_kwargs):
                return [replace(line, text="bản dịch") for line in input_lines]

        get_process_log().start_run(run_fingerprint(self.FILM_B))
        events = [
            SubtitleEvent(index=i, text=text, interval=TimeInterval(i, i + 0.9))
            for i, text in enumerate(self.FILM_A, start=1)
        ]
        use_case = TranslateSubtitlesUseCase(translator=_Translator())  # type: ignore
        use_case.execute(
            TranslateSubtitlesRequest(
                events=events,
                stages=[TranslationStageConfig(
                    kind=TranslationStageKind.LITERAL, model_name="m"
                )],
                context=TranslationContext(target_lang="Vietnamese"),
            )
        )
        assert get_process_log().snapshot()["owner"] == run_fingerprint(self.FILM_A)
        assert _bundle(self.FILM_A)["process_log"]["counts"]["stage_completed"] == 1
        assert _bundle(self.FILM_B)["process_log"]["events"] == []
