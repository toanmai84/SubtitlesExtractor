"""[v3.23.572] Hồi quy: model chỉ ghi Visual Cues ở dòng ĐẦU lượt lời, dòng sau bỏ trống.

Chuẩn đoán thật v3.23.571 (phân tích bằng gemini-3.8-flash): cues phủ 74,5% số dòng —
14/22 đoạn video đủ 98-100%, 8 đoạn chỉ 19-43% vì model chỉ ghi cue khi ĐỔI người nói.
994 dòng (25%) dịch mà không có người nói/người nghe; dòng không nhãn sai quy tắc xưng hô
gấp 11 lần dòng có nhãn. Tỉ lệ phủ toàn phim vẫn trên ngưỡng 60% nên bước quét bổ sung
không chạy.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

from subtitles_extractor.application.services.context_diagnostics import (
    build_context_diagnostics,
)
from subtitles_extractor.application.services.cue_run_filler import (
    MIN_COVERAGE_FOR_RUN_FILL,
    RUN_BREAK_SILENCE_MS,
    LineSpan,
    fill_cue_runs,
)
from subtitles_extractor.application.services.visual_cue_serializer import (
    deserialize_visual_cues,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
    VisualCue,
)
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

# Nguyên văn đoạn 16 của lượt phân tích thật: số dòng → (câu gốc, bắt đầu ms, kết thúc ms,
# nhãn model ghi hoặc None).
REAL: dict[int, tuple[str, int, int, tuple[str, str] | None]] = {
    2576: ("宋神医", 5268267, 5269067, None),
    2577: ("我的愿望终于实现了", 5269333, 5270733, None),
    2578: ("宋神医\uff01", 5274333, 5275000, None),
    2579: ("宋神医这是你要的脚链", 5276333, 5277733, ("沈明月", "宋钱")),
    2580: ("哦不宋钱弟弟", 5278267, 5279400, None),
    2581: ("这是第四块元神碎片", 5280267, 5282533, ("宋钱", "宋钱")),
    2582: ("再收集两块", 5282800, 5284200, None),
    2583: ("就能解除三年危机", 5284333, 5285867, None),
    2584: ("复活神仙姐姐了", 5285933, 5287667, None),
    2585: ("其实不光是脚链", 5289533, 5291800, ("沈明月", "宋钱")),
    2586: ("刚才答应你的", 5292133, 5293067, None),
    2587: ("也都可以做到", 5293133, 5293933, None),
    2588: ("我们是先从", 5294267, 5295200, None),
    2589: ("吃饭逛街看电影开始", 5295333, 5297067, None),
    2590: ("还是直接一步到位", 5297733, 5299200, None),
    2591: ("去楼上的总统套房", 5299667, 5300867, None),
    2592: ("深入了解一下彼此啊?", 5301200, 5302533, None),
    2593: ("沈明月", 5307400, 5308067, ("柳慕云", "沈明月")),
    2594: ("我们来这是有", 5308467, 5309133, None),
    2595: ("正经事要办的", 5309267, 5310000, None),
    2596: ("你别在这胡闹了", 5310133, 5311267, None),
    2597: ("我告诉你", 5311667, 5312333, None),
    2598: ("岂有此理", 5316067, 5317133, ("沈明月", "宋钱")),
    2599: ("徐俊豪和安雪漫", 5317267, 5318267, None),
}
# Đoạn 6 của cùng lượt phân tích: sau 3,3 giây lặng là lời bàn tán của dân làng.
VILLAGE: dict[int, tuple[str, int, int, tuple[str, str] | None]] = {
    861: ("我会让你后悔的\uff01", 1924800, 1926067, ("吴二狗", "宋钱")),
    862: ("是吗", 1926867, 1927867, ("宋钱", "吴二狗")),
    863: ("我正想见识一下", 1928000, 1929200, None),
    864: ("你能把我怎么样", 1930600, 1932000, None),
    865: ("这小子谁啊", 1935267, 1936267, None),
    866: ("连吴二狗都敢打", 1936400, 1937933, None),
    867: ("初生牛犊不怕虎", 1938400, 1939933, None),
    868: ("这小伙", 1940067, 1941133, None),
    869: ("怕是要栽了", 1941267, 1942800, None),
    870: ("狗哥", 1943200, 1944200, None),
    871: ("没事吧狗哥", 1945067, 1945733, None),
    872: ("狗哥狗哥好狗哥", 1952267, 1953800, None),
    873: ("你敢掐我胳膊?", 1955667, 1956800, ("吴二狗", "宋钱")),
}

Rows = dict[int, tuple[str, int, int, tuple[str, str] | None]]

#: Số dòng của đoạn DÀY ghép sau đoạn thưa — như 14/22 đoạn của phim thật, model ghi cue
#: cho mọi dòng. Phim thật phủ 74,5%; chỉ đoạn thưa thì chưa tới ngưỡng 60%.
DENSE_LINES = 60


def _film(rows: Rows) -> Rows:
    """Đoạn thưa thật nối tiếp một đoạn dày (mọi dòng có cue), cách nhau 1,5 giây."""
    last = max(rows)
    end = rows[last][2]
    dense = {
        last + step: (
            f"\u53e5\u5b50{step}",
            end + step * 1500,
            end + step * 1500 + 1200,
            ("宋钱", "许桃花"),
        )
        for step in range(1, DENSE_LINES + 1)
    }
    return {**rows, **dense}


FILM = _film(REAL)
ORDER = sorted(FILM)


def _spans(rows: Rows = FILM) -> list[LineSpan]:
    return [LineSpan(line, rows[line][1], rows[line][2]) for line in sorted(rows)]


def _cues(rows: Rows = FILM) -> list[VisualCue]:
    return [
        VisualCue(line_no=line, speaker=label[0], addressee=label[1], scene="")
        for line in sorted(rows)
        if (label := rows[line][3]) is not None
    ]


def _labels(cues: list[VisualCue]) -> dict[int, tuple[str, str]]:
    return {cue.line_no: (cue.speaker, cue.addressee) for cue in cues}


class TestFillCueRuns:
    def test_following_lines_take_the_run_head_label(self) -> None:
        filled, lines = fill_cue_runs(_cues(), _spans())
        labels = _labels(filled)
        for line in range(2586, 2593):
            assert labels[line] == ("沈明月", "宋钱")
        for line in range(2594, 2598):
            assert labels[line] == ("柳慕云", "沈明月")
        assert labels[2580] == ("沈明月", "宋钱")
        assert labels[2599] == ("沈明月", "宋钱")
        assert 2586 in lines and 2585 not in lines

    def test_model_labels_are_kept(self) -> None:
        labels = _labels(fill_cue_runs(_cues(), _spans())[0])
        assert labels[2593] == ("柳慕云", "沈明月")
        assert labels[2581] == ("宋钱", "宋钱")

    def test_silence_ends_the_run(self) -> None:
        """Dòng 865 đến sau 3,3 giây lặng: dân làng bàn tán, không phải Tống Tiền nói."""
        assert VILLAGE[865][1] - VILLAGE[864][2] > RUN_BREAK_SILENCE_MS
        village = _film(VILLAGE)
        filled, lines = fill_cue_runs(_cues(village), _spans(village))
        assert lines == [863, 864]
        assert _labels(filled)[864] == ("宋钱", "吴二狗")
        assert all(line not in _labels(filled) for line in range(865, 873))

    def test_lines_before_the_first_cue_stay_empty(self) -> None:
        labels = _labels(fill_cue_runs(_cues(), _spans())[0])
        assert 2576 not in labels and 2577 not in labels

    def test_dense_region_is_not_filled(self) -> None:
        """Vùng mà model ghi gần đủ từng dòng: dòng trống là cố ý, không điền."""
        spans = [LineSpan(i, i * 1000, i * 1000 + 900) for i in range(1, 41)]
        cues = [
            VisualCue(line_no=i, speaker="宋钱", addressee="许桃花", scene="")
            for i in range(1, 41)
            if i != 20
        ]
        assert fill_cue_runs(cues, spans) == (cues, [])

    def test_run_head_without_speaker_is_not_copied(self) -> None:
        spans = [LineSpan(i, i * 1000, i * 1000 + 900) for i in range(1, 11)]
        cues = [VisualCue(line_no=1, speaker="", addressee="宋钱", scene="")]
        assert fill_cue_runs(cues, spans) == (cues, [])

    def test_real_segment_counts(self) -> None:
        filled, lines = fill_cue_runs(_cues(), _spans())
        assert len(_cues(REAL)) == 5
        assert lines == [
            2580, 2582, 2583, 2584, 2586, 2587, 2588, 2589, 2590, 2591, 2592,
            2594, 2595, 2596, 2597, 2599,
        ]
        assert len(filled) == len(FILM) - 3

    def test_too_sparse_film_is_left_to_the_scanner(self) -> None:
        """Cả phim phủ dưới 60%: không phải kiểu ghi đầu lượt lời — để bộ quét bổ sung."""
        cues = _cues(REAL)
        assert len(cues) / len(REAL) < MIN_COVERAGE_FOR_RUN_FILL
        assert fill_cue_runs(cues, _spans(REAL)) == (cues, [])


def _request(cues: str, **kwargs: object) -> TranslateSubtitlesRequest:
    return TranslateSubtitlesRequest(
        events=[],
        stages=[
            TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="m")
        ],
        context=TranslationContext(target_lang="Vietnamese", visual_cues=cues),
        **kwargs,  # type: ignore[arg-type]
    )


def _positional_cues_json(rows: Rows = FILM) -> str:
    return json.dumps(
        [
            {"id": pos, "spk": label[0], "to": label[1], "cue": ""}
            for pos, line in enumerate(sorted(rows), start=1)
            if (label := rows[line][3]) is not None
        ],
        ensure_ascii=False,
    )


def _source_lines() -> list[TranslationLine]:
    return [
        TranslationLine(
            index=pos, start_ms=FILM[line][1], end_ms=FILM[line][2], text=FILM[line][0]
        )
        for pos, line in enumerate(ORDER, start=1)
    ]


def _events(rows: Rows = FILM) -> list[SubtitleEvent]:
    return [
        SubtitleEvent(
            index=pos,
            text=rows[line][0],
            interval=TimeInterval(rows[line][1] / 1000, rows[line][2] / 1000),
        )
        for pos, line in enumerate(sorted(rows), start=1)
    ]


class TestUseCase:
    def test_request_cues_are_filled(self) -> None:
        request = _request(_positional_cues_json())
        filled = TranslateSubtitlesUseCase._fill_cue_runs(request, _source_lines())
        assert filled is not request
        cues = _labels(deserialize_visual_cues(filled.context.visual_cues))
        assert cues[ORDER.index(2588) + 1] == ("沈明月", "宋钱")
        assert ORDER.index(2578) + 1 not in cues

    def test_nothing_to_fill_returns_same_request(self) -> None:
        request = _request("")
        assert TranslateSubtitlesUseCase._fill_cue_runs(request, _source_lines()) is (
            request
        )

    def test_translator_sees_filled_cues_and_they_are_persisted(self) -> None:
        seen: list[str] = []

        class _Translator:
            def translate_stage(self, *, input_lines, context, **_kwargs):
                seen.append(context.visual_cues)
                return [replace(line, text=f"dịch {line.index}") for line in input_lines]

        use_case = TranslateSubtitlesUseCase(translator=_Translator())  # type: ignore[arg-type]
        request = replace(_request(_positional_cues_json()), events=_events())
        response = use_case.execute(request)
        position = ORDER.index(2590) + 1
        assert _labels(deserialize_visual_cues(seen[0]))[position] == ("沈明月", "宋钱")
        persisted = _labels(deserialize_visual_cues(response.visual_cues))
        assert persisted[position] == ("沈明月", "宋钱")

    def test_top_up_still_runs_before_the_fill(self) -> None:
        """Cues quá thưa (dưới 60%) vẫn được QUÉT lại — quét tốt hơn điền."""
        scanned: list[list[int]] = []

        class _Scanner:
            def analyze_visual_cues(self, source_lines, target_lang, **_kwargs):
                scanned.append([line.index for line in source_lines])
                return [
                    VisualCue(line_no=line.index, speaker="QUÉT", addressee="QUÉT")
                    for line in source_lines
                ]

            def translate_stage(self, *, input_lines, **_kwargs):
                return list(input_lines)

        use_case = TranslateSubtitlesUseCase(translator=_Scanner())  # type: ignore[arg-type]
        request = replace(
            _request(_positional_cues_json(REAL)),
            events=_events(REAL),
            video_refs=(object(),),
            enable_visual_cues=True,
        )
        response = use_case.execute(request)
        position = sorted(REAL).index(2588) + 1
        assert scanned and position in scanned[0]
        persisted = _labels(deserialize_visual_cues(response.visual_cues))
        assert persisted[position] == ("QUÉT", "QUÉT")


class TestContextDiagnostics:
    def test_fillable_lines_are_counted(self) -> None:
        context = SimpleNamespace(
            target_lang="Vietnamese", source_lang="zh", overview="Tóm tắt.",
            characters="", glossary="", visual_cues=_positional_cues_json(),
            enable_tags=False, include_desc=True,
        )
        report = build_context_diagnostics(context=context, source_events=_events())
        assert report["visual_cues"]["run_fill_lines"] == 16

    def test_events_without_timing_are_accepted(self) -> None:
        village = _film(VILLAGE)
        context = SimpleNamespace(
            target_lang="Vietnamese", source_lang="zh", overview="Tóm tắt.",
            characters="", glossary="", visual_cues=_positional_cues_json(village),
            enable_tags=False, include_desc=True,
        )
        events = [
            SimpleNamespace(index=pos, text=village[line][0])
            for pos, line in enumerate(sorted(village), start=1)
        ]
        report = build_context_diagnostics(context=context, source_events=events)
        # Không có mốc thời gian thì không biết khoảng lặng: lượt lời chạy tới dòng 872.
        assert report["visual_cues"]["run_fill_lines"] == 10
