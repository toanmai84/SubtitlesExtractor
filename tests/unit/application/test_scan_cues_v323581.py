"""[v3.23.581] Nhãn quét bổ sung "lạc cảnh"; bộ quét có mốc thời gian + mạch hội thoại.

Chuẩn đoán thật (phim "he thong sung the", 5540 dòng, phân tích lại ở v3.23.578): cues của
khâu phân tích phủ 57% nên khâu dịch quét bổ sung 2358 dòng. Nhãn quét (luôn kèm mô tả)
sai người nghe ở 29% dòng gọi tên, nhãn của khâu phân tích chỉ 7%. Kiểu sai "lạc cảnh":
giữa cuộc gọi Lâm Dật ↔ chị dâu (1303, 1308), dòng 1304-1307 mang nhãn 付金奎 → 王大福 kèm
"Phó Kim Khuê hét lên đe dọa". Bộ quét chỉ gửi ``id`` + chữ của các dòng thiếu nhãn —
không mốc thời gian, không dòng nào quanh chúng.

Cùng phim: "三百四十五万元" (3.450.000) dịch "ba triệu bốn trăm năm mươi vạn tệ" (7,5
triệu) ở cả hai dòng của số tiền đó — lưới số không có cách đọc đúng nào để chọn.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

from subtitles_extractor.application.services.cue_contradiction_guard import (
    contradicted_cue_lines,
    described_scan_lines,
)
from subtitles_extractor.application.services.cue_label_canonicalizer import (
    canonicalize_cue_labels,
)
from subtitles_extractor.application.services.cue_run_filler import LineSpan
from subtitles_extractor.application.services.number_consistency_guard import (
    find_number_fixes,
    vietnamese_number_value,
    vietnamese_number_words,
)
from subtitles_extractor.application.services.translation_process_log import (
    get_process_log,
)
from subtitles_extractor.application.services.visual_cue_batching import (
    clip_timestamp,
    known_context_items,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationLine,
    VisualCue,
)

ROSTER = "\n".join([
    "林逸 (Lâm Dật) [nam, ~24] — nhân vật chính",
    "宋欣爽 / 小爽 (Tống Hân Sảng / Tiểu Sảng) [nữ, ~27] — chị dâu của Lâm Dật",
    "付金奎 (Phó Kim Khuê) [nam, ~50] — phó giám đốc xưởng",
    "钟俊涛 (Chung Tuấn Đào) [nam, ~26] — thiếu gia",
    "王大福 / 胖子 / 三胖子 (Vương Đại Phúc / Bàn Tử) [nam, ~24] — bạn thân Lâm Dật",
])

# (dòng, câu gốc, bắt đầu ms, kết thúc ms, người nói, người nghe, mô tả) — cues thật.
ROWS = [
    (1300, "得把这个消息", 2451867, 2452867, "林逸", "宋欣爽 / 小爽", ""),
    (1301, "告诉嫂子啊", 2452933, 2454333, "付金奎", "钟俊涛",
     "Phó Kim Khuê tức tối nói về việc đánh đấm"),
    (1302, "小逸", 2460067, 2460800, "王大福 / 胖子 / 三胖子", "林逸",
     "Bàn Tử gọi Lâm Dật với vẻ mặt tức giận"),
    (1303, "嫂子", 2462133, 2462800, "林逸", "宋欣爽 / 小爽", ""),
    (1304, "你在医院吗", 2463000, 2464133, "付金奎", "王大福 / 胖子 / 三胖子",
     "Phó Kim Khuê hét lên đe dọa"),
    (1305, "我有个好消息", 2464200, 2464733, "付金奎", "王大福 / 胖子 / 三胖子",
     "Phó Kim Khuê lớn tiếng đe dọa Bàn Tử"),
    (1306, "要告诉你", 2464800, 2465533, "付金奎", "王大福 / 胖子 / 三胖子",
     "Phó Kim Khuê tiếp tục đe dọa"),
    (1307, "我来找你", 2465800, 2466533, "付金奎", "王大福 / 胖子 / 三胖子",
     "Phó Kim Khuê quát tháo"),
    (1308, "不了", 2467000, 2467533, "宋欣爽 / 小爽", "林逸", ""),
    (1309, "我今天有点不太舒服", 2467933, 2469333, "付金奎", "钟俊涛",
     "Phó Kim Khuê quay sang nói với Chung Tuấn Đào với vẻ hậm hực"),
    (1310, "请假了", 2469733, 2470533, "付金奎", "钟俊涛", "Phó Kim Khuê phàn nàn"),
    (1311, "不舒服", 2471000, 2471667, "钟俊涛", "付金奎",
     "Chung Tuấn Đào lạnh lùng ngắt lời"),
    (1312, "嫂子", 2471733, 2472600, "林逸", "宋欣爽 / 小爽", ""),
    (1313, "你怎么了", 2472667, 2473267, "钟俊涛", "付金奎",
     "Chung Tuấn Đào hất hàm hỏi"),
    (1314, "其实也没什么", 2473400, 2475000, "宋欣爽 / 小爽", "林逸", ""),
]
LOST_IN_SCENE = [1301, 1304, 1305, 1306, 1307, 1309, 1310, 1311, 1313]


def _cues() -> list[VisualCue]:
    return canonicalize_cue_labels(
        [VisualCue(r[0], r[4], r[5], r[6]) for r in ROWS], ROSTER
    )


def _spans() -> dict[int, LineSpan]:
    return {r[0]: LineSpan(r[0], r[2], r[3]) for r in ROWS}


def _lines() -> list[TranslationLine]:
    return [
        TranslationLine(index=r[0], start_ms=r[2], end_ms=r[3], text=r[1]) for r in ROWS
    ]


class TestContradictionGuard:
    def test_real_lost_scene_labels_are_found(self) -> None:
        cues = _cues()
        untrusted = described_scan_lines(cues)
        assert contradicted_cue_lines(cues, untrusted, _spans()) == LOST_IN_SCENE

    def test_label_sharing_a_person_is_kept(self) -> None:
        # 1302 "小逸" (王大福 → 林逸) có chung Lâm Dật với cuộc gọi — không bỏ.
        cues = _cues()
        lost = contradicted_cue_lines(cues, described_scan_lines(cues), _spans())
        assert 1302 not in lost

    def test_single_source_file_is_untouched(self) -> None:
        described = [VisualCue(r[0], r[4], r[5], "mô tả") for r in ROWS]
        plain = [VisualCue(r[0], r[4], r[5], "") for r in ROWS]
        assert described_scan_lines(described) == set()
        assert described_scan_lines(plain) == set()

    def test_third_party_naming_the_listener_is_kept(self) -> None:
        roster = ROSTER + "\n秦可秀 / 阿秀 (Tần Khả Tú) [nữ, ~25] — vợ"
        cues = canonicalize_cue_labels([
            VisualCue(1, "林逸", "付金奎", ""),
            VisualCue(2, "钟俊涛", "秦可秀 / 阿秀", "mô tả"),
            VisualCue(3, "付金奎", "林逸", ""),
        ], roster)
        spans = {i: LineSpan(i, i * 1000, i * 1000 + 800) for i in (1, 2, 3)}
        sources = {1: "你给我滚", 2: "阿秀你没事吧", 3: "你敢"}
        assert contradicted_cue_lines(cues, {2}, spans, sources, roster) == []
        assert contradicted_cue_lines(cues, {2}, spans) == [2]


class TestScanPayload:
    def test_clip_relative_timestamp(self) -> None:
        # Dòng 1304 bắt đầu 2463,0 s; đoạn video 8 của phim bắt đầu 2446 s.
        assert clip_timestamp(2463000, 2446.0) == "00:17"
        assert clip_timestamp(2446000 + 3725000, 2446.0) == "1:02:05"
        assert clip_timestamp(1000, 5.0) == "00:00"

    def test_known_lines_between_missing_ones(self) -> None:
        lines = _lines()
        missing = [line for line in lines if line.index in {1304, 1305, 1306, 1307}]
        known = [cue for cue in _cues() if cue.line_no in {1300, 1303, 1308, 1312}]
        items = known_context_items(missing, lines, known, 2446.0)
        assert [item["id"] for item in items] == [1303, 1308]
        assert items[0] == {
            "id": 1303, "text": "嫂子", "spk": "林逸", "to": "宋欣爽 / 小爽",
            "t": "00:16",
        }


class TestAdapterScanPrompt:
    def _adapter(self, prompts: list[str], reply: dict):
        from subtitles_extractor.infrastructure.translation import (
            gemini_translation_adapter as module,
        )

        adapter = module.GeminiSubtitleTranslator(api_key="dummy")
        adapter._ensure_available = lambda: None  # type: ignore[assignment]
        adapter._build_config = lambda *a, **k: None  # type: ignore[assignment]
        adapter._resolve_handles_with_heal = (  # type: ignore[assignment]
            lambda *a, **k: []
        )
        adapter._handles_for_batch = lambda *a, **k: []  # type: ignore[assignment]

        def _call(_model, prompt, *_args, **_kwargs):
            prompts.append(prompt)
            return reply

        adapter._call_gemini = _call  # type: ignore[assignment]
        return adapter

    def test_prompt_has_timestamps_and_known_context(self) -> None:
        prompts: list[str] = []
        reply = {"cues": [
            {"id": 1305, "spk": "林逸", "to": "宋欣爽 / 小爽"},
            {"id": 1303, "spk": "付金奎", "to": "钟俊涛"},
        ]}
        adapter = self._adapter(prompts, reply)
        lines = _lines()
        missing = [line for line in lines if line.index in {1304, 1305, 1306, 1307}]
        known = [cue for cue in _cues() if cue.line_no in {1300, 1303, 1308, 1312}]
        cues = adapter.analyze_visual_cues(
            missing, "Vietnamese",
            video_refs=[SimpleNamespace(start_sec=2446.0, end_sec=2795.0)],
            sleep_between_s=0.0, known_cues=known, context_lines=lines,
        )
        prompt = prompts[0]
        lines_block = prompt.split("<lines>")[1].split("</lines>")[0]
        known_block = prompt.split("<known>")[1].split("</known>")[0]
        first = {"id": 1304, "text": "你在医院吗", "t": "00:17"}
        assert json.loads(lines_block)[0] == first
        assert [item["id"] for item in json.loads(known_block)] == [1303, 1308]
        # Nhãn model trả cho dòng tham khảo (1303) không được nhận.
        assert [cue.line_no for cue in cues] == [1305]

    def test_no_timestamp_without_exactly_one_clip(self) -> None:
        prompts: list[str] = []
        adapter = self._adapter(prompts, {"cues": []})
        adapter.analyze_visual_cues(_lines()[:2], "Vietnamese", sleep_between_s=0.0)
        assert '"t"' not in prompts[0]
        assert "<known>" not in prompts[0]


class _Scanner:
    def __init__(self, reply: list[VisualCue]) -> None:
        self.reply = reply
        self.kwargs: dict = {}

    def analyze_visual_cues(self, source_lines, target_lang, **kwargs):
        self.kwargs = kwargs
        wanted = {line.index for line in source_lines}
        return [cue for cue in self.reply if cue.line_no in wanted]

    def translate_stage(self, *, input_lines, **_kwargs):
        return [replace(line, text="bản dịch") for line in input_lines]

    def is_available(self) -> bool:
        return True


def _request(cues: list[VisualCue], enable: bool):
    from subtitles_extractor.application.services.visual_cue_serializer import (
        serialize_visual_cues,
    )
    from subtitles_extractor.application.use_cases.translate_subtitles import (
        TranslateSubtitlesRequest,
    )
    from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
    from subtitles_extractor.domain.ports.subtitle_translator_port import (
        TranslationContext,
        TranslationStageConfig,
        TranslationStageKind,
    )
    from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

    # Pipeline đánh số dòng theo vị trí (1..N): cues cũng dùng số vị trí.
    events = [
        SubtitleEvent(
            index=pos, text=r[1], interval=TimeInterval(r[2] / 1000, r[3] / 1000)
        )
        for pos, r in enumerate(ROWS, start=1)
    ]
    shifted = [replace(cue, line_no=cue.line_no - ROWS[0][0] + 1) for cue in cues]
    return TranslateSubtitlesRequest(
        events=events,
        stages=[
            TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="m")
        ],
        context=TranslationContext(
            target_lang="Vietnamese", characters=ROSTER,
            visual_cues=serialize_visual_cues(shifted),
        ),
        video_refs=("video",) if enable else (),
        enable_visual_cues=enable,
    )


class TestPipeline:
    def test_saved_lost_scene_labels_are_dropped_before_translation(self) -> None:
        from subtitles_extractor.application.services.visual_cue_serializer import (
            deserialize_visual_cues,
        )
        from subtitles_extractor.application.use_cases.translate_subtitles import (
            TranslateSubtitlesUseCase,
        )

        get_process_log().clear()
        request = _request(_cues(), enable=False)
        use_case = TranslateSubtitlesUseCase(translator=_Scanner([]))  # type: ignore
        response = use_case.execute(request)
        kept = {cue.line_no + ROWS[0][0] - 1 for cue in
                deserialize_visual_cues(response.visual_cues)}
        assert not kept & set(LOST_IN_SCENE)
        assert {1300, 1302, 1303, 1308, 1312, 1314} <= kept
        events = [e for e in get_process_log().snapshot()["events"]
                  if e["event"] == "scan_cues_dropped"]
        assert events and len(events[0]["lines"]) == len(LOST_IN_SCENE)

    def test_top_up_drops_lost_scene_scan_and_sends_context(self) -> None:
        from subtitles_extractor.application.use_cases.translate_subtitles import (
            TranslateSubtitlesUseCase,
        )

        plain = [cue for cue in _cues() if not cue.scene]
        # Bản quét mới (vị trí 1..N): dòng 1305 lạc cảnh, dòng 1302 hợp cảnh.
        reply = [
            VisualCue(6, "付金奎", "王大福 / 胖子 / 三胖子", "Phó Kim Khuê đe dọa"),
            VisualCue(3, "王大福 / 胖子 / 三胖子", "林逸", "Bàn Tử gọi Lâm Dật"),
        ]
        scanner = _Scanner(reply)
        request = _request(plain, enable=True)
        use_case = TranslateSubtitlesUseCase(translator=scanner)  # type: ignore[arg-type]
        topped = use_case._top_up_context_cues(
            use_case._events_to_lines(request.events), request, None, None, None
        )
        assert scanner.kwargs["known_cues"]
        assert len(scanner.kwargs["context_lines"]) == len(ROWS)
        labelled = {
            cue["id"] for cue in json.loads(topped.context.visual_cues)
            if cue.get("spk") or cue.get("to")
        }
        assert 3 in labelled
        assert 6 not in labelled


class TestStandardReading:
    def test_real_value_error_without_reference_reading(self) -> None:
        sources = {665: "三百四十五万元", 667: "三百四十五万"}
        translations = {
            665: "ba triệu bốn trăm năm mươi vạn tệ",
            667: "Ba triệu bốn trăm năm mươi vạn...",
        }
        fixes = {f.index: f.text_after for f in find_number_fixes(sources, translations)}
        assert fixes == {
            665: "ba triệu bốn trăm năm mươi nghìn tệ",
            667: "Ba triệu bốn trăm năm mươi nghìn...",
        }

    def test_reading_round_trips(self) -> None:
        for value in (1_000, 15_000, 21_000, 100_000, 345_000, 960_000, 1_500_000,
                      3_450_000, 10_000_000, 34_300_000, 394_928_520, 2_000_000_000):
            assert vietnamese_number_value(vietnamese_number_words(value)) == value

    def test_digit_written_amount_keeps_its_writing(self) -> None:
        # [v3.23.582] Trước đây cụm viết bằng chữ số bị bỏ qua dù sai giá trị (ca thật
        # "3.455 vạn" cho 3,45 triệu). Nay sửa PHẦN CHỮ SỐ, giữ đơn vị và lối viết chữ số
        # của phim — không đổi sang chữ.
        fixes = find_number_fixes({1: "三千四百三十万元"}, {1: "3,43 triệu tệ"})
        assert [fix.text_after for fix in fixes] == ["34,3 triệu tệ"]
