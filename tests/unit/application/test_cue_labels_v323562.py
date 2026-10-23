"""[v3.23.562] Hồi quy: cues quét bổ sung mang nhãn vai vế chung chung, không khớp bảng.

Chuẩn đoán thật v3.23.561: cues "phủ" 99,7% số dòng, nhưng bộ quét không được đưa bảng
nhân vật nên 95% nhãn của phần quét bổ sung là vai vế chung chung, nhiều nhãn chép nguyên
ví dụ trong prompt ("Nữ tỳ" 208 dòng, "Nam đệ tử" 131, "Sư phụ" 24 — trong phim đô thị).
Độ phủ DÙNG ĐƯỢC chỉ 26%; cờ xưng hô trôi thì báo ảo cho cặp "Nữ chính → Nam chính".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from subtitles_extractor.application.services.context_diagnostics import (
    build_context_diagnostics,
)
from subtitles_extractor.application.services.cue_label_canonicalizer import (
    canonicalize_cue_labels,
    roster_dialogue_cues_json,
    roster_label_map,
    usable_cue_lines,
)
from subtitles_extractor.application.services.translation_diagnostics import (
    detect_quality_flags,
)
from subtitles_extractor.application.services.visual_cue_serializer import (
    deserialize_visual_cues,
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
    VisualCue,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
    _visual_cues_system_instruction,
)

ROSTER = (
    "宋钱 (Tống Tiền) [nam, ~22] — nhân vật chính\n"
    "许桃花 (Hứa Đào Hoa) [nữ, ~25] — góa phụ trong làng\n"
    "吴二狗 / 土狗 (Ngô Nhị Cẩu) [nam, ~30] — đầu gấu chợ"
)


def _cue(line_no: int, speaker: str, addressee: str = "") -> VisualCue:
    return VisualCue(line_no=line_no, speaker=speaker, addressee=addressee)


class TestRosterLabelMap:
    def test_name_aliases_and_pronunciation_map_to_canonical_name(self) -> None:
        label_map = roster_label_map(ROSTER)
        assert label_map["宋钱"] == "宋钱"
        assert label_map["tống tiền"] == "宋钱"
        assert label_map["土狗"] == "吴二狗 / 土狗"
        assert label_map["吴二狗 / 土狗"] == "吴二狗 / 土狗"

    def test_shared_spelling_is_dropped_instead_of_guessing(self) -> None:
        roster = "林逸 (Lâm) — anh\n林妹 (Lâm) — em"
        assert "lâm" not in roster_label_map(roster)

    def test_no_roster_gives_empty_map(self) -> None:
        assert roster_label_map("") == {}


class TestCanonicalizeAndUsable:
    def test_transliteration_and_alias_become_roster_names(self) -> None:
        cues = [_cue(1, "Tống Tiền", "hứa đào hoa"), _cue(2, "土狗", "Nữ phụ")]
        result = canonicalize_cue_labels(cues, ROSTER)
        assert (result[0].speaker, result[0].addressee) == ("宋钱", "许桃花")
        assert (result[1].speaker, result[1].addressee) == ("吴二狗 / 土狗", "Nữ phụ")

    def test_generic_labels_are_not_usable(self) -> None:
        cues = [_cue(1, "Nữ tỳ", "Nam đệ tử"), _cue(2, "宋钱", "Nữ phụ"), _cue(3, "", "")]
        assert usable_cue_lines(cues, ROSTER) == {2}

    def test_without_roster_every_labeled_cue_counts(self) -> None:
        cues = [_cue(1, "Nữ tỳ", "Nam đệ tử"), _cue(3, "", "")]
        assert usable_cue_lines(cues, "") == {1}

    def test_roster_dialogue_keeps_only_named_pairs(self) -> None:
        raw = json.dumps(
            [
                {"id": 1, "spk": "Tống Tiền", "to": "许桃花"},
                {"id": 2, "spk": "Nữ chính", "to": "Nam chính"},
                {"id": 3, "spk": "宋钱", "to": "Mọi người"},
            ]
        )
        kept = deserialize_visual_cues(roster_dialogue_cues_json(raw, ROSTER))
        assert [(c.line_no, c.speaker, c.addressee) for c in kept] == [
            (1, "宋钱", "许桃花")
        ]


class _Scanner:
    def __init__(self, labels: dict[int, tuple[str, str]]) -> None:
        self.labels = labels
        self.scanned: list[int] = []
        self.characters: list[str] = []

    def analyze_visual_cues(self, lines, _lang, characters: str = "", **_kwargs):
        self.scanned = [line.index for line in lines]
        self.characters.append(characters)
        return [
            _cue(line.index, *self.labels[line.index])
            for line in lines
            if line.index in self.labels
        ]


def _lines(count: int = 10) -> list[TranslationLine]:
    return [
        TranslationLine(index=i, start_ms=i * 1000, end_ms=i * 1000 + 900, text=f"句{i}")
        for i in range(1, count + 1)
    ]


def _request(cues: list[dict[str, Any]]) -> TranslateSubtitlesRequest:
    return TranslateSubtitlesRequest(
        events=[],
        stages=[
            TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="m")
        ],
        context=TranslationContext(
            target_lang="Vietnamese", characters=ROSTER, visual_cues=json.dumps(cues)
        ),
        video_refs=(object(),),
        enable_visual_cues=True,
    )


class TestTopUpRescansGenericLabels:
    def test_generic_only_lines_count_as_missing_and_roster_is_sent(self) -> None:
        cues = [{"id": i, "spk": "Nữ tỳ", "to": "Nam đệ tử"} for i in range(1, 11)]
        cues[0] = {"id": 1, "spk": "宋钱", "to": "许桃花"}
        scanner = _Scanner({i: ("Tống Tiền", "Hứa Đào Hoa") for i in range(2, 11)})
        result = TranslateSubtitlesUseCase(translator=scanner)._top_up_context_cues(
            _lines(), _request(cues), None, None
        )
        assert scanner.scanned == list(range(2, 11))
        assert scanner.characters == [ROSTER]
        merged = deserialize_visual_cues(result.context.visual_cues)
        assert {c.speaker for c in merged} == {"宋钱"}

    def test_generic_cue_kept_where_new_scan_is_empty(self) -> None:
        cues = [{"id": i, "spk": "Nữ tỳ", "to": "Nam đệ tử"} for i in range(1, 11)]
        scanner = _Scanner({i: ("宋钱", "许桃花") for i in range(1, 6)})
        result = TranslateSubtitlesUseCase(translator=scanner)._top_up_context_cues(
            _lines(), _request(cues), None, None
        )
        merged = {
            c.line_no: c.speaker
            for c in deserialize_visual_cues(result.context.visual_cues)
        }
        assert merged[1] == "宋钱"
        assert merged[9] == "Nữ tỳ"

    def test_roster_labeled_coverage_above_threshold_costs_nothing(self) -> None:
        cues = [{"id": i, "spk": "Tống Tiền", "to": "Nữ phụ"} for i in range(1, 8)]
        scanner = _Scanner({})
        request = _request(cues)
        result = TranslateSubtitlesUseCase(translator=scanner)._top_up_context_cues(
            _lines(), request, None, None
        )
        assert result is request
        assert scanner.characters == []


class TestScannerPrompt:
    def test_roster_is_in_the_prompt_and_old_examples_are_gone(self) -> None:
        text = _visual_cues_system_instruction("Vietnamese", ROSTER)
        assert "宋钱 (Tống Tiền)" in text
        assert "TÊN GỐC" in text
        for parroted in ("Nữ tỳ", "Nam đệ tử", "Sư phụ"):
            assert parroted not in text

    def test_without_roster_asks_for_consistent_role_labels(self) -> None:
        text = _visual_cues_system_instruction("Vietnamese", "")
        assert "BẢNG NHÂN VẬT" not in text
        assert "Nữ tỳ" not in text

    def test_analyze_visual_cues_builds_config_with_roster(self) -> None:
        adapter = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
        adapter._ensure_available = lambda: None
        seen: list[str] = []

        def _config(_temperature, _schema, system_instr, **_kwargs):
            seen.append(system_instr)

        adapter._build_config = _config
        adapter._resolve_handles_with_heal = lambda refs, _extra: []
        adapter._call_gemini = lambda *args, **kwargs: {"cues": []}
        adapter.analyze_visual_cues(_lines(2), "Vietnamese", characters=ROSTER)
        assert "宋钱 (Tống Tiền)" in seen[0]


@dataclass
class _Event:
    index: int
    text: str
    interval: Any = field(default_factory=lambda: SimpleNamespace(start_sec=0, end_sec=1))


class TestContextDiagnosticsUsableCoverage:
    def test_generic_labels_do_not_count_as_coverage(self) -> None:
        cues = [{"id": i, "spk": "Nữ tỳ", "to": "Nam đệ tử"} for i in range(1, 401)]
        for i in range(1, 101):
            cues[i - 1] = {"id": i, "spk": "宋钱", "to": "许桃花"}
        ctx = SimpleNamespace(
            target_lang="Vietnamese",
            source_lang="zh",
            overview="x",
            characters=ROSTER,
            glossary="",
            visual_cues=json.dumps(cues),
            enable_tags=False,
            include_desc=True,
        )
        events = [_Event(i, "宋钱说") for i in range(1, 401)]
        report = build_context_diagnostics(context=ctx, source_events=events)
        assert report["visual_cues"]["coverage_ratio"] == 1.0
        assert report["visual_cues"]["roster_labeled_ratio"] == 0.25
        flag = next(
            f for f in report["quality_flags"] if f["code"] == "cues_low_coverage"
        )
        assert flag["severity"] == "error"
        assert "nhãn vai vế chung chung" in flag["message"]


class TestTranslationDiagnosticsIgnoreGenericPairs:
    @staticmethod
    def _events(texts: list[str]) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(index=i, text=t, start_sec=i, end_sec=i + 0.9)
            for i, t in enumerate(texts, start=1)
        ]

    def test_generic_pair_drift_is_not_reported(self) -> None:
        sources = ["我来了"] * 6
        translations = ["Em tới rồi", "Tôi tới rồi"] * 3
        cues = json.dumps(
            [{"id": i, "spk": "Nữ chính", "to": "Nam chính"} for i in range(1, 7)]
        )
        flags = detect_quality_flags(
            self._events(sources), self._events(translations), "", ROSTER, cues
        )
        assert flags["address_drift"] == []
        assert flags["counts"]["address_cue_coverage"] == 0.0

    def test_named_pair_drift_is_still_reported(self) -> None:
        sources = ["我来了"] * 6
        translations = ["Em tới rồi", "Tôi tới rồi"] * 3
        cues = json.dumps(
            [{"id": i, "spk": "Tống Tiền", "to": "许桃花"} for i in range(1, 7)]
        )
        flags = detect_quality_flags(
            self._events(sources), self._events(translations), "", ROSTER, cues
        )
        assert [(f["speaker"], f["addressee"]) for f in flags["address_drift"]] == [
            ("宋钱", "许桃花")
        ]


class TestSwappedAliasOrder:
    def test_swapped_alias_order_still_matches(self) -> None:
        roster = "可可 / 赵可可 (Khả Khả) [nữ, ~20] — trợ lý"
        cues = canonicalize_cue_labels([_cue(1, "赵可可 / 可可", "宋钱")], roster)
        assert cues[0].speaker == "可可 / 赵可可"

    def test_aliases_of_different_people_do_not_merge(self) -> None:
        cues = canonicalize_cue_labels([_cue(1, "宋钱 / 许桃花", "")], ROSTER)
        assert cues[0].speaker == "宋钱 / 许桃花"
