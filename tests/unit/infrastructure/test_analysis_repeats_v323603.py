"""[v3.23.603] Phân tích kèm Visual Cues gửi CẢ dòng lặp liền nhau để có nhãn từng dòng.

Lượt thật 11/10 (灭门惨案II, gemini-3.8-flash): "1246 dòng gốc → 1240 dòng sau khử trùng
→ gửi AI"; 6/6 dòng phải hỏi lại người nói đúng là 6 dòng lặp bị bỏ (207, 217, 356, 366,
455, 714). Bước hỏi lại mất 5 phút 36 giây và 4 lời gọi (cộng các lượt 503) của model 20
lượt/ngày — key đang dùng hết hạn mức ngay lời gọi đầu.

Dữ liệu: dòng 204-218 thật (mốc giây), đoạn video thứ 3 và 4 thật (17 đoạn ~344,18 s).
"""

from __future__ import annotations

import json
import re
from unittest.mock import MagicMock, patch

from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    analysis_source_lines,
)
from tests.unit.infrastructure.gemini_call_harness import bare_analysis_adapter

SEGMENT = 344.1764705882353
SEGMENTS = [(2 * SEGMENT, 3 * SEGMENT), (3 * SEGMENT, 4 * SEGMENT)]
LINES = [
    TranslationLine(index=index, start_ms=int(start * 1000), end_ms=int(end * 1000),
                    text=text)
    for index, start, end, text in [
        (204, 848.133, 849.267, "而女死者是有梅毒的"),
        (205, 849.333, 850.133, "那麼，這是誰的"),  # noqa: RUF001 — chữ thật
        (206, 850.667, 851.4, "朱正軍"),
        (207, 852.533, 854.067, "朱正軍"),
        (208, 856.133, 857.333, "你跟我回去準備行動"),
        (209, 859.067, 859.8, "所長"),
        (210, 860.6, 863.067, "怎麼這個芒果特别香滑的"),
        (211, 863.133, 865.733, "當然，這是我家傳秘製的"),  # noqa: RUF001 — chữ thật
        (212, 865.8, 867.0, "西米露芒果"),
        (213, 867.067, 869.4, "是嗎甚麼時候可以再吃呀"),
        (214, 870.467, 872.067, "如果這一次你表現得好"),
        (215, 872.333, 874.333, "我親自泡製一個你吃"),
        (216, 874.4, 874.8, "真的好捧啊"),
        (217, 875.133, 876.133, "真的好捧啊"),
        (218, 880.6, 881.267, "對不起"),
    ]
]


def test_giu_dong_lap_khi_can_nhan_tung_dong() -> None:
    kept = analysis_source_lines(LINES, keep_repeats=True)
    assert [line.index for line in kept] == list(range(204, 219))


def test_van_khu_trung_khi_khong_can_nhan() -> None:
    kept = analysis_source_lines(LINES, keep_repeats=False)
    assert 207 not in {line.index for line in kept}
    assert 217 not in {line.index for line in kept}
    assert len(kept) == len(LINES) - 2


def test_dong_rong_luon_bi_bo() -> None:
    blank = TranslationLine(index=219, start_ms=882_000, end_ms=883_000, text="  ")
    assert analysis_source_lines([*LINES, blank], keep_repeats=True)[-1].index == 218


def _ref(index: int) -> MagicMock:
    ref = MagicMock()
    ref.chunk_index = index
    ref.remote_name = f"files/f{index}"
    ref.start_sec, ref.end_sec = SEGMENTS[index]
    return ref


def _analyse(with_cues: bool) -> list[str]:
    adapter = bare_analysis_adapter()
    prompts: list[str] = []

    def fake_call(_model, prompt, _config, _validator, video_refs=None, **_kw):
        prompts.append(prompt)
        return {"source_lang": "zh", "characters": [], "overview": "x", "cues": []}

    with (
        patch.object(adapter, "_resolve_video_handles", lambda refs: ["h0", "h1"]),
        patch.object(adapter, "_build_config", lambda *a, **k: MagicMock()),
        patch.object(adapter, "_call_gemini", fake_call),
        patch.object(adapter, "_ensure_available", lambda: None),
    ):
        adapter.analyze_global_context(
            LINES, "Vietnamese", video_refs=[_ref(0), _ref(1)],
            with_visual_cues=with_cues,
        )
    return prompts


def _listed_ids(prompt: str) -> list[int]:
    match = re.search(r"<segment_lines>\n(.*)\n</segment_lines>", prompt, re.S)
    assert match, "prompt thiếu danh sách dòng của đoạn"
    return [item["id"] for item in json.loads(match.group(1))]


def test_doan_video_liet_ke_ca_dong_lap() -> None:
    first, _second = _analyse(with_cues=True)
    assert _listed_ids(first) == list(range(204, 219))
    assert "207. 朱正軍" in first
    assert "217. 真的好捧啊" in first


def test_khong_visual_cues_thi_noi_dung_van_khu_trung() -> None:
    first, _second = _analyse(with_cues=False)
    assert "206. 朱正軍" in first
    assert "207. 朱正軍" not in first
