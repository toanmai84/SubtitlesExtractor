"""[v3.23.578] Dòng SỐ không bị coi là "dịch thiếu"; báo bảng nhân vật thiếu xưng hô.

Chuẩn đoán thật (phim "he thong sung the", 5540 dòng, v3.23.577):

* 14/29 cờ dịch thiếu ở bản dịch thô là dòng số: số viết bằng chữ Hán ("三十四点五万元",
  7 chữ) dịch ra CHỮ SỐ ("345.000 Tệ." — 2 chữ cái) nên tỉ lệ chỉ 7-14% trung vị. Cờ này
  đẩy vào ``needs_expansion`` của giai đoạn sau, và model "bổ sung" bằng câu dòng kề:
  665 "3.450.000 Tệ" → "3.450.000 Tệ đã vào tài khoản." trong khi 666 là "đã vào tài
  khoản.".
* Phân tích đã lưu của phim được khôi phục từ bản cũ: bảng 48 nhân vật, 0 mệnh đề xưng hô
  → mọi lưới xưng hô im lặng, ``address_rule_violations = 0`` trông như tốt.
* Sự kiện ``monologue_lines`` (v3.23.577) không có trong file chuẩn đoán của lần chạy nào
  vì nhật ký bị xoá SAU các bước chuẩn bị nhãn.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import replace

from subtitles_extractor.application.services.roster_format import (
    lacks_address_rules,
)
from subtitles_extractor.application.services.translation_diagnostics import (
    detect_quality_flags,
)
from subtitles_extractor.application.services.translation_process_log import (
    get_process_log,
)
from subtitles_extractor.application.services.under_translation_guard import (
    find_under_translated,
)
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

# Bản dịch thô thật, dòng 650-675 (v3.23.577).
LITERAL_ROWS = [
    (650, "喜欢吗", "Thích không?"),
    (651, "你的了", "Là của chị rồi đấy."),
    (652, "我的", "Của... của tôi á?"),
    (653, "林先生", "Lâm tiên sinh,"),
    (654, "手续帮您办好了", "thủ tục đã làm xong cho anh rồi."),
    (655, "车价总共", "Tổng giá xe là"),
    (656, "三十四点五万元", "345.000 Tệ."),
    (657, "恭喜您成为尊贵的", "Chúc mừng anh đã trở thành"),
    (658, "梅赛德斯车主", "chủ sở hữu xe Mercedes-Benz quý giá."),
    (659, "忙去吧", "Cô đi làm việc đi."),
    (660, "好嘞", "Vâng ạ."),
    (661, "检测到宿主", "Phát hiện ký chủ"),
    (662, "为目标赵红娟", "tiêu dùng cho mục tiêu Triệu Hồng Quyên"),
    (663, "消费三十四点五万元", "345.000 Tệ,"),
    (664, "激活十倍返现", "kích hoạt hoàn tiền gấp 10 lần,"),
    (665, "三百四十五万元", "3.450.000 Tệ"),
    (666, "已到账", "đã vào tài khoản."),
    (667, "三百四十五万", "3.450.000 Tệ..."),
    (668, "加上之前的", "Cộng thêm trước đó,"),
    (669, "我现在", "bây giờ mình"),
    (670, "也是百万富翁了", "cũng là triệu phú rồi."),
    (671, "三十四万五", "345.000 Tệ..."),
    (672, "林逸眼睛都不眨", "Lâm Dật chớp mắt một cái"),
    (673, "就买下了", "đã mua luôn rồi sao?"),
    (674, "这还是胖子那个", "Đây có phải là cái tên bạn thân kiết xác"),
    (675, "一年接济到头的", "quanh năm được Tam Béo"),
]
NUMERIC_LINES = {656, 663, 665, 667, 671}

# Ba dòng đầu thật của bảng nhân vật đã khôi phục (không có mục "xưng hô:" nào).
STALE_ROSTER = "\n".join([
    "林逸 (Lâm Dật) [nam, ~24] — nhân vật chính: nam thanh niên sở hữu Hệ thống Cưng "
    "Chiều Vợ, quyết đoán, thông minh",
    "宋欣爽 (Tống Hân Sảng) [nữ, ~26] — nhân vật chính: chị dâu của Lâm Dật, dịu dàng, "
    "nhân hậu",
    "秦月 (Tần Nguyệt) [nữ, ~24] — phản diện: bạn gái cũ hám tài của Lâm Dật",
])
RULED_ROSTER = "\n".join([
    "林逸 (Lâm Dật) [nam, ~24] — nhân vật chính; xưng hô: với Tống Hân Sảng tự xưng "
    "'em' gọi 'chị dâu'",
    "宋欣爽 (Tống Hân Sảng) [nữ, ~26] — chị dâu của Lâm Dật; xưng hô: với Lâm Dật tự "
    "xưng 'chị' gọi 'em'",
])


def _flagged(rows: list[tuple[int, str, str]]) -> set[int]:
    return {line.index for line in find_under_translated(rows)}


class TestNumericLines:
    def test_real_numeric_lines_are_not_under_translated(self) -> None:
        assert not _flagged(LITERAL_ROWS) & NUMERIC_LINES

    def test_real_percentage_is_not_under_translated(self) -> None:
        rows = [*LITERAL_ROWS, (300, "达到百分之两百", "đạt 200%.")]
        assert 300 not in _flagged(rows)

    def test_numeral_words_translated_as_letters_still_count(self) -> None:
        # "一点点" = "một chút" (chữ cái, không phải chữ số) → vẫn phải đo đủ độ dài.
        rows = [*LITERAL_ROWS, (1692, "这叫一点点钱吗", "Hả?")]
        assert 1692 in _flagged(rows)

    def test_digits_elsewhere_do_not_hide_a_truncated_line(self) -> None:
        # Chữ số chỉ bỏ CỤM số Hán, phần chữ còn lại vẫn được đo.
        rows = [*LITERAL_ROWS, (9001, "三百万我一定要让他们全部还回来", "300 vạn.")]
        assert 9001 in _flagged(rows)

    def test_other_lines_keep_their_verdict(self) -> None:
        truncated = (9002, "恭喜您成为尊贵的梅赛德斯车主", "Chúc")
        assert 9002 in _flagged([*LITERAL_ROWS, truncated])


class TestStaleRoster:
    def test_real_restored_roster_lacks_rules(self) -> None:
        assert lacks_address_rules(STALE_ROSTER)

    def test_roster_with_rules(self) -> None:
        assert not lacks_address_rules(RULED_ROSTER)

    def test_single_character_or_empty(self) -> None:
        assert not lacks_address_rules(STALE_ROSTER.splitlines()[0])
        assert not lacks_address_rules("")

    def test_diagnostics_report_rule_count(self) -> None:
        events = [
            SubtitleEvent(index=i, text=src, interval=TimeInterval(i, i + 0.9))
            for i, src, _ in LITERAL_ROWS
        ]
        translated = [replace(ev, text=text) for ev, (_, _, text) in
                      zip(events, LITERAL_ROWS, strict=True)]
        stale = detect_quality_flags(events, translated, characters=STALE_ROSTER)
        ruled = detect_quality_flags(events, translated, characters=RULED_ROSTER)
        assert stale["counts"]["address_rules"] == 0
        assert ruled["counts"]["address_rules"] == 2
        assert "counts.address_rules" in stale["note"]
        assert not {row["index"] for row in stale["under_translation"]} & NUMERIC_LINES


class _Translator:
    def translate_stage(self, *, input_lines, **_kwargs):
        return [replace(line, text="bản dịch") for line in input_lines]


def _run(characters: str, cues: str = "") -> list[dict]:
    events = [
        SubtitleEvent(index=pos, text=src, interval=TimeInterval(pos, pos + 0.9))
        for pos, (_, src, _) in enumerate(LITERAL_ROWS, start=1)
    ]
    request = TranslateSubtitlesRequest(
        events=events,
        stages=[
            TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="m")
        ],
        context=TranslationContext(
            target_lang="Vietnamese", characters=characters, visual_cues=cues
        ),
        video_refs=("video",) if cues else (),
        enable_visual_cues=bool(cues),
    )
    get_process_log().record("left_over_from_previous_run")
    TranslateSubtitlesUseCase(translator=_Translator()).execute(request)  # type: ignore[arg-type]
    return get_process_log().snapshot()["events"]


class TestPipelineEvents:
    def test_stale_roster_is_recorded(self) -> None:
        events = _run(STALE_ROSTER)
        recorded = [e for e in events if e["event"] == "roster_without_address_rules"]
        assert recorded == [{"event": "roster_without_address_rules", "characters": 3}]

    def test_ruled_roster_is_not_recorded(self) -> None:
        events = _run(RULED_ROSTER)
        assert not [e for e in events if e["event"] == "roster_without_address_rules"]

    def test_previous_run_is_cleared(self) -> None:
        events = _run(RULED_ROSTER)
        assert not [e for e in events if e["event"] == "left_over_from_previous_run"]

    def test_monologue_event_survives_to_the_diagnostics(self) -> None:
        # Dòng 669-670 "我现在 / 也是百万富翁了" — Lâm Dật nghĩ thầm (nhãn tự-nói).
        cues = json.dumps(
            [{"id": 20, "spk": "林逸", "to": "林逸", "cue": ""},
             {"id": 21, "spk": "林逸", "to": "林逸", "cue": ""}],
            ensure_ascii=False,
        )
        events = _run(RULED_ROSTER, cues)
        recorded = [e for e in events if e["event"] == "monologue_lines"]
        assert recorded and recorded[0]["monologue"] == 2


class TestRestoreWarning:
    def test_view_model_counts_stale_roster(self) -> None:
        from subtitles_extractor.presentation.view_models.translate_page_view_model import (  # noqa: E501
            TranslatePageViewModel,
        )

        assert TranslatePageViewModel.roster_without_address_rules(STALE_ROSTER) == 3
        assert TranslatePageViewModel.roster_without_address_rules(RULED_ROSTER) == 0

    def test_page_warns_with_translated_strings(self) -> None:
        data_dir = pathlib.Path("src/subtitles_extractor/data")
        for name in ("strings_vi.json", "strings_en.json"):
            strings = json.loads((data_dir / name).read_text(encoding="utf-8"))
            assert strings["translate"]["toast_restored_no_address_title"]
            assert "{count}" in strings["translate"]["toast_restored_no_address_body"]
        page = pathlib.Path(
            "src/subtitles_extractor/presentation/pages/translate_page.py"
        ).read_text(encoding="utf-8")
        assert "roster_without_address_rules(" in page
        assert "translate.toast_restored_no_address_body" in page
