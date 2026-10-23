# ruff: noqa: RUF001 — dấu câu fullwidth CJK thuộc câu gốc chép từ phim
"""[v3.23.612] Tuỳ chọn xưng hô thuần Việt: "anh Kiếm" thay "Kiếm ca".

Bảng nhân vật, câu gốc và bản dịch chép từ các lượt chạy thật:

* phim "灭门惨案II", bản 3.23.609 (``068fe16b``, 13/10): 28 lần "Kiếm ca", 8 lần "đại
  tẩu", "Chân tiên sinh";
* phim "Hệ thống sủng thê" (``069d0c24``): 28 lần "Lâm tiên sinh", "Cường ca", "Biểu ca",
  "đại tẩu".

Hàng bảng nhân vật được rút gọn phần mô tả, giữ nguyên tên, giới tính, tuổi và các mệnh
đề xưng hô cần cho phép thử.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.application.services.native_address import (
    LOCATION_GLOSSARY,
    LOCATION_OVERVIEW,
    LOCATION_ROSTER,
    describe_changes,
    name_syllables,
    nativize_context,
    nativize_lines,
)
from subtitles_extractor.application.services.roster_format import address_rule_map
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleContextAnalysis,
    TranslationContext,
)

LONG = (
    "龍傳人 (Long Truyền Nhân) [nam, ~40] — nhân vật chính: Đồn trưởng đồn công an Đông "
    "Long; xưng hô: với 朱正軍 tự xưng 'tao' gọi 'mày'; với 秦可秀 tự xưng 'tôi' gọi "
    "'cô' (danh xưng: Chân đại tẩu)"
)
HONG = (
    "洪震 (Hồng Chấn) [nam, ~50] — ác bá dâm loạn tống tiền khét tiếng địa phương; xưng "
    "hô: với 洪大嫂 tự xưng 'tôi' gọi 'bà'; với 秦可秀 tự xưng 'tao' gọi 'mày'; "
    "với 甄劍聲 trước khi thua xưng 'tao' gọi 'mày', sau khi thua xưng 'tôi' gọi 'ông' "
    "(danh xưng: Chân tiên sinh)"
)
HONG_WIFE = (
    "洪大嫂 / 洪妻 (Hồng đại tẩu / Vợ Hồng Chấn) [nữ, ~45] — vợ của Hồng Chấn; xưng hô: "
    "với 洪震 tự xưng 'tôi' gọi 'ông'; với 秦可秀 tự xưng 'chị' gọi 'mày'"
)
TU = (
    "秦可秀 / 阿秀 (Tần Khả Tú / A Tú) [nữ, ~23] — vợ của Chân Kiếm Thanh; xưng hô: với "
    "甄劍聲 tự xưng 'em' gọi 'anh' (danh xưng: Kiếm ca); với 朱正軍 trước tự xưng 'tôi' "
    "gọi 'chú', sau xưng 'em' gọi 'anh' (danh xưng: anh Quân); với 洪大嫂 tự xưng 'tôi' "
    "gọi 'chị' (danh xưng: Hồng đại tẩu)"
)
KIEM = (
    "甄劍聲 / 劍哥 (Chân Kiếm Thanh / Kiếm ca) [nam, ~35] — chồng của Tần Khả Tú; xưng "
    "hô: với 秦可秀 tự xưng 'anh' gọi 'em'; với 朱正軍 tự xưng 'tôi' gọi 'cậu' (A Quân)"
)
QUAN = (
    "朱正軍 / 阿軍 (Chu Chính Quân / A Quân) [nam, ~30] — thợ sửa xe cựu binh; xưng hô: "
    "với 甄劍聲 tự xưng 'em' gọi 'anh' (danh xưng: Kiếm ca); với 龍傳人 tự xưng 'tôi' "
    "gọi 'đại đội trưởng'"
)
ROSTER = "\n".join([LONG, HONG, HONG_WIFE, TU, KIEM, QUAN])
OVERVIEW = (
    "Bị vợ bắt gặp đang cười đùa, hai cha con nói dối là vừa xem phim hài, sau đó Hồng "
    "đại tẩu bưng yến sào vào phòng hỏi han an ủi Khả Tú.\n[1:03:06–1:08:50] Tại nhà "
    "Hồng Chấn, Hồng đại tẩu ban đầu giả vờ rơi lệ.\nHồng đại tẩu vừa hút thuốc vừa răn "
    "đe Khả Tú."
)

# (dòng, câu gốc, bản dịch thật, người nói, người nghe, bản đúng)
MM_LINES: list[tuple[int, str, str, str, str, str]] = [
    (330, "給我一點反應好不好，大哥！", "Cho tao chút phản ứng đi chứ, đại ca!",
     "龍傳人", "朱正軍", "Cho tao chút phản ứng đi chứ, đại ca!"),
    (344, "甄大嫂，這兒群山環抱", "Chân đại tẩu, ở đây non nước bao quanh,",
     "龍傳人", "秦可秀", "Cô Chân, ở đây non nước bao quanh,"),
    (351, "甄大嫂麻煩你跟我回公安局", "Chân đại tẩu, phiền cô theo tôi về đồn công an.",
     "龍傳人", "秦可秀", "Cô Chân, phiền cô theo tôi về đồn công an."),
    (360, "劍哥", "Kiếm ca.", "秦可秀", "甄劍聲", "Anh Kiếm."),
    (460, "劍哥你打過越戰嗎", "Kiếm ca, anh từng đánh chiến tranh Việt Nam ư?",
     "朱正軍", "甄劍聲", "Anh Kiếm, anh từng đánh chiến tranh Việt Nam ư?"),
    (621, "我還進去看看劍哥", "Tôi vào xem Kiếm ca thế nào đã.",
     "秦可秀", "朱正軍", "Tôi vào xem anh Kiếm thế nào đã."),
    (699, "我愛劍哥劍哥也很愛我", "Tôi yêu Kiếm ca, Kiếm ca cũng rất yêu tôi.",
     "秦可秀", "朱正軍", "Tôi yêu anh Kiếm, anh Kiếm cũng rất yêu tôi."),
    (837, "洪大嫂我不想吃", "Hồng đại tẩu, tôi không muốn ăn.",
     "秦可秀", "洪大嫂", "Chị Hồng, tôi không muốn ăn."),
    (841, "洪大嫂，你幹甚麼呀", "Hồng đại tẩu, chị làm gì thế?",
     "秦可秀", "洪大嫂", "Chị Hồng, chị làm gì thế?"),
    (1126, "不要，甄先生，不要", "Đừng mà, Chân tiên sinh, đừng mà!",
     "洪震", "甄劍聲", "Đừng mà, ông Chân, đừng mà!"),
]

# Phim "Hệ thống sủng thê" (069d0c24) — rút gọn.
LIN_YI = (
    "林逸 (Lâm Dật) [nam, ~24] — nhân vật chính; xưng hô: với Vương Đại Phúc tự xưng "
    "'tao' gọi 'mày'"
)
HE_QIANG = (
    "贺强 (Hạ Cường / Cường ca) [nam, ~25] — bạn học cũ của Lâm Dật; xưng hô: với Lâm "
    "Dật tự xưng 'tao' gọi 'mày'; với Hà đội trưởng tự xưng 'em' gọi 'anh'"
)
ZHANG_CHAO = (
    "张超 / 张经理 (Trương Siêu / Trương kinh lý) [nam, ~30] — quản lý phòng bán hàng; "
    "xưng hô: với Lâm Dật tự xưng 'tôi' gọi 'Lâm tiên sinh'"
)
HE_CAPTAIN = (
    "何队长 (Hà đội trưởng) [nam, ~35] — anh họ của Hạ Cường; xưng hô: với Hạ Cường tự "
    "xưng 'anh' gọi 'chú em'"
)
ST_ROSTER = "\n".join([LIN_YI, HE_QIANG, ZHANG_CHAO, HE_CAPTAIN])
ST_LINES: list[tuple[int, str, str, str, str, str]] = [
    (653, "林先生", "Lâm tiên sinh,", "张超", "林逸", "Anh Lâm,"),
    (1088, "你赶紧给林先生道歉", "Cậu mau xin lỗi Lâm tiên sinh đi!",
     "张超", "贺强", "Cậu mau xin lỗi anh Lâm đi!"),
    (3903, "你找的那个林先生", "Vị Lâm tiên sinh mà cậu tìm",
     "何队长", "孙经理", "Anh Lâm mà cậu tìm"),
    (3315, "恭喜啊强哥", "Chúc mừng nhé Cường ca,", "张铭", "贺强",
     "Chúc mừng nhé anh Cường,"),
    (3783, "表哥你也看到了", "Biểu ca, anh cũng thấy rồi đấy,", "贺强", "何队长",
     "Anh họ, anh cũng thấy rồi đấy,"),
    (703, "你敢勾引大嫂", "Mày dám dụ dỗ đại tẩu,", "王大福", "林逸",
     "Mày dám dụ dỗ chị dâu,"),
]


def _run(
    rows: list[tuple[int, str, str, str, str, str]], roster: str
) -> dict[int, str]:
    changed, _ = nativize_lines(
        {row[0]: row[2] for row in rows},
        {row[0]: row[1] for row in rows},
        roster,
        speakers={row[0]: row[3] for row in rows},
        listeners={row[0]: row[4] for row in rows},
    )
    return {row[0]: changed.get(row[0], row[2]) for row in rows}


class TestRealLines:
    @pytest.mark.parametrize("row", MM_LINES, ids=[str(row[0]) for row in MM_LINES])
    def test_mm_line(self, row: tuple[int, str, str, str, str, str]) -> None:
        assert _run(MM_LINES, ROSTER)[row[0]] == row[5]

    @pytest.mark.parametrize("row", ST_LINES, ids=[str(row[0]) for row in ST_LINES])
    def test_st_line(self, row: tuple[int, str, str, str, str, str]) -> None:
        assert _run(ST_LINES, ST_ROSTER)[row[0]] == row[5]

    def test_kin_word_follows_speakers_address_rule(self) -> None:
        """Cảnh sát gọi vợ Kiếm Thanh là 'cô' → "cô Chân", không phải "chị Chân"."""
        fixed = _run(MM_LINES, ROSTER)
        assert fixed[344].startswith("Cô Chân")

    def test_conditional_rule_uses_the_call_of_the_title(self) -> None:
        """Quy tắc "trước khi thua 'mày', sau khi thua 'ông' (danh xưng: Chân tiên
        sinh)" → tiếng gọi "Chân tiên sinh" đi với 'ông'."""
        assert _run(MM_LINES, ROSTER)[1126] == "Đừng mà, ông Chân, đừng mà!"

    def test_young_sir_is_anh(self) -> None:
        """"先生" gọi người dưới 40 tuổi là "anh" (Lâm Dật ~24)."""
        assert _run(ST_LINES, ST_ROSTER)[653] == "Anh Lâm,"

    def test_changes_are_described(self) -> None:
        _, changes = nativize_lines(
            {row[0]: row[2] for row in MM_LINES},
            {row[0]: row[1] for row in MM_LINES},
            ROSTER,
            speakers={row[0]: row[3] for row in MM_LINES},
            listeners={row[0]: row[4] for row in MM_LINES},
        )
        assert "Kiếm ca → anh Kiếm" in describe_changes(changes)
        assert {change.index for change in changes} == {
            344, 351, 360, 460, 621, 699, 837, 841, 1126,
        }


class TestSafety:
    def test_source_without_suffix_is_untouched(self) -> None:
        changed, _ = nativize_lines(
            {1: "Khả Tú ca một bài."}, {1: "可秀唱了一首歌"}, ROSTER
        )
        assert changed == {}

    def test_disciple_compound_is_untouched(self) -> None:
        """"Tiểu Tiền đệ tử" — "đệ tử" (đồ đệ) không phải tiếng gọi "đệ"."""
        roster = "宋钱 (Tống Tiền / Tiểu Tiền) [nam, ~23] — nhân vật chính"
        changed, _ = nativize_lines({1: "Tiểu Tiền đệ tử,"}, {1: "小钱弟弟"}, roster)
        assert changed == {}

    def test_non_name_before_suffix_is_untouched(self) -> None:
        changed, _ = nativize_lines(
            {1: "Hai tỷ đồng đấy, chị ạ."}, {1: "二十亿啊姐"}, ROSTER
        )
        assert changed == {}

    def test_title_after_unknown_name_is_untouched(self) -> None:
        """Tên không có trong bảng: không thành "Mạc chị dâu"."""
        changed, _ = nativize_lines(
            {1: "Mạc đại tẩu, chị về đi."}, {1: "莫大嫂你回去吧"}, ROSTER
        )
        assert changed == {}

    def test_nickname_prefix_is_dropped(self) -> None:
        roster = "宋钱 (Tống Tiền / Tiểu Tiền) [nam, ~23] — nhân vật chính"
        changed, _ = nativize_lines({1: "Tiểu Tiền đệ đệ,"}, {1: "小钱弟弟"}, roster)
        assert changed == {1: "Em Tiền,"}

    def test_empty_roster_changes_nothing(self) -> None:
        assert nativize_lines({1: "Kiếm ca."}, {1: "劍哥"}, "") == ({}, [])


class TestContext:
    def _context(self) -> TranslationContext:
        return TranslationContext(
            target_lang="Vietnamese", characters=ROSTER, overview=OVERVIEW,
            glossary="劍哥 = Kiếm ca\n大哥 = đại ca",
        )

    def test_roster_aliases_and_titles(self) -> None:
        fixed, changes = nativize_context(self._context())
        roster = fixed.characters
        assert "甄劍聲 / 劍哥 (Chân Kiếm Thanh / anh Kiếm)" in roster
        assert "洪大嫂 / 洪妻 (chị Hồng / Vợ Hồng Chấn)" in roster
        # Danh xưng lấy đúng cách gọi của mệnh đề chứa nó.
        assert "gọi 'cô' (danh xưng: cô Chân)" in roster
        assert "gọi 'ông' (danh xưng: ông Chân)" in roster
        assert "gọi 'anh' (danh xưng: anh Kiếm)" in roster
        assert "gọi 'chị' (danh xưng: chị Hồng)" in roster
        assert "Kiếm ca" not in roster
        assert "đại tẩu" not in roster
        assert {change.location for change in changes} == {
            LOCATION_ROSTER, LOCATION_OVERVIEW, LOCATION_GLOSSARY,
        }

    def test_overview_capitalizes_sentence_start(self) -> None:
        fixed, _ = nativize_context(self._context())
        assert "sau đó chị Hồng bưng yến sào" in fixed.overview
        assert "Tại nhà Hồng Chấn, chị Hồng ban đầu" in fixed.overview
        assert "\nChị Hồng vừa hút thuốc" in fixed.overview

    def test_glossary(self) -> None:
        fixed, _ = nativize_context(self._context())
        assert fixed.glossary == "劍哥 = anh Kiếm\n大哥 = đại ca"

    def test_address_rules_still_resolve(self) -> None:
        """Đổi tiếng gọi không được làm mất quy tắc xưng hô nào."""
        fixed, _ = nativize_context(self._context())
        assert set(address_rule_map(fixed.characters)) == set(address_rule_map(ROSTER))

    def test_lines_follow_the_same_way_after_context_conversion(self) -> None:
        """Bảng nhân vật đã đổi ("anh Kiếm", "danh xưng: ông Chân") vẫn cho kết quả dòng
        y như bảng gốc — dòng dịch còn sót "Kiếm ca" vẫn được nhận ra."""
        fixed, _ = nativize_context(self._context())
        assert _run(MM_LINES, fixed.characters) == _run(MM_LINES, ROSTER)

    def test_idempotent(self) -> None:
        fixed, _ = nativize_context(self._context())
        again, changes = nativize_context(fixed)
        assert again is fixed
        assert changes == []

    def test_names_survive(self) -> None:
        fixed, _ = nativize_context(self._context())
        assert name_syllables(fixed.characters) >= {"kiếm", "chân", "hồng", "tú"}

    def test_analysis_result_type_is_kept(self) -> None:
        analysis = SubtitleContextAnalysis(
            source_lang="zh", characters=ROSTER, overview="", glossary=""
        )
        fixed, _ = nativize_context(analysis)
        assert isinstance(fixed, SubtitleContextAnalysis)
        assert fixed.source_lang == "zh"

    def test_call_form_calque_uses_age(self) -> None:
        """"gọi 'Lâm tiên sinh'" (Lâm Dật ~24) → "gọi 'anh Lâm'"."""
        context = TranslationContext(target_lang="Vietnamese", characters=ST_ROSTER)
        fixed, _ = nativize_context(context)
        assert "gọi 'anh Lâm'" in fixed.characters
        assert "(Hạ Cường / anh Cường)" in fixed.characters

    def test_place_word_in_glossary_is_untouched(self) -> None:
        """"北妹 = Bắc muội" — "Bắc" không phải tên nhân vật."""
        context = TranslationContext(
            target_lang="Vietnamese", characters=ROSTER, glossary="北妹 = Bắc muội"
        )
        fixed, _ = nativize_context(context)
        assert fixed.glossary == "北妹 = Bắc muội"

    def test_capitalized_suffix_needs_cjk_name(self) -> None:
        """"Lý Thẩm" có thể là họ tên (沈 cũng đọc Thẩm): chỉ đổi khi tên chữ Hán là
        嬸/婶."""
        aunt = "王二婶 (Vương Nhị Thẩm) [nữ, ~55] — hàng xóm"
        name = "李沈 (Lý Thẩm) [nữ, ~30] — đồng nghiệp"
        fixed, _ = nativize_context(
            TranslationContext(target_lang="Vietnamese", characters=f"{aunt}\n{name}")
        )
        assert "(thím Vương Nhị)" in fixed.characters
        assert "(Lý Thẩm)" in fixed.characters
