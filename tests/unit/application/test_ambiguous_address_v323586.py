"""[v3.23.586] Cờ "bảng xưng hô MƠ HỒ" chỉ báo cặp nhân vật thật sự mơ hồ.

Chuẩn đoán phân tích thật v3.23.585 (phim "灭门惨案II", gemini-3.7-flash) báo 6 cặp mơ hồ
kèm lời khuyên phân tích lại ngữ cảnh — thứ tốn hạn mức ngày nhất của ứng dụng. Soát từng
cặp thì chỉ MỘT cặp mơ hồ thật; năm cặp còn lại là người nghe thuộc một LOẠI người (gọi
'cậu/cô', 'anh/chị' theo giới là đúng), nhiều người nghe trong một mệnh đề, hoặc hai cách
viết của cùng một từ ('ba/lão ba').
"""

from __future__ import annotations

from subtitles_extractor.application.services.context_diagnostics import (
    build_context_diagnostics,
)
from subtitles_extractor.application.services.roster_format import (
    ambiguous_address_rules,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
)

# Các dòng CHÉP NGUYÊN từ bảng nhân vật của lượt phân tích thật.
REAL_ROSTER = "\n".join([
    "龍傳人 (Long Truyền Nhân) [nam, ~40] — Sở trưởng kiêm Đại đội trưởng đồn công an "
    "Đông Long; xưng hô: với cấp dưới tự xưng 'tôi' gọi 'cậu/cô', với dân làng tự xưng "
    "'tôi' gọi 'các người/bà con', với Tần Khả Tú tự xưng 'tôi' gọi 'cô' (danh xưng: "
    "Chân đại tẩu)",
    "甄劍聲 / 劍哥 (Chân Kiếm Thanh / Kiếm ca) [nam, ~35] — Chồng của Tần Khả Tú; xưng "
    "hô: với Tần Khả Tú tự xưng 'anh' gọi 'em/A Tú', với Chu Chính Quân tự xưng 'anh' "
    "gọi 'chú/A Quân', với Hồng Chấn xưng 'tôi' gọi 'anh/ông'",
    "秦可秀 / 阿秀 / 甄大嫂 (Tần Khả Tú / A Tú / Chân đại tẩu) [nữ, ~25] — Vợ góa của "
    "Chân Kiếm Thanh; xưng hô: với Chân Kiếm Thanh tự xưng 'em' gọi 'anh'",
    "朱正軍 / 阿軍 (Chu Chính Quân / A Quân) [nam, ~28] — Cựu chiến binh; xưng hô: với "
    "Chân Kiếm Thanh tự xưng 'em' gọi 'anh'",
    "洪震 (Hồng Chấn) [nam, ~50] — Tên ác bá trong làng, cha của Hồng Chí Quyền; xưng "
    "hô: với vợ con tự xưng 'tao' gọi 'mày', với người ngoài tự xưng 'tao' gọi 'mày'",
    "洪志權 / 阿權 (Hồng Chí Quyền / A Quyền) [nam, ~24] — Con trai Hồng Chấn; xưng hô: "
    "với Hồng Chấn tự xưng 'con' gọi 'ba/lão ba', với mẹ tự xưng 'con' gọi 'má', với "
    "Kiếm Thanh/Khả Tú tự xưng 'tao' gọi 'mày/con đàn bà kia'",
    "攤販 / 賣洋娃娃者 (Tiểu thương bán búp bê / Tiểu thương) [nữ, ~50] — Người bán búp "
    "bê tại chợ trấn; xưng hô: với khách hàng tự xưng 'tôi' gọi 'tiểu thư/cô/chị'",
    "旅店職員 (Nhân viên nhà trọ) [nam, ~40] — Người phụ trách phòng tại nhà trọ; xưng "
    "hô: với khách tự xưng 'tôi' gọi 'anh/chị'",
])

TRULY_AMBIGUOUS = "甄劍聲 / 劍哥 → Hồng Chấn: xưng 'tôi' gọi 'anh/ông'"


def test_real_roster_reports_only_the_truly_ambiguous_pair() -> None:
    assert ambiguous_address_rules(REAL_ROSTER) == [TRULY_AMBIGUOUS]


def test_kind_of_people_may_be_addressed_by_gender() -> None:
    roster = (
        "店員 (Nhân viên) [nữ, ~20] — bán hàng; xưng hô: với khách hàng tự xưng 'em' "
        "gọi 'anh/chị'\n"
        "林逸 (Lâm Dật) [nam, ~24] — nam chính"
    )
    assert ambiguous_address_rules(roster) == []


def test_two_spellings_of_one_form_are_not_ambiguous() -> None:
    roster = (
        "吴二狗 (Ngô Nhị Cẩu) [nam, ~30] — côn đồ; xưng hô: với Trương Y Bối tự xưng "
        "'anh' gọi 'em/cô em'\n"
        "张依贝 (Trương Y Bối) [nữ, ~24] — nữ chính"
    )
    assert ambiguous_address_rules(roster) == []


def test_one_listener_with_two_pronouns_is_still_reported() -> None:
    """Ca thật v3.23.550: hai đại từ khác hẳn nhau cho CÙNG một người nghe."""
    roster = (
        "柳慕云 (Liễu Mộ Vân) [nữ, ~26] — nữ chính; xưng hô: với Tống Tiền tự xưng "
        "'tôi/chị' gọi 'cậu/Tống thần y'\n"
        "宋钱 (Tống Tiền) [nam, ~24] — nam chính"
    )
    assert ambiguous_address_rules(roster) == [
        "柳慕云 → Tống Tiền: xưng 'tôi/chị' gọi 'cậu/Tống thần y'"
    ]


def test_diagnostics_warning_counts_one_pair() -> None:
    context = TranslationContext(
        target_lang="Vietnamese", source_lang="zh", characters=REAL_ROSTER,
        overview="tóm tắt",
    )
    report = build_context_diagnostics(context=context)
    flags = {flag["code"]: flag["message"] for flag in report["quality_flags"]}
    assert flags["roster_ambiguous_address"].startswith("1 cặp nhân vật")
    assert report["roster"]["ambiguous_address"] == [TRULY_AMBIGUOUS]
