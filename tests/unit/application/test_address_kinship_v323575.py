"""[v3.23.575] Hồi quy: quy tắc xưng hô ghi người nghe bằng QUAN HỆ/TÊN GỌI → bị bỏ.

Chuẩn đoán thật phim "灭门惨案II借种" (phân tích lại bằng gemini-3.8-flash, v3.23.574):
bảng nhân vật ghi "với con trai tự xưng 'ba' gọi 'con'; với vợ tự xưng 'tôi' gọi 'bà
nó'" (洪震), "với cha …; với mẹ …" (洪志權), "với Khả Tú khi đe dọa …" (洪大嫂). Bộ tra
chỉ biết tên đầy đủ nên 7/49 mệnh đề không gắn được vào cặp nhân vật nào: 洪震 → 洪志權
(32 dòng), 洪大嫂 → 秦可秀 (32), 洪震 → 洪大嫂 (31), 洪志權 → 洪震 (19) đều dịch KHÔNG
có quy tắc; bản dịch trộn "Tao bảo mày…" với "Ba không được thì con…" trong một cảnh.

Cùng lỗi ở phim khác: "với mẹ ruột (Bà Chu)", "với em trai (Tiểu Phi)" (tên trong
ngoặc), "gọi Đào Hoa là 'mày'" (tên gọi của Hứa Đào Hoa).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from subtitles_extractor.application.services.address_rule_guard import (
    find_address_rule_violations,
)
from subtitles_extractor.application.services.context_diagnostics import (
    build_context_diagnostics,
)
from subtitles_extractor.application.services.roster_format import (
    address_clauses,
    address_rule_map,
    lookup_address_rule,
)
from subtitles_extractor.application.services.roster_kinship import (
    KinEdge,
    KinPerson,
    KinshipGraph,
    kin_edges,
    split_relation,
)

# Nguyên văn 13/24 dòng của bảng nhân vật thật (đủ mọi nhân vật mà các dòng gọi tới).
FILM_ROSTER = "\n".join([
    "龍傳人 (Long Truyền Nhân) [nam, ~38] — Sở trưởng đồn công an xã Đông Long, "
    "tính hám danh và cơ hội nhưng sâu thẳm giàu lòng trắc ẩn; cảm động trước"
    " bi kịch của Khả Tú nên quyết định bao che cho cô làm lại cuộc đời; xưng"
    " hô: với cấp dưới tự xưng 'tôi' gọi 'cậu/cô'; với Dư Tiểu Liên khi tán "
    "tỉnh tự xưng 'anh' gọi 'em'; với Bác sĩ phòng khám tự xưng 'tôi' gọi "
    "'ông'; với Bác sĩ bệnh viện tự xưng 'tôi' gọi 'bác sĩ'; với Chu Chính "
    "Quân tự xưng 'tao' gọi 'mày'; với Tần Khả Tú tự xưng 'tôi' gọi 'chị "
    "Chân'",
    "余小蓮 / 當當 (Dư Tiểu Liên / Đương Đương) [nữ, ~22] — Nữ công an trẻ tại đồn"
    " Đông Long, tính tình ngây thơ, nhút nhát, bị cấp trên ép cởi cúc áo "
    "khêu gợi để dân làng nộp mẫu tinh dịch; xưng hô: với Long Truyền Nhân tự"
    " xưng 'tôi' gọi 'Sở trưởng'; với dân làng tự xưng 'tôi' gọi 'bà con'",
    "老莫 (Lão Mạc) [nam, ~48] — Dân làng Đông Long, người phát hiện hiện "
    "trường án mạng nhà Hồng Chấn và cõng vợ ngất xỉu tới đồn; xưng hô: với "
    "Long Truyền Nhân tự xưng 'tôi' gọi 'ông'; với vợ tự xưng 'tôi' gọi 'bà "
    "nó'",
    "莫大嫂 (Mạc đại tẩu) [nữ, ~45] — Vợ của Lão Mạc, ngất xỉu vì quá kinh hoàng"
    " khi tận mắt thấy cảnh tượng đẫm máu tại nhà Hồng Chấn; xưng hô: với Lão"
    " Mạc tự xưng 'tôi' gọi 'ông nó'",
    "洪震 (Hồng Chấn) [nam, ~50] — Kẻ lưu manh tống tiền ác độc ở địa phương, "
    "cùng con trai cưỡng hiếp tập thể Tần Khả Tú và mưu sát Chân Kiếm Thanh; "
    "bị Khả Tú vung liềm chém chết trong thảm án báo thù; xưng hô: với con "
    "trai tự xưng 'ba' gọi 'con'; với vợ tự xưng 'tôi' gọi 'bà nó'; với Chân "
    "Kiếm Thanh bình thường xưng 'tao' gọi 'mày', lúc van xin xưng 'tôi' gọi "
    "'ông Chân'; với Chu Chính Quân tự xưng 'tao' gọi 'mày'; với Tần Khả Tú "
    "tự xưng 'tao' gọi 'mày'",
    "洪志權 / 阿權 (Hồng Chí Quyền / A Quyền) [nam, ~23] — Con trai ngỗ ngược của "
    "Hồng Chấn, đồng phạm cưỡng bức Khả Tú và sát hại Kiếm Thanh; bị Khả Tú "
    "chém đứt tay rồi treo cổ trước cửa nhà; xưng hô: với cha tự xưng 'con' "
    "gọi 'ba'; với mẹ tự xưng 'con' gọi 'mẹ'; với Tần Khả Tú tự xưng 'tao' "
    "gọi 'mày'",
    "洪大嫂 (Hồng đại tẩu) [nữ, ~48] — Vợ của Hồng Chấn, đồng lõa xảo quyệt dung"
    " túng chồng con phạm tội ác; bị Khả Tú đè bẫy chông kẹp nát đầu trong "
    "gian bếp; xưng hô: với Hồng Chấn tự xưng 'tôi' gọi 'ông nó'; với con "
    "trai tự xưng 'mẹ' gọi 'con'; với Khả Tú khi đe dọa tự xưng 'tao' gọi "
    "'mày', khi đóng kịch tự xưng 'chị' gọi 'A Tú'; với Chân Kiếm Thanh tự "
    "xưng 'tôi' gọi 'anh bạn'",
    "甄劍聲 / 劍哥 (Chân Kiếm Thanh / anh Kiếm) [nam, ~32] — Người chồng quá cố "
    "của Tần Khả Tú, cựu chiến binh bị liệt dương sau chiến tranh biên giới; "
    "vì bảo vệ sản nghiệp dòng họ nên thuê Chu Chính Quân mượn giống; khi "
    "biết vợ bị làm nhục và con chết đã tìm nhà Hồng Chấn báo thù nhưng bị "
    "sát hại dã man; xưng hô: với Tần Khả Tú tự xưng 'anh' gọi 'em'; với Chu "
    "Chính Quân tự xưng 'anh' gọi 'chú Quân'; với Hồng Chấn tự xưng 'tao' gọi"
    " 'mày'; với Bác sĩ phòng khám tự xưng 'tôi' gọi 'bác sĩ'",
    "秦可秀 / 阿秀 (Tần Khả Tú / A Tú) [nữ, ~24] — Vợ Chân Kiếm Thanh, người phụ "
    "nữ chịu bi kịch tột cùng; sau khi biết thai nhi bị nhiễm giang mai từ "
    "cha con Hồng Chấn, cô đã tự nung móc sắt tự phá thai trong đau đớn tột "
    "cùng; dứt khoát cự tuyệt tình cảm của Chu Chính Quân; xưng hô: với Chân "
    "Kiếm Thanh tự xưng 'em' gọi 'anh Kiếm'; với Chu Chính Quân tự xưng 'tôi'"
    " gọi 'chú Quân'; với Hồng Chấn và Chí Quyền tự xưng 'tao' gọi 'chúng "
    "mày'; với Hồng đại tẩu tự xưng 'tôi' gọi 'Hồng đại tẩu'; với Long Truyền"
    " Nhân tự xưng 'tôi' gọi 'đồng chí công an'",
    "朱正軍 / 阿軍 (Chu Chính Quân / anh Quân) [nam, ~28] — Cựu chiến binh từng "
    "tham gia chiến tranh biên giới, sống đơn độc một mình, nhiệt tình cưu "
    "mang vợ chồng Kiếm Thanh; về sau dính líu đến thảm án và bị liệt toàn "
    "thân khi trốn chạy công an; xưng hô: với Long Truyền Nhân tự xưng 'tôi' "
    "gọi 'Đội trưởng'; với Tần Khả Tú tự xưng 'tôi' gọi 'chị dâu'; với Chân "
    "Kiếm Thanh tự xưng 'em' gọi 'anh Kiếm'",
    "醫生 / 診所醫生 (Bác sĩ phòng khám) [nam, ~52] — Thầy thuốc khám chữa bệnh "
    "kiêm trị bệnh hoa liễu tại thị trấn; chẩn đoán Khả Tú có thai nhưng mắc "
    "bệnh giang mai giai đoạn đầu và khuyên cô nên phá thai; xưng hô: với "
    "Chân Kiếm Thanh tự xưng 'tôi' gọi 'anh'; với Tần Khả Tú tự xưng 'tôi' "
    "gọi 'cô'",
    "醫院醫生 (Bác sĩ bệnh viện) [nam, ~35] — Bác sĩ điều trị cho Chu Chính Quân "
    "sau tai nạn ngã lầu, xác nhận nghi phạm bị liệt hoàn toàn; xưng hô: với "
    "Long Truyền Nhân tự xưng 'tôi' gọi 'đồng chí'",
    "小販 (Tiểu thương chợ quê) [nữ, ~45] — Tiểu thương bán hàng rong, búp bê "
    "và trang sức tại chợ thị trấn; xưng hô: với khách mua tự xưng 'tôi' gọi "
    "'cô/anh'",
])


def _rules() -> dict[tuple[str, str], str]:
    return address_rule_map(FILM_ROSTER)


class TestKinshipListeners:
    def test_father_to_son(self) -> None:
        assert _rules()[("洪震", "洪志權")] == "tự xưng 'ba' gọi 'con'"

    def test_husband_to_wife(self) -> None:
        rules = _rules()
        assert rules[("洪震", "洪大嫂")] == "tự xưng 'tôi' gọi 'bà nó'"
        assert rules[("老莫", "莫大嫂")] == "tự xưng 'tôi' gọi 'bà nó'"

    def test_son_to_father_read_from_the_sons_own_description(self) -> None:
        """"Con trai ngỗ ngược của Hồng Chấn" nằm trong mô tả của chính người nói."""
        assert _rules()[("洪志權", "洪震")] == "tự xưng 'con' gọi 'ba'"

    def test_mother_is_the_fathers_wife(self) -> None:
        """Bảng không ghi 洪大嫂 là mẹ 洪志權 — chỉ ghi "Vợ của Hồng Chấn"."""
        rules = _rules()
        assert rules[("洪志權", "洪大嫂")] == "tự xưng 'con' gọi 'mẹ'"
        assert rules[("洪大嫂", "洪志權")] == "tự xưng 'mẹ' gọi 'con'"

    def test_cue_labels_with_aliases_find_the_rule(self) -> None:
        assert lookup_address_rule(_rules(), "洪志權 / 阿權", "洪震") == (
            "tự xưng 'con' gọi 'ba'"
        )
        assert lookup_address_rule(_rules(), "洪震", "阿權") == "tự xưng 'ba' gọi 'con'"

    def test_only_group_listeners_stay_unresolved(self) -> None:
        clauses = address_clauses(FILM_ROSTER)
        unresolved = [c.raw_listener for c in clauses if not c.listeners]
        assert unresolved == ["cấp dưới", "dân làng", "khách mua"]
        assert len(clauses) == 39


class TestGivenName:
    def test_given_name_with_phase_condition(self) -> None:
        assert _rules()[("洪大嫂", "秦可秀")] == (
            "khi đe dọa tự xưng 'tao' gọi 'mày', khi đóng kịch tự xưng 'chị' gọi 'A Tú'"
        )

    def test_given_name_in_call_list_format(self) -> None:
        """Bảng thật phim khác: "tự xưng 'mẹ' gọi Đào Hoa là 'mày/đồ không có não'"."""
        roster = "\n".join([
            "许桃花 (Hứa Đào Hoa) [nữ, ~25] — góa phụ xinh đẹp trong làng",
            "周姨 (Bà Chu) [nữ, ~50] — mẹ của Hứa Đào Hoa: trọng nam khinh nữ; xưng hô: "
            "tự xưng 'mẹ' gọi Đào Hoa là 'mày/đồ không có não', gọi Tống Tiền là "
            "'thằng ngốc'",
            "小飞 (Tiểu Phi) [nam, ~20] — em trai của Hứa Đào Hoa; xưng hô: tự xưng "
            "'em' gọi Đào Hoa là 'chị', gọi Vương Ma Tử là 'Vương ca'",
        ])
        rules = address_rule_map(roster)
        assert rules[("周姨", "许桃花")] == "tự xưng 'mẹ' gọi 'mày/đồ không có não'"
        assert rules[("小飞", "许桃花")] == "tự xưng 'em' gọi 'chị'"

    def test_shared_given_name_is_not_guessed(self) -> None:
        roster = "\n".join([
            "甲 (Lý Minh Châu) [nữ] — chị",
            "乙 (Trần Minh Châu) [nữ] — em",
            "丙 (Vương Đại) [nam] — xưng hô: với Minh Châu tự xưng 'tôi' gọi 'cô'",
        ])
        assert address_rule_map(roster) == {}

    def test_title_is_not_a_given_name(self) -> None:
        """"Mạc đại tẩu" là danh xưng: "đại tẩu" không phải tên gọi của riêng ai."""
        roster = "\n".join([
            "莫大嫂 (Mạc đại tẩu) [nữ] — dân làng",
            "丙 (Vương Đại) [nam] — xưng hô: với đại tẩu tự xưng 'tôi' gọi 'chị'",
        ])
        assert address_rule_map(roster) == {}

    def test_full_alias_beats_given_name(self) -> None:
        roster = "\n".join([
            "秦可秀 (Tần Khả Tú) [nữ] — vợ",
            "可秀 (Khả Tú) [nữ] — người trùng tên",
            "丙 (Vương Đại) [nam] — xưng hô: với Khả Tú tự xưng 'tôi' gọi 'cô'",
        ])
        assert set(address_rule_map(roster)) == {("丙", "可秀")}


class TestParenthesisedName:
    ROSTER = "\n".join([
        "许桃花 (Hứa Đào Hoa) [nữ, ~25] — góa phụ xinh đẹp trong làng; xưng hô: với "
        "mẹ ruột (Bà Chu) tự xưng 'con' gọi 'mẹ'; với em trai (Tiểu Phi) tự xưng 'chị' "
        "gọi 'em'",
        "周姨 (Bà Chu) [nữ, ~50] — hám tiền sính lễ",
        "小飞 (Tiểu Phi) [nam, ~20] — hùa theo mẹ ép chị gái; xưng hô: với mẹ (Bà Chu) "
        "tự xưng 'con' gọi 'mẹ'",
    ])

    def test_name_inside_parentheses_is_the_listener(self) -> None:
        rules = address_rule_map(self.ROSTER)
        assert rules[("许桃花", "周姨")] == "tự xưng 'con' gọi 'mẹ'"
        assert rules[("许桃花", "小飞")] == "tự xưng 'chị' gọi 'em'"
        assert rules[("小飞", "周姨")] == "tự xưng 'con' gọi 'mẹ'"


def _person(key: str, gender: str, role: str) -> KinPerson:
    return KinPerson(key=key, gender=gender, role=role)


NAMES = {
    "hồng chấn": "洪震",
    "tần khả tú": "秦可秀",
    "chân kiếm thanh": "甄劍聲",
    "tống tiền": "宋钱",
    "tống dao": "宋瑶",
    "liễu mộ vân": "柳慕云",
    "hứa đào hoa": "许桃花",
}


def _find(text: str) -> str | None:
    return NAMES.get(text.strip().lower())


class TestKinEdges:
    @pytest.mark.parametrize(
        ("role", "expected"),
        [
            ("Con trai ngỗ ngược của Hồng Chấn, đồng phạm", ("child", "洪震")),
            ("Vợ Chân Kiếm Thanh, người phụ nữ chịu bi kịch", ("spouse", "甄劍聲")),
            ("Người chồng quá cố của Tần Khả Tú, cựu chiến binh", ("spouse", "秦可秀")),
            ("người dân trong làng: chồng của Tần Khả Tú", ("spouse", "秦可秀")),
            ("em gái ngoan ngoãn hiếu thảo của Tống Tiền", ("younger", "宋钱")),
        ],
    )
    def test_relation_at_clause_start(self, role: str, expected: tuple[str, str]) -> None:
        edges = kin_edges([_person("X", "", role)], _find)
        assert [(edge.relation, edge.of) for edge in edges] == [expected]

    def test_several_people_after_cua(self) -> None:
        edges = kin_edges(
            [_person("宋母", "nữ", "mẹ của Tống Tiền và Tống Dao: bị liệt chân")], _find
        )
        assert edges == [
            KinEdge("宋母", "parent", "宋钱"),
            KinEdge("宋母", "parent", "宋瑶"),
        ]

    @pytest.mark.parametrize(
        "role",
        [
            # Giữa câu, nói về người KHÁC (bảng thật: 王麻子, 洪震).
            "kẻ lưu manh, dựa vào 18 vạn tệ sính lễ đưa cho mẹ Hứa Đào Hoa",
            "Kẻ lưu manh, cùng con trai cưỡng hiếp tập thể Tần Khả Tú",
            # Quan hệ khác hẳn.
            "mẹ kế của Liễu Mộ Vân: cay nghiệt",
            "mẹ vợ của Tống Tiền",
            "chị dâu cùng làng: vợ hàng xóm",
            # Chỉ có trong phần xưng hô, không phải mô tả.
            "dân làng; xưng hô: với vợ Hồng Chấn tự xưng 'tôi' gọi 'chị'",
            # Có từ bổ nghĩa mà không có "của" — không đoán.
            "con trai trưởng Tống Tiền",
        ],
    )
    def test_no_edge(self, role: str) -> None:
        assert kin_edges([_person("X", "", role)], _find) == []

    def test_split_relation(self) -> None:
        assert split_relation("con trai khi say") == ("con trai", "khi say")
        assert split_relation("Con gái") == ("con gái", "")
        assert split_relation("mẹ vợ") is None
        assert split_relation("Hồng Chấn") is None


class TestResolveSafety:
    def _graph(self, people: list[KinPerson]) -> KinshipGraph:
        names = {"hồng chấn": "洪震", "tống tiền": "宋钱"}
        return KinshipGraph.build(people, lambda text: names.get(text.strip().lower()))

    def test_two_sons_are_ambiguous(self) -> None:
        graph = self._graph([
            _person("洪震", "nam", "lưu manh"),
            _person("A", "nam", "con trai của Hồng Chấn"),
            _person("B", "nam", "con trai của Hồng Chấn"),
        ])
        assert graph.resolve("con trai", "洪震") is None

    def test_gender_filters_parents(self) -> None:
        graph = self._graph([
            _person("洪震", "nam", "lưu manh"),
            _person("洪大嫂", "nữ", "Vợ của Hồng Chấn"),
            _person("A", "nam", "con trai của Hồng Chấn"),
        ])
        assert graph.resolve("cha", "A") == ("洪震", "")
        assert graph.resolve("mẹ ruột", "A") == ("洪大嫂", "")
        assert graph.resolve("con gái", "洪震") is None

    def test_remainder_is_returned_for_the_caller(self) -> None:
        graph = self._graph([
            _person("洪震", "nam", "lưu manh"),
            _person("洪大嫂", "nữ", "Vợ của Hồng Chấn"),
        ])
        assert graph.resolve("vợ khi say", "洪震") == ("洪大嫂", "khi say")

    def test_non_phase_remainder_is_not_a_rule(self) -> None:
        roster = "\n".join([
            "洪震 (Hồng Chấn) [nam] — lưu manh; xưng hô: với vợ hai tự xưng 'anh' gọi "
            "'em'; với vợ khi say tự xưng 'tao' gọi 'mụ'",
            "洪大嫂 (Hồng đại tẩu) [nữ] — Vợ của Hồng Chấn",
        ])
        assert address_rule_map(roster) == {
            ("洪震", "洪大嫂"): "khi say tự xưng 'tao' gọi 'mụ'"
        }


# Nguyên văn phim này, dòng 886-911: (dòng, người nói, người nghe, câu gốc, bản dịch).
SCENE_ROWS = [
    (886, "洪志權", "洪震", "幹甚麼", "Làm gì thế?"),
    (887, "洪震", "洪志權", "你媽說的\uff0c我作主呀", "Mẹ mày đã nói rồi, tao làm chủ."),
    (888, "洪震", "洪志權", "這次我先呀", "Lần này tao lên trước nhé."),
    (889, "洪震", "洪志權", "你要不要零用錢", "Mày có muốn tiền tiêu vặt không?"),
    (890, "洪志權", "洪震", "你每次都是這樣子的", "Lần nào ba cũng thế cả."),
    (895, "洪震", "洪志權", "按住她", "Đè nó xuống."),
    (902, "洪震", "洪志權", "十七\uff0c十八\uff0c十九....",
     "Mười bảy, mười tám, mười chín..."),
    (903, "洪震", "洪志權", "沒有了\uff01", "Hết rồi!"),
    (904, "洪志權", "洪震", "都說你十九的了\uff01", "Đã bảo là mười chín mà!"),
    (905, "洪志權", "洪震", "看的吧\uff0c到我了", "Thấy chưa, đến lượt con."),
    (906, "洪志權", "洪震", "到我了", "Đến lượt con."),
    (907, "洪震", "洪志權", "别攪我\uff0c我休息一下\uff0c休息一下",
     "Đừng quấy tao, để tao nghỉ một lát, nghỉ một lát."),
    (908, "洪志權", "洪震", "休息甚麼\uff1f下來呀", "Nghỉ gì chứ? Xuống đi chứ."),
    (909, "洪志權", "洪震", "到我了", "Đến lượt con."),
    (910, "洪震", "洪志權", "我有九十次\uff0c都算不錯了吧\uff1f",
     "Tao được chín mươi lần, thế là không tồi chứ?"),
    (911, "洪志權", "洪震", "真是沒用的老傢伙\uff0c看我吧",
     "Đúng là lão già vô tích sự, để con xem nào."),
]


class TestDownstream:
    def test_translation_batch_carries_the_family_rule(self) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (  # noqa: E501
            GeminiSubtitleTranslator,
        )

        translator = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
        translator._active_characters = FILM_ROSTER
        batch = [
            SimpleNamespace(index=826, speaker="洪震", addressee="洪志權"),
            SimpleNamespace(index=823, speaker="洪志權 / 阿權", addressee="洪震"),
        ]
        payload = [{"line_no": 826}, {"line_no": 823}]
        translator._attach_address_rules_once(payload, batch)
        assert payload[0]["address"] == "tự xưng 'ba' gọi 'con'"
        assert payload[1]["address"] == "tự xưng 'con' gọi 'ba'"

    def test_guard_now_sees_the_father_son_lines(self) -> None:
        cues = json.dumps(
            [{"id": i, "spk": s, "to": t, "cue": ""} for i, s, t, _, _ in SCENE_ROWS],
            ensure_ascii=False,
        )
        sources = {row[0]: row[3] for row in SCENE_ROWS}
        translations = {row[0]: row[4] for row in SCENE_ROWS}
        violations = find_address_rule_violations(
            sources, translations, cues, FILM_ROSTER
        )
        assert [(v.index, v.pronoun_found, v.pronoun_expected) for v in violations] == [
            (888, "tao", "ba"),
            (907, "tao", "ba"),
            (910, "tao", "ba"),
        ]

    def test_diagnostics_count_the_resolved_rules(self) -> None:
        context = SimpleNamespace(
            target_lang="Vietnamese", source_lang="zh", overview="Tóm tắt.",
            characters=FILM_ROSTER, glossary="", visual_cues="",
            enable_tags=False, include_desc=True,
        )
        report = build_context_diagnostics(context=context, source_events=[])
        stats = report["roster"]["address_rules"]
        assert (stats["clauses"], stats["resolved"]) == (39, 36)
        assert stats["unresolved_listeners"] == ["cấp dưới", "dân làng", "khách mua"]
