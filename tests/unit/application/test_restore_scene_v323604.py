"""[v3.23.604] Lưới khôi phục xưng hô không tạo một dòng lệch giữa cảnh đã đổi xưng hô.

Chuẩn đoán thật 11/10 16:01 (灭门惨案II): giai đoạn bản địa hoá chạy gemini-3.8-flash viết
lại cảnh 694-703 (Chu Chính Quân tặng vòng cho Tần Khả Tú khi chồng cô còn sống) thành
"tôi … cô" — sát bảng nhân vật ('lúc đầu tự xưng tôi', 'trước khi chồng chết tự xưng tôi')
hơn bản tinh chỉnh "em … anh". Lưới chỉ khôi phục được dòng 699: "rất hợp với cô mà" →
"rất hợp với **em** mà" — một dòng "em" giữa các dòng "tôi tặng cô".

Trên 153 dòng khôi phục của mọi file chuẩn đoán có nhật ký diễn biến: 4-5 dòng thuộc loại
này (699; 450 "chú" giữa bốn dòng "cậu"; 786 "tôi" giữa các dòng "tao"; 1143 "tôi" trong
cảnh "anh/em"); các dòng còn lại vẫn được khôi phục như cũ.
"""

from __future__ import annotations

import json

from subtitles_extractor.application.services.address_regression_guard import (
    find_address_regressions,
)

ROSTER = "\n".join([
    "秦可秀 (Tần Khả Tú) [nữ, ~25] — vợ của Chân Kiếm Thanh; xưng hô: với Chân Kiếm "
    "Thanh tự xưng 'em' gọi 'anh' (anh Kiếm), với Chu Chính Quân trước khi chồng chết "
    "tự xưng 'tôi' gọi 'chú Quân', sau khi chồng chết tự xưng 'em' gọi 'anh Quân'",
    "甄劍聲 (Chân Kiếm Thanh) [nam, ~45-50] — chồng của Tần Khả Tú; xưng hô: với Tần Khả "
    "Tú tự xưng 'anh' gọi 'em', với Chu Chính Quân tự xưng 'anh' gọi 'chú' (A Quân)",
    "朱正軍 (Chu Chính Quân) [nam, ~30-35] — thợ sửa xe; xưng hô: với Chân Kiếm Thanh "
    "tự xưng 'em' gọi 'anh Kiếm', với Tần Khả Tú lúc đầu tự xưng 'tôi' gọi 'chị dâu', "
    "về sau tự xưng 'anh' gọi 'em'",
])

#: Dòng 694-703 thật: (người nói, người nghe, câu gốc, bản tinh chỉnh, bản địa hoá).
SCENE_699 = [
    (694, "朱正軍", "秦可秀", "嫂子我來幫你吧", "Chị dâu, để em giúp chị nhé.",
     "Chị dâu, để tôi giúp chị."),
    (695, "秦可秀", "朱正軍", "不用了我已經洗完了", "Không cần đâu, em rửa xong rồi.",
     "Không cần đâu, tôi rửa xong rồi."),
    (696, "朱正軍", "秦可秀", "秀，這個玉鐲子，送給你",  # noqa: RUF001 — chữ thật
     "A Tú, chiếc vòng ngọc này tặng cho em.",
     "A Tú, chiếc vòng ngọc này, tôi tặng cô."),
    (697, "秦可秀", "朱正軍", "我沒有理由要你的東西的",
     "Tôi không có lý do gì để nhận đồ của anh cả.",
     "Tôi không có lý do gì để nhận đồ của chú cả."),
    (698, "朱正軍", "秦可秀", "不是呀這鐲子好漂亮", "Không đâu, chiếc vòng này rất đẹp,",
     "Đâu có, chiếc vòng này đẹp lắm,"),
    (699, "朱正軍", "秦可秀", "蠻配你呀", "rất hợp với em.", "rất hợp với cô mà."),
    (700, "秦可秀", "朱正軍", "這玉鐲子這麼漂亮", "Vòng ngọc này đẹp như thế,",
     "Vòng ngọc này đẹp như thế,"),
    (701, "秦可秀", "朱正軍", "不如留下送給你的愛人吧",
     "chi bằng giữ lại để tặng người yêu tương lai của anh đi.",
     "chi bằng giữ lại tặng vợ tương lai của chú đi."),
    (702, "朱正軍", "秦可秀", "就是這個原因，我才送給你",  # noqa: RUF001 — chữ thật
     "Chính vì lý do này nên anh mới tặng em.",
     "Chính vì lý do đó tôi mới tặng cô."),
    (703, "秦可秀", "朱正軍", "我愛劍哥劍哥也很愛我",
     "Em yêu anh Kiếm, anh Kiếm cũng rất yêu em.",
     "Tôi yêu anh Kiếm, anh Kiếm cũng rất yêu tôi."),
]

#: Dòng 583-599 thật (cảnh uống rượu): bản dịch thô và bản tinh chỉnh TRƯỚC khi khôi phục
#: (nhật ký thật: "587 (tôi→anh), 592 (tôi→anh), 594 (tôi→anh), 598 (tôi→anh)").
SCENE_587 = [
    (583, "甄劍聲", "朱正軍", "喝歸喝先說好哦", "Uống thì uống nhưng phải nói trước.",
     "Uống thì uống nhưng phải nói trước."),
    (585, "甄劍聲", "朱正軍", "就不許進房間睡覺", "thì không được vào phòng đi ngủ.",
     "thì không được vào phòng đi ngủ."),
    (587, "甄劍聲", "朱正軍", "酒喝光了我進去拿", "Hết rượu thì anh vào lấy.",
     "Hết rượu thì tôi vào lấy."),
    (592, "甄劍聲", "朱正軍", "我明白了", "Anh hiểu rồi.", "Tôi hiểu rồi."),
    (593, "甄劍聲", "朱正軍", "你自己不行了", "Cậu không uống nổi nữa",
     "Cậu uống không nổi nữa"),
    (594, "甄劍聲", "朱正軍", "就叫我去休息對不對", "nên mới bảo anh vào nghỉ chứ gì.",
     "nên mới bảo tôi vào nghỉ chứ gì."),
    (598, "甄劍聲", "朱正軍", "我就隨便進去休息一下", "Anh cứ vào nghỉ ngơi một lát.",
     "Tôi cứ vào nghỉ ngơi một lát."),
    (599, "甄劍聲", "朱正軍", "一會兒出來跟你再喝", "Lát nữa ra uống tiếp với cậu.",
     "Lát nữa ra uống tiếp với cậu."),
]


def _inputs(scene: list[tuple[int, str, str, str, str, str]]):
    cues = json.dumps(
        [{"id": line, "spk": speaker, "to": listener}
         for line, speaker, listener, *_rest in scene],
        ensure_ascii=False,
    )
    sources = {line: source for line, _s, _l, source, _p, _c in scene}
    previous = {line: before for line, _s, _l, _src, before, _c in scene}
    current = {line: after for line, _s, _l, _src, _p, after in scene}
    return sources, previous, current, cues


def test_dong_699_khong_bi_keo_ve_em_giua_canh_toi_co() -> None:
    sources, previous, current, cues = _inputs(SCENE_699)
    assert find_address_regressions(sources, previous, current, cues, ROSTER) == []


def test_canh_chi_hong_vai_dong_van_duoc_khoi_phuc() -> None:
    sources, previous, current, cues = _inputs(SCENE_587)
    found = find_address_regressions(sources, previous, current, cues, ROSTER)
    assert [(item.index, item.text_after) for item in found] == [
        (587, "Hết rượu thì anh vào lấy."),
        (592, "Anh hiểu rồi."),
        (594, "nên mới bảo anh vào nghỉ chứ gì."),
        (598, "Anh cứ vào nghỉ ngơi một lát."),
    ]
