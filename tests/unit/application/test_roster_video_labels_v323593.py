# ruff: noqa: RUF001 — dấu câu fullwidth CJK thuộc câu gốc chép từ phim
"""[v3.23.593] Cảnh báo nhầm: nhãn vai trò của nhân vật CHỈ thấy qua hình bị coi là "tên
ghi bằng ngôn ngữ đích".

Chuẩn đoán ngữ cảnh thật v3.23.592 (phim "灭门惨案II", gemini-3.5-flash):

    5 tên trong bảng nhân vật được ghi bằng NGÔN NGỮ ĐÍCH thay vì tên gốc (vd: Cậu bé
    trong thôn, Lão nhân trong thôn, Lão nhân) — bảng quy đổi tên gửi cho khâu dịch sẽ
    không tra được, và nhãn người nói trong Visual Cues cũng không khớp. Nên sửa lại…

Cả năm là nhân vật không có tên trong lời thoại (cùng file đó xếp họ vào
``identified_from_video``): không có tên gốc nào để ghi, và Visual Cues dùng đúng các nhãn
ấy (99,4% dòng có nhãn khớp bảng nhân vật). Trên kho chuẩn đoán, 8/9 tên từng bị cờ này
báo là nhãn như vậy; tên thứ chín là mảnh vỡ "Xưng hô" — không tên thật nào.

Dòng bảng nhân vật, Visual Cues và câu gốc chép nguyên văn từ lượt dịch ``0c0c7c9a``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from subtitles_extractor.application.services.context_diagnostics import (
    build_context_diagnostics,
)

ROSTER = "\n".join([
    "阿積 (A Tích) [nam, ~25] — công an viên đồn Đông Long, tính nịnh bợ. Xuất hiện ngắn "
    "ngủi ở đầu đoạn khi chuẩn bị đi nhậu ăn mừng cùng Long Truyền Nhân. Cách xưng hô: "
    "tự xưng 'tôi' gọi Long Truyền Nhân là 'Trưởng đồn'.",
    "Cậu bé trong thôn [nam, ~10] — cậu bé thông minh trong thôn, tự tin đứng ra giải "
    "thích cặn kẽ tiếng lóng dâm dục cho người dân hiểu.",
    "Lão nhân trong thôn [nam, ~70] — cụ già trong thôn, mang theo chiếc xô to để "
    "nộp tinh dịch và lo lắng không biết có đủ định lượng hay không.",
    "Công an viên 1 [nam, ~25] — công an viên đồn Đông Long, lén giám sát nhà Chu Chính "
    "Quân.",
    "Công an viên 2 [nam, ~25] — công an viên đồn Đông Long, cùng đồng nghiệp giám sát "
    "nhà Chu Chính Quân.",
])
#: (dòng, câu gốc, người nói, người nghe, cue)
LINES = [
    (85, "我知道", "Cậu bé trong thôn", "Người dân trong thôn", ""),
    (100, "隊長，我可以的，要不要裝滿？", "Lão nhân trong thôn", "龍傳人", ""),
    (183, "阿積有結果沒有", "龍傳人", "阿積", ""),
    (284, "所長叫我們來監視而已", "Công an viên 1", "Công an viên 2", ""),
    (
        285, "沒有叫我們行動呀", "Công an viên 2", "Công an viên 1",
        "Công an viên theo dõi thắc mắc",
    ),
]


def _report(characters: str, with_cues: bool = True) -> dict:
    cues = [
        {"id": index, "spk": speaker, "to": listener, "cue": cue}
        for index, _source, speaker, listener, cue in LINES
    ]
    context = SimpleNamespace(
        target_lang="Vietnamese", source_lang="zh", overview="Tóm tắt. " * 400,
        characters=characters, glossary="",
        visual_cues=json.dumps(cues, ensure_ascii=False) if with_cues else "",
    )
    events = [
        SimpleNamespace(index=index, text=source) for index, source, *_rest in LINES
    ]
    return build_context_diagnostics(context=context, source_events=events)


def _codes(report: dict) -> list[str]:
    return [flag["code"] for flag in report["quality_flags"]]


class TestVideoOnlyRoleLabels:
    def test_real_case_role_labels_are_not_flagged(self) -> None:
        report = _report(ROSTER)
        assert report["roster"]["identified_from_video"] == [
            "Cậu bé trong thôn", "Lão nhân trong thôn", "Công an viên 1",
            "Công an viên 2",
        ]
        assert "roster_names_not_in_source_language" not in _codes(report)

    def test_label_seen_nowhere_is_still_flagged(self) -> None:
        """Không có trong phụ đề, không có trong Visual Cues: không có bằng chứng đó là
        nhãn của một nhân vật thật — vẫn báo như trước (ca "Cậu bé lanh lợi",
        v3.23.536)."""
        report = _report(ROSTER, with_cues=False)
        assert "roster_names_not_in_source_language" in _codes(report)

    def test_named_character_written_in_vietnamese_is_still_flagged(self) -> None:
        """Nhân vật CÓ tên trong lời thoại mà bảng ghi tên tiếng Việt: vẫn là lỗi."""
        report = _report(ROSTER.replace("阿積 (A Tích)", "A Tích (A Tích)"))
        flag = next(
            flag for flag in report["quality_flags"]
            if flag["code"] == "roster_names_not_in_source_language"
        )
        assert flag["message"].startswith("1 tên")
        assert "A Tích" in flag["message"]
