"""[v3.23.679] Kiểm tra phát âm (AI nghe lại từng câu) + TTS lại câu đọc sai (tối đa 2 lượt).

Bộ chấm cũ chỉ có trong công cụ đo: VieNeu bản 3.23.663 còn 20/296 câu đọc sai (CER > 0,15) — người dùng phải nghe dò.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from subtitles_extractor.application.services import pronunciation_check as pc


# ── Hàm thuần ───────────────────────────────────────────────────────────────────────────────


def test_chuan_hoa_va_ti_le_sai_chu() -> None:
    assert pc.normalize_spoken("[Lâm:] Sư thúc, đi thôi!") == "sưthúcđithôi"
    assert pc.char_error_rate("Sư thúc, đi thôi!", "sư thúc đi thôi") == 0.0
    # "ta" → "bà": 2 ký tự khác (t→b, a→à) trên 9 ký tự sau chuẩn hoá — đúng lỗi thật của Gemini 2.5 (3.23.662).
    assert pc.char_error_rate("không có ta", "không có bà") == pytest.approx(2 / 9)
    assert pc.char_error_rate("", "") == 0.0 and pc.char_error_rate("", "a") == 1.0
    assert pc.char_error_rate("Đi thôi", "") == 1.0                       # không nghe được gì = sai hẳn


def test_chon_cau_can_kiem(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    results = [
        SimpleNamespace(event_index=0, text="Con đường tu luyện", segment_audio_path=str(wav), was_skipped=False),
        SimpleNamespace(event_index=1, text="Hừ!", segment_audio_path=str(wav), was_skipped=False),          # ngắn
        SimpleNamespace(event_index=2, text="Câu bị bỏ qua", segment_audio_path=str(wav), was_skipped=True),
        SimpleNamespace(event_index=3, text="Mất tệp âm thanh", segment_audio_path=str(tmp_path / "x.wav"),
                        was_skipped=False),
        SimpleNamespace(event_index=4, text="Không có audio riêng", segment_audio_path="", was_skipped=False),
        SimpleNamespace(event_index=5, text="Trải qua Trúc Cơ", segment_audio_path=str(wav), was_skipped=False),
    ]
    assert [x.event_index for x in pc.lines_to_check(results)] == [0, 5]
    assert [x.event_index for x in pc.lines_to_check(results, only={5})] == [5]


def test_doc_json_tra_ve_ben_vung() -> None:
    payload = {"items": [{"i": 0, "heard": "a"}, {"i": "1", "heard": None}, {"i": 9, "heard": "x"}, "rác", {"i": "?"}]}
    assert pc.parse_heard(payload, 2) == {0: "a", 1: ""}
    assert pc.parse_heard([{"i": 0, "heard": "b"}], 1) == {0: "b"}
    assert pc.parse_heard("hỏng", 3) == {}


# ── Bộ chấm Gemini (giả lập client) ─────────────────────────────────────────────────────────


class _Client:
    def __init__(self, replies: list) -> None:
        self.models = self
        self._replies = replies

    def generate_content(self, **_kw):
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(text=reply)


def _lines(tmp_path: Path, texts: list[str]) -> list[pc.LineToCheck]:
    out = []
    for i, text in enumerate(texts):
        wav = tmp_path / f"{i}.wav"
        wav.write_bytes(b"RIFF")
        out.append(pc.LineToCheck(i, text, str(wav)))
    return out


def test_cham_theo_lo_xoay_key_khi_loi(tmp_path: Path) -> None:
    pytest.importorskip("google.genai")
    from subtitles_extractor.infrastructure.tts.pronunciation_judge import GeminiPronunciationJudge

    replies = {
        "k1": [RuntimeError("503 UNAVAILABLE")],
        "k2": ['{"items":[{"i":0,"heard":"con đường tu luyện"},{"i":1,"heard":"trải qua trúc kê"}]}'],  # 2/13 chữ sai
    }
    made: list[str] = []

    def factory(key: str):
        made.append(key)
        return _Client(replies[key])

    judge = GeminiPronunciationJudge(["k1", "k2"], client_factory=factory, sleep=lambda _s: None)
    verdicts = judge.check(_lines(tmp_path, ["Con đường tu luyện", "Trải qua Trúc Cơ"]))
    assert [v.wrong for v in verdicts] == [False, True]           # "kê"/"cơ": 2/13 chữ = 0,154 > 0,15
    assert verdicts[1].cer == pytest.approx(round(2 / 13, 3))
    assert made == ["k1", "k2"]                                   # mỗi key một client, dùng lại


def test_het_han_muc_bao_ro(tmp_path: Path) -> None:
    pytest.importorskip("google.genai")
    from subtitles_extractor.infrastructure.tts.pronunciation_judge import (
        GeminiPronunciationJudge,
        PronunciationJudgeError,
    )

    client = _Client([RuntimeError("429 RESOURCE_EXHAUSTED")] * 50)     # mọi model, mọi key đều hết hạn mức
    judge = GeminiPronunciationJudge(["k"], client_factory=lambda _k: client, sleep=lambda _s: None)
    with pytest.raises(PronunciationJudgeError, match="hạn mức"):
        judge.check(_lines(tmp_path, ["Con đường tu luyện"]))
    with pytest.raises(PronunciationJudgeError, match="key Gemini"):
        GeminiPronunciationJudge([])


def test_het_han_muc_giua_chung_giu_phan_da_cham(tmp_path: Path) -> None:
    """Bản đầu: hết hạn mức ở lô sau → ném lỗi, mất cả phần đã chấm (đo thật: 200+ câu)."""
    pytest.importorskip("google.genai")
    from subtitles_extractor.infrastructure.tts.pronunciation_judge import GeminiPronunciationJudge

    first = '{"items":[' + ",".join(f'{{"i":{i},"heard":"con đường tu luyện"}}' for i in range(10)) + "]}"
    client = _Client([first] + [RuntimeError("429 RESOURCE_EXHAUSTED")] * 50)
    judge = GeminiPronunciationJudge(["k"], client_factory=lambda _k: client, sleep=lambda _s: None)
    verdicts = judge.check(_lines(tmp_path, ["Con đường tu luyện"] * 15))
    assert len(verdicts) == 10 and "hạn mức" in judge.stop_reason


# ── Vòng "chấm → TTS lại → chấm lại" trên trang ─────────────────────────────────────────────


def test_tts_lai_cau_sai_toi_da_hai_luot(tmp_path: Path, qtbot) -> None:
    from PySide6.QtCore import QObject, Qt, Signal
    from PySide6.QtWidgets import QLabel, QTableWidget, QTableWidgetItem

    from subtitles_extractor.presentation.pages.tts_line_edits import COLUMN_STATUS
    from subtitles_extractor.presentation.pages.tts_pronunciation import PronunciationController

    texts = ["Con đường tu luyện", "Trải qua Trúc Cơ", "Bắt đầu từ Luyện Khí"]
    lines = _lines(tmp_path, texts)
    results = [SimpleNamespace(event_index=x.event_index, text=x.text, segment_audio_path=x.audio_path,
                               was_skipped=False) for x in lines]
    table = QTableWidget(len(texts), 7)
    qtbot.addWidget(table)
    for row in range(len(texts)):
        head = QTableWidgetItem(str(row + 1))
        head.setData(Qt.ItemDataRole.UserRole, row)
        table.setItem(row, 0, head)
        table.setItem(row, COLUMN_STATUS, QTableWidgetItem("OK"))

    class _VM(QObject):
        result_ready = Signal(object)

        def __init__(self) -> None:
            super().__init__()
            self.is_busy = False
            self.can_resynthesize = True
            self.resynth: list[set[int]] = []

        def resynthesize_lines(self, positions, line_voices=None):
            self.resynth.append(set(positions))
            return True

    vm = _VM()
    label = QLabel()
    page = QObject()
    page._view_model, page._last_results, page._table, page._stage_lbl = vm, results, table, label
    page._ai_store = SimpleNamespace(gemini_keys=lambda: ["k"])
    page._resynthesis_line_voices = lambda: None
    # Lượt 1: câu 1 và 2 sai; lượt 2: câu 2 vẫn sai; lượt 3: câu 2 vẫn sai (chữ khó) → dừng sau 2 lượt TTS lại.
    heard_rounds = [
        {0: "con đường tu luyện", 1: "trải qua cái gì đó", 2: "bắt đầu thôi"},
        {1: "trải qua cái gì đó", 2: "bắt đầu từ luyện khí"},
        {1: "trải qua cái gì đó"},
    ]

    class _Worker(QObject):
        progress = Signal(int, int)
        finished_ok = Signal(object)
        failed = Signal(str)

        def __init__(self, todo) -> None:
            super().__init__()
            self._todo = todo

        def start(self) -> None:
            heard = heard_rounds.pop(0)
            self.finished_ok.emit([pc.PronunciationVerdict(x.event_index, x.text, heard[x.event_index],
                                                           pc.char_error_rate(x.text, heard[x.event_index]))
                                   for x in self._todo])

        def request_cancel(self) -> None:
            pass

    tr = lambda key, **kw: key + (" " + repr(sorted(kw.items())) if kw else "")  # noqa: E731
    controller = PronunciationController(page, tr, judge_factory=lambda keys: object(),
                                         worker_factory=lambda judge, todo: _Worker(todo))
    def repopulate(_results) -> None:     # trang thật dựng lại bảng sau mỗi lượt TTS (xoá dấu cũ)
        for row in range(table.rowCount()):
            table.item(row, COLUMN_STATUS).setText("OK")

    vm.result_ready.connect(repopulate)
    vm.result_ready.connect(controller.on_result_ready)
    assert controller.start()
    assert vm.resynth == [{1, 2}]
    vm.result_ready.emit(results)
    qtbot.waitUntil(lambda: len(vm.resynth) == 2)
    assert vm.resynth[1] == {1}
    vm.result_ready.emit(results)
    qtbot.waitUntil(lambda: not controller.is_running)
    assert len(vm.resynth) == 2                                    # không quá 2 lượt
    assert controller.wrong_indices() == {1}
    assert "('left', 1)" in label.text() and "('wrong', 2)" in label.text()
    assert table.item(1, COLUMN_STATUS).text().startswith("tts.pron_mark")
    assert table.item(2, COLUMN_STATUS).text() == "OK"
    assert [i.row() for i in table.selectionModel().selectedRows()] == [1]


def test_khong_co_key_bao_ro(qtbot) -> None:
    from PySide6.QtCore import QObject
    from PySide6.QtWidgets import QLabel

    from subtitles_extractor.presentation.pages.tts_pronunciation import PronunciationController

    label = QLabel()
    page = QObject()
    page._view_model = SimpleNamespace(is_busy=False)
    page._last_results = [SimpleNamespace(event_index=0, text="a b c", segment_audio_path=__file__, was_skipped=False)]
    page._stage_lbl = label
    page._ai_store = SimpleNamespace(gemini_keys=lambda: [])
    controller = PronunciationController(page, lambda key, **_kw: key)
    assert not controller.start() and label.text() == "tts.pron_no_key"


def test_doc_so_thanh_chu_khi_so_loi() -> None:
    # Đo thật: AI chép "chưa được 2 năm" cho câu "Chưa được hai năm" — bản đầu báo nhầm là đọc sai (CER 0,21).
    assert pc.char_error_rate("Chưa được hai năm", "chưa được 2 năm,") == 0.0
    assert [pc.read_number_vi(n) for n in (15, 21, 105, 2024)] == [
        "mười lăm", "hai mươi mốt", "một trăm lẻ năm", "hai nghìn không trăm hai mươi bốn"]


def test_cau_sai_duoc_cham_lai_rieng(tmp_path: Path) -> None:
    """Đo thật: trong lô, lời câu 109 bị gán sang câu 108 → câu đúng bị báo sai. Chấm lại riêng thì hết."""
    pytest.importorskip("google.genai")
    from subtitles_extractor.infrastructure.tts.pronunciation_judge import GeminiPronunciationJudge

    replies = [
        '{"items":[{"i":0,"heard":"muội nhất định phải hạnh phúc đấy"},{"i":1,"heard":"biết không"}]}',
        '{"items":[{"i":0,"heard":"muội nhất định phải hạnh phúc đấy, biết không?"}]}',   # chấm riêng câu 0
        '{"items":[{"i":0,"heard":"muội nhất định phải hạnh phúc đấy."}]}',               # chấm riêng câu 1
    ]
    client = _Client(replies)
    judge = GeminiPronunciationJudge(["k"], client_factory=lambda _k: client, sleep=lambda _s: None)
    verdicts = judge.check(_lines(tmp_path, [
        "Muội nhất định phải hạnh phúc đấy, biết không?", "Muội nhất định phải hạnh phúc đấy."]))
    assert [v.wrong for v in verdicts] == [False, False] and not replies


def test_phu_am_dau_dong_am_khong_bi_coi_la_doc_sai() -> None:
    # Đo thật: "Dao Dao" giọng Nam bị chép thành "giao giao" — 4/12 câu báo sai là vì vậy.
    assert pc.char_error_rate("đối với Dao Dao,", "Đối với giao dao") == 0.0
    assert pc.char_error_rate("Tên Luyện Khí nhỏ bé", "Tên lửa khí nhỏ bé") > pc.WRONG_CER   # lỗi đọc thật vẫn bắt
