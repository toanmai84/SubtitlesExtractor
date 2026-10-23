"""[v3.23.585] Giao diện không còn báo "Phân tích hoàn tất" cho lượt phân tích dở.

Nhật ký thật v3.23.584: lần phân tích thứ hai dừng ở 16/17 đoạn, lần đầu ở 1/17 — cả hai
đều kết thúc bằng thông báo thành công; chỉ nhật ký ứng dụng mới ghi là thiếu đoạn.
"""

from __future__ import annotations

from types import SimpleNamespace

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    ANALYSIS_STOP_OVERLOAD,
    SubtitleContextAnalysis,
)
from subtitles_extractor.presentation.view_models.translate_page_view_model import (
    TranslatePageViewModel,
)


class _Signal:
    def __init__(self) -> None:
        self.emitted: list[object] = []

    def emit(self, value: object) -> None:
        self.emitted.append(value)


def _finish(result: SubtitleContextAnalysis) -> SimpleNamespace:
    """Chạy ``_on_analyze_finished`` trên một view model tối giản (không cần Qt)."""
    view_model = SimpleNamespace(
        analyze_context_ready=_Signal(),
        status_message=_Signal(),
        _persist_analysis_result=lambda _result: None,
        _close_token_report=lambda: "",  # [v3.23.610] báo token của lượt phân tích
    )
    TranslatePageViewModel._on_analyze_finished(view_model, result)
    return view_model


INCOMPLETE = SubtitleContextAnalysis(
    characters="甄劍聲 (Chân Kiếm Thanh) — nam chính",
    missing_segments=(14,),
    total_segments=17,
    stop_reason=ANALYSIS_STOP_OVERLOAD,
)


def test_incomplete_analysis_is_announced_as_incomplete() -> None:
    view_model = _finish(INCOMPLETE)
    assert view_model.status_message.emitted == [
        "Phân tích ngữ cảnh CHƯA đủ — còn thiếu 1 đoạn video."
    ]
    assert view_model.analyze_context_ready.emitted == [INCOMPLETE]


def test_complete_analysis_keeps_the_usual_message() -> None:
    view_model = _finish(SubtitleContextAnalysis(characters="甄劍聲 (Chân Kiếm Thanh)"))
    assert view_model.status_message.emitted == ["Phân tích ngữ cảnh hoàn tất."]


def test_message_for_the_warning_toast() -> None:
    message = TranslatePageViewModel.incomplete_analysis_message(INCOMPLETE)
    assert message.startswith("Mới phân tích 16/17 đoạn video — còn thiếu đoạn 14.")
    assert TranslatePageViewModel.incomplete_analysis_message(object()) == ""


def _toast(result: SubtitleContextAnalysis) -> list[tuple[str, str, str]]:
    """Chạy ``_on_analyze_context_ready`` của trang dịch với hàm hiện thông báo giả."""
    from subtitles_extractor.presentation.pages.translate_page import TranslatePage

    shown: list[tuple[str, str, str]] = []
    page = SimpleNamespace(
        _fill_context_fields=lambda _result: None,
        # [v3.23.600] Trang hỏi thêm độ phủ người nói (cần phụ đề nguồn của view model).
        _view_model=SimpleNamespace(
            incomplete_analysis_message=TranslatePageViewModel.incomplete_analysis_message,
            speaker_coverage_message=lambda _visual_cues: "",
            last_token_report="",  # [v3.23.610]
        ),
        _show_warning=lambda title, body: shown.append(("warning", title, body)),
        _show_success=lambda title, body: shown.append(("success", title, body)),
    )
    TranslatePage._on_analyze_context_ready(page, result)
    return shown


def test_page_warns_instead_of_reporting_success() -> None:
    [(kind, title, body)] = _toast(INCOMPLETE)
    assert (kind, title) == ("warning", "Phân tích CHƯA đủ")
    assert "16/17" in body


def test_page_still_reports_success_for_a_complete_analysis() -> None:
    [(kind, title, _body)] = _toast(SubtitleContextAnalysis(source_lang="zh"))
    assert (kind, title) == ("success", "Phân tích hoàn tất")
