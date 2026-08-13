"""Thực thể :class:`ProjectRecord` — một dự án Auto-Dubbing theo hash video.

Mỗi video (định danh bằng hash nội dung, độc lập tên/thư mục) tương ứng MỘT
bản ghi dự án, tích luỹ kết quả và cài đặt của từng khâu:

    - OCR:        dữ liệu thô + cài đặt OCR.
    - Phụ đề gốc: nội dung phụ đề đã trích xuất (định dạng SRT/ASS).
    - Bản dịch:   nội dung phụ đề đã dịch + cài đặt/phân tích liên quan.
    - TTS:        đường dẫn kết quả + cài đặt liên quan.

Nhờ vậy người dùng có thể mở lại, tiếp tục công việc dở dang ở bất kỳ khâu
nào, hoặc xoá; và cùng một video sẽ không bị xử lý lại từ đầu chỉ vì đổi tên
hay di chuyển thư mục.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class WorkflowStage(IntEnum):
    """Khâu đã hoàn thành xa nhất của một dự án (để hiển thị tiến độ).

    [v3.23.316] Thêm :attr:`PUBLISHED` — trước đây quy trình KHÔNG có mốc kết thúc, nên
    nhìn vào Thư viện không biết phim nào đã xuất bản xong.
    """

    NEW = 0
    EXTRACTED = 1
    EDITED = 2
    TRANSLATED = 3
    TTS_DONE = 4
    PUBLISHED = 5

    @property
    def label_vi(self) -> str:
        """Nhãn tiếng Việt hiển thị trong Thư viện."""
        return {
            WorkflowStage.NEW: "Mới tạo",
            WorkflowStage.EXTRACTED: "Đã trích xuất",
            WorkflowStage.EDITED: "Đã chỉnh sửa",
            WorkflowStage.TRANSLATED: "Đã biên dịch",
            WorkflowStage.TTS_DONE: "Đã tạo TTS",
            WorkflowStage.PUBLISHED: "Đã xuất bản",
        }[self]

    @property
    def next_action_vi(self) -> str:
        """Việc NÊN LÀM TIẾP để hoàn thành phim — dùng cho gợi ý trên giao diện.

        Trước v3.23.316, ứng dụng không hề gợi ý bước tiếp theo ở bất kỳ đâu; người dùng
        phải tự biết thứ tự Trích xuất → Biên tập → Dịch → TTS → Xuất bản.
        """
        return {
            WorkflowStage.NEW: "Trích xuất phụ đề từ video",
            WorkflowStage.EXTRACTED: "Kiểm tra và chỉnh sửa phụ đề",
            WorkflowStage.EDITED: "Dịch phụ đề sang tiếng Việt",
            WorkflowStage.TRANSLATED: "Tạo giọng đọc (TTS)",
            WorkflowStage.TTS_DONE: "Xuất bản phim hoàn chỉnh",
            WorkflowStage.PUBLISHED: "Đã hoàn thành — không còn bước nào",
        }[self]

    @property
    def next_page_key(self) -> str | None:
        """Khoá trang cần chuyển tới cho bước tiếp theo; ``None`` nếu đã xong.

        Khoá khớp ``objectName`` của trang để tầng giao diện tự điều hướng được.
        """
        return {
            WorkflowStage.NEW: "extractPage",
            WorkflowStage.EXTRACTED: "editorPage",
            WorkflowStage.EDITED: "translatePage",
            WorkflowStage.TRANSLATED: "ttsPage",
            WorkflowStage.TTS_DONE: "publishPage",
            WorkflowStage.PUBLISHED: None,
        }[self]

    @property
    def progress_ratio(self) -> float:
        """Tỉ lệ hoàn thành 0.0–1.0 để vẽ thanh tiến độ quy trình."""
        return self.value / WorkflowStage.PUBLISHED.value

    @property
    def is_complete(self) -> bool:
        """``True`` khi dự án đã đi hết quy trình."""
        return self is WorkflowStage.PUBLISHED


@dataclass
class ProjectRecord:
    """Bản ghi dự án Auto-Dubbing gắn với một video (khoá chính = video_hash).

    Attributes:
        video_hash: Hash nội dung video (định danh duy nhất, từ video_hasher).
        video_path: Đường dẫn video lần cuối được biết (có thể thay đổi).
        video_name: Tên hiển thị (thường là tên file video).
        stage: Khâu hoàn thành xa nhất.
        published_video_path: Đường dẫn video hoàn chỉnh đã xuất bản (nếu có).
        ocr_settings_json: Cài đặt OCR (JSON).
        ocr_raw_json: Dữ liệu OCR thô (JSON; có thể lớn).
        original_subtitle: Nội dung phụ đề gốc đã trích xuất (text SRT/ASS).
        subtitle_format: Định dạng phụ đề ("srt"/"ass").
        translated_subtitle: Nội dung phụ đề đã dịch (text SRT/ASS).
        target_lang: Ngôn ngữ đích bản dịch (vd "vi").
        translation_settings_json: Cài đặt/phân tích bản dịch (JSON).
        tts_audio_path: Đường dẫn file WAV kết quả TTS.
        tts_settings_json: Cài đặt TTS (JSON).
        created_at: Mốc tạo (ISO 8601).
        updated_at: Mốc cập nhật gần nhất (ISO 8601).
    """

    video_hash: str
    video_path: str = ""
    video_name: str = ""
    stage: WorkflowStage = WorkflowStage.NEW

    ocr_settings_json: str = ""
    ocr_raw_json: str = ""

    original_subtitle: str = ""
    subtitle_format: str = "srt"

    translated_subtitle: str = ""
    target_lang: str = ""
    translation_settings_json: str = ""

    tts_audio_path: str = ""
    tts_settings_json: str = ""

    # [v3.23.316] Tệp video hoàn chỉnh đã xuất (để mở lại từ Thư viện).
    published_video_path: str = ""

    created_at: str = ""
    updated_at: str = ""

    extra: dict = field(default_factory=dict)


__all__ = ["ProjectRecord", "WorkflowStage"]
