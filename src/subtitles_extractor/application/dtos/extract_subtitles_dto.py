"""DTO cho use case "Trích xuất phụ đề"."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.frame_sampler_port import FrameSamplingConfig
from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
from subtitles_extractor.domain.value_objects.device_kind import SubtitleFormat
from subtitles_extractor.domain.value_objects.roi import Roi


@dataclass(frozen=True, slots=True)
class SubtitleBuilderConfig:
    similarity_threshold: float = 0.75
    min_duration_sec: float = 0.15
    max_duration_sec: float = 7.0
    merge_gap_sec: float = 0.60
    min_confidence: float = 0.30
    use_viterbi: bool = False
    """[v3.0] Đổi default từ True → False. Profiling thực nghiệm trên 8 file
    test cho thấy greedy đạt F1 tương đương Viterbi (0.989) nhưng nhanh hơn
    5-10× (Viterbi chiếm 82% thời gian build). Người dùng có thể bật lại
    Viterbi qua cài đặt nếu cần độ chính xác cao hơn trên video OCR nhiễu mạnh.
    """
    viterbi_open_penalty: float = 0.35
    min_text_chars: int = 2
    line_similarity_threshold: float = 0.75
    sample_step_sec: float = 0.05

    temporal_padding_sec: float = 0.0   # [v3.23.720] đệm đuôi câu: 0 như cài đặt ứng dụng (chủ dự án chốt 13/11/2026)
    y_clustering_tolerance_ratio: float = 0.30
    y_clustering_tolerance_min_px: float = 5.0
    alignment_center_tolerance_ratio: float = 0.10
    alignment_margin_tolerance_ratio: float = 0.20
    alignment_tolerance_min_px: float = 60.0
    merge_progressive: bool = False
    """[v3.23.654] Gộp phụ đề "hiện dần" (``我`` → ``我只是``) thành một câu (xem ``merge_progressive_events``)."""

@dataclass(frozen=True, slots=True)
class ExtractSubtitlesRequest:
    video_path: Path
    output_path: Path
    output_format: SubtitleFormat = SubtitleFormat.SRT
    roi: Roi | None = None
    sampling: FrameSamplingConfig = field(default_factory=FrameSamplingConfig)
    ocr: OcrEngineConfig = field(default_factory=OcrEngineConfig)
    builder: SubtitleBuilderConfig = field(default_factory=SubtitleBuilderConfig)
    auto_tune_batch: bool = False
    save_debug_frames: bool = False
    debug_frames_dir: str = ""
    keep_temp_files: bool = False
    skip_export: bool = False
    use_frame_cache: bool = True
    """[v3.23.645] Lượt chạy kiểu Re-OCR (không xuất tệp) giữ khung đã giải mã trong bộ nhớ đệm để lần sau khỏi giải mã lại.
    Phép đo tốc độ OCR đặt False: nén JPEG khung vào bộ đệm sẽ tính vào giờ đo và xoá bộ đệm của người dùng."""
    raw_ocr_output_path: Path | None = None
    verify_suspect_lines: bool = True
    """[v3.23.708] Cuối lượt trích xuất, tự OCR lại (ảnh gốc, không tiền xử lý) các câu mà hình học hộp chữ cho thấy
    đọc thiếu / thừa một chữ; chỉ thay chữ khi bản đọc lại khớp đúng dấu hiệu đó."""
    verify_with_vl: bool = False
    """[v3.23.733] Lượt soát câu nghi ngờ đọc lại bằng PaddleOCR-VL-1.6 (đúng 146/148 dòng phụ đề thật so với 143/148 của
    PP-OCRv6 small) thay cho cùng mô hình trên ảnh gốc; thêm cả câu chữ Hán tin cậy < 0,90."""
    export_roi_clip: bool = False
    """[v3.23.728] Cùng ``raw_ocr_output_path``: xuất thêm VIDEO cắt đúng vùng ROI (``<tên>.roi.mp4`` + ``.roi.json``)
    để gỡ lỗi — thay cho ảnh JPG từng khung trước đây (hàng trăm nghìn ảnh, nhiều GB). Trang Gỡ lỗi OCR đọc khung từ
    video này và tự vẽ hộp chữ."""
    export_qa_data: bool = False
    """[v3.23.737] Cùng ``export_roi_clip``: xuất thêm GÓI KIỂM ĐỊNH CHẤT LƯỢNG — mọi câu đọc lại bằng PaddleOCR-VL trên
    video ROI (``<tên>.qa.json.gz``) + tiếng phim (``<tên>.audio.opus``) — để đối chiếu phụ đề với hardsub mà không
    phải dò mắt từng dòng."""
    verify_max_lines: int = 400
    """[v3.23.738] Số câu tối đa được soát lại (Cài đặt); vượt thì soát câu đáng ngờ nhất trước."""
    vl_low_confidence: float = 0.90
    """[v3.23.738] Câu chữ Hán tin cậy dưới ngưỡng này được PaddleOCR-VL soát lại (Cài đặt)."""
    roi_clip_quality: int = 24
    """[v3.23.738] Chất lượng video vùng ROI (thang CRF x264; Cài đặt)."""
    qa_reader: str = "paddle_vl"
    """[v3.23.745] Bộ đọc gói kiểm định: ``"paddle_vl"`` hoặc ``"glm_ocr"`` (tải khi dùng; lỗi → lùi về PaddleOCR-VL)."""

@dataclass(frozen=True, slots=True)
class ExtractSubtitlesResponse:
    events: list[SubtitleEvent]
    output_path: Path
    elapsed_seconds: float
    frames_processed: int
    raw_ocr_path: Path | None = None
    """Đường dẫn file dữ liệu thô đã lưu (None nếu không yêu cầu lưu)."""

__all__ = [
    "ExtractSubtitlesRequest",
    "ExtractSubtitlesResponse",
    "SubtitleBuilderConfig",
]
