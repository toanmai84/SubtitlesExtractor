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
    max_duration_sec: float = 30.0
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

    temporal_padding_sec: float = 0.05
    y_clustering_tolerance_ratio: float = 0.30
    y_clustering_tolerance_min_px: float = 5.0
    alignment_center_tolerance_ratio: float = 0.10
    alignment_margin_tolerance_ratio: float = 0.20
    alignment_tolerance_min_px: float = 60.0

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
    raw_ocr_output_path: Path | None = None
    export_raw_images: bool = False
    """Nếu True và raw_ocr_output_path được cấu hình, hệ thống sẽ xuất thêm
    ảnh Input và Output (vẽ sẵn Box + Text) để dễ dàng Debug."""

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
