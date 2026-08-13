"""Cấu hình toàn ứng dụng — sử dụng :mod:`pydantic_settings`."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from subtitles_extractor.domain.value_objects.device_kind import (
    DeviceKind,
    PrecisionMode,
    SubtitleFormat,
)


class NlpSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SE_NLP_", extra="ignore")

    enable_vector_embeddings: bool = Field(default=False)
    model_name: str = Field(default="BAAI/bge-small-zh-v1.5")
    similarity_mode: str = Field(default="hybrid", pattern="^(hybrid|semantic)$")

class OcrSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SE_OCR_", extra="ignore")

    version: str = Field(default="PP-OCRv6_medium")
    language: str = Field(default="ch")
    detection_model: str = Field(default="PP-OCRv6_medium_det")
    recognition_model: str = Field(default="PP-OCRv6_medium_rec")
    score_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

    limit_side_len: int = Field(default=0, ge=0, le=4096)
    limit_type: str = Field(default="min", pattern="^(max|min)$")
    det_thresh: float = Field(default=0.3, ge=0.0, le=1.0)
    det_box_thresh: float = Field(default=0.6, ge=0.0, le=1.0)
    det_unclip_ratio: float = Field(default=1.5, ge=0.5, le=3.0)

    use_textline_orientation: bool = Field(default=True)
    use_doc_orientation_classify: bool = Field(default=False)
    use_doc_unwarping: bool = Field(default=False)

class HardwareSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SE_HW_", extra="ignore")

    device: DeviceKind = DeviceKind.GPU
    batch_size_ocr: int = Field(default=128, ge=1, le=512)
    batch_size_roi: int = Field(default=32, ge=1, le=128)
    workers: int = Field(default=8, ge=1, le=16)
    enable_mkldnn: bool = False
    auto_tune_batch: bool = False
    use_tensorrt: bool = False
    precision: PrecisionMode = PrecisionMode.FP32

    frame_decoder_backend: str = Field(default="pynvvideocodec")
    metadata_reader_backend: str = Field(default="pyav")

class MpvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SE_MPV_", extra="ignore")

    hwdec_mode: str = Field(default="auto-safe")
    hwdec_codecs: str = Field(default="h264,vc1,hevc,vp8,vp9,av1,mpeg2video,mpeg4")
    video_output: str = Field(default="auto")
    gpu_api: str = Field(default="auto")
    gpu_context: str = Field(default="auto")
    cache: str = Field(default="yes")
    cache_secs: int = Field(default=10, ge=0, le=600)
    demuxer_max_bytes: int = Field(default=150, ge=10, le=2048)
    video_sync_mode: str = Field(default="audio")
    deinterlace: bool = Field(default=False)
    profile: str = Field(default="default")
    log_level: str = Field(default="no")

class RoiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SE_ROI_", extra="ignore")

    default_preset: str = Field(default="auto_subtitle")
    remember_last_roi: bool = Field(default=True)
    auto_detect_on_load: bool = Field(default=False)
    auto_detect_step_ms: int = Field(default=500, ge=100, le=60000)
    # [v3.19] Tham số tinh chỉnh lõi AI dò ROI (BBoxAnalyzer) — expose cho người dùng.
    auto_enable_band_refinement: bool = Field(default=True)
    auto_band_keep_ratio: float = Field(default=0.50, ge=0.05, le=0.95)
    auto_band_extend_ratio: float = Field(default=0.50, ge=0.02, le=0.95)
    auto_bottom_padding_factor: float = Field(default=1.6, ge=1.0, le=3.0)
    auto_sensitivity_multiplier: float = Field(default=1.0, ge=0.3, le=2.0)

class ThresholdSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SE_TH_", extra="ignore")

    ocr_min_confidence: float = Field(default=0.30, ge=0.0, le=1.0)
    text_similarity: float = Field(default=0.75, ge=0.0, le=1.0)
    line_similarity: float = Field(default=0.75, ge=0.0, le=1.0)
    drop_short_text_chars: int = Field(default=0, ge=0, le=20)

class FrameSamplingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SE_FRAME_", extra="ignore")

    sample_step_sec: float = Field(default=0.04, ge=0.01, le=5.0)
    phash_distance: int = Field(default=5, ge=0, le=64)
    pixel_diff_ratio: float = Field(default=0.01, ge=0.0, le=1.0)
    skip_intro_sec: float = Field(default=0.0, ge=0.0, le=600.0)
    skip_outro_sec: float = Field(default=0.0, ge=0.0, le=600.0)

class PostProcessSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SE_POST_", extra="ignore")

    similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    min_duration_sec: float = Field(default=0.15, ge=0.0, le=60.0)
    max_duration_sec: float = Field(default=5.0, ge=0.1, le=600.0)
    merge_gap_sec: float = Field(default=0.20, ge=0.0, le=10.0)
    output_format: SubtitleFormat = SubtitleFormat.SRT
    use_viterbi: bool = Field(default=True)
    viterbi_open_penalty: float = Field(default=0.35, ge=0.0, le=2.0)

    temporal_padding_sec: float = Field(default=0.00, ge=0.0, le=1.0)
    y_clustering_tolerance_ratio: float = Field(default=0.30, ge=0.0, le=1.0)
    y_clustering_tolerance_min_px: float = Field(default=5.0, ge=0.0, le=50.0)
    alignment_center_tolerance_ratio: float = Field(default=0.10, ge=0.0, le=0.5)
    alignment_margin_tolerance_ratio: float = Field(default=0.20, ge=0.0, le=0.5)
    alignment_tolerance_min_px: float = Field(default=60.0, ge=0.0, le=200.0)

class PreprocessSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SE_PRE_", extra="ignore")

    upscale_small_text: bool = Field(default=False)
    upscale_target_height_px: int = Field(default=128, ge=32, le=512)
    add_white_border: bool = Field(default=False)
    border_thickness_px: int = Field(default=8, ge=0, le=64)
    apply_sharpen: bool = Field(default=False)
    apply_contrast_boost: bool = Field(default=False)
    contrast_factor: float = Field(default=1.20, ge=0.1, le=3.0)

    apply_clahe: bool = Field(default=False)
    clahe_clip_limit: float = Field(default=3.0, ge=1.0, le=10.0)
    clahe_tile_size: int = Field(default=8, ge=2, le=32)

    apply_median_blend: bool = Field(default=False)
    median_blend_frames: int = Field(default=3, ge=3, le=7)

class UiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SE_UI_", extra="ignore")

    theme: str = Field(default="auto", pattern="^(auto|light|dark)$")
    locale: str = Field(default="vi", min_length=2, max_length=5)
    font_size: int = Field(default=10, ge=8, le=18)
    show_ocr_overlay: bool = Field(default=True)
    show_waveform: bool = Field(default=True)

    @property
    def safe_font_size(self) -> int:
        return max(8, min(18, self.font_size or 10))

class AdvancedSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SE_ADV_", extra="ignore")

    log_level: str = Field(default="DEBUG", pattern="^(DEBUG|INFO|WARNING|ERROR)$")
    save_debug_frames: bool = False
    debug_frames_dir: str = Field(default="")
    keep_temp_files: bool = False
    disable_paddle_network_check: bool = Field(default=True)

class VideoContextSettings(BaseSettings):
    """Tham số xử lý video ngữ cảnh gửi lên Gemini (nén + cắt đoạn + dịch).

    Các thông số này trước đây hardcode; nay expose để người dùng cân bằng giữa chất
    lượng/độ chính xác và chi phí token/thời gian.
    """

    model_config = SettingsConfigDict(env_prefix="SE_VIDCTX_", extra="ignore")

    # Nén video ngữ cảnh trước khi upload.
    resolution_height: int = Field(default=360, ge=144, le=1080)
    fps: float = Field(default=1.0, ge=0.2, le=5.0)
    nvenc_cq: int = Field(default=32, ge=18, le=45)
    cpu_crf: int = Field(default=30, ge=18, le=45)

    # Cắt đoạn theo token (mỗi đoạn = một request phân tích tuần tự).
    # [v3.23.141] Mặc định 230K (< TPM free-tier 250K, chừa chỗ cho phần text): đảm bảo
    # MỘT đoạn không bao giờ tự vượt TPM kể cả khi phân tích ở medium/high (token/frame
    # cao hơn) -> tránh 429 ngay từ đoạn đầu. Ở low, coverage-split theo max_chunk_minutes
    # vẫn cho đoạn ~106K nên trần này không ảnh hưởng.
    tokens_per_chunk: int = Field(default=230_000, ge=50_000, le=900_000)
    max_chunk_minutes: float = Field(default=18.0, ge=1.0, le=60.0)

    # Token ước tính mỗi giây video ở độ phân giải thấp (Gemini media_resolution=LOW).
    tokens_per_second: int = Field(default=100, ge=50, le=400)

    # [v3.23.166] Ngân sách CACHE FILE ĐOẠN nén cục bộ. Từ v157 file đoạn được GIỮ để
    # tái dùng khi xoay API key / chạy lại cùng phim (tránh nén lại ~5 phút), nên phải
    # dọn tự động theo ngân sách kẻo phình ổ cứng. 0 = không giới hạn theo tiêu chí đó.
    chunk_cache_max_total_mb: int = Field(default=4096, ge=0, le=131_072)
    chunk_cache_max_age_hours: int = Field(default=72, ge=0, le=8_760)


class TranslationSettings(BaseSettings):
    """Tham số điều tiết quá trình dịch nhiều giai đoạn qua Gemini."""

    model_config = SettingsConfigDict(env_prefix="SE_TRANS_", extra="ignore")

    # Số dòng mỗi lô gửi lên model. Lô NHỎ → model giữ đúng số dòng/line_no tốt hơn,
    # ít bỏ sót; lô LỚN → nhanh hơn nhưng dễ lệch (nhất là model nhẹ flash-lite).
    default_batch_size: int = Field(default=50, ge=10, le=200)
    # Số câu ngữ cảnh kèm hai bên mỗi lô (giúp dịch mạch lạc).
    default_context_size: int = Field(default=10, ge=0, le=40)
    # Số lần thử lại khi model trả lỗi tạm thời (429/503/timeout).
    retry_count: int = Field(default=3, ge=1, le=10)
    # Timeout đọc kết quả mỗi request (giây). Tăng cho video ngữ cảnh lớn (đoạn dài).
    request_timeout_sec: int = Field(default=120, ge=30, le=600)
    # [v3.23.129] Độ phân giải VIDEO khi PHÂN TÍCH ngữ cảnh & Visual Cues (Gemini 3.x):
    # 'low' tiết kiệm token, 'medium' cân bằng (mặc định), 'high' nhìn rõ mặt/biểu cảm
    # nhất (cues siêu chính xác) nhưng tốn token & chậm hơn.
    analysis_media_resolution: str = Field(default="low", pattern="^(low|medium|high)$")
    analysis_thinking_level: str = Field(default="medium", pattern="^(low|medium|high)$")
    # [v3.23.149] Số BATCH dịch chạy SONG SONG trong mỗi giai đoạn (1 = tuần tự như cũ).
    # Song song che độ trễ chờ model (10-30s/request) -> nhanh 2-3x; quota manager đặt
    # chỗ trước theo từng request nên tổng RPM/TPM vẫn được điều tiết chính xác. Lịch sử
    # dịch (history) neo theo ĐỢT đã chốt để giữ mạch xưng hô/giọng điệu.
    translation_parallel_batches: int = Field(default=1, ge=1, le=4)

class ApplicationSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    nlp: NlpSettings = Field(default_factory=NlpSettings)
    ocr: OcrSettings = Field(default_factory=OcrSettings)
    hardware: HardwareSettings = Field(default_factory=HardwareSettings)
    mpv: MpvSettings = Field(default_factory=MpvSettings)
    roi: RoiSettings = Field(default_factory=RoiSettings)
    threshold: ThresholdSettings = Field(default_factory=ThresholdSettings)
    frame: FrameSamplingSettings = Field(default_factory=FrameSamplingSettings)
    post_process: PostProcessSettings = Field(default_factory=PostProcessSettings)
    preprocess: PreprocessSettings = Field(default_factory=PreprocessSettings)
    ui: UiSettings = Field(default_factory=UiSettings)
    advanced: AdvancedSettings = Field(default_factory=AdvancedSettings)
    video_context: VideoContextSettings = Field(default_factory=VideoContextSettings)
    translation: TranslationSettings = Field(default_factory=TranslationSettings)

__all__ = [
    "AdvancedSettings",
    "ApplicationSettings",
    "FrameSamplingSettings",
    "HardwareSettings",
    "MpvSettings",
    "NlpSettings",
    "OcrSettings",
    "PostProcessSettings",
    "PreprocessSettings",
    "RoiSettings",
    "ThresholdSettings",
    "TranslationSettings",
    "UiSettings",
    "VideoContextSettings",
]
