"""Dựng lệnh ffmpeg để xuất video hoàn chỉnh (thuần, không chạy tiến trình).

VÌ SAO tồn tại
==============
Quy trình của ứng dụng là: trích phụ đề cháy → dịch sang tiếng Việt → tổng hợp giọng
đọc. Nhưng **bước cuối vẫn thiếu**: tạo ra tệp video hoàn chỉnh để giao. Module này
dựng lệnh ffmpeg cho ba kiểu xuất:

* **Ghép mềm** (``SOFT_SUB``) — nhúng phụ đề dạng track, KHÔNG mã hoá lại. Rất nhanh
  (chỉ sao chép luồng), người xem bật/tắt được phụ đề.
* **Cháy vào hình** (``HARD_SUB``) — vẽ phụ đề trực tiếp lên khung hình bằng libass,
  phải mã hoá lại. Xem được ở mọi nơi kể cả thiết bị không hỗ trợ phụ đề.
* **Thay tiếng** (``DUB_AUDIO``) — thay/thêm track tiếng Việt do TTS tạo.

Tách phần DỰNG LỆNH khỏi phần CHẠY để kiểm thử được đầy đủ — đặc biệt phần thoát ký
tự đường dẫn Windows cho bộ lọc ``subtitles=``, vốn là nguồn lỗi kinh điển.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

# Định dạng chứa được phụ đề dạng văn bản khi ghép mềm.
_MATROSKA_SUFFIXES: Final[frozenset[str]] = frozenset({".mkv", ".mka"})
_MP4_SUFFIXES: Final[frozenset[str]] = frozenset({".mp4", ".m4v", ".mov"})

# Chất lượng mặc định khi phải mã hoá lại (CRF thấp hơn = nét hơn, tệp lớn hơn).
_DEFAULT_CRF: Final[int] = 20
_DEFAULT_PRESET: Final[str] = "medium"
# NVENC dùng thang -cq thay cho -crf.
_NVENC_PRESET: Final[str] = "p5"

# ── [v3.23.315] Tham số trộn thuyết minh (ducking) ──────────────────────────
# Các NGƯỠNG dưới đây được DO THỰC NGHIỆM bằng ffmpeg (nền 200Hz + thuyết minh 1kHz,
# tách bằng FFT), với ratio=12:
#     ngưỡng 0.020 -> hạ  -7.2 dB
#     ngưỡng 0.010 -> hạ -12.7 dB   (đúng dải chuẩn của thuyết minh phát thanh)
#     ngưỡng 0.005 -> hạ -18.2 dB
_DUCK_RATIO: Final[int] = 12
_DUCK_THRESHOLDS: Final[dict[str, float]] = {
    "gentle": 0.020,   # ~ -7 dB  — nền còn khá rõ, phù hợp phim nhiều nhạc
    "medium": 0.010,   # ~ -13 dB — mặc định, cân bằng nghe rõ thuyết minh + còn nền
    "strong": 0.005,   # ~ -18 dB — ưu tiên nghe rõ thuyết minh tối đa
}
# Bám nhanh (ms) để không bị "trôi" mất chữ đầu câu.
_DUCK_ATTACK_MS: Final[int] = 20
# Nhả chậm (ms) để nền dâng lại êm, không bị "phập phù" giữa các từ.
_DUCK_RELEASE_MS: Final[int] = 350
# Âm lượng nền khi KHÔNG có thuyết minh (1.0 = giữ nguyên).
_DEFAULT_BACKGROUND_GAIN: Final[float] = 1.0
# Âm lượng giọng thuyết minh (1.0 = giữ nguyên mức TTS đã chuẩn hoá).
_DEFAULT_NARRATION_GAIN: Final[float] = 1.0


class DuckLevel(StrEnum):
    """Mức hạ tiếng gốc khi thuyết minh nói (kèm mức đo thực nghiệm)."""

    GENTLE = "gentle"
    """Nhẹ — hạ khoảng 7 dB. Giữ nhạc/tiếng động khá rõ."""

    MEDIUM = "medium"
    """Vừa — hạ khoảng 13 dB. Mặc định, đúng dải chuẩn thuyết minh."""

    STRONG = "strong"
    """Mạnh — hạ khoảng 18 dB. Ưu tiên nghe rõ thuyết minh."""

    @property
    def threshold(self) -> float:
        """Ngưỡng sidechain tương ứng (đã đo thực nghiệm)."""
        return _DUCK_THRESHOLDS[self.value]

    @property
    def approx_reduction_db(self) -> float:
        """Mức hạ xấp xỉ (dB) — dùng để hiển thị cho người dùng."""
        return {"gentle": -7.2, "medium": -12.7, "strong": -18.2}[self.value]


class SubtitleMode(StrEnum):
    """[v3.23.326] Cách xử lý PHỤ ĐỀ — độc lập với cách xử lý âm thanh."""

    NONE = "none"
    """Không kèm phụ đề."""

    SOFT = "soft"
    """Track phụ đề bật/tắt được — không mã hoá lại, rất nhanh."""

    BURNED = "burned"
    """Vẽ thẳng vào khung hình (libass) — phải mã hoá lại video."""


class AudioMode(StrEnum):
    """[v3.23.326] Cách xử lý ÂM THANH — độc lập với cách xử lý phụ đề."""

    ORIGINAL = "original"
    """Giữ nguyên tiếng gốc."""

    VOICE_OVER = "voice_over"
    """Trộn thuyết minh lên tiếng gốc, tự hạ tiếng gốc khi thuyết minh nói."""

    REPLACE_TRACK = "replace_track"
    """Thêm tiếng Việt thành track riêng (đặt mặc định), giữ tiếng gốc làm track 2."""


class RenderMode(StrEnum):
    """Kiểu xuất video."""

    SOFT_SUB = "soft_sub"
    """Ghép phụ đề dạng track — không mã hoá lại, rất nhanh."""

    HARD_SUB = "hard_sub"
    """Vẽ phụ đề vào khung hình (libass) — phải mã hoá lại."""

    DUB_AUDIO = "dub_audio"
    """Thay/thêm track tiếng Việt — không mã hoá lại phần hình."""

    VOICE_OVER = "voice_over"
    """[v3.23.315] TRỘN thuyết minh tiếng Việt LÊN TRÊN tiếng gốc, có tự động hạ tiếng
    gốc (ducking) mỗi khi giọng thuyết minh nói — kiểu phim tài liệu / thuyết minh.

    Khác :attr:`DUB_AUDIO` (chỉ THÊM track riêng, người xem chọn 1 trong 2), chế độ này
    tạo ra MỘT track đã trộn: nghe thuyết minh rõ mà vẫn còn nhạc/tiếng động nền.
    """


class VideoRenderError(Exception):
    """Tham số xuất video không hợp lệ."""


@dataclass(frozen=True, slots=True)
class RenderRequest:
    """Yêu cầu xuất một tệp video.

    Attributes:
        video_path: Video nguồn.
        output_path: Tệp đích.
        mode: Kiểu xuất.
        subtitle_path: Tệp phụ đề (bắt buộc với SOFT_SUB và HARD_SUB).
        audio_path: Tệp tiếng Việt (bắt buộc với DUB_AUDIO).
        keep_original_audio: Với DUB_AUDIO, giữ tiếng gốc thành track thứ hai.
        video_encoder: Bộ mã hoá khi cần mã hoá lại (``libx264`` / ``h264_nvenc`` /
            ``libx265``…).
        quality: CRF (hoặc CQ với NVENC). Thấp hơn = nét hơn.
        subtitle_language: Mã ngôn ngữ ghi vào metadata track (ISO 639-2, vd ``vie``).
        background_gain: [VOICE_OVER] Hệ số âm lượng tiếng gốc khi không có thuyết minh.
        narration_gain: [VOICE_OVER] Hệ số âm lượng giọng thuyết minh.
        duck_level: Mức hạ tiếng gốc khi thuyết minh nói.
        downmix_multichannel: [v3.23.349] Trộn nguồn nhiều kênh (5.1/7.1) xuống stereo.
            Nguồn 6 kênh giữ nguyên khi sao chép có thể **không nghe thấy gì** trên máy
            người xem: thiết bị ra chỉ 2 kênh mà trình phát không tự trộn xuống, hoặc
            không giải mã được 5.1 trong định dạng chứa đó. Trộn sẵn cho chắc.
        subtitle_mode: [v3.23.326] Cách xử lý phụ đề. ``None`` = suy từ ``mode`` (giữ
            tương thích với mã cũ).
        audio_mode: [v3.23.326] Cách xử lý âm thanh. ``None`` = suy từ ``mode``.
    """

    video_path: Path
    output_path: Path
    mode: RenderMode
    subtitle_path: Path | None = None
    audio_path: Path | None = None
    keep_original_audio: bool = True
    video_encoder: str = "libx264"
    quality: int = _DEFAULT_CRF
    subtitle_language: str = "vie"
    background_gain: float = _DEFAULT_BACKGROUND_GAIN
    narration_gain: float = _DEFAULT_NARRATION_GAIN
    duck_level: DuckLevel = DuckLevel.MEDIUM
    downmix_multichannel: bool = True
    subtitle_mode: SubtitleMode | None = None
    audio_mode: AudioMode | None = None


def escape_filter_path(path: Path) -> str:
    """Thoát ký tự đường dẫn cho bộ lọc ffmpeg (``subtitles=``…).

    Đây là nguồn lỗi kinh điển trên Windows: bộ lọc ffmpeg phân tách tham số bằng
    ``:`` và ``,``, nên ổ đĩa ``C:`` và dấu ``\\`` phải được thoát, nếu không lệnh sẽ
    hỏng hoặc ffmpeg hiểu sai tham số.

    Quy tắc áp dụng (theo tài liệu ffmpeg filter syntax):
        1. ``\\`` → ``/`` (ffmpeg nhận ``/`` trên cả Windows).
        2. ``\\`` (còn lại), ``'``, ``[``, ``]``, ``,``, ``;`` → thêm ``\\`` phía trước.
        3. ``:`` → ``\\:`` (quan trọng nhất — ký tự ổ đĩa).

    Args:
        path: Đường dẫn cần thoát.

    Returns:
        Chuỗi đã thoát, dùng được trực tiếp trong biểu thức bộ lọc.
    """
    text = str(path).replace("\\", "/")
    for char in ("\\", "'", "[", "]", ",", ";"):
        text = text.replace(char, f"\\{char}")
    return text.replace(":", r"\:")


#: Bộ lọc trộn nhiều kênh xuống stereo, giữ nguyên độ lớn kênh giữa (thoại).
#:
#: ``aformat`` đứng trước để chuẩn hoá bố cục; ``pan`` chỉ có tác dụng khi nguồn nhiều
#: kênh. Với nguồn stereo/mono thì chuỗi này gần như không đổi gì.
DOWNMIX_FILTER: Final[str] = (
    "aresample=async=1,"
    "pan=stereo|FL=FL+0.707*FC+0.707*BL|FR=FR+0.707*FC+0.707*BR"
)

#: Định dạng phụ đề có định dạng phong phú (màu, style, vị trí).
_STYLED_SUBTITLE_SUFFIXES: Final[frozenset[str]] = frozenset({".ass", ".ssa"})


def _subtitle_codec_for(output_path: Path, subtitle_path: Path | None = None) -> str:
    """Chọn codec phụ đề theo CẢ định dạng nguồn LẪN định dạng chứa.

    [v3.23.327] SỬA LỖI MẤT ĐỊNH DẠNG: bản trước chọn codec CHỈ theo tệp đích, nên
    phụ đề ``.ass`` luôn bị chuyển sang ``srt`` — **mất sạch màu, style và vị trí**.
    Đã đo thực tế: mux ``.ass`` bằng codec ``srt`` làm mất ``&H0000FFFF`` (màu vàng),
    tên style, và thẻ ``{\\an8}``; dùng codec ``ass`` thì giữ nguyên cả ba.

    Args:
        output_path: Tệp đích (quyết định định dạng chứa).
        subtitle_path: Tệp phụ đề nguồn (để giữ định dạng nếu chứa được).

    Returns:
        Tên codec cho ``-c:s``.

    Raises:
        VideoRenderError: Khi định dạng đích không chứa được phụ đề văn bản.
    """
    suffix = output_path.suffix.lower()
    source_suffix = subtitle_path.suffix.lower() if subtitle_path is not None else ""

    if suffix in _MATROSKA_SUFFIXES:
        # Matroska chứa được ASS nguyên vẹn -> giữ định dạng khi nguồn có định dạng.
        if source_suffix in _STYLED_SUBTITLE_SUFFIXES:
            return "ass"
        return "srt"

    if suffix in _MP4_SUFFIXES:
        # MP4 KHÔNG chứa được ASS — buộc phải hạ về mov_text (mất định dạng).
        return "mov_text"

    raise VideoRenderError(
        f"Định dạng '{suffix}' không chứa được phụ đề dạng track. "
        "Hãy dùng .mkv (khuyến nghị) hoặc .mp4, hoặc chọn kiểu 'cháy vào hình'."
    )


def subtitle_styling_warning(
    output_path: Path, subtitle_path: Path | None
) -> str | None:
    """Cảnh báo nếu lựa chọn hiện tại sẽ LÀM MẤT định dạng phụ đề.

    Args:
        output_path: Tệp đích.
        subtitle_path: Tệp phụ đề nguồn.

    Returns:
        Thông điệp cảnh báo, hoặc ``None`` nếu không mất gì.
    """
    if subtitle_path is None:
        return None
    if subtitle_path.suffix.lower() not in _STYLED_SUBTITLE_SUFFIXES:
        return None
    if output_path.suffix.lower() in _MATROSKA_SUFFIXES:
        return None
    return (
        "Phụ đề .ass có màu/kiểu chữ/vị trí riêng, nhưng MP4 không chứa được định dạng "
        "này — sẽ bị hạ thành chữ trơn. Xuất ra .mkv để giữ nguyên, hoặc chọn “Phụ đề "
        "cháy vào hình”."
    )


def _quality_args(encoder: str, quality: int) -> list[str]:
    """Tham số chất lượng theo bộ mã hoá.

    NVENC dùng ``-cq`` và ``-preset pN``; các bộ mã hoá CPU dùng ``-crf`` và preset chữ.
    """
    if "nvenc" in encoder:
        return ["-preset", _NVENC_PRESET, "-cq", str(quality)]
    return ["-preset", _DEFAULT_PRESET, "-crf", str(quality)]


#: Ánh xạ chế độ CŨ sang cặp (phụ đề, âm thanh) — giữ tương thích ngược.
_LEGACY_MODE_MAP: Final[dict[RenderMode, tuple[SubtitleMode, AudioMode]]] = {
    RenderMode.SOFT_SUB: (SubtitleMode.SOFT, AudioMode.ORIGINAL),
    RenderMode.HARD_SUB: (SubtitleMode.BURNED, AudioMode.ORIGINAL),
    RenderMode.VOICE_OVER: (SubtitleMode.NONE, AudioMode.VOICE_OVER),
    RenderMode.DUB_AUDIO: (SubtitleMode.NONE, AudioMode.REPLACE_TRACK),
}


def resolve_modes(request: RenderRequest) -> tuple[SubtitleMode, AudioMode]:
    """Suy ra cặp (phụ đề, âm thanh) từ yêu cầu.

    Ưu tiên hai trường mới; nếu chưa đặt thì suy từ ``mode`` cũ để mã hiện có vẫn chạy.

    Args:
        request: Yêu cầu xuất.

    Returns:
        Cặp ``(SubtitleMode, AudioMode)``.
    """
    if request.subtitle_mode is not None and request.audio_mode is not None:
        return request.subtitle_mode, request.audio_mode
    legacy = _LEGACY_MODE_MAP[request.mode]
    return (
        request.subtitle_mode if request.subtitle_mode is not None else legacy[0],
        request.audio_mode if request.audio_mode is not None else legacy[1],
    )


def build_voice_over_filter(
    request: RenderRequest, audio_input_index: int = 1
) -> str:
    """Dựng biểu thức filter_complex trộn thuyết minh lên tiếng gốc, có ducking.

    Nguyên lý (kỹ thuật chuẩn của thuyết minh phát thanh/truyền hình):
    dùng chính giọng thuyết minh làm **tín hiệu điều khiển** (sidechain) cho một bộ nén
    đặt trên tiếng gốc. Khi thuyết minh nói → tiếng gốc bị hạ; khi im → tiếng gốc dâng
    lại. Nhờ vậy **nhạc và tiếng động nền vẫn còn**, chỉ nhỏ đi đúng lúc cần.

    Args:
        request: Yêu cầu xuất (dùng ``background_gain``/``narration_gain``/``duck_level``).
        audio_input_index: Chỉ số input của tệp thuyết minh. Trước đây cố định là 1,
            nhưng từ v3.23.326 thứ tự input thay đổi theo tổ hợp chế độ nên phải truyền
            vào — nếu để cứng sẽ trộn nhầm luồng.

    Returns:
        Chuỗi ``filter_complex``, đầu ra gắn nhãn ``[aout]``.

    Notes:
        GIỚI HẠN QUAN TRỌNG: bộ nén hạ **toàn bộ** tiếng gốc (cả nhạc lẫn thoại gốc),
        chứ không tách riêng thoại. Muốn xoá HẲN thoại gốc mà giữ nguyên nhạc/tiếng động
        thì cần tách nguồn âm bằng AI (vd Demucs) — xem ghi chú ở cuối module.
    """
    background = f"volume={request.background_gain}"
    narration = f"volume={request.narration_gain}"
    return (
        # Nhân đôi giọng thuyết minh: một bản để nghe, một bản làm tín hiệu điều khiển.
        f"[{audio_input_index}:a]aformat=sample_fmts=fltp:sample_rates=48000:"
        f"channel_layouts=stereo,"
        f"{narration},asplit=2[narr][key];"
        # [v3.23.349] Trộn xuống stereo có nhấn kênh giữa TRƯỚC khi chuẩn hoá — nếu để
        # aformat tự trộn, kênh thoại (FC) bị hạ ~3dB và lời nói mờ đi dưới nhạc nền.
        f"[0:a]{DOWNMIX_FILTER},"
        f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
        f"{background}[bg];"
        # Nén tiếng gốc theo tín hiệu điều khiển -> đây chính là ducking.
        f"[bg][key]sidechaincompress="
        f"threshold={request.duck_level.threshold}:ratio={_DUCK_RATIO}:"
        f"attack={_DUCK_ATTACK_MS}:release={_DUCK_RELEASE_MS}[ducked];"
        # Trộn nền đã hạ với giọng thuyết minh. normalize=0 để không tự giảm 1/2 âm lượng.
        f"[ducked][narr]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
    )


def _validate(request: RenderRequest) -> None:
    """Kiểm tra yêu cầu hợp lệ trước khi dựng lệnh.

    Raises:
        VideoRenderError: Khi thiếu tệp đầu vào bắt buộc hoặc đích trùng nguồn.
    """
    if not request.video_path.is_file():
        raise VideoRenderError(f"Không tìm thấy video nguồn: {request.video_path}")

    if request.mode in (RenderMode.SOFT_SUB, RenderMode.HARD_SUB):
        if request.subtitle_path is None:
            raise VideoRenderError("Kiểu xuất này cần tệp phụ đề.")
        if not request.subtitle_path.is_file():
            raise VideoRenderError(f"Không tìm thấy phụ đề: {request.subtitle_path}")

    if request.mode in (RenderMode.DUB_AUDIO, RenderMode.VOICE_OVER):
        if request.audio_path is None:
            raise VideoRenderError("Kiểu này cần tệp âm thanh tiếng Việt.")
        if not request.audio_path.is_file():
            raise VideoRenderError(f"Không tìm thấy âm thanh: {request.audio_path}")

    try:
        same = request.output_path.resolve() == request.video_path.resolve()
    except OSError:
        same = str(request.output_path) == str(request.video_path)
    if same:
        raise VideoRenderError("Tệp đích không được trùng tệp nguồn.")


def _validate_modes(
    request: RenderRequest, subtitle_mode: SubtitleMode, audio_mode: AudioMode
) -> None:
    """Kiểm tệp đầu vào cần thiết cho cặp chế độ đã chọn.

    Raises:
        VideoRenderError: Khi thiếu tệp phụ đề hoặc tệp âm thanh cần dùng.
    """
    if subtitle_mode is not SubtitleMode.NONE:
        if request.subtitle_path is None:
            raise VideoRenderError("Kiểu xuất này cần tệp phụ đề.")
        if not request.subtitle_path.is_file():
            raise VideoRenderError(f"Không tìm thấy phụ đề: {request.subtitle_path}")

    if audio_mode is not AudioMode.ORIGINAL:
        if request.audio_path is None:
            raise VideoRenderError("Kiểu này cần tệp âm thanh tiếng Việt.")
        if not request.audio_path.is_file():
            raise VideoRenderError(f"Không tìm thấy âm thanh: {request.audio_path}")


def build_render_command(ffmpeg_binary: str, request: RenderRequest) -> list[str]:
    """Dựng lệnh ffmpeg cho một yêu cầu xuất video.

    [v3.23.326] Phụ đề và âm thanh nay là HAI CHIỀU ĐỘC LẬP, nên kết hợp được mọi cặp —
    đặc biệt là **phụ đề + thuyết minh trong cùng một tệp**, thứ mà bốn chế độ cũ không
    làm được (chúng loại trừ nhau).

    Args:
        ffmpeg_binary: Đường dẫn tới ffmpeg.
        request: Yêu cầu xuất.

    Returns:
        Danh sách tham số dòng lệnh, truyền thẳng cho ``subprocess``.

    Raises:
        VideoRenderError: Khi tham số không hợp lệ.
    """
    _validate(request)
    subtitle_mode, audio_mode = resolve_modes(request)
    _validate_modes(request, subtitle_mode, audio_mode)

    command: list[str] = [
        ffmpeg_binary, "-hide_banner", "-loglevel", "error", "-stats", "-y",
        "-i", str(request.video_path),
    ]

    # Thứ tự input quyết định chỉ số dùng trong -map: 0=video, rồi lần lượt.
    audio_input_index: int | None = None
    subtitle_input_index: int | None = None
    next_index = 1

    if audio_mode is not AudioMode.ORIGINAL:
        command += ["-i", str(request.audio_path)]
        audio_input_index = next_index
        next_index += 1

    # Phụ đề chỉ cần làm INPUT khi ghép mềm; kiểu cháy dùng bộ lọc đọc thẳng tệp.
    if subtitle_mode is SubtitleMode.SOFT:
        command += ["-i", str(request.subtitle_path)]
        subtitle_input_index = next_index
        next_index += 1

    needs_video_filter = subtitle_mode is SubtitleMode.BURNED
    needs_audio_filter = audio_mode is AudioMode.VOICE_OVER

    # KHÔNG thể dùng đồng thời -vf và -filter_complex. Khi cần cả hai, gộp vào một
    # filter_complex duy nhất.
    if needs_video_filter and needs_audio_filter:
        escaped = escape_filter_path(request.subtitle_path)
        audio_filter = build_voice_over_filter(request, audio_input_index or 1)
        command += [
            "-filter_complex",
            f"[0:v]subtitles='{escaped}'[vout];{audio_filter}",
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", request.video_encoder,
            *_quality_args(request.video_encoder, request.quality),
            "-c:a", "aac", "-b:a", "192k",
        ]
    elif needs_audio_filter:
        command += [
            "-filter_complex", build_voice_over_filter(request, audio_input_index or 1),
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy",          # hình không đụng tới -> rất nhanh
            "-c:a", "aac", "-b:a", "192k",
        ]
    elif needs_video_filter:
        escaped = escape_filter_path(request.subtitle_path)
        command += [
            "-vf", f"subtitles='{escaped}'",
            "-map", "0:v",
            "-c:v", request.video_encoder,
            *_quality_args(request.video_encoder, request.quality),
        ]
        command += _plain_audio_args(request, audio_mode, audio_input_index)
    else:
        command += ["-map", "0:v", "-c:v", "copy"]
        command += _plain_audio_args(request, audio_mode, audio_input_index)

    # Ghép track phụ đề (chỉ với kiểu mềm).
    if subtitle_input_index is not None:
        command += [
            "-map", f"{subtitle_input_index}:s?",
            "-c:s", _subtitle_codec_for(request.output_path, request.subtitle_path),
            "-metadata:s:s:0", f"language={request.subtitle_language}",
            "-disposition:s:0", "default",
        ]

    command.append(str(request.output_path))
    return command


def _plain_audio_args(
    request: RenderRequest, audio_mode: AudioMode, audio_input_index: int | None
) -> list[str]:
    """Tham số âm thanh cho các trường hợp KHÔNG cần trộn (không dùng filter).

    Args:
        request: Yêu cầu xuất.
        audio_mode: Cách xử lý âm thanh.
        audio_input_index: Chỉ số input của tệp tiếng Việt (nếu có).

    Returns:
        Danh sách tham số ffmpeg cho phần âm thanh.
    """
    if audio_mode is AudioMode.REPLACE_TRACK and audio_input_index is not None:
        args = [
            "-map", f"{audio_input_index}:a",  # tiếng Việt đứng TRƯỚC -> thành mặc định
            "-metadata:s:a:0", f"language={request.subtitle_language}",
            "-disposition:a:0", "default",
        ]
        if request.keep_original_audio:
            # '?' để video CÂM không làm cả lệnh thất bại.
            args += ["-map", "0:a?"]
        return [*args, "-c:a", "aac", "-b:a", "192k"]

    if request.downmix_multichannel:
        # [v3.23.349] Trộn xuống stereo có NHẤN KÊNH GIỮA. Ở bố cục 5.1, kênh giữa (FC,
        # kênh thứ 3) chứa gần như toàn bộ thoại. Công thức trộn mặc định của ffmpeg hạ
        # FC khoảng -3dB; ta giữ nguyên hệ số 1.0 để lời thoại rõ hơn.
        # `pan` chỉ áp dụng khi nguồn thật sự nhiều kênh; ffmpeg tự bỏ qua nếu là stereo.
        return [
            "-map", "0:a?",
            "-af", DOWNMIX_FILTER,
            "-c:a", "aac", "-b:a", "192k", "-ac", "2",
        ]

    # Giữ nguyên luồng gốc — nhanh nhất, nhưng nguồn nhiều kênh có thể không phát được.
    return ["-map", "0:a?", "-c:a", "copy"]


# ── Ghi chú: muốn XOÁ HẲN thoại gốc mà giữ nhạc/tiếng động? ─────────────────
# Ducking ở trên hạ TOÀN BỘ tiếng gốc (nhạc + thoại) khi thuyết minh nói. Đây là kỹ
# thuật chuẩn của thuyết minh và nghe rất tự nhiên, nhưng thoại gốc VẪN CÒN nghe được
# nhỏ bên dưới.
#
# Muốn tách riêng: cần **tách nguồn âm bằng AI** — chia tiếng gốc thành "giọng hát/thoại"
# và "nhạc đệm", bỏ phần thoại, rồi trộn thuyết minh với phần nhạc đệm. Thư viện phù hợp:
#   * Demucs (MIT, facebookresearch/demucs) — chất lượng tốt nhất hiện nay, cần torch.
# Chưa tích hợp vì cần torch (~2GB, giống WhisperX). Nếu bật WhisperX thì đã có torch,
# lúc đó thêm Demucs là rẻ.

__all__ = [
    "DuckLevel",
    "RenderMode",
    "RenderRequest",
    "VideoRenderError",
    "build_render_command",
    "build_voice_over_filter",
    "subtitle_styling_warning",
    "escape_filter_path",
]
