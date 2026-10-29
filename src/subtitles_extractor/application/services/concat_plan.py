"""[v3.23.366] Lập kế hoạch NỐI các tập phim thành MỘT video trọn bộ.

Tách phần quyết định (thuần, test được) khỏi phần chạy ffmpeg (adapter). Nhiệm vụ:
tìm các tệp video trong thư mục, SẮP THỨ TỰ TỰ NHIÊN theo số tập (để "第2集" đứng trước
"第10集", không bị sắp theo bảng chữ cái), và dựng nội dung danh sách concat cho ffmpeg.

[v3.23.617] Thêm phần quyết định CÁCH nối: chỉ sao chép luồng khi mọi tập cùng thông số;
lệch (tốc độ khung, time base, bộ tham số mã hoá…) thì chuẩn hoá từng tập rồi nối bằng
concat filter. Trước đây bộ phim 86 tập có 6 tập 30 fps lẫn giữa các tập 25 fps: nối
sao chép luồng "thành công" nhưng tệp ra có khoảng trống 41.5s không có khung hình nào
ở đầu tập 30 fps, hình bị giãn 1.2 lần và tiếng lệch hình ngày càng xa.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: Đuôi tệp video hợp lệ để nối.
_VIDEO_SUFFIXES: frozenset[str] = frozenset(
    {".mp4", ".mkv", ".avi", ".mov", ".ts", ".webm", ".m4v", ".flv"}
)

_NUMBER_RE = re.compile(r"\d+")


@dataclass(frozen=True, slots=True)
class ConcatPlan:
    """Kế hoạch nối video.

    Attributes:
        videos: Danh sách tệp đã sắp thứ tự tự nhiên.
        output_path: Tệp video trọn bộ sẽ ghi ra.
    """

    videos: list[Path]
    output_path: Path

    @property
    def is_valid(self) -> bool:
        """Cần tối thiểu 2 tệp mới có ý nghĩa nối."""
        return len(self.videos) >= 2


def natural_sort_key(path: Path) -> tuple:
    """Khoá sắp xếp TỰ NHIÊN: tách xen kẽ chữ và SỐ (số so sánh theo giá trị).

    Ví dụ thứ tự đúng: ``第1集 < 第2集 < 第10集 < 第84集`` (thay vì 1,10,2,84...).
    """
    parts = _NUMBER_RE.split(path.stem)
    numbers = _NUMBER_RE.findall(path.stem)
    # Xen kẽ: text, number, text, number… — số ép kiểu int để so theo giá trị.
    key: list[object] = []
    for index, text in enumerate(parts):
        key.append(text.lower())
        if index < len(numbers):
            key.append(int(numbers[index]))
    return tuple(key)


def find_concat_videos(
    folder: Path,
    *,
    name_filter: str | None = None,
    exclude_names: set[str] | None = None,
) -> list[Path]:
    """Tìm và SẮP THỨ TỰ TỰ NHIÊN các video trong thư mục để nối.

    Args:
        folder: Thư mục chứa các tập.
        name_filter: Nếu khác ``None``, chỉ lấy tệp có chứa chuỗi này trong tên (vd
            ``"_phude_thuyetminh"`` để chỉ nối các bản ĐÃ xuất bản).
        exclude_names: Tập tên tệp cần loại (vd chính tệp trọn bộ đầu ra nếu đã tồn tại).

    Returns:
        Danh sách ``Path`` đã sắp thứ tự tự nhiên (rỗng nếu không có).
    """
    if not folder.is_dir():
        return []
    exclude = exclude_names or set()
    candidates = [
        entry
        for entry in folder.iterdir()
        if entry.is_file()
        and entry.suffix.lower() in _VIDEO_SUFFIXES
        and entry.name not in exclude
        and not is_full_series_output(entry)
        and not _is_partial_or_preview(entry)
        # [v3.23.659] Đúng HẬU TỐ (trước là "chứa chuỗi": "_phude" khớp cả "_phude_thuyetminh", "_phude_xemthu").
        and (name_filter is None or entry.stem.endswith(name_filter))
    ]
    return sorted(candidates, key=natural_sort_key)


def _is_partial_or_preview(path: Path) -> bool:
    """[v3.23.659] Tệp đang ghi dở (``.partial``) hoặc bản xuất thử (``_xemthu``) — không phải một tập."""
    stem = path.stem.lower()
    return stem.endswith(".partial") or "_xemthu" in stem


#: Hậu tố tên tệp trọn bộ do chính ứng dụng tạo (xem :func:`default_concat_output`).
FULL_SERIES_SUFFIX = "_tron_bo"


def is_full_series_output(path: Path) -> bool:
    """[v3.23.619] Tệp trọn bộ do ứng dụng tạo từ lần nối trước — KHÔNG phải một tập.

    Trước đây trang Xuất bản nối lần 2 lấy luôn ``<thư mục>_tron_bo.mkv`` của lần 1 làm
    đầu vào, trong khi đó cũng là tệp đầu ra → ffmpeg lỗi (mã 234), "Nén lại" cũng lỗi.
    """
    return path.stem.lower().endswith(FULL_SERIES_SUFFIX)


def build_concat_list_content(videos: list[Path]) -> str:
    """Dựng nội dung tệp danh sách cho ffmpeg concat demuxer.

    Định dạng mỗi dòng: ``file '<đường dẫn tuyệt đối>'``. Dấu nháy đơn trong đường dẫn
    được escape theo quy tắc của ffmpeg (``'`` → ``'\\''``).
    """
    lines = []
    for video in videos:
        safe = str(video.resolve()).replace("'", "'\\''")
        lines.append(f"file '{safe}'")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class ConcatInputInfo:
    """[v3.23.617] Thông số một tập (đọc bằng ffprobe) để quyết định cách nối.

    Attributes:
        path: Tệp video.
        duration_sec: Thời lượng tệp (format duration).
        video_codec / width / height / pix_fmt / sar: Thông số luồng hình.
        frame_rate: ``r_frame_rate`` dạng phân số ``"25/1"``.
        time_base: Time base luồng hình, ví dụ ``"1/12800"``.
        extradata_hash: Băm của extradata (VPS/SPS/PPS) — khác nhau nghĩa là bộ tham số
            mã hoá khác nhau dù cùng codec.
        audio_codec / sample_rate / channels: Thông số luồng tiếng; ``audio_codec`` rỗng
            khi tập không có tiếng.
    """

    path: Path
    duration_sec: float
    video_codec: str
    width: int
    height: int
    pix_fmt: str
    frame_rate: str
    time_base: str
    extradata_hash: str = ""
    sar: str = "1:1"
    audio_codec: str = ""
    sample_rate: int = 0
    channels: int = 0

    @property
    def has_audio(self) -> bool:
        return bool(self.audio_codec)

    @property
    def fps(self) -> float:
        return parse_frame_rate(self.frame_rate)

    def stream_signature(self) -> tuple:
        """Mọi thông số phải GIỐNG HỆT nhau giữa các tập thì mới sao chép luồng được."""
        return (
            self.video_codec, self.width, self.height, self.pix_fmt, self.sar,
            self.frame_rate, self.time_base, self.extradata_hash,
            self.audio_codec, self.sample_rate, self.channels,
        )


@dataclass(frozen=True, slots=True)
class ConcatTarget:
    """[v3.23.617] Thông số chung mà mọi tập được chuẩn hoá về trước khi nối."""

    width: int
    height: int
    frame_rate: str
    sample_rate: int
    channels: int

    @property
    def channel_layout(self) -> str:
        return "mono" if self.channels == 1 else "stereo"


def parse_frame_rate(value: str) -> float:
    """``"30000/1001"`` → 29.97; giá trị hỏng → 0.0."""
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            return float(num) / float(den) if float(den) else 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def copy_mismatch_reasons(infos: list[ConcatInputInfo]) -> list[str]:
    """[v3.23.617] Lý do KHÔNG được sao chép luồng (rỗng = sao chép an toàn).

    Concat demuxer + ``-c copy`` chỉ đúng khi mọi tập cùng thông số. Lệch tốc độ khung
    hay time base làm nhãn thời gian của các tập sau bị tính sai: hình bị giãn, xuất hiện
    khoảng trống hàng chục giây không có khung nào, tiếng lệch hình — mà ffmpeg KHÔNG báo
    lỗi. Lệch bộ tham số HEVC/H.264 (SPS) khiến tệp ra chỉ mang SPS của tập đầu.
    """
    if len(infos) < 2:
        return []
    reference = infos[0]
    labels = (
        "codec hình", "chiều rộng", "chiều cao", "định dạng điểm ảnh", "tỉ lệ điểm ảnh",
        "tốc độ khung", "time base", "bộ tham số mã hoá (SPS/PPS)",
        "codec tiếng", "tần số lấy mẫu tiếng", "số kênh tiếng",
    )
    reasons: list[str] = []
    ref_signature = reference.stream_signature()
    for index, label in enumerate(labels):
        differing = [
            info.path.name for info in infos[1:]
            if info.stream_signature()[index] != ref_signature[index]
        ]
        if differing:
            preview = ", ".join(differing[:5]) + ("…" if len(differing) > 5 else "")
            reasons.append(f"{label} khác tập đầu ở {len(differing)} tập ({preview})")
    return reasons


def _majority(values: list, *, prefer_larger: bool = True):
    """Giá trị xuất hiện nhiều nhất; hoà thì lấy giá trị lớn hơn (giữ chất lượng)."""
    counts: dict = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=lambda v: (counts[v], v if prefer_larger else 0))


def choose_concat_target(infos: list[ConcatInputInfo]) -> ConcatTarget:
    """[v3.23.617] Chọn thông số chung theo SỐ ĐÔNG các tập.

    Lấy theo số đông để phần lớn các tập không phải đổi tốc độ khung / kích thước (chỉ
    vài tập lệch bị chuẩn hoá), tránh nhân đôi khung hay phóng to vô ích.
    """
    sizes = _majority([(info.width, info.height) for info in infos])
    rates = [info.frame_rate for info in infos if parse_frame_rate(info.frame_rate) > 0]
    frame_rate = (
        max(set(rates), key=lambda r: (rates.count(r), parse_frame_rate(r)))
        if rates else "25/1"
    )
    with_audio = [info for info in infos if info.has_audio]
    sample_rate = _majority([i.sample_rate for i in with_audio]) if with_audio else 48000
    channels = _majority([min(i.channels, 2) for i in with_audio]) if with_audio else 2
    return ConcatTarget(
        width=sizes[0], height=sizes[1], frame_rate=frame_rate,
        sample_rate=sample_rate or 48000, channels=channels or 2,
    )


def build_normalize_filter(infos: list[ConcatInputInfo], target: ConcatTarget) -> str:
    """[v3.23.617] Dựng ``filter_complex`` chuẩn hoá từng tập rồi nối bằng concat FILTER.

    Concat filter nối theo khung đã giải mã, tự đặt lại nhãn thời gian liên tục — không
    phụ thuộc time base hay tốc độ khung của từng tệp như concat demuxer. Mỗi tập:
    * hình: về 0 (``setpts``), đổi tốc độ khung về đích (``fps``), khớp kích thước giữ tỉ lệ
      (``scale`` + ``pad`` — chỉ khi lệch), ``setsar=1``, ``format`` chung;
    * tiếng: về 0, ``aresample`` (bù trôi nhỏ), định dạng chung; tập KHÔNG có tiếng được
      chèn khoảng lặng đúng bằng thời lượng tập để hình-tiếng không lệch dồn.

    Đầu ra có nhãn ``[vout]`` và ``[aout]``.
    """
    parts: list[str] = []
    joined: list[str] = []
    for index, info in enumerate(infos):
        video_chain = ["setpts=PTS-STARTPTS", f"fps={target.frame_rate}"]
        if (info.width, info.height) != (target.width, target.height):
            video_chain.append(
                f"scale={target.width}:{target.height}:force_original_aspect_ratio=decrease"
            )
            video_chain.append(
                f"pad={target.width}:{target.height}:(ow-iw)/2:(oh-ih)/2"
            )
        video_chain.append("setsar=1")
        video_chain.append("format=yuv420p")
        parts.append(f"[{index}:v:0]{','.join(video_chain)}[v{index}]")

        audio_format = (
            f"aformat=sample_fmts=fltp:sample_rates={target.sample_rate}"
            f":channel_layouts={target.channel_layout}"
        )
        if info.has_audio:
            parts.append(
                f"[{index}:a:0]asetpts=PTS-STARTPTS,"
                f"aresample={target.sample_rate}:async=1,{audio_format}[a{index}]"
            )
        else:
            duration = max(info.duration_sec, 0.0)
            parts.append(
                f"anullsrc=r={target.sample_rate}:cl={target.channel_layout},"
                f"atrim=duration={duration:.3f},{audio_format}[a{index}]"
            )
        joined.append(f"[v{index}][a{index}]")
    parts.append(f"{''.join(joined)}concat=n={len(infos)}:v=1:a=1[vout][aout]")
    return ";".join(parts)


def duration_mismatch(expected_sec: float, actual_sec: float) -> bool:
    """[v3.23.617] Tệp nối dài/ngắn bất thường so với tổng các tập → nối hỏng.

    Dung sai: 1 giây hoặc 0.5% tổng (lấy lớn hơn) — đủ rộng cho làm tròn ở ranh giới tập,
    đủ chặt để bắt lỗi nhãn thời gian (lỗi thật làm lệch hàng chục giây).
    """
    if expected_sec <= 0 or actual_sec <= 0:
        return False
    tolerance = max(1.0, expected_sec * 0.005)
    return abs(actual_sec - expected_sec) > tolerance


def default_concat_output(folder: Path, videos: list[Path]) -> Path:
    """Tên tệp trọn bộ mặc định: ``<tên thư mục>_trọn bộ.mkv`` trong cùng thư mục.

    Dùng MKV để bao được mọi codec khi sao chép luồng (``-c copy``) mà không phải nén lại.
    """
    base = folder.name or (videos[0].parent.name if videos else "video")
    return folder / f"{base}{FULL_SERIES_SUFFIX}.mkv"
