"""Adapter trích phụ đề nhúng bằng ffprobe/ffmpeg.

Hiện thực :class:`EmbeddedSubtitlePort`:
  * ``list_tracks``  — ffprobe liệt kê stream phụ đề (``-select_streams s``).
  * ``extract_track`` — ffmpeg rút track:
      - Text-based → xuất tạm SRT rồi tái dùng ``SrtImporter`` (đã chống rác).
      - Bitmap (PGS/VOBSUB) → tách từng ảnh PNG + mốc thời gian để use-case OCR.

Không phụ thuộc PaddleOCR ở đây — việc OCR bitmap do tầng use-case điều phối, giữ
adapter này thuần I/O (đúng Single Responsibility).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from loguru import logger

from subtitles_extractor.domain.exceptions import VideoDecodeError, VideoNotFoundError
from subtitles_extractor.domain.ports.embedded_subtitle_port import (
    BitmapSubtitleFrame,
    EmbeddedExtractionResult,
    EmbeddedSubtitleTrack,
)
from subtitles_extractor.infrastructure.subtitle.importers.ass_importer import AssImporter
from subtitles_extractor.infrastructure.subtitle.importers.srt_importer import SrtImporter

# Codec phụ đề dạng ẢNH (cần OCR) — phần còn lại coi là text-based.
_BITMAP_CODECS: frozenset[str] = frozenset(
    {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub", "pgssub"}
)


def _subprocess_flags() -> dict[str, int]:
    """Cờ ẩn cửa sổ console trên Windows (đồng bộ với waveform widget)."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}  # type: ignore[attr-defined]
    return {}


class FfmpegEmbeddedSubtitleAdapter:
    """Trích phụ đề nhúng qua ffprobe + ffmpeg."""

    def __init__(
        self,
        ffprobe_binary: str = "ffprobe",
        ffmpeg_binary: str = "ffmpeg",
        subprocess_timeout_sec: float = 120.0,
        text_extract_timeout_sec: float = 900.0,
        bitmap_extract_timeout_sec: float = 900.0,
    ) -> None:
        """Khởi tạo adapter trích phụ đề nhúng.

        Args:
            ffprobe_binary: Đường dẫn tới ffprobe.
            ffmpeg_binary: Đường dẫn tới ffmpeg.
            subprocess_timeout_sec: Timeout chung cho ffprobe (liệt kê track — nhanh).
            text_extract_timeout_sec: Timeout RIÊNG cho trích track VĂN BẢN — dài hơn
                vì subtitle stream trải dài cả phim, file lớn (Blu-ray remux) cần quét
                tuần tự lâu hơn timeout chung (mặc định 15 phút).
            bitmap_extract_timeout_sec: [v3.23.360] Timeout RIÊNG cho tách ảnh BITMAP
                (PGS/VobSub qua sub2video). Trước đây dùng chung ``subprocess_timeout_sec``
                (120s) → SAI: sub2video vừa DEMUX cả file (10GB Blu-ray) vừa RENDER từng
                ảnh PNG nên còn NẶNG HƠN trích văn bản → timeout 120s không đủ, thất bại
                trên file lớn. Đặt bằng mức trích văn bản (mặc định 15 phút).
        """
        self._ffprobe = ffprobe_binary
        self._ffmpeg = ffmpeg_binary
        self._timeout = subprocess_timeout_sec
        self._text_extract_timeout = text_extract_timeout_sec
        self._bitmap_extract_timeout = bitmap_extract_timeout_sec
        self._srt_importer = SrtImporter()
        self._ass_importer = AssImporter()

    def list_tracks(self, video_path: Path) -> list[EmbeddedSubtitleTrack]:
        if not video_path.exists():
            raise VideoNotFoundError(f"Không tìm thấy video: {video_path}.")
        command = [
            self._ffprobe, "-v", "error",
            "-select_streams", "s",
            "-show_entries", "stream=index:stream_tags=language,title:stream=codec_name",
            "-of", "json", str(video_path),
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=self._timeout, **_subprocess_flags(),
            )
        except FileNotFoundError as exc:
            raise VideoDecodeError("Chưa cài ffprobe (cần để đọc phụ đề nhúng).") from exc
        except subprocess.TimeoutExpired as exc:
            raise VideoDecodeError("ffprobe quá thời gian khi liệt kê phụ đề.") from exc
        if completed.returncode != 0:
            raise VideoDecodeError(f"ffprobe lỗi: {completed.stderr.strip()[:200]}")

        try:
            payload = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise VideoDecodeError("Không phân tích được kết quả ffprobe.") from exc

        tracks: list[EmbeddedSubtitleTrack] = []
        # ffprobe trả index TUYỆT ĐỐI trong container; ta cần index TƯƠNG ĐỐI trong
        # nhóm phụ đề (cho ffmpeg -map 0:s:N) → đánh số lại theo thứ tự xuất hiện.
        for relative_index, stream in enumerate(payload.get("streams", [])):
            codec = str(stream.get("codec_name", "")).lower()
            tags = stream.get("tags", {}) or {}
            tracks.append(
                EmbeddedSubtitleTrack(
                    track_index=relative_index,
                    codec=codec,
                    language=str(tags.get("language", "")),
                    title=str(tags.get("title", "")),
                    is_bitmap=codec in _BITMAP_CODECS,
                )
            )
        logger.info("Phát hiện {} track phụ đề nhúng trong {}.", len(tracks), video_path.name)
        return tracks

    def extract_track(
        self, video_path: Path, track: EmbeddedSubtitleTrack
    ) -> EmbeddedExtractionResult:
        if not video_path.exists():
            raise VideoNotFoundError(f"Không tìm thấy video: {video_path}.")
        if track.is_bitmap:
            return self._extract_bitmap(video_path, track)
        return self._extract_text(video_path, track)

    def _extract_text(
        self, video_path: Path, track: EmbeddedSubtitleTrack
    ) -> EmbeddedExtractionResult:
        """Trích track text-based → SubtitleEvent, dùng parser ĐÚNG ĐỊNH DẠNG GỐC.

        [v3.23.362] Tham khảo Subtitle Edit: giữ ĐỊNH DẠNG GỐC thay vì ép mọi thứ về SRT.
        ASS/SSA: sao chép nguyên track ``.ass`` (``-c:s copy`` — nhanh, không mất mát) rồi
        parse bằng :class:`AssImporter` (xoá override ``{...}``, đổi ``\\N`` thành xuống
        dòng, giữ đúng số câu). Tránh lỗi ffmpeg chuyển ass→srt hay gộp/đảo câu chồng lấn
        và làm rơi định dạng. Các codec text khác (subrip/mov_text/webvtt…) vẫn qua SRT.
        Mọi lỗi ở đường ASS đều rơi về đường SRT làm lưới an toàn (không bao giờ regress).
        """
        if track.codec in {"ass", "ssa"}:
            try:
                result = self._extract_ass_native(video_path, track)
            except (VideoDecodeError, OSError, ValueError, RuntimeError) as exc:
                logger.warning("Trích ASS gốc lỗi ({}); chuyển sang chuyển đổi SRT.", exc)
                result = None
            if result is not None and result.events:
                logger.info(
                    "Trích {} câu ASS (giữ định dạng gốc) từ track #{}.",
                    len(result.events), track.track_index,
                )
                return result
            logger.warning("ASS gốc không ra câu; chuyển sang chuyển đổi SRT.")
        return self._extract_text_as_srt(video_path, track)

    def _extract_ass_native(
        self, video_path: Path, track: EmbeddedSubtitleTrack
    ) -> EmbeddedExtractionResult | None:
        """[v3.23.362] Sao chép track ASS/SSA ra ``.ass`` rồi parse bằng AssImporter."""
        with tempfile.TemporaryDirectory(prefix="se_embed_ass_") as tmp_dir:
            ass_path = Path(tmp_dir) / "embedded.ass"
            command = [
                self._ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(video_path),
                "-map", f"0:s:{track.track_index}",
                "-map", "-0:d?", "-map", "-0:t?",  # loại data + attachment (font)
                "-c:s", "copy", str(ass_path),
            ]
            self._run_ffmpeg(
                command, "trích phụ đề ASS", timeout=self._text_extract_timeout
            )
            if not ass_path.exists() or ass_path.stat().st_size == 0:
                return None
            events = self._ass_importer.import_from(ass_path)
        return EmbeddedExtractionResult(events=events, is_bitmap=False)

    def _extract_text_as_srt(
        self, video_path: Path, track: EmbeddedSubtitleTrack
    ) -> EmbeddedExtractionResult:
        """Rút track text-based ra SRT tạm rồi parse bằng SrtImporter.

        [v3.23.170] Với file lớn (Blu-ray remux 40-60GB), ffmpeg phải DEMUX tuần tự để
        gom trọn subtitle stream (trải dài cả phim) nên có thể vượt timeout mặc định.
        Khắc phục: (1) ``-map -0:d? -map -0:t?`` LOẠI data/attachment (font đính kèm) —
        chúng không cần cho subtitle, tránh ffmpeg xử lý thừa; (2) dùng timeout RIÊNG
        dài hơn cho bước trích text (chỉ đọc-ghi text, rẻ nhưng cần quét hết file).
        """
        with tempfile.TemporaryDirectory(prefix="se_embed_") as tmp_dir:
            srt_path = Path(tmp_dir) / "embedded.srt"
            command = [
                self._ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(video_path),
                "-map", f"0:s:{track.track_index}",
                "-map", "-0:d?", "-map", "-0:t?",  # loại data + attachment (font)
                "-c:s", "srt", str(srt_path),
            ]
            self._run_ffmpeg(
                command, "trích phụ đề văn bản", timeout=self._text_extract_timeout
            )
            if not srt_path.exists() or srt_path.stat().st_size == 0:
                raise VideoDecodeError("ffmpeg không xuất được phụ đề (track rỗng?).")
            events = self._srt_importer.import_from(srt_path)
        logger.info("Trích {} câu từ track text #{}.", len(events), track.track_index)
        return EmbeddedExtractionResult(events=events, is_bitmap=False)

    def _extract_bitmap(
        self, video_path: Path, track: EmbeddedSubtitleTrack
    ) -> EmbeddedExtractionResult:
        """Tách track bitmap thành PNG + timestamp (chờ OCR ở use-case).

        [v3.23.95] Với VobSub (``dvd_subtitle``): trích thẳng ``.idx/.sub`` (copy gói,
        KHÔNG render frame video) rồi giải mã RLE từng sự kiện - nhanh như Subtitle Edit.
        Các định dạng bitmap khác (PGS…) hoặc khi giải mã VobSub thất bại: quay lại cách
        sub2video của ffmpeg.
        """
        if track.codec == "dvd_subtitle":
            try:
                frames = self._extract_vobsub(video_path, track)
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning("Giải mã VobSub lỗi ({}), thử cách sub2video.", exc)
                frames = []
            if frames:
                logger.info("Trích {} ảnh VobSub (đường .idx/.sub nhanh).", len(frames))
                return EmbeddedExtractionResult(bitmap_frames=frames, is_bitmap=True)
            logger.warning("VobSub không ra ảnh; chuyển sang cách sub2video.")
        elif "pgs" in track.codec:
            # [v3.23.361] PGS (Blu-ray): demux thẳng stream ra .sup rồi tự giải mã trong
            # bộ nhớ (như Subtitle Edit/BDSup2Sub) — KHÔNG render frame video → nhanh hơn
            # rất nhiều so với sub2video. Fallback sub2video nếu đường này lỗi.
            try:
                frames = self._extract_pgs_via_sup(video_path, track)
            except (OSError, ValueError, RuntimeError, VideoDecodeError) as exc:
                logger.warning("Giải mã PGS trực tiếp lỗi ({}), thử cách sub2video.", exc)
                frames = []
            if frames:
                logger.info("Trích {} ảnh PGS (đường .sup trực tiếp, nhanh).", len(frames))
                return EmbeddedExtractionResult(bitmap_frames=frames, is_bitmap=True)
            logger.warning("PGS trực tiếp không ra ảnh; chuyển sang cách sub2video.")
        return self._extract_bitmap_via_sub2video(video_path, track)

    def _extract_pgs_via_sup(
        self, video_path: Path, track: EmbeddedSubtitleTrack
    ) -> list[BitmapSubtitleFrame]:
        """[v3.23.361] Trích PGS bằng cách demux ``.sup`` rồi giải mã (SE-style).

        Demux ``-c copy`` chỉ SAO CHÉP gói phụ đề (không giải mã/không render video) nên
        rất nhanh; sau đó :func:`pgs_decoder.parse_sup` dựng ảnh + mốc thời gian trong MỘT
        lượt (không cần ffprobe quét lại). Trả [] nếu không ra ảnh (để router fallback).

        Raises:
            VideoDecodeError: nếu ffmpeg demux lỗi/timeout.
        """
        from subtitles_extractor.infrastructure.video import pgs_decoder

        tmp_dir = Path(tempfile.mkdtemp(prefix="se_pgs_"))
        sup_path = tmp_dir / "subs.sup"
        command = [
            self._ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path),
            "-map", f"0:s:{track.track_index}",
            "-c", "copy", "-f", "sup", str(sup_path),
        ]
        self._run_ffmpeg(
            command, "tách stream PGS (.sup)", timeout=self._bitmap_extract_timeout
        )
        if not sup_path.exists() or sup_path.stat().st_size == 0:
            return []

        subtitles = pgs_decoder.parse_sup(sup_path.read_bytes())
        frames: list[BitmapSubtitleFrame] = []
        for position, sub in enumerate(subtitles):
            image_path = tmp_dir / f"pgs_{position:05d}.png"
            sub.image.save(image_path)
            frames.append(
                BitmapSubtitleFrame(
                    image_path=image_path,
                    start_sec=sub.start_ms / 1000.0,
                    end_sec=sub.end_ms / 1000.0,
                )
            )
        return frames

    def _extract_vobsub(
        self, video_path: Path, track: EmbeddedSubtitleTrack
    ) -> list[BitmapSubtitleFrame]:
        """[v3.23.98] Trích VobSub từ MKV (đọc thẳng track) -> giải mã SPU -> PNG.

        Mô phỏng Subtitle Edit: đọc trực tiếp container Matroska (CodecPrivate = palette,
        mỗi block = một SPU thô) - KHÔNG cần ffmpeg muxer 'vobsub' (vốn không tồn tại) và
        KHÔNG quét frame video -> rất nhanh. Chỉ áp dụng cho file ``.mkv``; định dạng khác
        trả [] để router fallback sang sub2video.
        """
        from subtitles_extractor.infrastructure.video import mkv_vobsub, vobsub_decoder

        if not mkv_vobsub.is_mkv(video_path):
            logger.info("Nguồn không phải MKV; bỏ qua đường VobSub thuần Python.")
            return []

        idx_text, blocks = mkv_vobsub.extract_vobsub_track(video_path)
        if not blocks:
            return []

        palette = vobsub_decoder.parse_idx(idx_text).palette
        # [v3.23.99] Log chẩn đoán vài block đầu để soi cấu trúc SPU thật khi giải mã lỗi.
        for diag_pos in range(min(3, len(blocks))):
            logger.info(
                "VobSub block #{} chẩn đoán: {}",
                diag_pos,
                vobsub_decoder.inspect_spu(blocks[diag_pos][1]),
            )
        logger.info("VobSub palette[0..3]: {}", palette[:4])

        tmp_dir = Path(tempfile.mkdtemp(prefix="se_vobsub_"))
        frames: list[BitmapSubtitleFrame] = []
        for position, (start_ms, spu_bytes) in enumerate(blocks):
            try:
                image = vobsub_decoder.decode_spu(spu_bytes, palette)
            except (IndexError, ValueError, OSError) as exc:
                logger.debug("Bỏ qua SPU VobSub #{}: {}", position, exc)
                continue
            if image is None:
                continue
            # [v3.23.101] Thời điểm tắt hiển thị lấy từ chính SPU (khớp Subtitle Edit);
            # nếu SPU không khai báo thì mới dùng mốc của block kế tiếp làm cận trên.
            start_offset, end_offset = vobsub_decoder.parse_spu_timing(spu_bytes)
            display_start_ms = start_ms + (start_offset or 0)
            if end_offset is not None and end_offset > (start_offset or 0):
                display_end_ms = start_ms + end_offset
            elif position + 1 < len(blocks):
                display_end_ms = blocks[position + 1][0]
            else:
                display_end_ms = display_start_ms + 3000
            image_path = tmp_dir / f"vob_{position:05d}.png"
            image.save(image_path)
            frames.append(
                BitmapSubtitleFrame(
                    image_path=image_path,
                    start_sec=display_start_ms / 1000.0,
                    end_sec=display_end_ms / 1000.0,
                )
            )
        return frames

    def _extract_bitmap_via_sub2video(
        self, video_path: Path, track: EmbeddedSubtitleTrack
    ) -> EmbeddedExtractionResult:
        """Tách bitmap qua sub2video của ffmpeg (fallback cho PGS / khi VobSub fail)."""
        tmp_dir = Path(tempfile.mkdtemp(prefix="se_embed_bmp_"))
        pattern = str(tmp_dir / "sub_%05d.png")
        # [v3.23.93] Stream phụ đề là KIỂU SUBTITLE (sost) nên KHÔNG map thẳng sang muxer
        # image2 được (-> "encoder selection failed ... codec none"); '-c:v' cũng
        # không áp vì stream không phải video. PHẢI chuyển subtitle -> video bằng cơ chế
        # "sub2video" của ffmpeg: đưa stream phụ đề qua filter_complex (filter video),
        # ffmpeg tự dựng frame video từ ảnh phụ đề, rồi mới encode PNG.
        command = [
            self._ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path),
            "-filter_complex", f"[0:s:{track.track_index}]copy[v]",
            "-map", "[v]",
            "-c:v", "png",
            "-f", "image2",
            pattern,
        ]
        self._run_ffmpeg(
            command, "tách ảnh phụ đề bitmap", timeout=self._bitmap_extract_timeout
        )

        frames: list[BitmapSubtitleFrame] = []
        all_images = sorted(tmp_dir.glob("sub_*.png"))
        # sub2video sinh thêm frame TRỐNG (thời điểm XOÁ phụ đề) xen kẽ -> lọc bỏ để số
        # ảnh khớp số sự kiện phụ đề và tránh OCR ảnh rỗng.
        images = [p for p in all_images if not self._is_blank_image(p)]
        if len(images) != len(all_images):
            logger.info(
                "Lọc {} frame trống (sub2video), còn {} ảnh phụ đề thực.",
                len(all_images) - len(images), len(images),
            )
        # Lấy mốc thời gian từng ảnh bằng ffprobe trên stream phụ đề (packet pts).
        timings = self._probe_bitmap_timings(video_path, track.track_index, len(images))
        for image_path, (start_sec, end_sec) in zip(images, timings):
            frames.append(
                BitmapSubtitleFrame(image_path=image_path, start_sec=start_sec, end_sec=end_sec)
            )
        logger.info("Tách {} ảnh phụ đề bitmap từ track #{}.", len(frames), track.track_index)
        return EmbeddedExtractionResult(bitmap_frames=frames, is_bitmap=True)

    @staticmethod
    def _is_blank_image(image_path: Path) -> bool:
        """[v3.23.93] Ảnh có rỗng không (frame xoá phụ đề do sub2video sinh ra).

        Rỗng khi: hoàn toàn trong suốt (alpha toàn 0) HOẶC mọi kênh màu đồng nhất một
        giá trị (nền đặc, không có nội dung). Lỗi đọc ảnh -> coi là KHÔNG rỗng (giữ lại,
        an toàn hơn là loại nhầm).
        """
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                extrema = img.convert("RGBA").getextrema()
        except (OSError, ValueError, ImportError) as exc:
            logger.debug("Không kiểm được ảnh trống {}: {}", image_path, exc)
            return False

        (r_lo, r_hi), (g_lo, g_hi), (b_lo, b_hi), (_a_lo, a_hi) = extrema
        if a_hi == 0:  # toàn trong suốt
            return True
        # Mọi kênh RGB đồng nhất -> ảnh một màu đặc, không có nội dung phụ đề.
        return r_lo == r_hi and g_lo == g_hi and b_lo == b_hi

    def _probe_bitmap_timings(
        self, video_path: Path, track_index: int, expected_count: int
    ) -> list[tuple[float, float]]:
        """Đọc (start, end) từng gói phụ đề bitmap qua ffprobe packets."""
        command = [
            self._ffprobe, "-v", "error",
            "-select_streams", f"s:{track_index}",
            "-show_entries", "packet=pts_time,duration_time",
            "-of", "json", str(video_path),
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                # [v3.23.360] ffprobe packets QUÉT CẢ FILE (giống sub2video) → dùng timeout
                # bitmap dài, không phải timeout ffprobe-liệt-kê ngắn (120s) vốn thất bại
                # trên Blu-ray lớn.
                timeout=self._bitmap_extract_timeout, **_subprocess_flags(),
            )
            packets = json.loads(completed.stdout or "{}").get("packets", [])
        except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
            logger.warning("Không đọc được mốc thời gian phụ đề bitmap: {}", exc)
            packets = []

        timings: list[tuple[float, float]] = []
        for packet in packets:
            try:
                start = float(packet.get("pts_time", 0.0))
                duration = float(packet.get("duration_time", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            end = start + (duration if duration > 0 else 2.0)
            timings.append((start, end))

        # Đệm nếu số mốc < số ảnh (đề phòng lệch); mỗi ảnh hiển thị mặc định 2s.
        while len(timings) < expected_count:
            previous_end = timings[-1][1] if timings else 0.0
            timings.append((previous_end, previous_end + 2.0))
        return timings[:expected_count]

    def _run_ffmpeg(
        self, command: list[str], action_label: str, timeout: float | None = None
    ) -> None:
        """Chạy ffmpeg với timeout (mặc định dùng timeout chung nếu không truyền).

        Args:
            command: Danh sách tham số ffmpeg.
            action_label: Nhãn hành động (đưa vào thông điệp lỗi tiếng Việt).
            timeout: Timeout riêng (giây); None -> dùng ``self._timeout``.

        Raises:
            VideoDecodeError: Khi thiếu ffmpeg, quá thời gian, hoặc ffmpeg trả mã lỗi.
        """
        effective_timeout = timeout if timeout is not None else self._timeout
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=effective_timeout, **_subprocess_flags(),
            )
        except FileNotFoundError as exc:
            raise VideoDecodeError("Chưa cài ffmpeg (cần để trích phụ đề nhúng).") from exc
        except subprocess.TimeoutExpired as exc:
            raise VideoDecodeError(
                f"ffmpeg quá thời gian khi {action_label} (>{effective_timeout:.0f}s). "
                "File có thể rất lớn — thử lại hoặc dùng track khác."
            ) from exc
        if completed.returncode != 0:
            raise VideoDecodeError(
                f"ffmpeg lỗi khi {action_label}: {completed.stderr.strip()[:200]}"
            )


__all__ = ["FfmpegEmbeddedSubtitleAdapter"]
