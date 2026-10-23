"""[v3.23.450] TTS LẠI một số dòng được chọn (có thể đã sửa nội dung) rồi ghép lại master.

Luồng:
1. Tổng hợp LẠI các dòng được chọn (sub-request) → segment WAV mới ghi đè segment cũ theo index.
2. Trộn kết quả mới vào danh sách kết quả cũ (thay theo ``event_index``).
3. GHÉP LẠI master từ TẤT CẢ segment (dòng cũ tái dùng file cũ) → ghi WAV output + SRT đồng bộ.

Không render lại toàn bộ → sửa 1 dòng chỉ tốn thời gian của 1 dòng.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from collections.abc import Callable
from pathlib import Path

import numpy as np

from subtitles_extractor.infrastructure.tts.segment_store import segment_path
from subtitles_extractor.application.services.tts_remerge import (
    SegmentPlacement,
    remerge_master,
)
from subtitles_extractor.infrastructure.torch_import_blocker import without_torch_block
from subtitles_extractor.domain.ports.subtitle_tts_port import (
    LineNeighbors,
    SubtitleTTSPort,
    TTSCancellationCallback,
    TTSProgressCallback,
    TTSRequest,
    TTSSegmentResult,
)

logger = logging.getLogger(__name__)

#: [v3.23.647] Tiền tố thư mục segment tạm của lượt TTS lại (nằm trong thư mục segment chính).
_RESYNTH_TMP_PREFIX = "resynth_seg_"


def resynthesis_neighbors(
    events: list,
    selected_positions: list[int],
    prior_results: list[TTSSegmentResult],
    media_duration_s: float | None,
) -> tuple[LineNeighbors, ...]:
    """[v3.23.596] Mốc dòng kề THẬT của từng dòng được TTS lại (hàm thuần).

    Engine chỉ thấy tập con dòng chọn, nên tự nó lấy dòng chọn kế tiếp làm "câu sau" và
    coi như không có câu trước. Hàm này đưa mốc thật: câu sau KHÔNG được chọn thì là mốc
    tiếng đã nằm trong bản trộn (cố định); câu trước không được chọn thì là lúc tiếng của
    nó dứt.

    Args:
        events: Toàn bộ dòng (đã cập nhật nội dung sửa).
        selected_positions: Vị trí (0-based) các dòng TTS lại, tăng dần.
        prior_results: Kết quả lượt trước (theo ``event_index``).
        media_duration_s: Thời lượng video — biên của dòng cuối phim.

    Returns:
        Tuple cùng thứ tự ``selected_positions``.
    """
    selected = set(selected_positions)
    prior_by_pos = {int(result.event_index): result for result in prior_results}
    neighbors: list[LineNeighbors] = []
    for position in selected_positions:
        following = position + 1
        if following >= len(events):
            next_start, next_fixed = media_duration_s, True
        elif following in selected:
            next_start, next_fixed = float(events[following].start_sec), False
        else:
            next_start = _audio_start(prior_by_pos.get(following), events[following])
            next_fixed = True
        neighbors.append(LineNeighbors(
            next_start_sec=next_start,
            next_fixed=next_fixed,
            prev_audio_end_sec=_previous_audio_end(position, selected, prior_by_pos),
        ))
    return tuple(neighbors)


def _audio_start(result: TTSSegmentResult | None, event: object) -> float:
    """Mốc tiếng thật của một dòng đã có trong bản trộn (mốc phụ đề nếu chưa có tiếng)."""
    if result is not None and not result.was_skipped and result.adjusted_start_sec >= 0:
        return float(result.adjusted_start_sec)
    return float(event.start_sec)  # type: ignore[attr-defined]


def _previous_audio_end(
    position: int, selected: set[int], prior_by_pos: dict[int, TTSSegmentResult]
) -> float | None:
    """Lúc tiếng của dòng có tiếng gần nhất phía trước dứt.

    ``None`` khi dòng đó cũng được TTS lại (engine tự theo dõi trong lượt); dòng bị bỏ
    qua (không tiếng) thì xét tiếp dòng trước nữa.
    """
    for earlier in range(position - 1, -1, -1):
        if earlier in selected:
            return None
        result = prior_by_pos.get(earlier)
        if result is None or result.was_skipped or result.audio_duration_s <= 0.0:
            continue
        start = (
            result.adjusted_start_sec if result.adjusted_start_sec >= 0
            else result.start_sec
        )
        return float(start) + float(result.audio_duration_s)
    return 0.0


class ResynthesizeLinesUseCase:
    """Tổng hợp lại các dòng được chọn và ghép lại master (giữ nguyên các dòng khác)."""

    def __init__(self, adapter: SubtitleTTSPort) -> None:
        self._adapter = adapter

    def execute(
        self,
        *,
        full_request: TTSRequest,
        selected_indices: set[int],
        prior_results: list[TTSSegmentResult],
        output_path: Path,
        sample_rate: int,
        total_length_samples: int,
        progress_cb: TTSProgressCallback | None = None,
        cancel_cb: TTSCancellationCallback | None = None,
        remerge_only: bool = False,
        checkpoint_cb: Callable[[list[TTSSegmentResult], int], object] | None = None,
    ) -> list[TTSSegmentResult]:
        """TTS lại các dòng ``selected_indices`` rồi ghi lại master + trả kết quả đã cập nhật.

        Args:
            full_request: Yêu cầu TTS gốc (chứa toàn bộ events ĐÃ cập nhật nội dung sửa).
            selected_indices: Tập ``event_index`` cần tổng hợp lại.
            prior_results: Kết quả TTS trước đó (để tái dùng segment các dòng không đổi).
            output_path: File WAV master cần ghi đè.
            sample_rate: Tần số lấy mẫu của các segment.
            total_length_samples: Độ dài master (giữ nguyên tổng thời lượng/đồng bộ video).
            progress_cb: Callback tiến độ.
            cancel_cb: Callback kiểm tra huỷ.
            remerge_only: [v3.23.625] Chỉ GHÉP LẠI master từ các segment hiện có (sau khi xoá
                dòng), không tổng hợp gì; master không còn dòng nào có tiếng thì là im lặng.
            checkpoint_cb: [v3.23.647] Nhận điểm kiểm tra định kỳ ``(kết quả các dòng ĐÃ XONG theo vị trí gốc, sample
                rate)`` khi engine hỗ trợ ``set_checkpoint_callback``; segment mới được dời về thư mục chính ngay
                lúc đó, nên lượt tiếp tục dài mà bị ngắt vẫn giữ được phần đã làm.

        Returns:
            Danh sách kết quả đã cập nhật (dòng chọn có segment mới).

        Raises:
            ValueError: Nếu không có dòng nào được chọn hoặc thiếu segment để ghép.
        """
        if remerge_only:
            return self._remerge(
                full_request, prior_results, output_path, sample_rate, total_length_samples
            )
        if not selected_indices:
            raise ValueError("Chưa chọn dòng nào để TTS lại.")

        events = full_request.events
        # ``selected_indices`` là VỊ TRÍ enumerate (= ``TTSSegmentResult.event_index``, 0-based),
        # KHÔNG phải ``event.index`` (1-based). Chọn đúng theo vị trí để tránh lệch n→n-1.
        selected_positions = sorted(
            p for p in selected_indices if 0 <= p < len(events)
        )
        if not selected_positions:
            raise ValueError("Các dòng được chọn không hợp lệ.")
        selected_events = [events[p] for p in selected_positions]

        # 1. Tổng hợp lại CHỈ các dòng chọn vào thư mục segment TẠM (tránh ghi đè segment của
        #    dòng khác — subset bị đánh số lại 0,1,… nên nếu ghi thẳng vào thư mục chính sẽ đè
        #    nhầm seg_000000/seg_000001 của dòng 0/1).
        import shutil
        import tempfile

        from subtitles_extractor.infrastructure.tts.segment_store import (
            segments_dir_for,
        )

        # [v3.23.647] Thư mục tạm NẰM TRONG thư mục segment chính (cùng ổ đĩa ⇒ dời file là đổi tên, không chép; trước
        # đây ở thư mục tạm hệ thống: lượt tiếp tục ~1 GB WAV đổ vào ổ C:). Thư mục tạm sót lại sau sập thì dọn.
        main_segdir = segments_dir_for(output_path, getattr(full_request, "segments_dir", ""))
        main_segdir.mkdir(parents=True, exist_ok=True)
        for leftover in main_segdir.glob(_RESYNTH_TMP_PREFIX + "*"):
            shutil.rmtree(leftover, ignore_errors=True)
        tmp_segdir = Path(tempfile.mkdtemp(prefix=_RESYNTH_TMP_PREFIX, dir=main_segdir))
        sub_request = replace(
            full_request,
            events=selected_events,
            # [v3.23.593] Lồng tiếng: giọng của vai đi theo dòng — tập con đánh số lại
            # từ 0 nên phải cắt ``line_voices`` theo đúng vị trí gốc, không thì dòng TTS
            # lại mượn giọng của dòng 0, 1, … của cả phim.
            line_voices=full_request.line_voices_for(selected_positions),
            # [v3.23.596] Mốc dòng kề thật — không thì dòng TTS lại đè lên câu không chọn.
            neighbors=resynthesis_neighbors(
                list(events), selected_positions, prior_results,
                getattr(full_request, "media_duration_s", None),
            ),
            keep_line_audio=True,
            segments_dir=str(tmp_segdir),
        )
        tmp_output = output_path.with_name(output_path.stem + ".resynth_tmp.wav")
        prior_by_pos = {int(r.event_index): r for r in prior_results}
        # Dòng KHÔNG chọn là kết quả thật (dòng chọn trong ``prior_results`` chỉ là chỗ giữ chỗ "bỏ qua").
        settled = {pos: r for pos, r in prior_by_pos.items() if pos not in selected_indices}
        adopted: dict[int, TTSSegmentResult] = {}

        def adopt(new_result: TTSSegmentResult) -> TTSSegmentResult | None:
            """Dời segment của một dòng vừa xong về thư mục chính, trả kết quả theo vị trí gốc (None = bỏ qua)."""
            sub_pos = int(new_result.event_index)         # 0..k-1 trong subset
            if sub_pos < 0 or sub_pos >= len(selected_positions):
                return None
            orig_pos = selected_positions[sub_pos]
            dest = segment_path(main_segdir, orig_pos)
            new_seg_path = ""
            src = new_result.segment_audio_path
            if src and Path(src).exists():
                try:
                    shutil.move(src, dest)  # ghi đè đúng file của dòng gốc
                    new_seg_path = str(dest)
                except OSError as exc:
                    logger.warning("Không dời được segment %s → %s: %s", src, dest, exc)
                    new_seg_path = src
            elif orig_pos in prior_by_pos:
                new_seg_path = prior_by_pos[orig_pos].segment_audio_path
            replaced = replace(new_result, event_index=orig_pos, segment_audio_path=new_seg_path)
            prior_by_pos[orig_pos] = replaced
            return replaced

        def on_adapter_checkpoint(sub_results: list[TTSSegmentResult], rate: int) -> None:
            for new_result in sub_results:
                if int(new_result.event_index) in seen_sub_positions:
                    continue
                replaced = adopt(new_result)
                seen_sub_positions.add(int(new_result.event_index))
                if replaced is not None:
                    adopted[int(replaced.event_index)] = replaced
            if checkpoint_cb is not None:
                checkpoint_cb(sorted(
                    [*settled.values(), *adopted.values()], key=lambda r: int(r.event_index)
                ), rate)

        seen_sub_positions: set[int] = set()
        setter = getattr(self._adapter, "set_checkpoint_callback", None)
        if setter is not None:
            setter(on_adapter_checkpoint if checkpoint_cb is not None else None)
        try:
            new_results = self._adapter.generate(
                request=sub_request,
                output_path=tmp_output,
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
            )
        except BaseException:
            # [v3.23.630] Huỷ/lỗi giữa lượt: dọn thư mục segment tạm (lượt tăng dần dài nên
            # hay bị huỷ — trước đây mỗi lần huỷ để lại hàng trăm MB WAV tạm).
            # [v3.23.647] Segment đã dời về thư mục chính ở các điểm kiểm tra thì GIỮ (cache đã trỏ tới chúng).
            shutil.rmtree(tmp_segdir, ignore_errors=True)
            self._safe_unlink(tmp_output)
            raise
        finally:
            if setter is not None:
                setter(None)
        self._safe_unlink(tmp_output)

        # 2. Dời segment mới về thư mục segment CHÍNH với tên ổn định ``seg_<vị trí gốc>.wav`` và remap
        #    ``event_index`` về vị trí gốc để merge đúng (dòng đã dời ở điểm kiểm tra thì giữ đường dẫn mới).
        for new_result in new_results:
            adopt(new_result)
        shutil.rmtree(tmp_segdir, ignore_errors=True)

        # 3. Trộn kết quả mới vào cũ (thay theo vị trí gốc) → danh sách đầy đủ theo thứ tự cũ.
        merged: list[TTSSegmentResult] = [
            prior_by_pos.get(int(r.event_index), r) for r in prior_results
        ]

        # 3. Ghép lại master từ tất cả segment còn tiếng.
        placements = self._load_placements(merged, sample_rate)
        master = remerge_master(placements, sample_rate, total_length_samples)
        self._write_master(master, sample_rate, full_request, output_path)

        # [v3.23.454] Xuất LẠI phụ đề đính kèm để khớp nội dung vừa sửa (trước đây re-TTS chỉ
        # ghi WAV → phụ đề giữ nội dung cũ).
        from subtitles_extractor.application.services.tts_subtitle_export import (
            export_tts_subtitle,
        )
        export_tts_subtitle(merged, output_path, full_request)

        logger.info(
            "TTS lại %d dòng xong → ghép lại master (%d dòng) → %s",
            len(selected_events), len(merged), output_path,
        )
        return merged

    def _remerge(
        self,
        full_request: TTSRequest,
        prior_results: list[TTSSegmentResult],
        output_path: Path,
        sample_rate: int,
        total_length_samples: int,
    ) -> list[TTSSegmentResult]:
        """[v3.23.625] Ghép lại master + xuất lại phụ đề từ kết quả hiện có (không TTS)."""
        merged = list(prior_results)
        placements = self._load_placements(merged, sample_rate, allow_empty=True)
        master = remerge_master(placements, sample_rate, total_length_samples)
        self._write_master(master, sample_rate, full_request, output_path)
        from subtitles_extractor.application.services.tts_subtitle_export import (
            export_tts_subtitle,
        )
        export_tts_subtitle(merged, output_path, full_request)
        logger.info("Ghép lại master (%d dòng, không TTS) → %s", len(merged), output_path)
        return merged

    @staticmethod
    def _load_placements(
        results: list[TTSSegmentResult], sample_rate: int, allow_empty: bool = False
    ) -> list[SegmentPlacement]:
        """Đọc audio từng segment WAV, đặt tại mốc đọc thật của dòng."""
        import soundfile as sf

        placements: list[SegmentPlacement] = []
        for result in results:
            if result.was_skipped or not result.segment_audio_path:
                continue
            path = Path(result.segment_audio_path)
            if not path.exists():
                logger.warning("Thiếu segment %s (bỏ qua khi ghép).", path)
                continue
            try:
                audio, seg_sr = sf.read(str(path), dtype="float32")
            except (OSError, RuntimeError) as exc:
                logger.warning("Không đọc được segment %s: %s", path, exc)
                continue
            if audio.ndim > 1:  # phòng stereo → lấy kênh trái
                audio = audio[:, 0]
            if seg_sr != sample_rate:
                logger.warning(
                    "Segment %s có sr=%d ≠ %d — vẫn đặt theo mẫu (có thể lệch nhẹ).",
                    path, seg_sr, sample_rate,
                )
            start = (
                result.adjusted_start_sec
                if result.adjusted_start_sec >= 0
                else result.start_sec
            )
            placements.append(SegmentPlacement(start_sec=start, audio=audio))
        if not placements and not allow_empty:
            raise ValueError("Không có segment nào để ghép lại master.")
        return placements

    @staticmethod
    @without_torch_block
    def _write_master(
        master: np.ndarray, sample_rate: int, request: TTSRequest, output_path: Path
    ) -> None:
        """Chuẩn hoá (nếu bật) + ghi master WAV — tái dùng mastering như đường TTS chính."""
        from subtitles_extractor.infrastructure.tts import audio_mastering as mastering

        if getattr(request, "normalize", False):
            master = mastering.master_finalize(
                master,
                sample_rate,
                target_lufs=getattr(request, "target_lufs", -16.0),
                apply_clarity=getattr(request, "voice_clarity", True),
            )
        else:
            master = mastering.soft_limit(master)
        # [v3.23.630] Ghi theo ĐỊNH DẠNG đã chọn (FLAC/MP3/M4A…) như lượt đầy đủ — trước đây
        # luôn ghi WAV PCM dù tên file là .mp3/.m4a nên TTS lại dòng chọn hỏng file xuất.
        mastering.write_audio(
            master, sample_rate, output_path,
            fmt=getattr(request, "output_format", "wav") or "wav",
            subtype=getattr(request, "wav_subtype", "PCM_16"),
            bitrate_kbps=getattr(request, "output_bitrate_kbps", 320),
        )

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


__all__ = ["ResynthesizeLinesUseCase"]
