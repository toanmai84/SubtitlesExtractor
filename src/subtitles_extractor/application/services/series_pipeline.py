"""[v3.23.658] Dây chuyền phim bộ — phần THUẦN: trạng thái từng tập theo khâu, kế hoạch "Chạy tiếp tất cả", trạng thái lượt chạy.

Vì sao: mỗi khâu (Trích xuất, Dịch, TTS, Xuất bản) có chế độ "cả bộ" riêng; với bộ 86 tập người dùng phải tự chạy 4 lượt
hàng loạt và tự nhớ tập nào đang ở khâu nào. Module này:

* đọc TỆP có sẵn cạnh từng video (cùng quy ước tên tệp với các khâu "cả bộ") → trạng thái từng tập, không cần CSDL;
* lập kế hoạch chạy THEO KHÂU (trích xuất mọi tập thiếu → dịch mọi tập thiếu → …): model OCR/TTS nạp một lần cho cả bộ,
  khâu Dịch tự dừng êm khi hết hạn mức ngày;
* mặc định DỪNG SAU KHÂU DỊCH để soát người nói/bản dịch trước khi TTS (theo yêu cầu của chủ dự án);
* lưu trạng thái lượt chạy cạnh các tập để tắt app giữa chừng vẫn "chạy tiếp" được — kết quả của từng khâu đã nằm trong
  tệp nên chạy lại tự bỏ qua phần đã xong.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from subtitles_extractor.application.services.batch_publish_plan import (
    VIDEO_SUFFIXES,
    find_episode_videos,
    find_tts_audio,
)
from subtitles_extractor.application.services.batch_translate_plan import (
    find_existing_translation,
    find_source_subtitle,
)

logger = logging.getLogger(__name__)

#: Tệp trạng thái lượt chạy, đặt trong thư mục phim bộ.
STATE_FILENAME = ".subtitles_extractor_series.json"
STATE_VERSION = 1

#: Hậu tố tên tệp mà khâu Xuất bản có thể tạo (xem ``PublishPage._current_output_suffix``).
PUBLISH_SUFFIXES: tuple[str, ...] = (
    "_phude_thuyetminh", "_phude_tiengviet", "_phudechay_thuyetminh", "_phudechay_tiengviet",
    "_phude", "_phudechay", "_thuyetminh", "_tiengviet", "_xuatban",
)


class SeriesStage(StrEnum):
    """Các khâu của dây chuyền, theo thứ tự chạy."""

    EXTRACT = "extract"
    TRANSLATE = "translate"
    TTS = "tts"
    PUBLISH = "publish"


STAGE_ORDER: tuple[SeriesStage, ...] = (
    SeriesStage.EXTRACT, SeriesStage.TRANSLATE, SeriesStage.TTS, SeriesStage.PUBLISH,
)


class StageState(StrEnum):
    """Trạng thái một khâu của một tập."""

    DONE = "done"          # đã có tệp kết quả
    READY = "ready"        # khâu trước đã xong — chạy được
    BLOCKED = "blocked"    # khâu trước chưa xong
    FAILED = "failed"      # lượt chạy gần nhất báo lỗi ở khâu này (vẫn chạy lại được)


@dataclass(frozen=True, slots=True)
class EpisodeStatus:
    """Trạng thái một tập: khâu → trạng thái, khâu → tệp kết quả (nếu có)."""

    video: Path
    states: dict[SeriesStage, StageState]
    outputs: dict[SeriesStage, Path | None]

    @property
    def name(self) -> str:
        return self.video.name

    def next_stage(self) -> SeriesStage | None:
        """Khâu đầu tiên chưa xong (``None`` = đã xong cả bốn khâu)."""
        for stage in STAGE_ORDER:
            if self.states[stage] is not StageState.DONE:
                return stage
        return None


def find_published_output(video: Path, suffixes: Iterable[str] = PUBLISH_SUFFIXES) -> Path | None:
    """Tệp đã xuất bản của tập (``<tên><hậu tố>.<đuôi video>``), hoặc ``None``."""
    try:
        siblings = {p.name.lower(): p for p in video.parent.iterdir() if p.is_file()}
    except OSError:
        return None
    for suffix in suffixes:
        for ext in (".mkv", *sorted(VIDEO_SUFFIXES - {".mkv"})):
            found = siblings.get(f"{video.stem}{suffix}{ext}".lower())
            if found is not None:
                return found
    return None


def scan_episode(
    video: Path, *, target_language: str = "vi", publish_suffixes: Iterable[str] = PUBLISH_SUFFIXES,
    failures: dict[str, str] | None = None,
) -> EpisodeStatus:
    """Trạng thái một tập theo tệp có sẵn cạnh video.

    Args:
        video: Video nguồn.
        target_language: Ngôn ngữ đích của bản dịch.
        publish_suffixes: Hậu tố tệp xuất bản được coi là "đã xuất bản".
        failures: ``{khâu: thông báo}`` lỗi của lượt chạy gần nhất (khâu chưa có tệp kết quả thì hiện "lỗi").
    """
    source = find_source_subtitle(video)
    translation = find_existing_translation(video, target_language)
    # Cùng tiêu chí với TTS "cả bộ" (có giọng đọc là xong) — khác tiêu chí thì tập có audio mà thiếu ``.tts.srt``
    # bị dây chuyền chạy lại mãi trong khi kế hoạch TTS luôn bỏ qua nó.
    tts_ok = find_tts_audio(video) is not None
    published = find_published_output(video, publish_suffixes)
    outputs: dict[SeriesStage, Path | None] = {
        SeriesStage.EXTRACT: source,
        SeriesStage.TRANSLATE: translation,
        SeriesStage.TTS: find_tts_audio(video) if tts_ok else None,
        SeriesStage.PUBLISH: published,
    }
    done = {
        SeriesStage.EXTRACT: source is not None,
        SeriesStage.TRANSLATE: translation is not None,
        SeriesStage.TTS: tts_ok,
        SeriesStage.PUBLISH: published is not None,
    }
    states: dict[SeriesStage, StageState] = {}
    previous_done = True
    for stage in STAGE_ORDER:
        if done[stage]:
            states[stage] = StageState.DONE
        elif failures and stage.value in failures:
            states[stage] = StageState.FAILED
        elif previous_done:
            states[stage] = StageState.READY
        else:
            states[stage] = StageState.BLOCKED
        previous_done = done[stage]
    return EpisodeStatus(video=video, states=states, outputs=outputs)


def scan_series(
    folder: Path, *, target_language: str = "vi", state: SeriesRunState | None = None,
    publish_suffixes: Iterable[str] = PUBLISH_SUFFIXES,
) -> list[EpisodeStatus]:
    """Trạng thái mọi tập trong thư mục (sắp theo tên).

    Args:
        publish_suffixes: [v3.23.659] Hậu tố tệp xuất bản coi là "xong" — dây chuyền truyền đúng hậu tố của chế độ
            đang chọn ở trang Xuất bản (bản chỉ có phụ đề không được coi là đã xuất bản bản thuyết minh).
    """
    failures = state.failures if state is not None else {}
    suffixes = tuple(publish_suffixes)
    return [
        scan_episode(video, target_language=target_language, failures=failures.get(video.name),
                     publish_suffixes=suffixes)
        for video in find_episode_videos(Path(folder))
    ]


def stages_to_run(stop_after: SeriesStage) -> tuple[SeriesStage, ...]:
    """Các khâu sẽ chạy với điểm dừng ``stop_after`` (gồm cả khâu đó)."""
    return STAGE_ORDER[: STAGE_ORDER.index(stop_after) + 1]


def episodes_for_stage(statuses: Sequence[EpisodeStatus], stage: SeriesStage) -> list[Path]:
    """Các tập cần chạy ``stage`` NGAY BÂY GIỜ (khâu trước đã xong, khâu này chưa)."""
    return [
        status.video for status in statuses
        if status.states[stage] in (StageState.READY, StageState.FAILED) and _previous_done(status, stage)
    ]


def _previous_done(status: EpisodeStatus, stage: SeriesStage) -> bool:
    index = STAGE_ORDER.index(stage)
    return index == 0 or status.states[STAGE_ORDER[index - 1]] is StageState.DONE


def remaining_work(statuses: Sequence[EpisodeStatus], stop_after: SeriesStage) -> dict[SeriesStage, int]:
    """Số tập còn phải chạy ở từng khâu tới điểm dừng (kể cả tập đang kẹt ở khâu trước — sẽ chạy sau khi khâu đó xong)."""
    return {
        stage: sum(1 for status in statuses if status.states[stage] is not StageState.DONE)
        for stage in stages_to_run(stop_after)
    }


def first_stage_with_work(
    statuses: Sequence[EpisodeStatus], stop_after: SeriesStage, after: SeriesStage | None = None,
) -> SeriesStage | None:
    """Khâu kế tiếp (sau ``after``) có tập chạy được; ``None`` = đã tới điểm dừng / không còn gì chạy được."""
    stages = stages_to_run(stop_after)
    start = 0 if after is None else stages.index(after) + 1 if after in stages else len(stages)
    for stage in stages[start:]:
        if episodes_for_stage(statuses, stage):
            return stage
    return None


# ── Trạng thái lượt chạy (bền qua lần tắt app) ───────────────────────────────


@dataclass(slots=True)
class SeriesRunState:
    """Lượt "Chạy tiếp tất cả" đang dở của một thư mục phim bộ.

    Attributes:
        stop_after: Khâu dừng lại (mặc định Dịch).
        running: ``True`` khi lượt chưa kết thúc (tắt app giữa chừng → mở lại thấy "chạy dở").
        stage: Khâu đang chạy (rỗng nếu chưa bắt đầu).
        failures: ``{tên video: {khâu: thông báo lỗi}}`` của lượt gần nhất.
        updated_at: Mốc ``time.time()`` lần ghi cuối.
    """

    stop_after: SeriesStage = SeriesStage.TRANSLATE
    running: bool = False
    stage: str = ""
    failures: dict[str, dict[str, str]] = field(default_factory=dict)
    updated_at: float = 0.0

    def record_failures(self, stage: SeriesStage, names: Iterable[str], message: str = "") -> None:
        for name in names:
            self.failures.setdefault(name, {})[stage.value] = message or "lỗi"

    def clear_failures(self, stage: SeriesStage, names: Iterable[str]) -> None:
        for name in names:
            per = self.failures.get(name)
            if per is not None:
                per.pop(stage.value, None)
                if not per:
                    del self.failures[name]


def state_path(folder: Path) -> Path:
    return Path(folder) / STATE_FILENAME


def load_run_state(folder: Path) -> SeriesRunState:
    """Đọc trạng thái lượt chạy; tệp thiếu / hỏng → trạng thái mặc định (không lỗi)."""
    try:
        data = json.loads(state_path(folder).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return SeriesRunState()
    if not isinstance(data, dict) or int(data.get("version", 0) or 0) != STATE_VERSION:
        return SeriesRunState()
    try:
        stop_after = SeriesStage(str(data.get("stop_after", SeriesStage.TRANSLATE.value)))
    except ValueError:
        stop_after = SeriesStage.TRANSLATE
    raw_failures = data.get("failures", {})
    failures: dict[str, dict[str, str]] = {}
    if isinstance(raw_failures, dict):
        for name, per in raw_failures.items():
            if isinstance(per, dict):
                kept = {str(k): str(v) for k, v in per.items() if str(k) in {s.value for s in STAGE_ORDER}}
                if kept:
                    failures[str(name)] = kept
    stage = str(data.get("stage", "") or "")
    return SeriesRunState(
        stop_after=stop_after, running=bool(data.get("running", False)),
        stage=stage if stage in {s.value for s in STAGE_ORDER} else "",
        failures=failures, updated_at=float(data.get("updated_at", 0.0) or 0.0),
    )


def save_run_state(folder: Path, state: SeriesRunState, *, clock: object = time.time) -> bool:
    """Ghi trạng thái (ghi tệp tạm rồi đổi tên — tắt máy giữa chừng không để tệp hỏng). Lỗi ghi chỉ cảnh báo."""
    state.updated_at = float(clock())  # type: ignore[operator]
    payload = {
        "version": STATE_VERSION, "stop_after": state.stop_after.value, "running": state.running,
        "stage": state.stage, "failures": state.failures, "updated_at": state.updated_at,
    }
    target = state_path(folder)
    tmp = target.with_name(target.name + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, target)
    except OSError as exc:
        logger.warning("Không ghi được trạng thái dây chuyền phim bộ (%s): %s", target, exc)
        return False
    return True


__all__ = [
    "PUBLISH_SUFFIXES", "STAGE_ORDER", "STATE_FILENAME", "EpisodeStatus", "SeriesRunState", "SeriesStage",
    "StageState", "episodes_for_stage", "find_published_output", "first_stage_with_work", "load_run_state",
    "remaining_work", "save_run_state", "scan_episode", "scan_series", "stages_to_run", "state_path",
]
