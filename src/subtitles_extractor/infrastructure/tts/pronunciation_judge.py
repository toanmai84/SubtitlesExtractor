"""[v3.23.679] Gemini nghe lại các câu lồng tiếng (theo lô 10 câu) và chép lời — đầu vào cho kiểm tra phát âm.

Cùng cách chấm với công cụ đo ``bm/vn/judge_vn.py`` (gemini-3.1-flash-lite, nhiệt độ 0, trả JSON), nơi đã chấm 296 câu
VieNeu (CER trung bình 0,036; 20 câu > 0,15). Xoay key khi gặp 429/503 như các lời gọi Gemini khác.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from subtitles_extractor.application.services.pronunciation_check import (
    LineToCheck,
    PronunciationVerdict,
    char_error_rate,
    judge_prompt,
    parse_heard,
)

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "gemini-3.1-flash-lite"
#: [v3.23.679] Model dự phòng khi model chính quá tải (đo 04/11: 3.1-flash-lite trả 503 liên tục, 3.5-flash chép đúng).
FALLBACK_JUDGE_MODEL = "gemini-3.5-flash"
#: Chuỗi model mặc định: 3.5-flash-lite có hạn mức miễn phí lớn (500 lượt/ngày) — đo 04/11 còn dùng được trên cả 3 key
#: khi 3.1-flash-lite quá tải và 3.5-flash hết hạn mức ở một key.
DEFAULT_JUDGE_CHAIN = (DEFAULT_JUDGE_MODEL, "gemini-3.5-flash-lite", FALLBACK_JUDGE_MODEL)
BATCH_SIZE = 10
_ATTEMPTS_PER_MODEL = 3
#: Giới hạn chờ mỗi lời gọi (ms). SDK mặc định chờ VÔ HẠN: đo thật một lô treo > 10 phút.
_REQUEST_TIMEOUT_MS = 120_000


class PronunciationJudgeError(RuntimeError):
    """Không chấm được (hết key / lỗi mạng kéo dài) — thông điệp hiển thị cho người dùng."""


#: Tần số gửi cho AI: đủ nghe lời nói, tệp nhỏ hơn ~6 lần so với WAV 48 kHz float của VieNeu.
_SEND_RATE = 16000


def _mime_for(path: Path) -> str:
    return {".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg"}.get(path.suffix.lower(), "audio/wav")


def compact_audio(path: Path) -> tuple[bytes, str]:
    """Âm thanh gửi đi: mono 16 kHz PCM 16 bit (WAV). Đọc không được thì gửi nguyên tệp.

    [v3.23.679] Đo thật: lô 10 câu VieNeu (WAV 48 kHz float ~2 MB) gửi nguyên bản chậm và hay quá hạn chờ.
    """
    import io

    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly

        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        if rate != _SEND_RATE:
            from math import gcd

            g = gcd(int(rate), _SEND_RATE)
            mono = resample_poly(mono, _SEND_RATE // g, int(rate) // g).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, np.clip(mono, -1.0, 1.0), _SEND_RATE, format="WAV", subtype="PCM_16")
        return buf.getvalue(), "audio/wav"
    except Exception:  # noqa: BLE001 — định dạng lạ / thiếu thư viện: gửi nguyên tệp, Gemini vẫn đọc được
        return path.read_bytes(), _mime_for(path)


class GeminiPronunciationJudge:
    """Chép lời từng câu bằng Gemini. ``client_factory(key)`` trả client kiểu ``google.genai.Client`` (test giả được)."""

    def __init__(
        self,
        api_keys: Sequence[str],
        models: Sequence[str] = DEFAULT_JUDGE_CHAIN,
        *,
        client_factory: Callable[[str], Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._keys = [k for k in api_keys if k]
        if not self._keys:
            raise PronunciationJudgeError("Cần key Gemini (Cài đặt → Khoá API và AI) để kiểm tra phát âm.")
        self._models = list(dict.fromkeys(m for m in models if m)) or [DEFAULT_JUDGE_MODEL]
        #: Lý do dừng sớm (hết hạn mức…) khi vẫn trả được một phần kết quả; rỗng = chấm đủ.
        self.stop_reason = ""
        self._factory = client_factory or self._default_factory
        self._clients: dict[str, Any] = {}
        self._sleep = sleep
        self._turn = 0
        self._dead_models: set[str] = set()
        self._lock = threading.Lock()

    @staticmethod
    def _default_factory(key: str) -> Any:
        from google import genai
        from google.genai import types

        return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS))

    def _client(self, key: str) -> Any:
        with self._lock:
            if key not in self._clients:        # giữ client: client tạm bị thu gom thì lời gọi sau báo "client closed"
                self._clients[key] = self._factory(key)
            return self._clients[key]

    def check(
        self,
        lines: Sequence[LineToCheck],
        *,
        progress: Callable[[int, int], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> list[PronunciationVerdict]:
        """Chấm ``lines`` theo lô (song song mỗi key một lô); câu Gemini không trả lời thì bỏ qua (không coi là sai).

        [v3.23.679] Đo thật 237 câu VieNeu: chạy tuần tự + model chính quá tải → 30 phút. Nay song song tối đa
        ``len(keys)`` lô (≤ 3), model hỏng cả lô thì bỏ cho các lô sau. Câu bị chấm sai được chấm LẠI RIÊNG để loại lỗi
        lệch đoạn trong lô (đo thật: lời câu 109 bị gán sang câu 108).
        """
        batches = [list(lines[i:i + BATCH_SIZE]) for i in range(0, len(lines), BATCH_SIZE)]
        done = 0
        verdicts: list[PronunciationVerdict] = []

        stop = threading.Event()

        def run(batch: list[LineToCheck]) -> list[PronunciationVerdict]:
            if stop.is_set() or (cancelled is not None and cancelled()):
                return []
            try:
                return self._judge(batch)
            except PronunciationJudgeError as exc:
                # Hết hạn mức: dừng các lô còn lại nhưng GIỮ phần đã chấm (bản đầu ném lỗi → mất cả 200 câu đã chấm).
                self.stop_reason = str(exc)
                stop.set()
                return []

        with ThreadPoolExecutor(max_workers=max(1, min(3, len(self._keys)))) as pool:
            for result, batch in zip(pool.map(run, batches), batches, strict=True):
                verdicts.extend(result)
                done += len(batch)
                if progress is not None:
                    progress(min(len(lines), done), len(lines))
        if self.stop_reason and not verdicts:
            raise PronunciationJudgeError(self.stop_reason)
        if stop.is_set() or (cancelled is not None and cancelled()):
            return verdicts
        # Xác nhận: chấm lại riêng từng câu bị chấm sai, lấy kết quả riêng.
        by_index = {line.event_index: line for line in lines}
        confirmed: list[PronunciationVerdict] = []
        for verdict in verdicts:
            if verdict.wrong and verdict.event_index in by_index:
                try:
                    single = self._judge([by_index[verdict.event_index]])
                except PronunciationJudgeError:
                    single = []
                confirmed.append(single[0] if single else verdict)
            else:
                confirmed.append(verdict)
        return confirmed

    def _judge(self, batch: list[LineToCheck]) -> list[PronunciationVerdict]:
        heard = self._transcribe(batch)
        return [
            PronunciationVerdict(line.event_index, line.text, heard[offset],
                                 round(char_error_rate(line.text, heard[offset]), 3))
            for offset, line in enumerate(batch) if offset in heard
        ]

    def _transcribe(self, batch: list[LineToCheck]) -> dict[int, str]:
        from google.genai import types

        parts: list[Any] = [judge_prompt(len(batch))]
        for offset, line in enumerate(batch):
            data, mime = compact_audio(Path(line.audio_path))
            parts += [f"Đoạn {offset}:", types.Part.from_bytes(data=data, mime_type=mime)]
        config = types.GenerateContentConfig(temperature=0, response_mime_type="application/json")
        last_error = ""
        with self._lock:
            turn = self._turn
            self._turn += 1
        for model in self._models:
            if model in self._dead_models and model != self._models[-1]:
                continue
            for attempt in range(max(_ATTEMPTS_PER_MODEL, len(self._keys))):   # thử mọi key (hạn mức tính theo key)
                key = self._keys[(turn + attempt) % len(self._keys)]
                try:
                    response = self._client(key).models.generate_content(model=model, contents=parts, config=config)
                    heard = parse_heard(json.loads(response.text or "{}"), len(batch))
                except (ValueError, json.JSONDecodeError) as exc:
                    last_error = f"JSON hỏng: {exc}"
                    continue
                except Exception as exc:  # noqa: BLE001 — lỗi SDK (429/503/mạng/quá hạn) các loại: thử key/model khác
                    last_error = str(exc)[:200]
                    self._sleep(min(30.0, 2.0 * (attempt + 1)))
                    continue
                if len(heard) >= len(batch) - 1:     # thiếu tối đa 1 câu: chấp nhận (như bộ chấm đo)
                    return heard
                last_error = f"chỉ trả {len(heard)}/{len(batch)} câu"
            logger.info("Kiểm tra phát âm: %s không dùng được (%s) — thử model kế.", model, last_error)
            if "503" in last_error or "UNAVAILABLE" in last_error:
                with self._lock:
                    self._dead_models.add(model)        # quá tải / hết hạn mức: các lô sau đi thẳng model kế
        if last_error:
            logger.warning("Kiểm tra phát âm: bỏ qua lô %d câu (%s).", len(batch), last_error)
        if "429" in last_error or "RESOURCE_EXHAUSTED" in last_error:
            raise PronunciationJudgeError(
                f"Hết hạn mức Gemini khi kiểm tra phát âm ({', '.join(self._models)}). Thử lại sau.")
        return {}
