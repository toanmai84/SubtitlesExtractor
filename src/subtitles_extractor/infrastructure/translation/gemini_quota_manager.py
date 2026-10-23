"""Quản lý hạn mức (quota) gọi Gemini ở phía client để tránh vượt rate-limit.

Mô phỏng & đặt chỗ trước (reservation) theo ba giới hạn của Gemini:
* RPM — số request mỗi phút,
* TPM — số token mỗi phút,
* RPD — số request mỗi ngày.

Cơ chế: trước mỗi lần gọi, :meth:`acquire` ước lượng token và **đặt chỗ**; nếu sắp
vượt RPM/TPM thì tự chờ (điều tiết) đến khi cửa sổ trượt 60 giây giải phóng đủ chỗ;
nếu chạm RPD thì báo lỗi. Sau khi có phản hồi, :meth:`reconcile` cập nhật token thực
tế (thay cho ước lượng), còn :meth:`release` nhả chỗ khi gọi thất bại.

Port gọn từ ứng dụng dịch của người dùng: bỏ phần ghi đĩa QSettings, giữ lõi điều
tiết thuần (in-memory, thread-safe) — dễ kiểm thử và tái dùng.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from subtitles_extractor.infrastructure.translation.gemini_quota_clock import (
    describe_quota_reset,
    quota_day,
)

logger = logging.getLogger(__name__)

CancelCallback = Callable[[], bool]
WaitCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class RateLimit:
    """Giới hạn của một model Gemini."""

    rpm: int   # requests / phút
    tpm: int   # tokens / phút
    rpd: int   # requests / ngày


@dataclass(frozen=True)
class QuotaReservation:
    """Một chỗ token đã đặt trước cho một lần gọi."""

    model_key: str
    reservation_id: int
    reserved_tokens: int


MAX_SERVER_COOLDOWN_SEC = 180.0
"""[v3.23.534] Trần thời gian tôn trọng ``retryDelay`` của server (giây).

Chờ lâu hơn mức này thì ứng dụng trông như bị treo, trong khi thử lại sớm chỉ tốn một request để
biết server đã cho phép hay chưa."""


class QuotaExhaustedError(RuntimeError):
    """Đã chạm giới hạn ngày (RPD) — không thể chờ trong ngày."""


#: [v3.23.641] Dấu "key hỏng" tồn tại tối đa chừng này giây trong một phiên chạy.
INVALID_KEY_TTL_S = 600.0


def describe_wait(seconds: float) -> str:
    """Thời lượng dạng "10 giờ 12 phút" / "45 phút" (làm tròn xuống phút, tối thiểu 1).

    Args:
        seconds: Số giây.

    Returns:
        Chuỗi tiếng Việt cho người dùng đọc.
    """
    minutes = max(1, int(seconds // 60))
    hours, minutes = divmod(minutes, 60)
    if hours and minutes:
        return f"{hours} giờ {minutes} phút"
    return f"{hours} giờ" if hours else f"{minutes} phút"


def daily_quota_message(model_name: str, now_utc: datetime | None = None) -> str:
    """[v3.23.586] Thông báo khi SERVER từ chối vì hết hạn mức ngày và không còn key nào.

    [v3.23.607] Mốc hồi tính theo 00:00 giờ Thái Bình Dương (tài liệu Gemini API,
    khớp nhật ký thật), KHÔNG theo ``retryDelay`` server gửi kèm: con số đó luôn chỉ
    về 00:00 UTC (07:00 giờ Việt Nam) trong khi hạn mức hồi lúc 14:00/15:00 giờ Việt
    Nam. Thông báo cũ "Server báo hạn mức hồi sau khoảng 26 phút" lúc 06:33 sáng
    12/10 là sai — 11:06 vẫn bị từ chối. Xem :mod:`.gemini_quota_clock`.

    Args:
        model_name: Model vừa bị từ chối.
        now_utc: Thời điểm; ``None`` = bây giờ.

    Returns:
        Câu thông báo kèm giờ hạn mức hồi theo giờ máy.
    """
    return (
        f"Model '{model_name}' đã hết hạn mức request/ngày ở mọi API key "
        f"(server xác nhận). {describe_quota_reset(now_utc)}"
    )

# [v3.23.17] Giới hạn free tier Gemini (sau đợt cắt giảm 12/2025). Nguồn: tài liệu
# Google AI + tổng hợp cộng đồng (cập nhật ~Q1-Q2 2026). LƯU Ý:
#   * TPM dùng chung 250.000 cho hầu hết model free tier (KHÔNG phải 1.000.000).
# [v3.23.36] Cập nhật theo BẢNG QUOTA THỰC TẾ (ảnh tài khoản, tháng 6/2026):
#   gemini-3.1-flash-lite : RPM 15 · TPM 250K · RPD 500
#   gemini-3.5-flash-lite : RPD 500 (người dùng xác nhận 10/2026; tài liệu Google KHÔNG công bố RPD
#                           theo model — chỉ hiện trong AI Studio; RPM/TPM giữ như 3.1, và con số
#                           ngày THẬT được học từ lỗi 429 — xem ``learn_daily_limit_from_server``)
#   gemini-2.5-flash-lite : RPM 10 · TPM 250K · RPD 20
#   gemini-3.5/3/2.5-flash: RPM  5 · TPM 250K · RPD 20
#   *-pro (2.5/3.1)       : 0/0 — KHÔNG còn free
# Vì 'flash-lite' gồm cả 3.1 (RPD 500) lẫn 2.5 (RPD 20), lấy giá trị BẢO THỦ (thấp
# hơn) để KHÔNG vượt quota của model yếu nhất trong nhóm — tránh 429 bất ngờ.
#   * Quota tính theo PROJECT, reset 00:00 giờ Thái Bình Dương (KHÔNG phải UTC) — xem
#     ``gemini_quota_clock`` (v3.23.607: đã kiểm lại bằng nhật ký thật).
#   * RPD là ràng buộc dễ chạm nhất với tác vụ dài.
# Khớp theo tiền tố tên model (vd 'gemini-flash-lite-latest', 'gemini-3.5-flash').
_FREE_TIER_LIMITS: dict[str, RateLimit] = {
    "flash-lite-hi": RateLimit(rpm=15, tpm=250_000, rpd=500),  # 3.1/3.5+-flash-lite
    "flash-lite": RateLimit(rpm=10, tpm=250_000, rpd=20),       # 2.5-flash-lite (bảo thủ)
    "flash": RateLimit(rpm=5, tpm=250_000, rpd=20),             # *-flash
    "pro": RateLimit(rpm=5, tpm=250_000, rpd=20),               # pro không free → giữ thấp
}

# Mặc định an toàn (bảo thủ) khi không khớp model nào.
_DEFAULT_LIMIT = RateLimit(rpm=5, tpm=250_000, rpd=20)


def _flash_lite_generation(lowered_name: str) -> float:
    """Số đời của model flash-lite (``gemini-3.5-flash-lite`` → 3.5; ``gemini-3-1-…`` → 3.1); 0 nếu lạ."""
    match = re.search(r"gemini-(\d+)(?:[.\-](\d+))?", lowered_name)
    if not match:
        return 0.0
    return float(f"{match.group(1)}.{match.group(2) or 0}")


def _match_free_tier_limit(model_name: str) -> RateLimit:
    """Suy giới hạn free tier theo TÊN model (khớp tiền tố đặc trưng).

    Ưu tiên 'flash-lite' trước 'flash' (vì 'flash-lite' chứa 'flash'). Trong nhóm
    flash-lite, 3.1 có RPD cao (500); các flash-lite khác bảo thủ (RPD 20).
    """
    name = model_name.lower()
    if "flash-lite" in name or "flash_lite" in name:
        # [v3.23.635] flash-lite đời 3.x (3.1, 3.5, ...) có RPD cao (500); 2.5-flash-lite thấp (20).
        # Trước đây chỉ "3.1" được coi là cao nên 3.5-flash-lite bị ứng dụng tự giới hạn 20 request/ngày
        # dù tài khoản được 500.
        if _flash_lite_generation(name) >= 3.0:
            return _FREE_TIER_LIMITS["flash-lite-hi"]
        return _FREE_TIER_LIMITS["flash-lite"]
    if "flash" in name:
        return _FREE_TIER_LIMITS["flash"]
    if "pro" in name:
        return _FREE_TIER_LIMITS["pro"]
    return _DEFAULT_LIMIT


def _same_label(label: str, _quota_period: str) -> str:
    """Nhãn ngày đã đúng cách tính hiện hành: giữ nguyên."""
    return label


def _relabel_utc_day(label: str, quota_period: str) -> str:
    """[v3.23.607] Đổi nhãn ngày UTC (bản v3.23.606) sang ngày hạn mức hiện tại, nếu chắc.

    Thời điểm thuộc ngày UTC ``D`` rơi vào ngày Thái Bình Dương ``D-1`` (trước
    07:00/08:00 UTC) hoặc ``D``. Khi ngày hạn mức hiện tại là ``D-1`` thì mọi bản
    ghi nhãn ``D`` chắc chắn thuộc ``D-1`` (chưa thể có bản ghi nào của ngày ``D``)
    → giữ, đổi nhãn. Các trường hợp khác không phân biệt được → bỏ; key nào thật sự
    cạn thì server báo lại ngay.

    Args:
        label: Nhãn ngày UTC đã lưu.
        quota_period: Ngày hạn mức hiện tại (theo giờ Thái Bình Dương).

    Returns:
        ``quota_period`` nếu bản ghi thuộc ngày hạn mức hiện tại; chuỗi rỗng nếu bỏ.
    """
    try:
        utc_day = date.fromisoformat(label)
        current = date.fromisoformat(quota_period)
    except ValueError:
        return ""
    return quota_period if utc_day - timedelta(days=1) == current else ""


class GeminiQuotaManager:
    """Điều tiết quota Gemini phía client, an toàn nhiều luồng."""

    def __init__(
        self,
        rate_limits: dict[str, RateLimit] | None = None,
        default_limit: RateLimit = _DEFAULT_LIMIT,
        state_path: Path | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._limits: dict[str, RateLimit] = {
            k.lower(): v for k, v in (rate_limits or {}).items()
        }
        self._default_limit = default_limit
        self._request_windows: dict[str, deque[dict]] = defaultdict(deque)
        self._token_windows: dict[str, deque[dict]] = defaultdict(deque)
        self._token_totals: dict[str, int] = defaultdict(int)
        # [v3.23.141] Cooldown theo (key,model) khi server trả 429 kèm retryDelay: chặn MỌI
        # request tiếp theo tới cùng (key,model) đến hết retryDelay, thay vì để chúng lao
        # vào rồi lại 429 (server tính TPM/RPM chặt hơn ước lượng phía client).
        # [v3.23.526] (model,key) mà SERVER đã xác nhận hết quota ngày, theo từng mốc ngày.
        self._server_daily_block: dict[str, str] = {}
        # [v3.23.533] Hạn mức HỌC ĐƯỢC từ server, tách khỏi cấu hình người dùng (``_limits``) để
        # không ghi đè thứ người dùng cố ý đặt, và để lưu/nạp riêng qua tệp trạng thái.
        self._learned_limits: dict[str, RateLimit] = {}
        # [v3.23.639] RPM/TPM HỌC được từ server (``quotaId …PerMinute…``): ``{model: (rpm, tpm)}``,
        # 0 = chưa học. Tách riêng để không đụng ``_learned_limits`` (RPD) đang dùng ở nhiều nơi.
        self._learned_minute: dict[str, tuple[int, int]] = {}
        # Đã cảnh báo "bộ đếm chạm trần" cho (model,key) nào — chỉ log một lần cho đỡ nhiễu.
        self._soft_daily_warned: dict[str, bool] = {}
        self._cooldown_until: dict[str, float] = defaultdict(float)
        # [v3.23.639] Key bị server từ chối vĩnh viễn (không hợp lệ / hết hạn / bị vô hiệu hoá vì lộ
        # key): ``{vân tay: (lý do, nhãn ngày)}``. Lưu qua phiên trong ngày — key hỏng không tự lành,
        # sang ngày mới thì thử lại một lần (người dùng có thể đã thay key).
        self._invalid_keys: dict[str, tuple[str, str]] = {}
        # [v3.23.641] Thời điểm (``time.time()``) đánh dấu hỏng: dấu hết hạn sau ``INVALID_KEY_TTL_S`` để
        # key đã được sửa phía Google được thử lại (một request), thay vì bị bỏ qua cả ngày.
        self._invalid_marked_at: dict[str, float] = {}
        self._daily: dict[str, dict] = {}
        self._counter = 0
        # [v3.23.123] Lưu bộ đếm request/NGÀY qua các phiên: restart không reset count ảo
        # (server vẫn nhớ quota theo ngày). Chỉ nạp lại mục CÙNG NGÀY hiện tại.
        self._state_path = Path(state_path) if state_path else None
        self._load_state()

    def _load_state(self) -> None:
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Không đọc được trạng thái quota đã lưu: %s", exc)
            return
        if not isinstance(data, dict):
            # [v3.23.641] JSON hợp lệ nhưng không phải object (tệp sửa tay / hỏng): trước đây
            # ``.pop`` ném TypeError/AttributeError ngay trong hàm dựng → ứng dụng không mở được.
            logger.warning("Tệp trạng thái quota sai định dạng (%s) — bỏ qua.", type(data).__name__)
            return
        period = self._period_key()
        loaded = 0
        meta = data.pop("__meta__", None)
        # [v3.23.607] Tệp do v3.23.606 ghi: nhãn ngày tính theo 00:00 UTC (sai). Nhãn đó
        # trùng chữ với ngày theo giờ Thái Bình Dương bắt đầu lúc 14:00 giờ Việt Nam — giữ
        # nguyên là chặn oan model cả ngày hạn mức kế tiếp. Xem :func:`_relabel_utc_day`.
        utc_labelled = isinstance(meta, dict) and "daily_reset_offset_s" in meta
        if utc_labelled:
            logger.info(
                "Chuyển bộ đếm quota/ngày do bản 3.23.606 ghi (ngày theo UTC) về ngày "
                "theo giờ Thái Bình Dương."
            )
        accepted = _relabel_utc_day if utc_labelled else _same_label
        if isinstance(meta, dict):
            blocks = meta.get("server_daily_block")
            if isinstance(blocks, dict):
                # Chỉ giữ dấu chặn của ĐÚNG mốc ngày hiện tại — sang ngày mới là bỏ.
                self._server_daily_block = {
                    key: period for key, value in blocks.items()
                    if accepted(str(value), period) == period
                }
            invalid = meta.get("invalid_keys")
            if isinstance(invalid, dict):
                for key_id, item in invalid.items():
                    if (
                        isinstance(item, (list, tuple)) and len(item) == 2
                        and accepted(str(item[1]), period) == period
                    ):
                        self._invalid_keys[str(key_id)] = (str(item[0]), period)
                        self._invalid_marked_at[str(key_id)] = time.time()
            learned_minute = meta.get("learned_minute")
            if isinstance(learned_minute, dict):
                for name, pair in learned_minute.items():
                    try:
                        rpm_value, tpm_value = int(pair[0]), int(pair[1])
                    except (TypeError, ValueError, IndexError, KeyError, OverflowError):
                        continue
                    self._learned_minute[str(name)] = (max(0, rpm_value), max(0, tpm_value))
            learned = meta.get("learned_rpd")
            if isinstance(learned, dict):
                for name, rpd in learned.items():
                    try:
                        value = int(rpd)
                    except (TypeError, ValueError, OverflowError):
                        continue
                    if value > 0:
                        base = self._limits.get(name) or _match_free_tier_limit(name)
                        self._learned_limits[name] = RateLimit(
                            rpm=base.rpm, tpm=base.tpm, rpd=value
                        )
        for model_key, rec in data.items():
            if (
                isinstance(rec, dict)
                and accepted(str(rec.get("period_key")), period) == period
            ):
                try:
                    count = max(0, int(rec.get("count", 0)))
                except (TypeError, ValueError, OverflowError):
                    continue  # [v3.23.641] số đếm hỏng: bỏ riêng mục này, không sập cả khởi động
                self._daily[model_key] = {"period_key": period, "count": count}
                loaded += 1
        if loaded:
            logger.info("Đã khôi phục bộ đếm quota/ngày cho %d (model,key).", loaded)

    def _save_state_locked(self) -> None:
        """Ghi bộ đếm ngày ra đĩa (gọi trong lock, sau khi count thay đổi)."""
        if self._state_path is None:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            # [v3.23.533] Lưu kèm hai thứ trước đây chỉ nằm trong bộ nhớ và mất khi tắt app:
            #  - dấu SERVER đã từ chối key (không lưu → mở lại app là thử lại key đã cạn, đốt
            #    request vô ích, mà hạn mức chỉ 20/ngày);
            #  - hạn mức học được từ server (không lưu → lần sau lại dùng con số chép tay lạc hậu,
            #    khiến bản v3.23.530 gần như vô nghĩa giữa các phiên).
            # Giữ bộ đếm ở CẤP GỐC để tệp cũ/mới đọc lẫn nhau được.
            payload = dict(self._daily)
            payload["__meta__"] = {
                "invalid_keys": {k: list(v) for k, v in self._invalid_keys.items()},
                "learned_minute": {k: list(v) for k, v in self._learned_minute.items()},
                "server_daily_block": dict(self._server_daily_block),
                "learned_rpd": {
                    name: limit.rpd for name, limit in self._learned_limits.items()
                },
            }
            # [v3.23.641] Ghi nguyên tử (tệp tạm + replace): tắt máy giữa chừng không để lại
            # tệp cụt làm mất toàn bộ bộ đếm ngày.
            tmp_path = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp_path, self._state_path)
        except (OSError, ValueError) as exc:
            logger.warning("Không lưu được trạng thái quota: %s", exc)

    @staticmethod
    def key_fingerprint(api_key: str) -> str:
        """[v3.23.124] Vân tay ngắn, KHÔNG đảo ngược của API key (nguồn DUY NHẤT).

        Dùng để khoá quota theo từng key mà không lưu/è lộ key thật. Cả adapter lẫn
        tầng UI đều gọi hàm này để tính cùng một định danh.
        """
        import hashlib
        key = (api_key or "").strip()
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12] if key else ""

    @staticmethod
    def _compose_key(key_id: str, model_name: str) -> str:
        """Khoá đếm quota = (vân tay API key) | (tên model).

        Quota Gemini tính theo PROJECT/API key, nên mỗi key phải có bộ đếm riêng;
        nếu chỉ khoá theo model thì đổi key vẫn bị "nhớ" quota cũ của key trước.
        """
        kid = (key_id or "").strip()
        return f"{kid}|{model_name.lower()}" if kid else model_name.lower()

    def set_limit(self, model_name: str, rpm: int, tpm: int, rpd: int) -> None:
        with self._lock:
            self._limits[model_name.lower()] = RateLimit(rpm, tpm, rpd)

    def export_limits_dict(self) -> dict[str, dict[str, int]]:
        """[v3.23.122] Xuất các giới hạn TUỲ CHỈNH (do người dùng đặt) để lưu/hiển thị."""
        with self._lock:
            return {
                name: {"rpm": lim.rpm, "tpm": lim.tpm, "rpd": lim.rpd}
                for name, lim in self._limits.items()
            }

    def replace_limits_dict(self, limits: dict[str, dict[str, int]]) -> None:
        """[v3.23.122] Thay TOÀN BỘ giới hạn tuỳ chỉnh bằng tập mới.

        [v3.23.124] Dựng dict mới RỒI gán nguyên tử (thay vì clear-rồi-populate) để luồng
        khác đang đọc ``_limit_for`` không thấy trạng thái rỗng giữa chừng.
        """
        new_limits: dict[str, RateLimit] = {}
        for name, lim in (limits or {}).items():
            key = (name or "").strip().lower()
            if not key:
                continue
            try:
                new_limits[key] = RateLimit(
                    int(lim["rpm"]), int(lim["tpm"]), int(lim["rpd"])
                )
            except (KeyError, ValueError, TypeError):
                continue
        with self._lock:
            self._limits = new_limits

    @staticmethod
    def default_tier_limits() -> dict[str, dict[str, int]]:
        """[v3.23.122] Bảng giới hạn free-tier mặc định (theo tiền tố model)."""
        return {
            prefix: {"rpm": lim.rpm, "tpm": lim.tpm, "rpd": lim.rpd}
            for prefix, lim in _FREE_TIER_LIMITS.items()
        }

    def learn_daily_limit_from_server(self, model_name: str, server_rpd: int) -> bool:
        """[v3.23.530] Học hạn mức NGÀY THẬT từ thông báo lỗi của server.

        Vì sao cần: bảng giới hạn trong ứng dụng là con số CHÉP TAY và lạc hậu ngay khi Google đổi
        chính sách. Nhật ký thật cho thấy ứng dụng cấu hình 70 request/ngày trong khi server nêu rõ
        hạn mức là 20 — nên ứng dụng tưởng "còn 20 req/ngày" ở key vừa bị từ chối, xoay key rồi lại
        đâm vào tường ở key kế tiếp. Lỗi 429 của Google có kèm ``quotaValue``; lấy luôn con số đó
        làm hạn mức cho các lần sau thì không bao giờ lạc hậu nữa.

        Args:
            model_name: Tên model.
            server_rpd: Hạn mức ngày do server nêu (``quotaValue``).

        Returns:
            True nếu hạn mức đã được cập nhật (khác giá trị đang dùng).
        """
        if server_rpd <= 0:
            return False
        key = model_name.lower()
        with self._lock:
            current = (
                self._learned_limits.get(key)
                or self._limits.get(key)
                or _match_free_tier_limit(model_name)
            )
            changed = current.rpd != server_rpd
            if not changed and key in self._learned_limits:
                return False
            # [v3.23.585] Ghi nhận CẢ khi con số server nêu trùng với con số đang dùng: từ
            # đây hạn mức là số THẬT chứ không còn là phỏng đoán theo tên model (xem
            # is_daily_limit_confirmed). Nhật ký thật: server nêu 20 cho gemini-3.8-flash,
            # trùng bảng suy theo tên, nên trước đây không có gì được ghi lại.
            self._learned_limits[key] = RateLimit(
                rpm=current.rpm, tpm=current.tpm, rpd=server_rpd
            )
            # [v3.23.533] Ghi ngay ra đĩa — nếu không, tắt app là mất và lần sau lại dùng con số
            # chép tay lạc hậu.
            self._save_state_locked()
        if not changed:
            logger.info(
                "Server xác nhận hạn mức NGÀY của '%s' là %d request — đúng con số ứng "
                "dụng đang dùng.", model_name, server_rpd,
            )
            return False
        logger.warning(
            "Cập nhật hạn mức NGÀY của '%s': %d → %d (theo số server báo). Bảng giới hạn chép "
            "tay trong ứng dụng đã lạc hậu so với chính sách hiện hành của Google.",
            model_name, current.rpd, server_rpd,
        )
        return True

    def learn_minute_limit_from_server(
        self, model_name: str, *, rpm: int = 0, tpm: int = 0
    ) -> bool:
        """[v3.23.639] Học hạn mức PHÚT (RPM / TPM) thật từ lỗi 429 ``…PerMinute…`` của server.

        Bảng chép tay đoán RPM/TPM theo tên model (vd 3.x-flash = 5 RPM) và lạc hậu khi Google đổi
        chính sách: đoán thấp thì ứng dụng tự chờ vô ích, đoán cao thì vẫn ăn 429. Đo thật: server
        trả ``quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier``, ``quotaValue: 15``.

        Returns:
            True nếu giá trị đang dùng đã thay đổi.
        """
        if rpm <= 0 and tpm <= 0:
            return False
        key = model_name.lower()
        with self._lock:
            old_rpm, old_tpm = self._learned_minute.get(key, (0, 0))
            new = (rpm if rpm > 0 else old_rpm, tpm if tpm > 0 else old_tpm)
            if new == (old_rpm, old_tpm):
                return False
            before = self._limit_for(model_name)
            self._learned_minute[key] = new
            self._save_state_locked()
        after = self._limit_for(model_name)
        logger.info(
            "Server nêu hạn mức PHÚT của '%s': RPM %d → %d, TPM %d → %d (học từ lỗi 429).",
            model_name, before.rpm, after.rpm, before.tpm, after.tpm,
        )
        return True

    def estimate_wait(
        self, model_name: str, estimated_tokens: int, *, key_id: str = ""
    ) -> float:
        """[v3.23.639] Số giây ``acquire`` sẽ phải CHỜ cho (key, model) nếu gọi ngay; KHÔNG đặt chỗ.

        0.0 = gửi được ngay; ``math.inf`` = không gửi được hôm nay (server đã từ chối vì hết hạn
        mức ngày, hoặc key bị đánh dấu hỏng). Dùng để chọn key RẢNH thay vì ngồi chờ RPM/TPM trên key
        này trong khi key khác đang trống.
        """
        if self.is_key_invalid(key_id):
            return math.inf
        limit = self._limit_for(model_name)
        model_key = self._compose_key(key_id, model_name)
        reserved = max(1, int(estimated_tokens))
        with self._lock:
            now = time.time()
            self._prune_locked(model_key, now)
            daily = self._daily_locked(model_key)
            if (
                daily["count"] >= limit.rpd
                and self._server_daily_block.get(model_key) == daily.get("period_key")
            ):
                return math.inf
            wait = max(0.0, self._cooldown_until.get(model_key, 0.0) - now)
            rq = self._request_windows[model_key]
            if limit.rpm > 0 and len(rq) >= limit.rpm and rq:
                wait = max(wait, 60.0 - (now - rq[0]["ts"]))
            tw = self._token_windows[model_key]
            if self._token_totals[model_key] + reserved > limit.tpm and tw:
                wait = max(wait, 60.0 - (now - tw[0]["ts"]))
            return max(0.0, wait)

    # ── Key hỏng ([v3.23.639]) ─────────────────────────────────────────────────

    def mark_key_invalid(self, key_id: str, reason: str) -> None:
        """Ghi nhận server từ chối key VĨNH VIỄN (không hợp lệ / hết hạn / bị vô hiệu hoá).

        Mọi lượt chọn key sau đó bỏ qua key này (trong ngày) thay vì đâm vào cùng lỗi 400/403 ở mỗi
        lời gọi — trước đây MỘT key hỏng làm hỏng cả lượt dịch dù các key khác còn tốt.
        """
        key_id = (key_id or "").strip()
        if not key_id:
            return
        # [v3.23.641] Lý do là văn bản của server: bỏ ký tự không mã hoá được (surrogate lẻ làm
        # ghi tệp ném UnicodeEncodeError) và cắt ngắn.
        reason = (reason or "không rõ").encode("utf-8", "replace").decode("utf-8")[:300]
        with self._lock:
            self._invalid_keys[key_id] = (reason, self._period_key())
            self._invalid_marked_at[key_id] = time.time()
            self._save_state_locked()

    def is_key_invalid(self, key_id: str) -> bool:
        """True nếu key đã bị đánh dấu hỏng HÔM NAY (sang ngày mới thì thử lại)."""
        key_id = (key_id or "").strip()
        if not key_id:
            return False
        with self._lock:
            item = self._invalid_keys.get(key_id)
            if item is None:
                return False
            expired = (
                time.time() - self._invalid_marked_at.get(key_id, 0.0) > INVALID_KEY_TTL_S
            )
            if item[1] != self._period_key() or expired:
                del self._invalid_keys[key_id]
                self._invalid_marked_at.pop(key_id, None)
                return False
            return True

    def invalid_key_reason(self, key_id: str) -> str:
        """Lý do key bị đánh dấu hỏng (rỗng nếu không hỏng)."""
        if not self.is_key_invalid(key_id):
            return ""
        with self._lock:
            item = self._invalid_keys.get((key_id or "").strip())
            return item[0] if item else ""

    def clear_invalid_key(self, key_id: str) -> None:
        """Bỏ dấu hỏng của một key (vd người dùng vừa thay/sửa key)."""
        with self._lock:
            self._invalid_marked_at.pop((key_id or "").strip(), None)
            if self._invalid_keys.pop((key_id or "").strip(), None) is not None:
                self._save_state_locked()

    def is_daily_limit_confirmed(self, model_name: str) -> bool:
        """[v3.23.585] Hạn mức ngày của model là SỐ THẬT hay chỉ là phỏng đoán theo tên.

        Số thật = server đã nêu trong lỗi 429 (đã học) hoặc người dùng tự cấu hình. Bảng
        suy theo tên model thì cố ý bảo thủ (mọi 'flash-lite' ngoài 3.1 bị đoán là 20
        request/ngày trong khi thực tế có model 500) — không đủ căn cứ để siết số lượt
        thử lại.

        Args:
            model_name: Tên model.

        Returns:
            True khi hạn mức đến từ server hoặc từ cấu hình của người dùng.
        """
        key = model_name.lower()
        with self._lock:
            return key in self._learned_limits or key in self._limits

    def _limit_for(self, model_name: str) -> RateLimit:
        # [v3.23.533] Ưu tiên hạn mức HỌC ĐƯỢC từ server — đó là sự thật hiện hành, đáng tin hơn
        # cả bảng chép tay lẫn con số người dùng đoán khi cấu hình.
        learned = self._learned_limits.get(model_name.lower())
        if learned is not None:
            return self._with_learned_minute(model_name, learned)
        explicit = self._limits.get(model_name.lower())
        if explicit is not None:
            return explicit
        # [v3.23.17] Không cấu hình thủ công → suy theo tên model (free tier).
        return self._with_learned_minute(model_name, _match_free_tier_limit(model_name))

    def _with_learned_minute(self, model_name: str, base: RateLimit) -> RateLimit:
        """[v3.23.639] Thay RPM/TPM của ``base`` bằng số server đã nêu (nếu có)."""
        rpm, tpm = self._learned_minute.get(model_name.lower(), (0, 0))
        if rpm <= 0 and tpm <= 0:
            return base
        return RateLimit(rpm=rpm or base.rpm, tpm=tpm or base.tpm, rpd=base.rpd)

    @staticmethod
    def _period_key(now_utc: datetime | None = None) -> str:
        """Khoá ngày theo mốc reset của Google: 00:00 giờ Thái Bình Dương.

        [v3.23.123] Xử lý ĐÚNG giờ mùa hè (PST UTC-8 / PDT UTC-7). [v3.23.607] Uỷ quyền
        :func:`~.gemini_quota_clock.quota_day` — chạy đúng cả khi Windows thiếu dữ liệu
        múi giờ (trước đây lui về UTC-8, lệch một giờ suốt mùa hè).
        """
        return quota_day(now_utc)

    def _prune_locked(self, model_key: str, now: float) -> None:
        rq = self._request_windows[model_key]
        while rq and now - rq[0]["ts"] >= 60.0:
            rq.popleft()
        tw = self._token_windows[model_key]
        while tw and now - tw[0]["ts"] >= 60.0:
            expired = tw.popleft()
            self._token_totals[model_key] = max(
                0, self._token_totals[model_key] - expired["tokens"]
            )

    def _daily_locked(self, model_key: str) -> dict:
        period = self._period_key()
        cur = self._daily.get(model_key)
        if cur is None or cur.get("period_key") != period:
            cur = {"period_key": period, "count": 0}
            self._daily[model_key] = cur
        return cur

    def acquire(
        self,
        model_name: str,
        estimated_tokens: int,
        cancel_cb: CancelCallback | None = None,
        wait_cb: WaitCallback | None = None,
        *,
        key_id: str = "",
    ) -> QuotaReservation | None:
        """Đặt chỗ cho một lần gọi; tự chờ nếu cần để không vượt RPM/TPM.

        Args:
            key_id: Định danh (vân tay) của API KEY đang dùng. Quota RPD/RPM/TPM của
                Gemini tính theo PROJECT (tức theo API key), nên phải tách bộ đếm theo
                key — đổi sang key khác sẽ có hạn mức độc lập, không bị "nhớ" quota cũ.

        Returns: reservation, hoặc None nếu model không có giới hạn cấu hình.
        Raises:
            QuotaExhaustedError: nếu đã chạm giới hạn request/ngày.
            InterruptedError: nếu ``cancel_cb`` báo huỷ trong lúc chờ.
        """
        limit = self._limit_for(model_name)
        model_key = self._compose_key(key_id, model_name)
        reserved = max(1, int(estimated_tokens))

        while True:
            if cancel_cb is not None and cancel_cb():
                raise InterruptedError("Người dùng đã huỷ khi đang chờ quota.")
            wait_time = 0.0
            reason = ""
            with self._lock:
                now = time.time()
                self._prune_locked(model_key, now)
                daily = self._daily_locked(model_key)

                if daily["count"] >= limit.rpd:
                    # [v3.23.526] Bộ đếm nội bộ chỉ là ƯỚC LƯỢNG và có thể lệch xa thực tế: nhật
                    # ký thật cho thấy app báo "chạm giới hạn 50 request/ngày" trong khi bảng điều
                    # khiển Google ghi mới dùng 0/20, 1/20 và 17/20 trên ba dự án. Chặn cứng lúc
                    # đó là tự khoá mình dù còn quota. Nay chỉ chặn khi CHÍNH SERVER đã từ chối vì
                    # quota ngày; còn lại thì cứ gửi và để Google phán quyết.
                    if self._server_daily_block.get(model_key) == daily.get("period_key"):
                        raise QuotaExhaustedError(
                            f"Model '{model_name}' đã chạm giới hạn {limit.rpd} "
                            "request/ngày (server xác nhận). "
                            f"{describe_quota_reset()} Có thể đổi model."
                        )
                    if not self._soft_daily_warned.get(model_key):
                        self._soft_daily_warned[model_key] = True
                        logger.warning(
                            "Bộ đếm nội bộ cho '%s' đã chạm %d request/ngày nhưng server CHƯA "
                            "từ chối — vẫn gửi tiếp để Google quyết định (bộ đếm có thể lệch).",
                            model_name, limit.rpd,
                        )
                # [v3.23.141] Tôn trọng cooldown do server áp (429 + retryDelay): nếu còn
                # trong thời gian nghỉ, chờ cho hết trước khi xét RPM/TPM. Đây là lá chắn
                # chính chống chuỗi 429 nối tiếp khi ước lượng token client thấp hơn server.
                cooldown = self._cooldown_until.get(model_key, 0.0)
                if cooldown > now:
                    wait_time = max(wait_time, cooldown - now)
                    reason = "cooldown (429 server)"
                rq = self._request_windows[model_key]
                # [v3.23.534] ``rq`` có thể RỖNG khi ``limit.rpm`` <= 0 (cấu hình sai, hoặc học
                # nhầm từ server). Khi đó ``rq[0]`` ném IndexError làm sập cả tác vụ thay vì chờ.
                if limit.rpm > 0 and len(rq) >= limit.rpm and rq:
                    wait_time = max(wait_time, 60.0 - (now - rq[0]["ts"]))
                    reason = "RPM"
                tw = self._token_windows[model_key]
                if self._token_totals[model_key] + reserved > limit.tpm and tw:
                    wait_time = max(wait_time, 60.0 - (now - tw[0]["ts"]))
                    reason = "TPM"

                if wait_time <= 0:
                    self._counter += 1
                    rid = self._counter
                    rq.append({"id": rid, "ts": now})
                    tw.append({"id": rid, "ts": now, "tokens": reserved})
                    self._token_totals[model_key] += reserved
                    daily["count"] += 1
                    self._save_state_locked()
                    return QuotaReservation(model_key, rid, reserved)

            logger.info(
                "Điều tiết quota model '%s' theo %s: chờ %.1fs để tránh vượt giới hạn.",
                model_name, reason, wait_time,
            )
            if wait_cb is not None:
                wait_cb(wait_time, reason)
            # Chờ ngắt quãng để phản hồi huỷ kịp thời.
            slept = 0.0
            step = 0.2
            while slept < wait_time:
                if cancel_cb is not None and cancel_cb():
                    raise InterruptedError("Người dùng đã huỷ khi đang chờ quota.")
                time.sleep(min(step, wait_time - slept))
                slept += step

    def note_rate_limited(
        self, model_name: str, retry_after_s: float, *, key_id: str = ""
    ) -> None:
        """[v3.23.141] Ghi nhận server đã áp rate-limit (429) cho (key,model).

        Đặt mốc COOLDOWN = now + retry_after_s. Mọi ``acquire`` tiếp theo tới cùng
        (key,model) sẽ CHỜ tới hết mốc này trước khi gửi — chống chuỗi 429 nối tiếp khi
        server tính TPM/RPM chặt hơn ước lượng phía client. Lấy giá trị LỚN NHẤT nếu đã có
        cooldown (server có thể tăng dần retryDelay).

        Args:
            retry_after_s: Số giây server yêu cầu chờ (từ ``retryDelay``). Bỏ qua nếu <= 0.
        """
        if retry_after_s <= 0:
            return
        # [v3.23.534] KẸP TRẦN thời gian chờ. Server có thể yêu cầu chờ rất lâu (hàng chục phút
        # tới hàng giờ) khi hết hạn mức ngày; chờ đúng ngần ấy thì ứng dụng đứng im và người dùng
        # tưởng là TREO. Chờ tối đa vài phút rồi thử lại — nếu vẫn hết quota thì server báo tiếp
        # và các lớp xử lý phía trên (xoay key, báo lỗi tử tế) sẽ vào cuộc.
        capped = min(float(retry_after_s), MAX_SERVER_COOLDOWN_SEC)
        if capped < retry_after_s:
            logger.info(
                "Server yêu cầu chờ %.0fs cho '%s' — kẹp còn %.0fs để ứng dụng không đứng im "
                "quá lâu.", retry_after_s, model_name, capped,
            )
        model_key = self._compose_key(key_id, model_name)
        with self._lock:
            target = time.time() + capped
            if target > self._cooldown_until.get(model_key, 0.0):
                self._cooldown_until[model_key] = target

    def cooldown_remaining_s(self, model_name: str, *, key_id: str = "") -> float:
        """[v3.23.147] Số giây COOLDOWN còn lại của (key,model); 0.0 nếu không bị.

        Cho tầng adapter DỰ ĐOÁN key nào đang bị server rate-limit tạm thời (429) để
        tránh xoay vào key đang trong thời gian chờ — chọn key vừa còn quota ngày vừa
        không cooldown thì gửi được NGAY.
        """
        model_key = self._compose_key(key_id, model_name)
        with self._lock:
            return max(0.0, self._cooldown_until.get(model_key, 0.0) - time.time())

    def reconcile(self, reservation: QuotaReservation | None, actual_tokens: int) -> None:
        """Cập nhật token thực tế (thay ước lượng) sau khi có phản hồi."""
        if reservation is None:
            return
        actual = max(1, int(actual_tokens))
        with self._lock:
            tw = self._token_windows.get(reservation.model_key)
            if not tw:
                return
            for item in tw:
                if item["id"] == reservation.reservation_id:
                    self._token_totals[reservation.model_key] = max(
                        0,
                        self._token_totals[reservation.model_key]
                        - item["tokens"] + actual,
                    )
                    item["tokens"] = actual
                    break

    def release(
        self, reservation: QuotaReservation | None, *, request_counted: bool = False
    ) -> None:
        """Nhả chỗ đã đặt khi lần gọi thất bại (lỗi 503/429/mạng…).

        [v3.23.123] Trước đây chỉ nhả token; nay hoàn trả CẢ request/phút lẫn bộ đếm
        request/NGÀY. Đây là điểm mấu chốt: các lần thử thất bại (server không trừ quota)
        không được phép làm tăng RPD ảo → tránh báo "hết quota" oan khi gặp nhiều 429.

        [v3.23.585] Riêng lỗi QUÁ TẢI (503) thì server VẪN tính request: nhật ký thật có
        bảy lần server báo hết hạn mức 20 request/ngày khi "request ứng dụng đếm + số lần
        503" là 21, 20, 19, 21, 16, 21, 17 — còn ứng dụng chỉ đếm 0, 0, 1, 8, 5, 8, 4.
        Với ``request_counted=True`` chỉ nhả token (model chưa xử lý gì), GIỮ request
        trong bộ đếm phút và bộ đếm ngày.

        Args:
            reservation: Chỗ đã đặt; ``None`` thì bỏ qua.
            request_counted: True khi server đã nhận và tính lượt gọi dù trả lỗi.
        """
        if reservation is None:
            return
        with self._lock:
            mk = reservation.model_key
            rid = reservation.reservation_id
            # 1) Nhả token (TPM).
            tw = self._token_windows.get(mk)
            if tw:
                for item in list(tw):
                    if item["id"] == rid:
                        tw.remove(item)
                        self._token_totals[mk] = max(
                            0, self._token_totals[mk] - item["tokens"]
                        )
                        break
            if request_counted:
                return
            # 2) Nhả request (RPM).
            rq = self._request_windows.get(mk)
            if rq:
                for item in list(rq):
                    if item["id"] == rid:
                        rq.remove(item)
                        break
            # 3) Hoàn trả bộ đếm NGÀY (RPD) — quan trọng nhất.
            cur = self._daily.get(mk)
            if cur is not None and cur.get("count", 0) > 0:
                cur["count"] -= 1
                self._save_state_locked()

    def mark_daily_exhausted(self, model_name: str, *, key_id: str = "") -> None:
        """Đánh dấu (key, model) đã DÙNG HẾT quota ngày — dùng khi server trả 429 RPD.

        Đặt bộ đếm ngày bằng đúng giới hạn để lần ``acquire`` sau lập tức coi là hết,
        giúp tầng trên xoay sang API key khác thay vì chờ vô ích trên key đã cạn.
        """
        limit = self._limit_for(model_name)
        model_key = self._compose_key(key_id, model_name)
        with self._lock:
            daily = self._daily_locked(model_key)
            daily["count"] = max(daily["count"], limit.rpd)
            # [v3.23.526] Ghi nhận rằng chính SERVER đã từ chối vì hết quota NGÀY. Chỉ trường hợp
            # này mới được chặn cứng; bộ đếm nội bộ chạm trần thì chỉ là ƯỚC LƯỢNG.
            self._server_daily_block[model_key] = daily.get("period_key", "")
            self._save_state_locked()

    def is_server_daily_blocked(self, model_name: str, *, key_id: str = "") -> bool:
        """[v3.23.529] True nếu CHÍNH SERVER đã từ chối (model, key) vì hết quota ngày hôm nay.

        Dùng để phân biệt "Google thật sự hết" với "bộ đếm nội bộ tưởng hết" khi ghi nhật ký và
        khi quyết định xoay key.

        Args:
            model_name: Tên model.
            key_id: Dấu vân tay của API key.

        Returns:
            True khi có xác nhận từ server trong đúng mốc ngày hiện tại.
        """
        model_key = self._compose_key(key_id, model_name)
        with self._lock:
            daily = self._daily_locked(model_key)
            return self._server_daily_block.get(model_key) == daily.get("period_key")

    def get_remaining(self, model_name: str, *, key_id: str = "") -> dict[str, int]:
        """Trả về hạn mức còn lại hiện tại (để hiển thị/giám sát)."""
        limit = self._limit_for(model_name)
        model_key = self._compose_key(key_id, model_name)
        with self._lock:
            now = time.time()
            self._prune_locked(model_key, now)
            daily = self._daily_locked(model_key)
            rpm_used = len(self._request_windows[model_key])
            tpm_used = self._token_totals[model_key]
            return {
                "rpm_limit": limit.rpm, "rpm_used": rpm_used,
                "rpm_remaining": max(0, limit.rpm - rpm_used),
                "tpm_limit": limit.tpm, "tpm_used": tpm_used,
                "tpm_remaining": max(0, limit.tpm - tpm_used),
                "rpd_limit": limit.rpd, "rpd_used": daily["count"],
                "rpd_remaining": max(0, limit.rpd - daily["count"]),
            }

    def snapshot(self) -> list[dict]:
        """[v3.23.124] Ảnh chụp usage request/NGÀY của mọi (key,model) trong ngày nay.

        Mỗi mục: ``{"key_id", "model", "rpd_used", "rpd_limit", "rpd_remaining"}``.
        Dùng để hiển thị tình trạng quota cho người dùng (vd trong hộp thoại nhiều key).
        """
        period = self._period_key()
        with self._lock:
            items = [
                (mk, rec) for mk, rec in self._daily.items()
                if rec.get("period_key") == period
            ]
        out: list[dict] = []
        for model_key, rec in items:
            if "|" in model_key:
                kid, _, model = model_key.partition("|")
            else:
                kid, model = "", model_key
            limit = self._limit_for(model)
            used = int(rec.get("count", 0))
            out.append({
                "key_id": kid, "model": model,
                "rpd_used": used, "rpd_limit": limit.rpd,
                "rpd_remaining": max(0, limit.rpd - used),
            })
        return out
