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
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

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


class QuotaExhaustedError(RuntimeError):
    """Đã chạm giới hạn ngày (RPD) — không thể chờ trong ngày."""


# [v3.23.17] Giới hạn free tier Gemini (sau đợt cắt giảm 12/2025). Nguồn: tài liệu
# Google AI + tổng hợp cộng đồng (cập nhật ~Q1-Q2 2026). LƯU Ý:
#   * TPM dùng chung 250.000 cho hầu hết model free tier (KHÔNG phải 1.000.000).
# [v3.23.36] Cập nhật theo BẢNG QUOTA THỰC TẾ (ảnh tài khoản, tháng 6/2026):
#   gemini-3.1-flash-lite : RPM 15 · TPM 250K · RPD 500
#   gemini-2.5-flash-lite : RPM 10 · TPM 250K · RPD 20
#   gemini-3.5/3/2.5-flash: RPM  5 · TPM 250K · RPD 20
#   *-pro (2.5/3.1)       : 0/0 — KHÔNG còn free
# Vì 'flash-lite' gồm cả 3.1 (RPD 500) lẫn 2.5 (RPD 20), lấy giá trị BẢO THỦ (thấp
# hơn) để KHÔNG vượt quota của model yếu nhất trong nhóm — tránh 429 bất ngờ.
#   * Quota tính theo PROJECT, reset 00:00 giờ Thái Bình Dương (KHÔNG phải UTC).
#   * RPD là ràng buộc dễ chạm nhất với tác vụ dài.
# Khớp theo tiền tố tên model (vd 'gemini-flash-lite-latest', 'gemini-3.5-flash').
_FREE_TIER_LIMITS: dict[str, RateLimit] = {
    "flash-lite-hi": RateLimit(rpm=15, tpm=250_000, rpd=500),  # 3.1-flash-lite
    "flash-lite": RateLimit(rpm=10, tpm=250_000, rpd=20),       # 2.5-flash-lite (bảo thủ)
    "flash": RateLimit(rpm=5, tpm=250_000, rpd=20),             # *-flash
    "pro": RateLimit(rpm=5, tpm=250_000, rpd=20),               # pro không free → giữ thấp
}

# Mặc định an toàn (bảo thủ) khi không khớp model nào.
_DEFAULT_LIMIT = RateLimit(rpm=5, tpm=250_000, rpd=20)


def _match_free_tier_limit(model_name: str) -> RateLimit:
    """Suy giới hạn free tier theo TÊN model (khớp tiền tố đặc trưng).

    Ưu tiên 'flash-lite' trước 'flash' (vì 'flash-lite' chứa 'flash'). Trong nhóm
    flash-lite, 3.1 có RPD cao (500); các flash-lite khác bảo thủ (RPD 20).
    """
    name = model_name.lower()
    if "flash-lite" in name or "flash_lite" in name:
        # gemini-3.1-flash-lite có RPD cao hơn hẳn (500) so với 2.5-flash-lite (20).
        if "3.1" in name or "3-1" in name:
            return _FREE_TIER_LIMITS["flash-lite-hi"]
        return _FREE_TIER_LIMITS["flash-lite"]
    if "flash" in name:
        return _FREE_TIER_LIMITS["flash"]
    if "pro" in name:
        return _FREE_TIER_LIMITS["pro"]
    return _DEFAULT_LIMIT


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
        self._cooldown_until: dict[str, float] = defaultdict(float)
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
        period = self._period_key()
        loaded = 0
        for model_key, rec in (data or {}).items():
            if isinstance(rec, dict) and rec.get("period_key") == period:
                self._daily[model_key] = {
                    "period_key": period, "count": int(rec.get("count", 0))
                }
                loaded += 1
        if loaded:
            logger.info("Đã khôi phục bộ đếm quota/ngày cho %d (model,key).", loaded)

    def _save_state_locked(self) -> None:
        """Ghi bộ đếm ngày ra đĩa (gọi trong lock, sau khi count thay đổi)."""
        if self._state_path is None:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(
                json.dumps(self._daily, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
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

    def _limit_for(self, model_name: str) -> RateLimit:
        explicit = self._limits.get(model_name.lower())
        if explicit is not None:
            return explicit
        # [v3.23.17] Không cấu hình thủ công → suy theo tên model (free tier).
        return _match_free_tier_limit(model_name)

    @staticmethod
    def _period_key(now_utc: datetime | None = None) -> str:
        """Khoá ngày theo mốc reset của Google: 00:00 giờ Thái Bình Dương.

        [v3.23.123] Dùng zoneinfo('America/Los_Angeles') để xử lý ĐÚNG DST (PST UTC-8 /
        PDT UTC-7); nếu hệ thống thiếu dữ liệu tz thì lui về xấp xỉ UTC-8 như trước.
        """
        cur = now_utc or datetime.now(UTC)
        try:
            from zoneinfo import ZoneInfo

            pacific = cur.astimezone(ZoneInfo("America/Los_Angeles"))
            return pacific.date().isoformat()
        except Exception:  # thiếu tzdata → xấp xỉ UTC-8
            return (cur - timedelta(hours=8)).date().isoformat()

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
                    raise QuotaExhaustedError(
                        f"Model '{model_name}' đã chạm giới hạn {limit.rpd} "
                        "request/ngày. Hãy chờ sang ngày mới hoặc đổi model."
                    )
                # [v3.23.141] Tôn trọng cooldown do server áp (429 + retryDelay): nếu còn
                # trong thời gian nghỉ, chờ cho hết trước khi xét RPM/TPM. Đây là lá chắn
                # chính chống chuỗi 429 nối tiếp khi ước lượng token client thấp hơn server.
                cooldown = self._cooldown_until.get(model_key, 0.0)
                if cooldown > now:
                    wait_time = max(wait_time, cooldown - now)
                    reason = "cooldown (429 server)"
                rq = self._request_windows[model_key]
                if len(rq) >= limit.rpm:
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
        model_key = self._compose_key(key_id, model_name)
        with self._lock:
            target = time.time() + float(retry_after_s)
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

    def release(self, reservation: QuotaReservation | None) -> None:
        """Nhả TOÀN BỘ chỗ đã đặt khi lần gọi KHÔNG tiêu thụ quota (lỗi 503/429/mạng…).

        [v3.23.123] Trước đây chỉ nhả token; nay hoàn trả CẢ request/phút lẫn bộ đếm
        request/NGÀY. Đây là điểm mấu chốt: các lần thử thất bại (server không trừ quota)
        không được phép làm tăng RPD ảo → tránh báo "hết quota" oan khi gặp nhiều 503/429.
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
            self._save_state_locked()

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
