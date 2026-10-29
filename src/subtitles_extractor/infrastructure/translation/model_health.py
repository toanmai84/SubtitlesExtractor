"""[v3.23.631] Sức khoẻ MODEL Gemini: bộ ngắt mạch khi quá tải (503/504) + chọn model dự phòng.

Nhật ký thật của phim 86 tập (free tier, 3 API key): trong 3,06 giờ chờ thử lại, **142 phút
(46%) là chờ lỗi 503 UNAVAILABLE** của CÙNG một model. Một chuỗi 68 lần 503 liền trên
``gemini-3.1-flash-lite`` kéo dài 41 phút; hai lượt dịch hỏng hẳn sau 12 lần thử (13 phút) mà
không ai thử model khác. Chuỗi thử lại cũ chỉ biết đợi trên một model, và đổi API key không
giúp: 17 lần xoay key vì quá tải thì chỉ 2 lần được trả lời ngay (15/17 vẫn 503) — quá tải là
của MODEL, không phải của key, mà mỗi lần xoay key kèm video còn tốn một lần tải lại đoạn
video (~40 giây) và vài request bị tính vào hạn mức ngày.

Module này (hàm thuần + một lớp trạng thái nhỏ, không I/O) cho ``_call_gemini``:

* :class:`ModelHealth` — đếm lỗi quá tải LIÊN TIẾP theo model; đủ :data:`TRIP_AFTER` lần thì
  "ngắt mạch": model bị coi là đang bận trong ``90 s × 2^(lần ngắt-1)`` (trần 15 phút); hết
  hạn thì lượt đầu là lượt thăm dò — hỏng một lần là ngắt lại ngay, thành công là bình phục.
  Dùng chung cho mọi adapter của tiến trình, nên các lời gọi KẾ TIẾP không phải tự đốt lại hai
  lượt 503 mới biết model đang bận.
* :func:`rank_fallbacks` — thứ tự model dự phòng: cùng nhóm (lite/flash) trước, hạn mức ngày
  cao trước, đời mới trước.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Iterable

#: [v3.23.638] Giá trị lựa chọn model dự phòng của người dùng: tự động (mặc định) / không chuyển model.
AUTO_FALLBACK = ""
NO_FALLBACK = "-"
#: Số lỗi quá tải LIÊN TIẾP trên một model thì ngắt mạch model đó.
TRIP_AFTER = 3
#: Thời gian ngắt mạch lần đầu (giây); mỗi lần ngắt liên tiếp nhân đôi.
BASE_COOLDOWN_S = 90.0
#: Trần thời gian ngắt mạch (giây).
MAX_COOLDOWN_S = 900.0
#: Trong MỘT lời gọi, số lượt quá tải liên tiếp trước khi chuyển sang model dự phòng.
FAILOVER_AFTER_IN_CALL = 2

_VERSION_RE = re.compile(r"gemini-(\d+(?:\.\d+)?)")
#: Model KHÔNG dùng được làm dự phòng cho dịch/phân tích văn bản JSON: âm thanh gốc/trực tiếp,
#: đa phương thức đặc thù, bí danh ``*-latest`` (trỏ tới model không đoán trước được).
_UNSUITABLE = ("native-audio", "live", "omni", "image", "tts", "audio", "embedding", "-latest")


def model_tier(model: str) -> str:
    """Nhóm model: ``"lite"`` (``*-flash-lite``) hoặc ``"flash"``."""
    name = model.lower()
    return "lite" if "flash-lite" in name or "flash_lite" in name else "flash"


def model_family(model: str) -> str:
    """[v3.23.666] Họ model: phần trước dấu ``-`` đầu tiên (``gemini``, ``gpt``, ``claude``…).

    Bỏ tiền tố nhà cung cấp kiểu ``google/gemini-3.7-flash``.
    """
    name = (model or "").strip().lower().rsplit("/", 1)[-1]
    return name.split("-", 1)[0]


#: [v3.23.666] Dấu hiệu bậc giá trong tên model (nhà cung cấp tuỳ chỉnh bán nhiều hãng).
_PREMIUM_MARKERS = ("-pro", "-max", "-ultra", "opus", "-plus", "-large", "-high", "reasoner")
_BUDGET_MARKERS = ("-lite", "-mini", "-nano", "-small", "haiku", "-flash-8b", "-air", "-fast")


def cost_tier(model: str) -> str:
    """[v3.23.666] Bậc giá ước theo tên: ``"premium"`` (pro/max/opus…), ``"budget"`` (lite/mini…)
    hoặc ``"standard"``. Dùng để dự phòng tự động không nhảy sang model đắt hơn hẳn."""
    name = (model or "").strip().lower().rsplit("/", 1)[-1]
    if any(marker in name for marker in _PREMIUM_MARKERS):
        return "premium"
    if any(marker in name for marker in _BUDGET_MARKERS):
        return "budget"
    return "standard"


def model_version(model: str) -> float:
    """Số đời của model (``gemini-3.5-flash`` → 3.5); 0 nếu không đọc được."""
    match = _VERSION_RE.search(model.lower())
    return float(match.group(1)) if match else 0.0


def rank_fallbacks(
    model: str,
    known: Iterable[str],
    *,
    cooling: Callable[[str], float] = lambda _m: 0.0,
    daily_limit: Callable[[str], int] = lambda _m: 0,
    available: Callable[[str], bool] = lambda _m: True,
) -> list[str]:
    """Model dự phòng cho ``model``, tốt nhất trước.

    Loại: chính nó, model đang ngắt mạch, model ``available`` trả ``False`` (hết hạn mức
    ngày…), model không hợp (âm thanh/trực tiếp/bí danh ``-latest``). Xếp: cùng nhóm với ``model``
    trước; bản chính thức trước bản ``preview``; hạn mức ngày cao trước (dư dả hơn); đời mới
    trước; cuối cùng theo tên cho ổn định.

    Args:
        model: Model đang quá tải.
        known: Các model biết là dùng được (danh sách free-tier lấy từ API).
        cooling: ``model -> số giây còn bị ngắt mạch``.
        daily_limit: ``model -> hạn mức request/ngày``.
        available: ``model -> còn dùng được (hạn mức ngày…)``.

    Returns:
        Danh sách model, rỗng nếu không còn ai.
    """
    own_tier = model_tier(model)
    candidates = {
        name for name in known
        if name and name != model and cooling(name) <= 0 and available(name)
        and not any(marker in name.lower() for marker in _UNSUITABLE)
    }
    # Bản preview xếp SAU mọi bản chính thức (hạn mức của chúng chỉ đoán theo tên), rồi mới tới
    # hạn mức ngày cao → đời mới.
    return sorted(
        candidates,
        key=lambda name: (
            0 if model_tier(name) == own_tier else 1,
            1 if "preview" in name.lower() else 0,
            -daily_limit(name),
            -model_version(name),
            name,
        ),
    )


class ModelHealth:
    """Trạng thái quá tải theo model, dùng chung giữa các adapter (an toàn đa luồng)."""

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        seed_models: Iterable[str] = (),
        same_family_only: bool = False,
    ) -> None:
        """Khởi tạo.

        Args:
            clock: Đồng hồ đơn điệu (kiểm thử thay được).
            seed_models: Model biết chắc dùng được ngay từ đầu (khi chưa lấy danh sách API).
            same_family_only: [v3.23.666] Dự phòng TỰ ĐỘNG chỉ trong cùng họ model (``gemini-*``
                → ``gemini-*``) VÀ cùng bậc giá (:func:`cost_tier`). Nhà cung cấp tuỳ chỉnh bán cả
                ``claude-opus``/``gpt-5.6-pro`` giá gấp chục lần: quá tải một lúc không được lặng lẽ
                chuyển sang model đắt hơn. Model người dùng tự chọn làm dự phòng vẫn được dùng nếu
                nhà cung cấp có model đó (nằm trong danh sách đã tải).
        """
        self._same_family_only = bool(same_family_only)
        self._clock = clock
        self._lock = threading.Lock()
        self._consecutive: dict[str, int] = {}
        self._trips: dict[str, int] = {}
        self._cool_until: dict[str, float] = {}
        self._known: set[str] = {m for m in seed_models if m}
        self._preferred = AUTO_FALLBACK

    # ── Danh sách model biết ─────────────────────────────────────────────────

    def set_known_models(self, models: Iterable[str]) -> None:
        """Ghi danh sách model free-tier vừa lấy từ API (thay toàn bộ nếu không rỗng)."""
        names = {m for m in models if m}
        if names:
            with self._lock:
                self._known = names

    @property
    def known_models(self) -> tuple[str, ...]:
        """Các model biết là dùng được."""
        with self._lock:
            return tuple(sorted(self._known))

    # ── Lựa chọn của người dùng ([v3.23.638]) ────────────────────────────────

    def set_preferred_fallback(self, value: str) -> None:
        """Người dùng chọn model dự phòng: ``""`` = tự động, ``"-"`` = không chuyển model, còn lại
        là tên model (kể cả model gõ tay chưa có trong danh sách API)."""
        with self._lock:
            self._preferred = (value or "").strip()

    @property
    def preferred_fallback(self) -> str:
        """Lựa chọn dự phòng hiện tại của người dùng (xem :meth:`set_preferred_fallback`)."""
        with self._lock:
            return self._preferred

    # ── Ghi nhận kết quả ─────────────────────────────────────────────────────

    def record_overload(self, model: str) -> float:
        """Ghi một lượt quá tải; trả số giây ngắt mạch MỚI (0 nếu chưa ngắt).

        Đang trong thời gian ngắt thì không nhân đôi thêm (các lượt đang bay của luồng song
        song cùng báo lỗi một đợt quá tải).
        """
        with self._lock:
            now = self._clock()
            if self._cool_until.get(model, 0.0) > now:
                return 0.0
            trips = self._trips.get(model, 0)
            half_open = trips > 0                      # vừa hết hạn ngắt: lượt này là thăm dò
            count = self._consecutive.get(model, 0) + 1
            if half_open or count >= TRIP_AFTER:
                trips += 1
                cooldown = min(MAX_COOLDOWN_S, BASE_COOLDOWN_S * 2 ** min(trips - 1, 16))
                self._trips[model] = trips
                self._consecutive[model] = 0
                self._cool_until[model] = now + cooldown
                return cooldown
            self._consecutive[model] = count
            return 0.0

    def record_success(self, model: str) -> None:
        """Model trả lời bình thường → bình phục hoàn toàn."""
        with self._lock:
            self._consecutive.pop(model, None)
            self._trips.pop(model, None)
            self._cool_until.pop(model, None)

    def cooling_remaining(self, model: str) -> float:
        """Số giây model còn bị coi là đang bận (0 = dùng được)."""
        with self._lock:
            return max(0.0, self._cool_until.get(model, 0.0) - self._clock())

    # ── Chọn dự phòng ────────────────────────────────────────────────────────

    def fallbacks(
        self,
        model: str,
        *,
        daily_limit: Callable[[str], int] = lambda _m: 0,
        available: Callable[[str], bool] = lambda _m: True,
        exclude: Iterable[str] = (),
    ) -> list[str]:
        """Model dự phòng khả dụng cho ``model`` (xem :func:`rank_fallbacks`).

        ``exclude``: model đã thử trong lời gọi hiện tại (không quay lại).
        """
        skipped = set(exclude)
        preferred = self.preferred_fallback
        if preferred == NO_FALLBACK:
            return []
        family, tier = model_family(model), cost_tier(model)
        known = self.known_models
        ranked = rank_fallbacks(
            model,
            (
                m for m in known
                if m not in skipped
                and not (self._same_family_only and (
                    model_family(m) != family or cost_tier(m) != tier))
            ),
            cooling=self.cooling_remaining, daily_limit=daily_limit, available=available,
        )
        # [v3.23.638] Model người dùng chọn đứng ĐẦU khi dùng được (không phải chính nó, chưa thử, không
        # đang ngắt mạch, còn hạn mức); không dùng được thì rơi về thứ tự tự động chứ không bỏ cuộc.
        if self._same_family_only and preferred not in known:
            preferred = ""  # vd tên Gemini cũ mà nhà cung cấp tuỳ chỉnh không bán → 404 hỏng lời gọi
        if (
            preferred and preferred != model and preferred not in skipped
            and self.cooling_remaining(preferred) <= 0 and available(preferred)
        ):
            return [preferred, *[name for name in ranked if name != preferred]]
        return ranked


__all__ = [
    "AUTO_FALLBACK", "NO_FALLBACK", "BASE_COOLDOWN_S", "FAILOVER_AFTER_IN_CALL", "MAX_COOLDOWN_S", "TRIP_AFTER",
    "ModelHealth", "cost_tier", "model_family", "model_tier", "model_version", "rank_fallbacks",
]
