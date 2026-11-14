"""[v3.23.677] Kho DÙNG CHUNG khoá API + nhà cung cấp AI (trang Cài đặt sửa; trang Dịch, Lồng tiếng đọc).

Trước bản này khoá nằm ở hai nơi: trang Dịch (key Gemini chính + dự phòng + nhà cung cấp tuỳ chỉnh) và ô "API Key"
riêng của Gemini TTS ở trang Lồng tiếng. Nay chỉ còn MỘT chỗ sửa (Cài đặt → Khoá API & AI), mọi trang đọc cùng kho.

Lưu ở đúng các khoá QSettings cũ của trang Dịch (``api_key``, ``api_keys_extra``, ``ai_provider``,
``custom_base_url``, ``custom_api_key``) nên bản cũ nâng lên KHÔNG mất key. Mọi lần đọc đều đọc thẳng QSettings
(không cache) nên hai đối tượng kho bất kỳ luôn thấy cùng dữ liệu.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QSettings, Signal

from subtitles_extractor.domain.exceptions import ConfigurationError

#: Nhóm QSettings cũ của trang Dịch — giữ nguyên để không mất key khi nâng cấp.
SETTINGS_ORG = "SubtitlesExtractor"
SETTINGS_APP = "TranslatePage"

PROVIDER_GEMINI = "gemini"
PROVIDER_CUSTOM = "custom"

#: Các khoá QSettings thuộc kho (test dùng để dọn sau mỗi ca).
STORE_KEYS: tuple[str, ...] = ("api_key", "api_keys_extra", "ai_provider", "custom_base_url", "custom_api_key")


def _dedupe(keys: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in keys:
        key = (raw or "").strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def split_keys(text: str) -> list[str]:
    """Tách chuỗi nhiều key (xuống dòng / dấu phẩy) thành danh sách không trùng."""
    return _dedupe((text or "").replace(",", "\n").splitlines())


class AiAccessStore(QObject):
    """Đọc/ghi khoá API + nhà cung cấp; phát ``changed`` sau mỗi lần ghi."""

    changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

    @staticmethod
    def _settings() -> QSettings:
        return QSettings(SETTINGS_ORG, SETTINGS_APP)

    # ── Đọc ──
    def provider(self) -> str:
        value = str(self._settings().value("ai_provider", PROVIDER_GEMINI, type=str) or PROVIDER_GEMINI)
        return value if value in (PROVIDER_GEMINI, PROVIDER_CUSTOM) else PROVIDER_GEMINI

    def is_custom(self) -> bool:
        return self.provider() == PROVIDER_CUSTOM

    def gemini_keys(self) -> list[str]:
        """Key Gemini hiệu lực: key chính trước, rồi key dự phòng; bỏ trùng/rỗng."""
        s = self._settings()
        main = str(s.value("api_key", "", type=str) or "")
        extra = str(s.value("api_keys_extra", "", type=str) or "")
        # [v3.23.698] Tách cả dấu phẩy ở key chính: dán "key1, key2" vào ô key chính (hoặc cấu hình cũ lưu danh sách phẩy)
        # từng thành MỘT "key" — kiểm phát âm gửi nguyên chuỗi đó (key không hợp lệ), tóm tắt ghi "1 key", hạn mức luôn 0.
        return split_keys("\n".join((main, extra)))

    def custom_url(self) -> str:
        return str(self._settings().value("custom_base_url", "", type=str) or "").strip()

    def custom_key(self) -> str:
        return str(self._settings().value("custom_api_key", "", type=str) or "").strip()

    def custom_complete(self) -> bool:
        """[v3.23.698] Cùng phép kiểm với bộ dịch (``CustomProviderConfig.is_complete``): http(s) + tên máy + key ASCII.

        Trước đây chỉ kiểm "không rỗng": địa chỉ thiếu ``https://`` hay key dính ký tự vô hình vẫn hiện 🟢 trong Cài đặt
        trong khi mọi thao tác Dịch/Phân tích/Tải model đều từ chối và đẩy người dùng về Cài đặt.
        """
        from subtitles_extractor.infrastructure.translation.openai_compat_client import CustomProviderConfig

        return CustomProviderConfig(self.custom_url(), self.custom_key()).is_complete

    # ── Ghi ──
    def set_gemini_keys(self, keys: list[str]) -> None:
        clean = split_keys("\n".join(k or "" for k in keys))   # [v3.23.698] mỗi phần tử có thể chứa nhiều key
        s = self._settings()
        s.setValue("api_key", clean[0] if clean else "")
        s.setValue("api_keys_extra", "\n".join(clean[1:]))
        s.sync()
        self.changed.emit()

    def add_gemini_keys(self, keys: list[str]) -> int:
        """Thêm key chưa có vào cuối danh sách; trả số key mới thêm."""
        current = self.gemini_keys()
        new = [k for k in _dedupe(list(keys)) if k not in current]
        if new:
            self.set_gemini_keys(current + new)
        return len(new)

    def set_provider(self, provider: str) -> None:
        value = provider if provider in (PROVIDER_GEMINI, PROVIDER_CUSTOM) else PROVIDER_GEMINI
        if value == self.provider():
            return
        s = self._settings()
        s.setValue("ai_provider", value)
        s.sync()
        self.changed.emit()

    def set_custom(self, base_url: str, api_key: str) -> None:
        url, key = (base_url or "").strip(), (api_key or "").strip()
        if url == self.custom_url() and key == self.custom_key():
            return
        s = self._settings()
        s.setValue("custom_base_url", url)
        s.setValue("custom_api_key", key)
        s.sync()
        self.changed.emit()


_shared: AiAccessStore | None = None


def shared_ai_access_store() -> AiAccessStore:
    """Kho dùng chung toàn ứng dụng (tạo lần đầu khi cần)."""
    global _shared
    if _shared is None:
        _shared = AiAccessStore()
    return _shared


def migrate_legacy_ai_settings(store: AiAccessStore, settings_service: object | None) -> list[str]:
    """Chuyển cấu hình cũ một lần; trả danh sách việc đã làm (để ghi log).

    * Key "API Key" riêng của Gemini TTS (QSettings trang Lồng tiếng) → thêm vào danh sách key Gemini chung.
    * "Số lần thử lại" của trang Dịch (QSettings, 1–20, mặc định 5) → ``translation.retry_count`` trong Cài đặt
      (1–10). Trước đây ô trên trang Dịch ĐÈ cấu hình Cài đặt nên giá trị trang Dịch mới là giá trị thật đang dùng.
    """
    done: list[str] = []
    tts = QSettings("SubtitlesExtractorPaddleOCR", "TTSPage_v2")
    legacy_tts_key = str(tts.value("gemini_key", "", type=str) or "")
    if legacy_tts_key.strip():
        added = store.add_gemini_keys(split_keys(legacy_tts_key))
        tts.remove("gemini_key")
        tts.sync()
        done.append(f"key Gemini TTS → khoá chung (+{added})")
    page = QSettings(SETTINGS_ORG, SETTINGS_APP)
    if page.contains("retry_count"):
        try:
            value = max(1, min(10, int(str(page.value("retry_count", 5)))))
        except (TypeError, ValueError):
            value = 5
        update = getattr(settings_service, "update", None)
        if callable(update):
            try:
                update(translation={"retry_count": value})
                done.append(f"số lần thử lại trang Dịch → Cài đặt ({value})")
            except ConfigurationError:
                return done  # cấu hình hỏng không được chặn khởi động; giữ khoá cũ để lần sau thử lại
        page.remove("retry_count")
        page.sync()
    return done


__all__ = [
    "PROVIDER_CUSTOM",
    "PROVIDER_GEMINI",
    "STORE_KEYS",
    "AiAccessStore",
    "migrate_legacy_ai_settings",
    "shared_ai_access_store",
    "split_keys",
]
