"""Lớp thành phần giao diện Fluent — chọn được backend qfluentwidgets hoặc Qt thuần.

Lịch sử
-------
[v3.23.267] ``qfluentwidgets`` là GPL v3 nên khi còn hướng thương mại, app bỏ hẳn và
tự xây lại mọi component bằng Qt thuần trong gói này, **giữ NGUYÊN tên API** của
qfluentwidgets để code gọi không phải đổi.

[v3.23.313] Dự án đã chuyển sang **mã nguồn mở (GPL)** nên qfluentwidgets dùng được
lại. Nhờ việc giữ nguyên tên API từ đầu, giờ chuyển backend KHÔNG phải sửa 15 nơi gọi.

Cách bật qfluentwidgets
-----------------------
Đặt biến môi trường trước khi chạy::

    set SUBEXT_USE_QFLUENTWIDGETS=1

**Mặc định TẮT** — vì bản Qt thuần đang chạy ổn định và đạt WCAG 2.1, còn giao diện
là thứ hỏng thì thấy ngay. Bật lên nếu thấy đẹp hơn; không thích thì bỏ biến là về
nguyên trạng, không cần build lại.

Cơ chế lùi theo TỪNG ký hiệu
----------------------------
qfluentwidgets không chắc có đủ mọi tên (vd ``GroupBox`` là component tự thêm của dự
án). Nên thay vì nhập cả khối (dễ vỡ), module này phân giải **từng ký hiệu một**: có
trong qfluentwidgets thì dùng, không có thì lấy bản nội bộ. Nhờ vậy bật backend mới
không bao giờ gây ``ImportError``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Final

from subtitles_extractor.presentation.fluent_compat.icons import (
    FluentIcon as _LocalFluentIcon,
)
from subtitles_extractor.presentation.fluent_compat.theme import (
    InfoBar as _LocalInfoBar,
)
from subtitles_extractor.presentation.fluent_compat.theme import (
    InfoBarPosition as _LocalInfoBarPosition,
)
from subtitles_extractor.presentation.fluent_compat.theme import Theme as _LocalTheme
from subtitles_extractor.presentation.fluent_compat.theme import (
    isDarkTheme as _local_is_dark_theme,
)
from subtitles_extractor.presentation.fluent_compat.theme import (
    setTheme as _local_set_theme,
)
from subtitles_extractor.presentation.fluent_compat.theme import (
    setThemeColor as _local_set_theme_color,
)
from subtitles_extractor.presentation.fluent_compat.theme import (
    themeColor as _local_theme_color,
)
from subtitles_extractor.presentation.fluent_compat.widgets import (
    BodyLabel as _LocalBodyLabel,
)
from subtitles_extractor.presentation.fluent_compat.widgets import (
    CaptionLabel as _LocalCaptionLabel,
)
from subtitles_extractor.presentation.fluent_compat.widgets import (
    CardWidget as _LocalCardWidget,
)
from subtitles_extractor.presentation.fluent_compat.widgets import (
    CheckBox as _LocalCheckBox,
)
from subtitles_extractor.presentation.fluent_compat.widgets import (
    ComboBox as _LocalComboBox,
)
from subtitles_extractor.presentation.fluent_compat.widgets import (
    DoubleSpinBox as _LocalDoubleSpinBox,
)
from subtitles_extractor.presentation.fluent_compat.widgets import (
    GroupBox as _LocalGroupBox,
)
from subtitles_extractor.presentation.fluent_compat.widgets import (
    GroupHeaderCardWidget as _LocalGroupHeaderCardWidget,
)
from subtitles_extractor.presentation.fluent_compat.widgets import (
    HeaderCardWidget as _LocalHeaderCardWidget,
)
from subtitles_extractor.presentation.fluent_compat.widgets import (
    LineEdit as _LocalLineEdit,
)
from subtitles_extractor.presentation.fluent_compat.widgets import (
    NavigationItemPosition as _LocalNavigationItemPosition,
)
from subtitles_extractor.presentation.fluent_compat.widgets import (
    PrimaryPushButton as _LocalPrimaryPushButton,
)
from subtitles_extractor.presentation.fluent_compat.widgets import (
    ProgressBar as _LocalProgressBar,
)
from subtitles_extractor.presentation.fluent_compat.widgets import (
    PushButton as _LocalPushButton,
)
from subtitles_extractor.presentation.fluent_compat.widgets import (
    ScrollArea as _LocalScrollArea,
)
from subtitles_extractor.presentation.fluent_compat.widgets import (
    SimpleCardWidget as _LocalSimpleCardWidget,
)
from subtitles_extractor.presentation.fluent_compat.widgets import Slider as _LocalSlider
from subtitles_extractor.presentation.fluent_compat.widgets import (
    SpinBox as _LocalSpinBox,
)
from subtitles_extractor.presentation.fluent_compat.widgets import (
    StrongBodyLabel as _LocalStrongBodyLabel,
)
from subtitles_extractor.presentation.fluent_compat.widgets import (
    TextEdit as _LocalTextEdit,
)
from subtitles_extractor.presentation.fluent_compat.widgets import (
    ToolButton as _LocalToolButton,
)
from subtitles_extractor.presentation.fluent_compat.window import (
    FluentWindow as _LocalFluentWindow,
)

logger = logging.getLogger(__name__)

_BACKEND_ENV: Final[str] = "SUBEXT_USE_QFLUENTWIDGETS"
_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})


def _load_qfluentwidgets() -> Any | None:
    """Nạp ``qfluentwidgets`` nếu được yêu cầu và có sẵn.

    Returns:
        Module ``qfluentwidgets``, hoặc ``None`` khi không bật / chưa cài.
    """
    if os.environ.get(_BACKEND_ENV, "").strip().lower() not in _TRUTHY:
        return None
    try:
        import qfluentwidgets
    except ImportError as exc:
        logger.warning(
            "%s=1 nhưng chưa cài qfluentwidgets (dùng bản Qt thuần): %s.",
            _BACKEND_ENV,
            exc,
        )
        return None
    logger.info("Giao diện: dùng qfluentwidgets (GPL v3).")
    return qfluentwidgets


_qfluent = _load_qfluentwidgets()


def _pick(name: str, local_symbol: Any) -> Any:
    """Chọn ký hiệu từ qfluentwidgets nếu có, ngược lại dùng bản nội bộ.

    Args:
        name: Tên ký hiệu theo API qfluentwidgets.
        local_symbol: Bản dựng bằng Qt thuần trong gói này.

    Returns:
        Ký hiệu sẽ được xuất ra.
    """
    if _qfluent is None:
        return local_symbol
    candidate = getattr(_qfluent, name, None)
    if candidate is None:
        logger.debug("qfluentwidgets thiếu '%s' — dùng bản nội bộ.", name)
        return local_symbol
    return candidate


# ── Phân giải từng ký hiệu ───────────────────────────────────────────────────
FluentIcon = _pick("FluentIcon", _LocalFluentIcon)

Theme = _pick("Theme", _LocalTheme)
setTheme = _pick("setTheme", _local_set_theme)  # noqa: N816 — giữ tên API qfluentwidgets
isDarkTheme = _pick("isDarkTheme", _local_is_dark_theme)  # noqa: N816
themeColor = _pick("themeColor", _local_theme_color)  # noqa: N816
setThemeColor = _pick("setThemeColor", _local_set_theme_color)  # noqa: N816
InfoBar = _pick("InfoBar", _LocalInfoBar)
InfoBarPosition = _pick("InfoBarPosition", _LocalInfoBarPosition)

CheckBox = _pick("CheckBox", _LocalCheckBox)
ComboBox = _pick("ComboBox", _LocalComboBox)
LineEdit = _pick("LineEdit", _LocalLineEdit)
Slider = _pick("Slider", _LocalSlider)
SpinBox = _pick("SpinBox", _LocalSpinBox)
DoubleSpinBox = _pick("DoubleSpinBox", _LocalDoubleSpinBox)
ProgressBar = _pick("ProgressBar", _LocalProgressBar)
TextEdit = _pick("TextEdit", _LocalTextEdit)
ToolButton = _pick("ToolButton", _LocalToolButton)
ScrollArea = _pick("ScrollArea", _LocalScrollArea)
PushButton = _pick("PushButton", _LocalPushButton)
PrimaryPushButton = _pick("PrimaryPushButton", _LocalPrimaryPushButton)
CaptionLabel = _pick("CaptionLabel", _LocalCaptionLabel)
StrongBodyLabel = _pick("StrongBodyLabel", _LocalStrongBodyLabel)
BodyLabel = _pick("BodyLabel", _LocalBodyLabel)
HeaderCardWidget = _pick("HeaderCardWidget", _LocalHeaderCardWidget)
CardWidget = _pick("CardWidget", _LocalCardWidget)
SimpleCardWidget = _pick("SimpleCardWidget", _LocalSimpleCardWidget)
GroupHeaderCardWidget = _pick("GroupHeaderCardWidget", _LocalGroupHeaderCardWidget)
# GroupBox là component RIÊNG của dự án (qfluentwidgets không có) -> luôn dùng nội bộ.
GroupBox = _LocalGroupBox
NavigationItemPosition = _pick("NavigationItemPosition", _LocalNavigationItemPosition)

FluentWindow = _pick("FluentWindow", _LocalFluentWindow)


def active_backend() -> str:
    """Tên backend giao diện đang dùng — hữu ích để hiển thị trong Chẩn đoán."""
    return "qfluentwidgets" if _qfluent is not None else "fluent_compat (Qt thuần)"


__all__ = [  # noqa: RUF022 — nhóm theo loại component cho dễ đọc
    # icons
    "FluentIcon",
    # theme
    "Theme", "setTheme", "isDarkTheme", "themeColor", "setThemeColor",
    "InfoBar", "InfoBarPosition",
    # widgets
    "CheckBox", "ComboBox", "LineEdit", "Slider", "SpinBox", "DoubleSpinBox",
    "ProgressBar", "TextEdit", "ToolButton", "ScrollArea", "PushButton",
    "PrimaryPushButton", "CaptionLabel", "StrongBodyLabel", "BodyLabel",
    "HeaderCardWidget", "CardWidget", "SimpleCardWidget", "GroupHeaderCardWidget",
    "GroupBox", "NavigationItemPosition",
    # window
    "FluentWindow",
    # tiện ích
    "active_backend",
]
