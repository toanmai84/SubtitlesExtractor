# Kế hoạch chuyển PyQt6 → PySide6 & quyết định qfluentwidgets

**Ngày:** 2026-07-18

## 1. Khảo sát hiện trạng

| Hạng mục | Số lượng | Ghi chú |
|---|---|---|
| File dùng PyQt6 | 50 | Chủ yếu QtCore/QtWidgets/QtGui |
| `pyqtSignal` | 181 | → đổi thành `Signal` |
| `pyqtSlot` | 4 | → đổi thành `Slot` |
| `.exec()` | 13 | PySide6 giữ `.exec()` — OK, không cần đổi |
| File dùng qfluentwidgets | 15 | 24 component |

## 2. Phần PyQt6 → PySide6 (quyết định: LÀM — miễn phí license)

PySide6 là binding Qt chính thức, **LGPL v3** → dùng thương mại đóng KHÔNG cần công khai
mã nguồn (chỉ cần link động, Python thoả mãn tự nhiên).

### Các thay đổi cần thiết (có hệ thống):

| PyQt6 | PySide6 | Số chỗ |
|---|---|---|
| `from PyQt6.QtCore import ...` | `from PySide6.QtCore import ...` | 45 file |
| `from PyQt6.QtWidgets import ...` | `from PySide6.QtWidgets import ...` | 44 file |
| `from PyQt6.QtGui import ...` | `from PySide6.QtGui import ...` | 19 file |
| `pyqtSignal` | `Signal` | 181 |
| `pyqtSlot` | `Slot` | 4 |
| `pyqtProperty` | `Property` | (nếu có) |
| `.exec()` | `.exec()` | giữ nguyên (PySide6 6.x có) |

### Khác biệt tinh tế cần chú ý:
- **Enum:** PyQt6 bắt buộc `Qt.AlignmentFlag.AlignCenter`; PySide6 chấp nhận cả dạng ngắn
  `Qt.AlignCenter` lẫn dạng đầy đủ. Code hiện dùng dạng đầy đủ → **tương thích PySide6**.
- **Signal với overload:** cú pháp `Signal(int)` giống `pyqtSignal(int)`.
- **`@pyqtSlot` → `@Slot`:** decorator đổi tên, tham số giống.
- **QAction:** PyQt6 ở QtGui, PySide6 cũng QtGui → OK.
- **sip vs shiboken:** không dùng trực tiếp → không ảnh hưởng.

## 3. Quyết định qfluentwidgets (QUAN TRỌNG)

### Kết quả khảo sát thị trường:
**KHÔNG có thư viện Fluent Design nào cho PySide6 với license MIT/Apache miễn phí thương
mại đầy đủ.** Mọi bản (PyQt6-Fluent-Widgets, PySide6-Fluent-Widgets, kể cả các fork) đều
**GPL v3 cho phi thương mại / thương mại phải mua license**.

### 24 component đang dùng — phân loại:

**Nhóm A — widget cơ bản (16 cái), CÓ tương đương Qt thuần (miễn phí):**
CheckBox, ComboBox, LineEdit, Slider, SpinBox, DoubleSpinBox, PushButton,
PrimaryPushButton, ProgressBar, TextEdit, ToolButton, ScrollArea, CaptionLabel,
StrongBodyLabel, SpinBox → QCheckBox, QComboBox, QLineEdit... (chỉ khác thẩm mỹ).

**Nhóm B — đặc thù Fluent (8 cái), cần tự xây nếu bỏ:**
FluentWindow (cửa sổ + navigation), FluentIcon (bộ icon), HeaderCardWidget (thẻ),
InfoBar + InfoBarPosition (toast thông báo), setTheme/Theme/themeColor/isDarkTheme
(hệ thống theme sáng/tối).

### 3 lựa chọn:

#### Lựa chọn 1: MUA qfluentwidgets commercial license ⭐ (khuyến nghị nếu ưu tiên thẩm mỹ)
- Đổi `PyQt6-Fluent-Widgets` → `PySide6-Fluent-Widgets` (cùng tác giả, cùng API).
- Mua commercial license từ qfluentwidgets.com.
- **Công sức code: THẤP** (chỉ đổi import PyQt6→PySide6, giữ nguyên qfluentwidgets).
- **Chi phí: có** (license hàng năm, thường vài trăm CNY/USD — rẻ so với công xây lại UI).
- Giữ NGUYÊN giao diện đẹp.

#### Lựa chọn 2: BỎ qfluentwidgets → Qt thuần (PySide6 core, miễn phí)
- Thay 16 widget nhóm A bằng Qt thuần (dễ, chỉ đổi tên class).
- Tự xây 8 component nhóm B:
  - InfoBar → dùng QMessageBox hoặc tự viết toast widget (~100 dòng).
  - FluentWindow navigation → QMainWindow + QListWidget/QToolBar sidebar (~200 dòng).
  - HeaderCardWidget → QGroupBox + QSS (~50 dòng).
  - theme system → tự viết QSS sáng/tối + hàm switch (~150 dòng).
  - FluentIcon → dùng icon Qt tiêu chuẩn hoặc bộ icon MIT (vd Feather, Lucide).
- **Công sức code: CAO** (tự xây + tinh chỉnh QSS cho đẹp).
- **Chi phí: 0đ.**
- Giao diện: mất vẻ Fluent, cần đầu tư QSS để không xấu.

#### Lựa chọn 3: LAI — bỏ phần lớn, giữ tối thiểu tự xây
- Thay 16 widget nhóm A → Qt thuần.
- Tự xây theme + InfoBar + card (nhóm B) nhưng đơn giản hoá.
- Trung hoà giữa chi phí và công sức.

## 4. Khuyến nghị

**Nếu ứng dụng thương mại nghiêm túc, ưu tiên thẩm mỹ:** Lựa chọn 1 (mua license
qfluentwidgets). Chi phí license nhỏ so với giá trị giao diện chuyên nghiệp + tiết kiệm
hàng trăm giờ công.

**Nếu ngân sách bằng 0, chấp nhận đầu tư công sức:** Lựa chọn 2/3 (bỏ, tự xây Qt thuần).

## 5. Thứ tự thực hiện đề xuất

1. **Chuyển PyQt6 → PySide6 TRƯỚC** (áp dụng mọi lựa chọn). Đây là phần cơ học, an toàn,
   test bảo vệ. Làm từng bước:
   - Tạo lớp shim `qt_compat.py` re-export từ PySide6 (giảm rủi ro, dễ rollback).
   - Đổi import theo nhóm module, chạy test sau mỗi nhóm.
   - Đổi `pyqtSignal`→`Signal`, `pyqtSlot`→`Slot`.
2. **Sau khi PySide6 chạy ổn:** quyết định qfluentwidgets (mua license hoặc tự xây).

## 6. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| 181 pyqtSignal đổi sót | Dùng sed toàn cục + test bắt lỗi |
| Enum PyQt6-only vài chỗ | PySide6 khoan dung hơn; test phát hiện |
| qfluentwidgets cần PyQt6 | Bản PySide6-Fluent-Widgets cùng API |
| Test mock PyQt6 symbol | Cập nhật mock sang PySide6 |
