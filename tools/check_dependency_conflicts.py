"""Kiểm XUNG ĐỘT PHỤ THUỘC giữa các gói trước khi cài — tránh hỏng âm thầm.

VÌ SAO cần công cụ này
======================
v3.23.333 phát hiện ``whisperx`` ghim ``huggingface-hub<1.0.0`` trong khi ứng dụng dùng
``1.24.0``. Nguy hiểm ở chỗ: PaddleOCR/PaddleX/VieNeu **không ràng buộc phiên bản** gói
này, nên pip sẽ vui vẻ **hạ cấp** mà không báo lỗi — ứng dụng hỏng âm thầm, rất khó lần
ra nguyên nhân.

Công cụ này phát hiện sớm những trường hợp như vậy.

Điểm dễ sai khi đọc bằng mắt
----------------------------
Ràng buộc trong metadata thường KÈM ĐIỀU KIỆN::

    numpy (>=1.26)        ; python_version == "3.12"
    numpy (>=1.21,<2.3.0) ; python_version == "3.10"

Đọc lướt sẽ tưởng ``numpy<2.3.0`` là xung đột, nhưng nó chỉ áp dụng cho Python 3.10.
Công cụ này **đánh giá điều kiện** theo đúng môi trường đang dùng trước khi kết luận.

Cách dùng::

    python tools/check_dependency_conflicts.py
    python tools/check_dependency_conflicts.py whisperx pedalboard
"""

from __future__ import annotations

import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

#: Gói cố ý cài ở môi trường RIÊNG — không tính là xung đột với môi trường chính.
ISOLATED_PACKAGES: Final[frozenset[str]] = frozenset({"whisperx", "torch", "torchaudio", "torchvision"})


@dataclass(frozen=True, slots=True)
class Conflict:
    """Một xung đột phiên bản phát hiện được.

    Attributes:
        package: Gói yêu cầu ràng buộc.
        dependency: Gói bị ràng buộc.
        required: Chuỗi ràng buộc (vd ``"<1.0.0"``).
        installed: Phiên bản đang cài.
    """

    package: str
    dependency: str
    required: str
    installed: str

    def describe(self) -> str:
        """Mô tả xung đột kèm hệ quả."""
        return (
            f"{self.package} yêu cầu {self.dependency} {self.required} "
            f"nhưng đang cài {self.installed} → pip sẽ ĐỔI PHIÊN BẢN gói này."
        )


def normalise(name: str) -> str:
    """Chuẩn hoá tên gói theo quy ước PyPI (thường hoá, gạch dưới → gạch ngang)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def installed_versions(python_exe: str | None = None) -> dict[str, str]:
    """Đọc danh sách gói đang cài của một môi trường.

    Args:
        python_exe: Trình thông dịch cần kiểm; ``None`` = môi trường hiện tại.

    Returns:
        Dict ``{tên gói chuẩn hoá: phiên bản}``.
    """
    executable = python_exe or sys.executable
    try:
        result = subprocess.run(
            [executable, "-m", "pip", "list", "--format=freeze"],
            capture_output=True, text=True, timeout=120, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}

    versions: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "==" not in line:
            continue
        name, _, version = line.partition("==")
        versions[normalise(name)] = version.strip()
    return versions


def requirements_of(wheel_path: Path) -> list[str]:
    """Đọc danh sách ``Requires-Dist`` từ một tệp wheel.

    Args:
        wheel_path: Tệp ``.whl``.

    Returns:
        Các dòng ràng buộc, giữ nguyên phần điều kiện sau ``;``.
    """
    try:
        with zipfile.ZipFile(wheel_path) as archive:
            names = [n for n in archive.namelist() if n.endswith("METADATA")]
            if not names:
                return []
            text = archive.read(names[0]).decode("utf-8", errors="replace")
    except (OSError, zipfile.BadZipFile):
        return []

    return [
        line.replace("Requires-Dist:", "").strip()
        for line in text.splitlines()
        if line.startswith("Requires-Dist:")
    ]


def find_conflicts(
    package_name: str, requirement_lines: list[str], installed: dict[str, str]
) -> list[Conflict]:
    """Tìm ràng buộc KHÔNG thoả mãn với các gói đang cài.

    Đánh giá điều kiện môi trường (``python_version``, ``sys_platform``…) theo đúng môi
    trường hiện tại — bỏ qua ràng buộc không áp dụng, tránh báo động giả.

    Args:
        package_name: Tên gói đang xét.
        requirement_lines: Các dòng ``Requires-Dist``.
        installed: Phiên bản đang cài.

    Returns:
        Danh sách xung đột.
    """
    try:
        from packaging.requirements import Requirement
        from packaging.version import InvalidVersion, Version
    except ImportError:
        return []

    conflicts: list[Conflict] = []
    for line in requirement_lines:
        try:
            requirement = Requirement(line)
        except Exception:  # noqa: BLE001 — dòng lạ thì bỏ qua
            continue

        # Bỏ qua phần "extra" (chỉ cài khi người dùng yêu cầu tường minh).
        if requirement.marker is not None:
            try:
                if not requirement.marker.evaluate():
                    continue
            except Exception:  # noqa: BLE001 — marker có extra -> coi như không áp dụng
                continue

        dependency = normalise(requirement.name)
        current = installed.get(dependency)
        if current is None or not requirement.specifier:
            continue
        try:
            if requirement.specifier.contains(Version(current), prereleases=True):
                continue
        except InvalidVersion:
            continue

        conflicts.append(
            Conflict(package_name, dependency, str(requirement.specifier), current)
        )
    return conflicts


def installed_requirements(python_exe: str | None = None) -> dict[str, list[str]]:
    """Đọc ràng buộc của MỌI gói đang cài trong môi trường.

    Args:
        python_exe: Trình thông dịch cần kiểm; ``None`` = môi trường hiện tại.

    Returns:
        Dict ``{tên gói: danh sách Requires-Dist}``.
    """
    executable = python_exe or sys.executable
    code = (
        "import importlib.metadata as m, json;"
        "print(json.dumps({d.metadata['Name']: (d.requires or []) "
        "for d in m.distributions() if d.metadata['Name']}))"
    )
    try:
        result = subprocess.run(
            [executable, "-c", code],
            capture_output=True, text=True, timeout=120, check=False,
        )
        import json

        return json.loads(result.stdout or "{}")
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return {}


def check_installed_consistency(python_exe: str | None = None) -> list[Conflict]:
    """Kiểm MÔI TRƯỜNG HIỆN TẠI có tự mâu thuẫn không (tương đương ``pip check``).

    [v3.23.337] BỔ SUNG SAU MỘT SỰ CỐ THẬT. Bản trước chỉ hỏi "cài gói X có xung đột
    không" cho một danh sách gói cố định. Nhưng khi pip ĐÃ hạ cấp một gói, trạng thái
    trở nên tự nhất quán *với gói vừa cài*, nên câu hỏi đó trả lời "không xung đột" —
    trong khi các gói KHÁC đã hỏng.

    Log build thực tế: ``whisperx`` hạ ``huggingface-hub`` từ 1.25.1 xuống 0.36.2. Công
    cụ chạy sau đó vẫn báo sạch, dù pip đã in::

        gradio 6.20.0 requires huggingface-hub<2.0,>=1.2.0, but you have 0.36.2

    Hàm này quét NGƯỢC LẠI: với mọi gói đang cài, ràng buộc của nó có được thoả mãn bởi
    các gói khác đang cài không.

    Args:
        python_exe: Trình thông dịch cần kiểm.

    Returns:
        Danh sách gói đang bị hỏng phụ thuộc.
    """
    installed = installed_versions(python_exe)
    if not installed:
        return []

    conflicts: list[Conflict] = []
    for package, requirements in installed_requirements(python_exe).items():
        conflicts.extend(find_conflicts(package, list(requirements), installed))
    return conflicts


def download_metadata(package: str, destination: Path) -> Path | None:
    """Tải wheel của một gói (chỉ để đọc metadata, không cài).

    Args:
        package: Tên gói trên PyPI.
        destination: Thư mục lưu.

    Returns:
        Đường dẫn wheel, hoặc ``None`` nếu không tải được.
    """
    destination.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "download", "--no-deps", "-q",
             "-d", str(destination), package],
            capture_output=True, timeout=300, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    wheels = sorted(destination.glob("*.whl"))
    return wheels[-1] if wheels else None


def main() -> int:
    """Kiểm xung đột cho các gói truyền vào (mặc định: các gói tuỳ chọn của dự án)."""
    import tempfile

    packages = sys.argv[1:] or [
        "whisperx", "pedalboard", "fastembed", "scikit-learn",
        "edge-tts", "PySide6-Fluent-Widgets",
    ]

    print("=" * 68)
    print("  KIỂM XUNG ĐỘT PHỤ THUỘC")
    print("=" * 68)
    print(f"  Python: {sys.version.split()[0]}  ·  Nền tảng: {sys.platform}")

    installed = installed_versions()
    if not installed:
        print("\n  ⚠️ Không đọc được danh sách gói đang cài.")
        return 2
    print(f"  Đã cài: {len(installed)} gói\n")

    all_conflicts: list[Conflict] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for package in packages:
            wheel = download_metadata(package, Path(temp_dir) / normalise(package))
            if wheel is None:
                print(f"  {package:26} — không tải được metadata, bỏ qua")
                continue

            conflicts = find_conflicts(package, requirements_of(wheel), installed)
            isolated = normalise(package) in ISOLATED_PACKAGES
            if not conflicts:
                print(f"  {package:26} ✓ không xung đột")
            elif isolated:
                print(f"  {package:26} ⓘ có xung đột nhưng CÀI RIÊNG (không ảnh hưởng)")
                for conflict in conflicts:
                    print(f"      • {conflict.describe()}")
            else:
                print(f"  {package:26} ⛔ XUNG ĐỘT")
                for conflict in conflicts:
                    print(f"      • {conflict.describe()}")
                all_conflicts.extend(conflicts)

    # [v3.23.337] Kiểm tính nhất quán của MÔI TRƯỜNG — bắt hỏng đã xảy ra rồi.
    print("\n  Kiểm tính nhất quán môi trường hiện tại…")
    broken = check_installed_consistency()
    if broken:
        print(f"  ⛔ {len(broken)} gói ĐANG BỊ HỎNG phụ thuộc:")
        for conflict in broken:
            print(f"      • {conflict.describe()}")
        print("      → Môi trường đã bị một lần cài trước đó làm hỏng.")
    else:
        print("  ✓ Môi trường nhất quán.")
    all_conflicts.extend(broken)

    print("\n" + "=" * 68)
    if not all_conflicts:
        print("  ✓ Không có xung đột nào cần xử lý.")
        print("=" * 68)
        return 0
    print(f"  ⛔ {len(all_conflicts)} vấn đề phụ thuộc.")
    print("     Gói sắp cài: cân nhắc dùng môi trường riêng (như whisperx_env).")
    print("     Gói đã hỏng: chạy lại lệnh cài của gói đó để khôi phục phiên bản đúng.")
    print("=" * 68)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Conflict",
    "ISOLATED_PACKAGES",
    "check_installed_consistency",
    "find_conflicts",
    "installed_requirements",
    "installed_versions",
    "normalise",
    "requirements_of",
]
