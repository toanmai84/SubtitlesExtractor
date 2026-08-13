r"""[v3.23.397] Tải + cấu hình Python EMBEDDABLE vào vendor/python_embed/ (chạy lúc BUILD).

Mục đích: nhúng sẵn một Python nhẹ (~15-25MB) ĐÚNG phiên bản Python đang build, kèm pip, để
bản .exe TỰ LẬP chạy các bước tải-lúc-chạy (paddlepaddle-gpu, CUDA) mà KHÔNG cần Python cài sẵn
trên máy người dùng. Nhờ khớp phiên bản, wheel paddle luôn đúng ABI.

Chạy tay:
    python tools/setup_embedded_python.py            # dùng phiên bản Python hiện tại
    python tools/setup_embedded_python.py --force     # tải lại dù đã có

Idempotent: nếu vendor/python_embed/python.exe đã tồn tại thì bỏ qua (trừ khi --force).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

_GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"


def current_python_version() -> str:
    """Phiên bản Python ĐANG CHẠY dạng ``major.minor.micro`` (khớp Python sẽ bundle)."""
    v = sys.version_info
    return f"{v.major}.{v.minor}.{v.micro}"


def embedded_python_url(version: str) -> str:
    """URL chính thức của Python embeddable (Windows amd64) cho ``version``."""
    return (
        f"https://www.python.org/ftp/python/{version}/"
        f"python-{version}-embed-amd64.zip"
    )


def _url_exists(url: str) -> bool:
    """Kiểm URL có tồn tại (HTTP 200) bằng request HEAD nhẹ."""
    request = urllib.request.Request(url, method="HEAD")  # noqa: S310 — python.org
    try:
        with urllib.request.urlopen(request, timeout=20) as resp:  # noqa: S310
            return 200 <= resp.status < 300
    except OSError:
        return False


def find_available_embed_version(
    version: str, *, exists: object = None
) -> str | None:
    """Tìm phiên bản 3.x.y CÓ bản embeddable tải được, dò ngược micro từ ``version``.

    python.org KHÔNG phát hành embeddable cho mọi micro (bản security-only chỉ có source).
    Wheel Python là ``cpXY`` (chỉ phụ thuộc major.minor), nên BẤT KỲ micro nào cùng minor đều
    chạy được. Dò từ micro yêu cầu xuống 0, trả về bản đầu tiên tồn tại.

    Args:
        version: Phiên bản mong muốn ``major.minor.micro``.
        exists: Hàm kiểm tồn tại (mặc định :func:`_url_exists`); tách ra để test được.

    Returns:
        Chuỗi phiên bản tải được, hoặc ``None`` nếu không có micro nào của minor đó có embeddable.
    """
    check = exists or _url_exists
    major, minor, micro = (int(p) for p in version.split("."))
    for candidate_micro in range(micro, -1, -1):
        candidate = f"{major}.{minor}.{candidate_micro}"
        if check(embedded_python_url(candidate)):
            return candidate
    return None


def enable_site_in_pth(pth_path: Path) -> bool:
    """Bỏ chú thích ``import site`` trong file ``python3XX._pth`` để pip hoạt động.

    Bản embeddable mặc định TẮT site-packages (dòng ``#import site``). Phải bật để pip (cài
    vào ``Lib/site-packages``) import được.

    Args:
        pth_path: Đường dẫn file ``._pth``.

    Returns:
        ``True`` nếu đã sửa (hoặc vốn đã bật); ``False`` nếu không tìm thấy dòng.
    """
    lines = pth_path.read_text(encoding="utf-8").splitlines()
    changed = False
    has_site = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped in ("import site", "#import site", "# import site"):
            if stripped != "import site":
                lines[i] = "import site"
                changed = True
            has_site = True
    if not has_site:
        lines.append("import site")
        changed = True
    if changed:
        pth_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def _find_pth(target_dir: Path) -> Path | None:
    """Tìm file ``python3XX._pth`` trong thư mục embeddable."""
    for candidate in sorted(target_dir.glob("python*._pth")):
        return candidate
    return None


def setup_embedded_python(
    vendor_dir: Path, *, version: str | None = None, force: bool = False
) -> Path:
    """Tải + cấu hình Python embeddable vào ``vendor_dir/python_embed/``.

    Args:
        vendor_dir: Thư mục ``vendor/`` của dự án.
        version: Phiên bản Python (mặc định: phiên bản đang chạy).
        force: Tải lại dù đã có.

    Returns:
        Đường dẫn ``python.exe`` embeddable đã cấu hình.

    Raises:
        RuntimeError: Nếu tải/giải nén/cài pip thất bại.
    """
    version = version or current_python_version()
    target = vendor_dir / "python_embed"
    python_exe = target / "python.exe"

    if python_exe.is_file() and not force:
        print(f"[embed] Đã có {python_exe} — bỏ qua (dùng --force để tải lại).")
        return python_exe

    # python.org không phát hành embeddable cho mọi micro (bản security-only chỉ có source).
    # Wheel là cpXY nên bất kỳ micro cùng minor cũng chạy — dò xuống tới bản tải được.
    available = find_available_embed_version(version)
    if available is None:
        raise RuntimeError(
            f"Không tìm thấy bản Python embeddable nào cho dòng "
            f"{'.'.join(version.split('.')[:2])} (đã dò từ {version} xuống .0)."
        )
    if available != version:
        print(
            f"[embed] {version} không có embeddable → dùng {available} "
            f"(cùng dòng {'.'.join(version.split('.')[:2])}, wheel cpXY tương thích)."
        )
    version = available

    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / "_embed.zip"
    url = embedded_python_url(version)
    print(f"[embed] Tải Python embeddable {version}: {url}")
    try:
        urllib.request.urlretrieve(url, zip_path)  # noqa: S310 — URL cố định python.org
    except OSError as exc:
        raise RuntimeError(f"Không tải được Python embeddable: {exc}") from exc

    print("[embed] Giải nén…")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)
    zip_path.unlink(missing_ok=True)

    pth = _find_pth(target)
    if pth is None:
        raise RuntimeError("Không tìm thấy file python3XX._pth trong bản embeddable.")
    enable_site_in_pth(pth)

    print("[embed] Cài pip vào bản embeddable…")
    get_pip = target / "get-pip.py"
    try:
        urllib.request.urlretrieve(_GET_PIP_URL, get_pip)  # noqa: S310
    except OSError as exc:
        raise RuntimeError(f"Không tải được get-pip.py: {exc}") from exc
    result = subprocess.run(
        [str(python_exe), str(get_pip), "--no-warn-script-location"],
        capture_output=True,
        text=True,
        check=False,
    )
    get_pip.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Cài pip cho embeddable thất bại:\n"
            + (result.stderr or result.stdout or "")[-500:]
        )

    print(f"[embed] ✓ Xong: {python_exe}")
    return python_exe


def main() -> int:
    parser = argparse.ArgumentParser(description="Tải + cấu hình Python embeddable.")
    parser.add_argument("--force", action="store_true", help="Tải lại dù đã có.")
    parser.add_argument("--version", default=None, help="Phiên bản Python (mặc định: hiện tại).")
    args = parser.parse_args()
    vendor_dir = Path(__file__).resolve().parent.parent / "vendor"
    try:
        setup_embedded_python(vendor_dir, version=args.version, force=args.force)
    except RuntimeError as exc:
        print(f"[embed] LỖI: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
