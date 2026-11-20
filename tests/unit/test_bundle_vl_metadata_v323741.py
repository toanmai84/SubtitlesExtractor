"""[v3.23.741] Bản exe: PaddleOCR-VL lỗi ``name 'ImmutableSandboxedEnvironment' is not defined`` (log 18/11 09:12).

PaddleX không thử ``import`` mà hỏi METADATA gói cài đặt (``is_dep_available("Jinja2")`` → ``importlib.metadata``). Bản
exe chỉ chép metadata của các gói trong ``_METADATA_PACKAGES`` (``SubtitlesExtractor.spec``, chép đệ quy cả gói phụ
thuộc) — Jinja2 không có → PaddleX bỏ qua ``from jinja2.sandbox import ImmutableSandboxedEnvironment`` → NameError lúc
dựng mẫu hội thoại của bộ tách từ VL. Kế tiếp sẽ là ``require_deps("safetensors")`` khi đọc trọng số mô hình.

Test mô phỏng đúng điều kiện bản exe trong TIẾN TRÌNH CON: ``importlib.metadata`` chỉ thấy các gói ``.spec`` chép
metadata (cộng phụ thuộc đệ quy), rồi kiểm các điểm PaddleX hỏi trên đường chạy PaddleOCR-VL.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _spec_metadata_packages() -> list[str]:
    spec = (ROOT / "SubtitlesExtractor.spec").read_text(encoding="utf-8")
    block = spec.split("_METADATA_PACKAGES = (", 1)[1].split("\n)\n", 1)[0]
    block = re.sub(r"#[^\n]*", "", block)
    return re.findall(r'"([^"]+)"', block)


def test_vl_dependencies_listed_for_metadata_and_collection() -> None:
    names = {name.lower() for name in _spec_metadata_packages()}
    assert {"jinja2", "safetensors"} <= names
    spec = (ROOT / "SubtitlesExtractor.spec").read_text(encoding="utf-8")
    vl_block = spec.split("for _vl_pkg in (", 1)[1].split(")", 1)[0]
    for module in ("jinja2", "markupsafe", "safetensors"):
        assert f'"{module}"' in vl_block, module


SCRIPT = textwrap.dedent(
    """
    import importlib.metadata as md, re, sys
    listed = sys.argv[1].split(",")
    allowed, todo = set(), list(listed)
    while todo:                                   # chép metadata ĐỆ QUY như copy_metadata(recursive=True)
        name = todo.pop()
        key = name.lower().replace("_", "-")
        if key in allowed:
            continue
        allowed.add(key)
        try:
            reqs = md.requires(name) or []
        except md.PackageNotFoundError:
            continue
        for req in reqs:
            if "extra ==" in req:
                continue
            todo.append(re.split(r"[ ;<>=!~\\[(]", req, maxsplit=1)[0])
    real_version = md.version
    def bundle_version(dep):
        if dep.lower().replace("_", "-") not in allowed:
            raise md.PackageNotFoundError(dep)
        return real_version(dep)
    md.version = bundle_version
    from paddlex.utils import deps
    deps.is_dep_available.cache_clear() if hasattr(deps.is_dep_available, "cache_clear") else None
    from paddlex.inference.models.common.tokenizer import tokenizer_utils
    print("JINJA", hasattr(tokenizer_utils, "ImmutableSandboxedEnvironment"))
    print("SAFETENSORS", deps.is_dep_available("safetensors"))
    """
)


@pytest.mark.skipif(importlib.util.find_spec("paddlex") is None, reason="chưa cài paddlex")
def test_bundle_metadata_lets_paddlex_load_vl_tokenizer() -> None:
    packages = ",".join(_spec_metadata_packages())
    result = subprocess.run([sys.executable, "-c", SCRIPT, packages], capture_output=True, text=True, timeout=300,
                            env=dict(os.environ), check=False)
    found = dict(line.split() for line in result.stdout.splitlines() if line.split()[:1] in (["JINJA"], ["SAFETENSORS"]))
    assert found == {"JINJA": "True", "SAFETENSORS": "True"}, result.stderr[-1500:]
