"""[v3.23.443] Hồi quy: các method ghép ngữ cảnh KHÔNG được là staticmethod dùng self.

Bug v439→v442: refactor classmethod→instance để lộ _context_window/_batch_context vẫn là
@staticmethod nhưng thân đổi sang self._... → NameError 'self' is not defined khi DỊCH THẬT
(chỉ lộ lúc runtime, không lộ lúc compile). Test này gọi thẳng để chặn tái phát.
"""
from __future__ import annotations
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine


def _lines(n=10):
    return [TranslationLine(index=i, start_ms=i * 1000, end_ms=i * 1000 + 900,
                            text=f"câu {i}") for i in range(1, n + 1)]


def test_context_window_callable_on_instance():
    adapter = GeminiSubtitleTranslator(api_key="k")
    before, after = adapter._context_window(_lines(), 3, 2, 2)
    assert len(before) == 2 and len(after) == 2


def test_batch_context_callable_on_instance():
    adapter = GeminiSubtitleTranslator(api_key="k")
    lines = _lines()
    sb, sa, hb = adapter._batch_context(lines, 3, 2, 2, lines[:3])
    assert len(sb) == 2 and len(sa) == 2 and len(hb) == 2


def test_no_staticmethod_uses_self():
    """Quét AST: không method thật nào (không nested) là static/thiếu self mà dùng self."""
    import ast, inspect
    src = inspect.getsource(GeminiSubtitleTranslator)
    tree = ast.parse(src)
    parent = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}

    def nested(fn):
        p = parent.get(fn)
        while p is not None:
            if isinstance(p, ast.FunctionDef):
                return True
            p = parent.get(p)
        return False

    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and not nested(node):
            decos = [d.id for d in node.decorator_list if isinstance(d, ast.Name)]
            args = [a.arg for a in node.args.args]
            uses_self = any(isinstance(n, ast.Name) and n.id == "self"
                            for n in ast.walk(node))
            if uses_self and ("staticmethod" in decos or (args[:1] != ["self"])):
                bad.append(node.name)
    assert not bad, f"Method dùng self nhưng static/thiếu self: {bad}"
