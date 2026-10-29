"""[v3.23.666] Nhà cung cấp AI TUỲ CHỈNH tương thích OpenAI (vd OxionAI ``https://oxionai.com/v1``).

Mọi khâu văn bản của ứng dụng (phân tích ngữ cảnh, dịch nhiều giai đoạn, tự kiểm tra, rút gọn câu
cho lồng tiếng) gọi ĐÚNG MỘT chỗ: ``client.models.generate_content(model=, contents=, config=)`` của
SDK ``google-genai``. Module này cung cấp một "client" cùng dáng đó nhưng gửi tới
``POST {base_url}/chat/completions`` theo chuẩn OpenAI, rồi trả về một đối tượng phản hồi cùng dáng
SDK Gemini (``text``, ``candidates[0].finish_reason``, ``usage_metadata``, ``model_version``) — nên
toàn bộ vòng thử lại, sửa JSON, kiểm lô, sổ token… chạy y nguyên, không phải viết lại.

Ánh xạ:

* ``config.system_instruction`` → tin nhắn ``system``; ``response_schema`` (JSON) được chép vào
  cuối tin nhắn này kèm yêu cầu "chỉ trả MỘT đối tượng JSON" và gửi ``response_format =
  {"type": "json_object"}`` (OxionAI hỗ trợ ``json_object``, không hỗ trợ ``json_schema``).
* Phần chữ của ``contents`` → tin nhắn ``user``; ảnh ``inline_data`` → phần ``image_url`` dạng
  ``data:`` URL. Video / file đã tải lên Gemini (``file_data``) KHÔNG gửi được → lỗi rõ ràng.
* ``temperature`` / ``top_p`` / ``max_output_tokens`` → ``temperature`` / ``top_p`` /
  ``max_tokens``. Cấu hình suy nghĩ của Gemini (``thinking_config``) bị bỏ — model của nhà cung
  cấp tự quyết.
* Lỗi HTTP → :class:`CustomProviderError` có ``code`` = mã HTTP và cờ ``retryable`` để vòng thử
  lại của adapter phân loại đúng (429/5xx thử lại; 400/401/402/403/404 dừng ngay).

Chỉ dùng thư viện chuẩn (``urllib``) — không thêm phụ thuộc cho bản exe.
"""

from __future__ import annotations

import base64
import http.client
import json
import logging
import re
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)

#: Ví dụ địa chỉ hiện trong ô nhập (người dùng hỏi đúng nhà cung cấp này).
EXAMPLE_BASE_URL = "https://oxionai.com/v1"

#: Thời hạn mặc định mỗi lời gọi khi caller không truyền ``http_options``.
DEFAULT_TIMEOUT_S = 120.0

#: Trần token đầu ra gửi đi khi cấu hình không nêu. OxionAI mặc định chỉ 8192 — lô dịch 40 dòng
#: kèm ghi chú có thể vượt và bị cắt cụt (``finish_reason=length``). Máy chủ chê con số này thì
#: :class:`_Models` bỏ hẳn trường ``max_tokens`` và gửi lại (nhớ theo model).
DEFAULT_MAX_TOKENS = 32768

#: Trần dung lượng một ảnh / cả request theo tài liệu OxionAI (vượt thì máy chủ trả 413).
MAX_IMAGE_BYTES = 5 * 1024 * 1024

_STATUS_NAMES = {
    400: "BAD_REQUEST",
    401: "UNAUTHENTICATED",
    402: "PAYMENT_REQUIRED",
    403: "PERMISSION_DENIED",
    404: "NOT_FOUND",
    408: "DEADLINE_EXCEEDED",
    409: "CONFLICT",
    413: "PAYLOAD_TOO_LARGE",
    422: "UNPROCESSABLE",
    429: "RESOURCE_EXHAUSTED",
    500: "INTERNAL",
    502: "UNAVAILABLE",
    503: "UNAVAILABLE",
    504: "DEADLINE_EXCEEDED",
}
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: ``finish_reason`` chuẩn OpenAI → tên lý do Gemini mà ``empty_response_policy`` hiểu.
_FINISH_REASONS = {
    "stop": "STOP",
    "length": "MAX_TOKENS",
    "content_filter": "SAFETY",
    "tool_calls": "STOP",
    "function_call": "STOP",
}

_THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)

#: Model KHÔNG sinh văn bản (bỏ khỏi danh sách chọn model dịch).
_NON_TEXT_MARKERS = (
    "embedding", "embed-", "tts", "whisper", "transcribe", "dall-e", "imagen", "veo",
    "image-gen", "-image", "moderation", "rerank", "audio", "realtime", "sora",
)


@dataclass(frozen=True, slots=True)
class CustomProviderConfig:
    """Cấu hình nhà cung cấp tuỳ chỉnh: địa chỉ API gốc + API key.

    Attributes:
        base_url: Địa chỉ gốc kiểu OpenAI, vd ``https://oxionai.com/v1``.
        api_key: Khoá Bearer của nhà cung cấp.
    """

    base_url: str
    api_key: str

    @property
    def normalized_base_url(self) -> str:
        """Địa chỉ gốc đã chuẩn hoá (bỏ khoảng trắng, ``/`` cuối, đuôi ``/chat/completions``)."""
        return normalize_base_url(self.base_url)

    @property
    def is_complete(self) -> bool:
        """Đủ địa chỉ hợp lệ (http/https, có tên máy, cổng đọc được) và key ASCII để gọi.

        [v3.23.666] Key dán kèm ký tự lạ (khoảng trắng rộng 0, chữ Kirin…) không đặt được vào
        header HTTP; địa chỉ có khoảng trắng / cổng sai làm ``urllib`` ném lỗi khó hiểu ở mỗi lô.
        """
        return is_valid_base_url(self.base_url) and is_valid_api_key(self.api_key)


def is_valid_base_url(url: str) -> bool:
    """Địa chỉ gốc dùng được: http(s), có tên máy, không khoảng trắng, cổng (nếu có) là số."""
    text = normalize_base_url(url)
    if not text or any(ch.isspace() for ch in text):
        return False
    try:
        parts = urllib.parse.urlsplit(text)
        _ = parts.port  # ném ValueError nếu cổng không phải số / ngoài khoảng
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and bool(parts.hostname)


def is_valid_api_key(key: str) -> bool:
    """Key đặt được vào header ``Authorization`` (ASCII in được, không khoảng trắng giữa)."""
    text = (key or "").strip()
    return bool(text) and all(33 <= ord(ch) < 127 for ch in text)


def normalize_base_url(url: str) -> str:
    """Chuẩn hoá địa chỉ gốc người dùng dán vào.

    Chấp nhận cả khi người dùng dán nguyên ``…/v1/chat/completions`` hoặc ``…/v1/models``.

    Args:
        url: Chuỗi người dùng nhập.

    Returns:
        Địa chỉ gốc không có ``/`` cuối; rỗng nếu đầu vào rỗng.
    """
    text = (url or "").strip().rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/models"):
        if text.lower().endswith(suffix):
            text = text[: -len(suffix)].rstrip("/")
    return text


class CustomProviderError(Exception):
    """Lỗi từ nhà cung cấp tuỳ chỉnh, dáng giống lỗi SDK Gemini (``"429 RESOURCE_EXHAUSTED …"``).

    Attributes:
        code: Mã HTTP (0 = lỗi mạng, không có phản hồi).
        status: Tên trạng thái kiểu Gemini (``RESOURCE_EXHAUSTED``…).
        message: Thông điệp máy chủ (đã rút gọn).
        retryable: Đáng thử lại không (429/5xx/lỗi mạng: có; 4xx khác: không).
    """

    def __init__(self, code: int, message: str, *, retryable: bool | None = None,
                 status: str = "") -> None:
        self.code = int(code)
        self.status = status or _STATUS_NAMES.get(self.code, "ERROR" if self.code else "NETWORK")
        self.message = (message or "").strip()[:600]
        self.retryable = (self.code in _RETRYABLE_STATUS) if retryable is None else retryable
        super().__init__(f"{self.code} {self.status}. {self.message}")


class UnsupportedContentError(CustomProviderError):
    """Nội dung không gửi được qua API OpenAI (video/file đã tải lên Gemini)."""

    def __init__(self, message: str) -> None:
        super().__init__(400, message, retryable=False, status="UNSUPPORTED_CONTENT")


# ── Dựng request ─────────────────────────────────────────────────────────────


def _plain(value: Any) -> Any:
    """Đổi Schema/pydantic/enum về dict/list/str thuần để ``json.dumps`` được."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _plain(dump(exclude_none=True))
        except TypeError:
            return _plain(dump())
    name = getattr(value, "value", None)
    if isinstance(name, str):
        return name
    return str(value)


def _text_of(value: Any) -> str:
    """Chữ của một ``system_instruction`` (chuỗi, Content, Part, hoặc danh sách)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(t for t in (_text_of(v) for v in value) if t)
    parts = getattr(value, "parts", None)
    if parts:
        return _text_of(list(parts))
    text = getattr(value, "text", None)
    return text if isinstance(text, str) else ""


def _json_instruction(schema: Any) -> str:
    """Lời dặn trả JSON (kèm schema nếu có) nối vào cuối tin nhắn system."""
    lines = [
        "# ĐỊNH DẠNG ĐẦU RA (bắt buộc)",
        "Chỉ trả về DUY NHẤT một đối tượng JSON hợp lệ — không markdown, không ```, không lời "
        "giải thích trước hay sau.",
    ]
    plain = _plain(schema)
    if plain:
        lines.append("Đối tượng JSON phải đúng JSON Schema sau (đủ mọi trường bắt buộc):")
        lines.append(json.dumps(plain, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines)


def _image_part(mime: str, data: Any) -> dict[str, Any]:
    """Phần ``image_url`` dạng data URL từ byte ảnh (hoặc chuỗi base64 sẵn)."""
    if isinstance(data, (bytes, bytearray)):
        if len(data) > MAX_IMAGE_BYTES:
            raise UnsupportedContentError(
                f"Ảnh {len(data) / 1048576:.1f} MB vượt trần 5 MB của nhà cung cấp tuỳ chỉnh."
            )
        encoded = base64.b64encode(bytes(data)).decode("ascii")
    else:
        encoded = str(data)
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def _content_parts(contents: Any) -> list[dict[str, Any]]:
    """Các phần ``user`` (chữ / ảnh) từ ``contents`` dáng SDK Gemini."""
    items = contents if isinstance(contents, (list, tuple)) else [contents]
    parts: list[dict[str, Any]] = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, str):
            if item:
                parts.append({"type": "text", "text": item})
            continue
        nested = getattr(item, "parts", None)
        if nested and not isinstance(item, (bytes, bytearray)):
            parts.extend(_content_parts(list(nested)))
            continue
        if getattr(item, "file_data", None) is not None or getattr(item, "uri", None):
            raise UnsupportedContentError(
                "Nhà cung cấp tuỳ chỉnh (API kiểu OpenAI) không nhận video/tệp đã tải lên "
                "Gemini — chỉ nhận chữ và ảnh."
            )
        inline = getattr(item, "inline_data", None)
        if inline is not None:
            mime = str(getattr(inline, "mime_type", "") or "")
            if not mime.startswith("image/"):
                raise UnsupportedContentError(
                    f"Nhà cung cấp tuỳ chỉnh không nhận dữ liệu '{mime or 'không rõ'}' "
                    "(chỉ ảnh PNG/JPEG/WebP/GIF)."
                )
            parts.append(_image_part(mime, getattr(inline, "data", b"")))
            continue
        text = getattr(item, "text", None)
        if isinstance(text, str):
            if text:
                parts.append({"type": "text", "text": text})
            continue
        raise UnsupportedContentError(
            f"Không gửi được phần nội dung kiểu {type(item).__name__} tới nhà cung cấp tuỳ chỉnh."
        )
    return parts


def _wants_json(config: Any) -> bool:
    mime = str(getattr(config, "response_mime_type", "") or "").lower()
    return "json" in mime or getattr(config, "response_schema", None) is not None


def build_chat_payload(
    model: str, contents: Any, config: Any, *, max_tokens: int | None = DEFAULT_MAX_TOKENS,
    json_mode: bool = True,
) -> dict[str, Any]:
    """Thân request ``/chat/completions`` tương ứng một lời gọi ``generate_content`` (hàm thuần).

    Args:
        model: Tên model của nhà cung cấp (vd ``gemini-3.7-flash``).
        contents: ``contents`` dáng SDK Gemini (chuỗi / Part / danh sách).
        config: ``GenerateContentConfig`` (hoặc đối tượng cùng thuộc tính), có thể ``None``.
        max_tokens: Trần token đầu ra khi cấu hình không nêu; ``None`` = không gửi trường này.
        json_mode: Gửi ``response_format = json_object`` khi cấu hình đòi JSON.

    Returns:
        Dict sẵn sàng ``json.dumps``.

    Raises:
        UnsupportedContentError: ``contents`` có video/tệp/âm thanh.
    """
    system = _text_of(getattr(config, "system_instruction", None)).strip()
    wants_json = _wants_json(config)
    if wants_json:
        instruction = _json_instruction(getattr(config, "response_schema", None))
        system = f"{system}\n\n{instruction}" if system else instruction
    parts = _content_parts(contents)
    if not parts:
        parts = [{"type": "text", "text": ""}]
    user_content: Any
    if all(p["type"] == "text" for p in parts):
        # Chuỗi thuần hợp với MỌI máy chủ kiểu OpenAI (một số không nhận mảng phần).
        user_content = "\n\n".join(p["text"] for p in parts)
    else:
        user_content = parts
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})
    payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
    for name in ("temperature", "top_p"):
        value = getattr(config, name, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            payload[name] = float(value)
    configured_max = getattr(config, "max_output_tokens", None)
    if isinstance(configured_max, int) and configured_max > 0:
        payload["max_tokens"] = configured_max
    elif max_tokens:
        payload["max_tokens"] = int(max_tokens)
    seed = getattr(config, "seed", None)
    if isinstance(seed, int) and not isinstance(seed, bool):
        payload["seed"] = seed
    if wants_json and json_mode:
        payload["response_format"] = {"type": "json_object"}
    return payload


# ── Đọc phản hồi ─────────────────────────────────────────────────────────────


def _message_text(message: dict[str, Any]) -> str | None:
    content = message.get("content")
    if isinstance(content, list):
        texts = [
            str(p.get("text", "")) for p in content
            if isinstance(p, dict) and p.get("type") in (None, "text", "output_text")
        ]
        content = "".join(texts)
    if not isinstance(content, str):
        return None
    # Model suy luận kiểu DeepSeek có thể chèn khối <think>…</think> ngay trong content.
    return _THINK_BLOCK_RE.sub("", content, count=1)


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _usage(raw: Any) -> SimpleNamespace | None:
    if not isinstance(raw, dict):
        return None
    completion_details = raw.get("completion_tokens_details") or {}
    prompt_details = raw.get("prompt_tokens_details") or {}
    reasoning = _int(completion_details.get("reasoning_tokens")) if isinstance(
        completion_details, dict) else None
    cached = _int(prompt_details.get("cached_tokens")) if isinstance(prompt_details, dict) else None
    completion = _int(raw.get("completion_tokens"))
    # Gemini đếm suy nghĩ RIÊNG (thoughts_token_count), OpenAI gộp vào completion_tokens.
    candidates = (
        completion - reasoning if completion is not None and reasoning else completion
    )
    return SimpleNamespace(
        prompt_token_count=_int(raw.get("prompt_tokens")),
        candidates_token_count=candidates,
        thoughts_token_count=reasoning,
        cached_content_token_count=cached,
        total_token_count=_int(raw.get("total_tokens")),
    )


def parse_chat_response(data: Any) -> SimpleNamespace:
    """Phản hồi ``/chat/completions`` → đối tượng dáng phản hồi SDK Gemini (hàm thuần).

    Args:
        data: JSON đã giải mã.

    Returns:
        Đối tượng có ``text`` (``None`` khi không có chữ), ``candidates``, ``usage_metadata``,
        ``model_version``, ``prompt_feedback``.

    Raises:
        CustomProviderError: Thân phản hồi không đúng dạng chuẩn OpenAI.
    """
    if not isinstance(data, dict):
        raise CustomProviderError(502, "Phản hồi của nhà cung cấp không phải JSON object.", retryable=False,
                                  status="BAD_RESPONSE")
    if isinstance(data.get("error"), dict):
        error = data["error"]
        code = _int(error.get("status")) or _int(error.get("code")) or 500
        raise CustomProviderError(code, str(error.get("message") or error))
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise CustomProviderError(502, "Phản hồi của nhà cung cấp thiếu 'choices'.", retryable=False,
                                  status="BAD_RESPONSE")
    first = choices[0]
    raw_message = first.get("message")
    message: dict[str, Any] = raw_message if isinstance(raw_message, dict) else {}
    text = _message_text(message)
    if text is not None and not text.strip():
        text = None
    raw_reason = str(first.get("finish_reason") or "").lower()
    finish = _FINISH_REASONS.get(raw_reason, raw_reason.upper() or "")
    candidate = SimpleNamespace(
        finish_reason=finish or None,
        content=SimpleNamespace(parts=[SimpleNamespace(text=text)] if text else []),
    )
    return SimpleNamespace(
        text=text,
        candidates=[candidate],
        usage_metadata=_usage(data.get("usage")),
        model_version=str(data.get("model") or ""),
        prompt_feedback=None,
    )


def is_text_model(model_id: str, caps: Any = None) -> bool:
    """Model sinh văn bản dùng được cho dịch (loại embedding/TTS/ảnh…)."""
    name = (model_id or "").lower()
    if not name or any(marker in name for marker in _NON_TEXT_MARKERS):
        return False
    if isinstance(caps, dict) and caps.get("chat") is False:
        return False
    return True


# ── Gửi HTTP ─────────────────────────────────────────────────────────────────


def _error_message(body: bytes) -> str:
    try:
        data = json.loads(body.decode("utf-8", errors="replace"))
    except (ValueError, UnicodeDecodeError):
        return body.decode("utf-8", errors="replace")[:300]
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            code = error.get("code") or error.get("type") or ""
            message = str(error.get("message") or "")
            return f"{message} ({code})" if code else message
        if isinstance(error, str):
            return error
        if "message" in data:
            return str(data["message"])
    return str(data)[:300]


class _Transport:
    """Gửi một request JSON và đọc JSON trả về (``urllib``; tôn trọng HTTPS_PROXY của hệ thống)."""

    def __init__(self, base_url: str, api_key: str, timeout_s: float) -> None:
        self._base_url = normalize_base_url(base_url)
        self._api_key = api_key.strip()
        self._timeout_s = max(5.0, float(timeout_s))

    def _bad_reply(self, what: str) -> CustomProviderError:
        """[v3.23.666] Phản hồi sai giao thức (HTML, thiếu ``choices``…): KHÔNG thử lại.

        Thường là sai địa chỉ (thiếu ``/v1``) — thử lại 11 lần trong ngân sách "quá tải" 10 phút
        chỉ làm người dùng chờ vô ích.
        """
        return CustomProviderError(
            502, f"{what} — kiểm tra lại Địa chỉ API ({self._base_url}), thường phải kết thúc bằng /v1.",
            retryable=False, status="BAD_RESPONSE",
        )

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None,
                timeout_s: float | None = None) -> Any:
        timeout = max(5.0, float(timeout_s)) if timeout_s else self._timeout_s
        url = f"{self._base_url}/{path.lstrip('/')}"
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self._api_key}")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", "SubtitlesExtractor")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                content_type = str(resp.headers.get("Content-Type", "") or "").lower()
        except urllib.error.HTTPError as exc:
            body = b""
            try:
                body = exc.read() or b""
            except OSError:
                pass
            raise CustomProviderError(exc.code, _error_message(body) or str(exc.reason)) from exc
        except TimeoutError as exc:
            raise CustomProviderError(
                504, f"Request timed out sau {timeout:.0f} giây.", retryable=True,
            ) from exc
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, (TimeoutError, socket.timeout)):
                raise CustomProviderError(
                    504, f"Request timed out sau {timeout:.0f} giây.", retryable=True,
                ) from exc
            raise CustomProviderError(
                0, f"Network error: không kết nối được {self._base_url} ({reason}).",
                retryable=True,
            ) from exc
        except (ConnectionError, OSError) as exc:
            raise CustomProviderError(
                0, f"Network error: connection reset ({exc}).", retryable=True,
            ) from exc
        except http.client.HTTPException as exc:
            # IncompleteRead / RemoteDisconnected…: thân bị cắt giữa chừng — nhất thời.
            raise CustomProviderError(
                0, f"Network error: connection closed ({type(exc).__name__}).", retryable=True,
            ) from exc
        except (ValueError, UnicodeError) as exc:
            # Địa chỉ không phân tích được (InvalidURL là ValueError) / key có ký tự ngoài latin-1.
            raise CustomProviderError(
                400, f"Địa chỉ API hoặc API key không hợp lệ ({type(exc).__name__}).",
                retryable=False, status="INVALID_CONFIG",
            ) from exc
        if "html" in content_type:
            raise self._bad_reply("Máy chủ trả về trang web HTML thay vì JSON")
        try:
            data = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise self._bad_reply("Máy chủ trả về nội dung không phải JSON") from exc
        if path == "chat/completions" and isinstance(data, dict) and "error" not in data and not (
            isinstance(data.get("choices"), list) and data["choices"]
        ):
            raise self._bad_reply("Phản hồi không có 'choices' (không phải API kiểu OpenAI?)")
        return data


def _is_param_rejection(error: CustomProviderError, words: tuple[str, ...]) -> bool:
    if error.code not in (400, 422):
        return False
    text = error.message.lower()
    return any(word in text for word in words)


#: Trần token đầu ra thấp nhất chấp nhận khi đọc giới hạn máy chủ nêu trong lỗi.
_MIN_STATED_MAX_TOKENS = 1024


def stated_max_tokens(message: str, sent: int) -> int | None:
    """Giới hạn ``max_tokens`` máy chủ nêu trong lỗi (số lớn nhất nhỏ hơn số đã gửi), hoặc ``None``.

    Vd ``"max_tokens is too large: 32768 > 16384"`` → 16384.
    """
    numbers = [int(n) for n in re.findall(r"\d{3,7}", message.replace(",", ""))]
    candidates = [n for n in numbers if _MIN_STATED_MAX_TOKENS <= n < sent]
    return max(candidates) if candidates else None


class _Models:
    """``client.models`` dáng SDK Gemini: ``generate_content`` + ``list``."""

    def __init__(self, transport: _Transport, quirks: dict[str, dict[str, Any]],
                 lock: threading.Lock) -> None:
        self._transport = transport
        self._quirks = quirks
        self._lock = lock

    def _model_quirks(self, model: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._quirks.get(model, {}))

    def _remember(self, model: str, name: str, value: Any) -> None:
        with self._lock:
            self._quirks.setdefault(model, {})[name] = value

    def generate_content(self, *, model: str, contents: Any, config: Any = None) -> Any:
        """Gửi một lời gọi chat completion; trả phản hồi dáng Gemini.

        Máy chủ chê ``response_format`` (400) thì bỏ trường đó; chê ``max_tokens`` thì hạ xuống
        đúng giới hạn máy chủ nêu (không nêu thì bỏ trường) — gửi lại NGAY và nhớ cho model này.
        """
        timeout_s = _timeout_seconds(getattr(config, "http_options", None))
        for _ in range(4):
            quirks = self._model_quirks(model)
            payload = build_chat_payload(
                model, contents, config,
                max_tokens=quirks.get("max_tokens", DEFAULT_MAX_TOKENS),
                json_mode=quirks.get("json_mode", True),
            )
            try:
                return parse_chat_response(
                    self._transport.request("POST", "chat/completions", payload, timeout_s)
                )
            except CustomProviderError as exc:
                if "response_format" in payload and _is_param_rejection(
                    exc, ("response_format", "json_object", "json mode", "json_mode")
                ):
                    logger.info("Model '%s' không nhận response_format=json_object → bỏ, gửi lại.", model)
                    self._remember(model, "json_mode", False)
                    continue
                # Chỉ khi lỗi NÊU TÊN trường: "maximum context length" là prompt quá dài — bỏ
                # max_tokens không chữa được mà còn làm mọi lô sau bị cắt ở 8192 token.
                if "max_tokens" in payload and _is_param_rejection(
                    exc, ("max_tokens", "max_completion_tokens", "max_output_tokens")
                ):
                    sent = int(payload["max_tokens"])
                    limit = stated_max_tokens(exc.message, sent)
                    logger.info("Model '%s' chê max_tokens=%s → %s, gửi lại.", model, sent,
                                limit if limit else "để máy chủ tự đặt")
                    self._remember(model, "max_tokens", limit)
                    continue
                raise
        raise CustomProviderError(400, "Máy chủ liên tục từ chối tham số request.", retryable=False)

    def list(self) -> list[Any]:
        """Danh sách model (``GET /models``) dáng SDK Gemini: mỗi phần tử có ``name``."""
        data = self._transport.request("GET", "models")
        items = data.get("data") if isinstance(data, dict) else data
        out: list[Any] = []
        for item in items or []:
            if isinstance(item, dict) and item.get("id"):
                out.append(SimpleNamespace(
                    name=str(item["id"]), caps=item.get("capabilities") or item.get("caps"),
                    context_length=item.get("context_length") or item.get("context_window"),
                ))
            elif isinstance(item, str):
                out.append(SimpleNamespace(name=item, caps=None, context_length=None))
        return out


def _timeout_seconds(http_options: Any) -> float | None:
    """Thời hạn (giây) từ ``http_options`` dáng SDK (``{"timeout": ms}`` hoặc ``HttpOptions``)."""
    if http_options is None:
        return None
    if isinstance(http_options, dict):
        timeout_ms = http_options.get("timeout")
    else:
        timeout_ms = getattr(http_options, "timeout", None)
    if isinstance(timeout_ms, (int, float)) and not isinstance(timeout_ms, bool) and timeout_ms > 0:
        return float(timeout_ms) / 1000.0
    return None


class OpenAICompatClient:
    """Client dáng ``google.genai.Client`` cho một API kiểu OpenAI."""

    def __init__(self, base_url: str, api_key: str, timeout_s: float,
                 quirks: dict[str, dict[str, Any]], lock: threading.Lock) -> None:
        self.models = _Models(_Transport(base_url, api_key, timeout_s), quirks, lock)


class OpenAICompatModule:
    """Thay cho module ``google.genai`` trong adapter dịch: ``Client(api_key=, http_options=)``.

    Ghi nhớ "tật" của từng model (không nhận json_object / max_tokens) dùng chung giữa các client.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = normalize_base_url(base_url)
        self._quirks: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def Client(self, api_key: str, http_options: Any = None) -> OpenAICompatClient:  # noqa: N802 — cùng tên SDK
        timeout_s = _timeout_seconds(http_options) or DEFAULT_TIMEOUT_S
        return OpenAICompatClient(self.base_url, api_key, timeout_s, self._quirks, self._lock)


# ── types tối thiểu khi máy không có google-genai ────────────────────────────


class _KwObject:
    """Đối tượng giữ nguyên tham số khởi tạo làm thuộc tính (thay lớp pydantic của SDK)."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)

    def model_copy(self, update: dict[str, Any] | None = None) -> _KwObject:
        clone = type(self)(**self.__dict__)
        clone.__dict__.update(update or {})
        return clone

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not (exclude_none and v is None)}


class _GenerateContentConfig(_KwObject):
    pass


class _ThinkingConfig(_KwObject):
    pass


class _AutomaticFunctionCallingConfig(_KwObject):
    pass


class _Part(_KwObject):
    @classmethod
    def from_text(cls, text: str) -> _Part:
        return cls(text=text)

    @classmethod
    def from_bytes(cls, data: bytes, mime_type: str) -> _Part:
        return cls(inline_data=SimpleNamespace(data=data, mime_type=mime_type))


class _Blob(_KwObject):
    pass


class _HttpOptions(_KwObject):
    pass


#: ``types`` tối thiểu đủ để adapter dựng cấu hình khi không có ``google-genai`` (bản exe luôn có).
FALLBACK_TYPES = SimpleNamespace(
    GenerateContentConfig=_GenerateContentConfig,
    ThinkingConfig=_ThinkingConfig,
    AutomaticFunctionCallingConfig=_AutomaticFunctionCallingConfig,
    Part=_Part,
    Blob=_Blob,
    HttpOptions=_HttpOptions,
)


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "EXAMPLE_BASE_URL",
    "FALLBACK_TYPES",
    "CustomProviderConfig",
    "CustomProviderError",
    "OpenAICompatClient",
    "OpenAICompatModule",
    "UnsupportedContentError",
    "build_chat_payload",
    "is_text_model",
    "normalize_base_url",
    "parse_chat_response",
]
