#!/usr/bin/env python3
"""Sanitize evidence files before publication.

This module intentionally has no repository-specific paths.  Pass source paths
only as CLI arguments; manifests contain sanitized relative names and hashes,
never the input path itself.

The public API is useful for workers preparing individual records:
``sanitize_text`` handles markdown/code/log text, while ``sanitize_obj`` handles
JSON values and keys recursively.  The CLI handles files or directory trees.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import stat
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import SplitResult, urlsplit, urlunsplit

VERSION = "1.0"

# Deliberately recognizable markers make omissions auditable without retaining
# the sensitive value.  They are plain ASCII so they survive every log format.
EMAIL_MARKER = "[REDACTED_EMAIL]"
PATH_MARKER = "[REDACTED_LOCAL_PATH]"
FILE_URL_MARKER = "[REDACTED_LOCAL_FILE_URL]"
INTERNAL_MARKER = "[REDACTED_INTERNAL_ADDRESS]"
SECRET_MARKER = "[REDACTED_SECRET]"
BINARY_MARKER = "[OMITTED_BINARY_PAYLOAD]"
LISTING_MARKER = "[OMITTED_HOME_DIRECTORY_LISTING]"
KEY_MARKER = "[REDACTED_KEY]"
PORT_MARKER = "<PORT>"

# A URL is held while path/email patterns run so a public URL such as a GitHub
# raw path is not mistaken for a local absolute path.
_URL_HOLD_PREFIX = "__PUBLIC_URL_HOLD_"
_URL_HOLD_SUFFIX = "__"

_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+\-])(?:[A-Za-z0-9.!#$%&'*+/=?^_`{|}~\-]+)@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,63}(?![A-Za-z0-9._%+\-])"
)

# Common local path roots.  Do not redact ordinary relative project paths.
_LOCAL_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])/(?:home|root|Users|user|private|mnt|workspace|workspaces|"
    r"tmp|var/tmp|srv|opt/local)(?:/[A-Za-z0-9._+@%~=,:\-]+)+(?![A-Za-z0-9_/])"
)
# Also catch a home path ending at punctuation/whitespace when a component has
# characters outside the conservative component class above.
_HOME_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])/(?:home|Users|user)/[^\s\"'`<>\]}),;]+"
)
_LOCAL_ROOT_RE = re.compile(
    r"(?<![A-Za-z0-9_])/(?:home|root|Users|user|private|mnt|workspace|workspaces|tmp|var/tmp|srv|opt/local)"
    r"(?=$|[\s\"'`<>\]}),;:])"
)
_FILE_URL_RE = re.compile(r"(?i)\bfile:(?://)?[^\s\"'<>]+")
_URL_RE = re.compile(r"(?i)\bhttps?://[^\s\"'<>]+")
_PRIVATE_IPV4_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:10\.(?:\d{1,3}\.){2}\d{1,3}|"
    r"192\.168\.(?:\d{1,3}\.)?\d{1,3}|"
    r"172\.(?:1[6-9]|2\d|3[0-1])\.(?:\d{1,3}\.)?\d{1,3}|"
    r"100\.(?:6[4-9]|[7-9]\d)\.(?:\d{1,3}\.)?\d{1,3}|"
    r"169\.254\.(?:\d{1,3}\.)?\d{1,3})(?::\d{1,5})?(?![A-Za-z0-9])"
)
_INTERNAL_HOST_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_-])[A-Za-z0-9_-]+\.(?:ts\.net|internal|intranet|corp|lan)(?![A-Za-z0-9_.-])"
)
# Long base64/data URLs are almost always embedded binary or an opaque token in
# evidence.  Normal short hashes and ordinary prose are intentionally ignored.
_DATA_URL_RE = re.compile(r"(?is)data:[^;,\s]+;base64,[A-Za-z0-9+/=\r\n]{32,}")
_BASE64_RUN_RE = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{192,}={0,2}(?![A-Za-z0-9+/=])")
_DATA_IMAGE_START_RE = re.compile(r"(?is)data:image/[^,\s]+,")
_HEX_SECRET_RE = re.compile(r"(?<![A-Za-z0-9])[A-Fa-f0-9]{64,}(?![A-Za-z0-9])")
_UUID_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[89ab][0-9a-f]{3}-[0-9a-f]{12}(?![A-Za-z0-9])"
)
_OPAQUE_TEXT_ID_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(?:(?:chatcmpl|toolu|resp|req|run|session)-[A-Za-z0-9-]{8,}|call_[A-Za-z0-9-]{8,})(?![A-Za-z0-9_])"
)
_ASSIGN_SECRET_RE = re.compile(
    r"(?<![A-Za-z0-9_.$])(?P<key>(?:[A-Z][A-Z0-9_-]*_(?:API[_-]?KEY|ACCESS[_-]?TOKEN|REFRESH[_-]?TOKEN|"
    r"AUTH(?:ORIZATION)?|CLIENT[_-]?SECRET|PASSWORD|PASSWD|PRIVATE[_-]?KEY|CREDENTIALS?|SECRET)|"
    r"API[_-]?KEY|ACCESS[_-]?TOKEN|REFRESH[_-]?TOKEN|AUTH(?:ORIZATION)?|CLIENT[_-]?SECRET|"
    r"PASSWORD|PASSWD|PRIVATE[_-]?KEY|CREDENTIALS?|SECRET))\b"
    r"\s*(?:=|:)\s*(?P<value>\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;}\]]+)"
)
_ASSIGN_SECRET_WORD_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_.$])(?P<key>api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"auth(?:orization)?|client[_-]?secret|password|passwd|private[_-]?key|credentials?|secret)\b"
    r"\s*(?:=|:)\s*(?P<value>\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;}\]]+)"
)
_LISTING_HEADER_RE = re.compile(
    r"(?is)(?:directory|home|user|file)\s+(?:listing|contents)|"
    r"(?:listing|contents)\s+of\s+(?:/home|/root|~)|"
    r"(?:/home|/root)/[^\n]{0,120}\s*(?:\n|$)\s*(?:total\s+\d+|drwx|[-*]\s+)"
)
_LS_TOTAL_RE = re.compile(r"(?m)^\s*(?:total|合計)\s+\d+\s*$")
_LISTING_ENTRY_RE = re.compile(r"(?m)^\s*(?:[-*]\s+|[d\-][rwxst\-]{3,9}\s+|\d+\s+\S+)")
_MODE_LISTING_RE = re.compile(r"(?m)^\s*[d\-][rwxst\-]{9}\s+\d+\s+\S+\s+\S+\s+\d+\s+")
_FILENAME_LIST_ENTRY_RE = re.compile(r"(?m)^[A-Za-z0-9][A-Za-z0-9_.+@=~\- ]{0,119}$")

_SENSITIVE_KEY_RE = re.compile(
    r"(?i)(?:^|[_.-])(?:api[_-]?key|access[_-]?token|refresh[_-]?token|auth(?:orization)?|"
    r"client[_-]?secret|password|passwd|private[_-]?key|credential|credentials|secret)(?:$|[_.-])"
)
_SENSITIVE_EXACT_KEYS = {
    "api", "apikey", "api_key", "access_token", "refresh_token", "auth_token",
    "authorization", "password", "passwd", "secret", "token", "private_key", "credential",
    "credentials", "client_secret", "hf_token", "huggingface_token", "openai_key",
}

_DROP_METADATA_KEYS = {
    "cost", "signature",
}
_OPAQUE_ID_KEYS = {
    "id", "parent_id", "tool_call_id", "response_id", "thinking_signature",
    "call_id", "request_id",
}
_BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".tif", ".tiff",
    ".mp4", ".mov", ".webm", ".avi", ".mkv", ".wav", ".mp3", ".flac", ".ogg",
    ".zip", ".gz", ".bz2", ".xz", ".zst", ".7z", ".rar", ".tar", ".whl", ".so",
    ".dll", ".dylib", ".exe", ".bin", ".pt", ".pth", ".safetensors", ".npz",
    ".npy", ".sqlite", ".sqlite3", ".db", ".woff", ".woff2", ".ttf", ".otf",
    ".pdf",
}
_OMIT_DIRS = {
    ".git", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".cache", "cache", "caches", "browser-cache", "torch_compile_cache", "flashinfer_autotune_cache",
}


@dataclass
class SanitizationStats:
    """Counters emitted in optional manifests and useful to callers."""

    replacements: Counter[str] = field(default_factory=Counter)
    files_seen: int = 0
    files_written: int = 0
    files_omitted: int = 0
    id_map: dict[str, str] = field(default_factory=dict, repr=False)
    next_id: int = field(default=1, repr=False)

    def add(self, reason: str, amount: int = 1) -> None:
        self.replacements[reason] += amount

    def merge(self, other: "SanitizationStats") -> None:
        self.replacements.update(other.replacements)
        self.files_seen += other.files_seen
        self.files_written += other.files_written
        self.files_omitted += other.files_omitted

    def as_dict(self) -> dict[str, Any]:
        return {
            "replacements": dict(sorted(self.replacements.items())),
            "files_seen": self.files_seen,
            "files_written": self.files_written,
            "files_omitted": self.files_omitted,
        }


def _is_sensitive_key(key: str) -> bool:
    norm = key.strip().lower().replace("-", "_")
    return norm in _SENSITIVE_EXACT_KEYS or bool(_SENSITIVE_KEY_RE.search(norm))

def _metadata_key(key: str) -> str:
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    return re.sub(r"[^a-z0-9]+", "_", snake.lower()).strip("_")


def _redacted_id(value: Any, stats: SanitizationStats) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    raw = str(value)
    if raw.startswith("[REDACTED_ID_"):
        return raw
    replacement = stats.id_map.get(raw)
    if replacement is None:
        replacement = f"[REDACTED_ID_{stats.next_id}]"
        stats.next_id += 1
        stats.id_map[raw] = replacement
    stats.add("opaque_id")
    return replacement


def _sanitize_url(url: str, stats: SanitizationStats, source_name: str = "") -> str | None:
    """Return a safe URL or None when the endpoint is internal."""
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower()
    except ValueError:
        return None
    if not host:
        return None
    if parsed.scheme.lower() == "file":
        stats.add("local_file_url")
        return FILE_URL_MARKER
    loopback = host in {"localhost", "127.0.0.1", "::1", "[::1]"}
    private = bool(_PRIVATE_IPV4_RE.search(host))
    internal = bool(_INTERNAL_HOST_RE.search(host)) or host.endswith((".local", ".home"))
    if private or internal:
        stats.add("internal_address")
        return INTERNAL_MARKER
    if loopback:
        # Session histories retain the reserved localhost identity while
        # normalizing machine-specific ports/routes.  Source code and tests
        # keep valid numeric localhost URLs so the released app remains usable.
        if source_name.lower().endswith((".jsonl", ".ndjson")) or "session" in source_name.lower():
            stats.add("localhost_port_or_route")
            return f"{parsed.scheme.lower()}://localhost:{PORT_MARKER}"
        try:
            port = parsed.port
        except ValueError:
            port = None
        netloc = "localhost" + (f":{port}" if port is not None else "")
        stats.add("localhost_normalized")
        return urlunsplit(SplitResult(parsed.scheme.lower(), netloc, parsed.path, parsed.query, parsed.fragment))

    # Public URLs are retained, but credentials and secret query parameters are
    # not.  This preserves package/GitHub provenance links exactly otherwise.
    netloc = parsed.hostname or ""
    if ":" in netloc and not netloc.startswith("["):
        netloc = f"[{netloc}]"
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port is not None and port not in {80, 443}:
        netloc += f":{port}"
    query = parsed.query
    if query:
        pieces: list[str] = []
        for piece in query.split("&"):
            if "=" in piece:
                qkey, qvalue = piece.split("=", 1)
                if _is_sensitive_key(qkey):
                    qvalue = SECRET_MARKER
                    stats.add("secret_query_value")
                pieces.append(f"{qkey}={qvalue}")
            else:
                pieces.append(piece)
        query = "&".join(pieces)
    safe = urlunsplit(SplitResult(parsed.scheme, netloc, parsed.path, query, parsed.fragment))
    return safe


def _replace_urls(text: str, stats: SanitizationStats, source_name: str = "") -> tuple[str, dict[str, str]]:
    held: dict[str, str] = {}

    def repl(match: re.Match[str]) -> str:
        original = match.group(0)
        safe = _sanitize_url(original, stats, source_name)
        if safe is None:
            stats.add("malformed_url")
            return INTERNAL_MARKER
        # Internal/file URLs and localhost markers need no holding.  Holding
        # public URLs protects their path from local-path and email regexes.
        if safe.startswith((INTERNAL_MARKER, FILE_URL_MARKER)) or "localhost:" in safe:
            return safe
        token = f"{_URL_HOLD_PREFIX}{len(held)}{_URL_HOLD_SUFFIX}"
        held[token] = safe
        return token

    return _URL_RE.sub(repl, text), held

def _replace_data_image_urls(text: str, stats: SanitizationStats) -> str:
    """Replace complete data-image values without leaving SVG attributes."""
    pieces: list[str] = []
    cursor = 0
    for match in _DATA_IMAGE_START_RE.finditer(text):
        start = match.start()
        if start < cursor:
            continue
        quote = text[start - 1] if start and text[start - 1] in {"'", '"'} else None
        if quote is not None:
            end = start
            while True:
                end = text.find(quote, end)
                if end < 0:
                    end = len(text)
                    break
                # Ignore an escaped quote in a JSON/string representation.
                backslashes = 0
                probe = end - 1
                while probe >= start and text[probe] == "\\":
                    backslashes += 1
                    probe -= 1
                if backslashes % 2 == 0:
                    break
                end += 1
        else:
            delimiter = re.search(r"""[\s"'<>)]""", text[start:])
            end = start + (delimiter.start() if delimiter else len(text) - start)
        pieces.append(text[cursor:start])
        pieces.append(BINARY_MARKER)
        stats.add("embedded_image_url")
        cursor = end
    if cursor == 0:
        return text
    pieces.append(text[cursor:])
    return "".join(pieces)


def _looks_like_listing(text: str, source_name: str = "") -> bool:
    source_lower = source_name.lower()
    is_session = source_lower.endswith((".jsonl", ".ndjson"))
    is_listing_scope = is_session or source_lower.endswith((".log", ".txt"))
    if not is_listing_scope:
        return False
    lines = text.splitlines()
    entries = sum(1 for line in lines if _LISTING_ENTRY_RE.search(line))
    if is_session and any(_MODE_LISTING_RE.search(line) for line in lines):
        # A single `ls -ld` row still exposes the local owner and directory.
        return True
    has_header = bool(_LISTING_HEADER_RE.search(text))
    has_total = bool(_LS_TOTAL_RE.search(text))
    if has_header:
        # A single directory mention is not enough; this gate targets whole
        # tool results while preserving normal narrative references to /home.
        return entries >= 3 or len(lines) >= 12
    if has_total and entries >= 3:
        return True
    # `ls` output can be only bare names (including non-ASCII names), without
    # the total/header line.  Restrict this broader detector to session records
    # and require enough short filename-like lines to avoid hiding prose.
    if is_session and len(lines) >= 20:
        name_entries = sum(1 for line in lines if _FILENAME_LIST_ENTRY_RE.fullmatch(line.strip()))
        return name_entries >= 15
    return False


def _sanitize_secret_assignments(text: str, stats: SanitizationStats) -> str:
    safe_credential_values = {
        "include", "omit", "same-origin", "same_origin", "anonymous", "true", "false",
        "none", "use-credentials",
    }

    def repl(match: re.Match[str]) -> str:
        key = match.group("key")
        raw = match.group("value")
        normalized_key = key.lower().replace("-", "_")
        normalized_value = raw.strip("\"'").strip().lower()
        # The two assignment patterns intentionally overlap for exact
        # uppercase names; do not sanitize an already-redacted value twice.
        if normalized_value.startswith(SECRET_MARKER[:-1].lower()):
            return match.group(0)
        # Browser fetch's `credentials: \"include\"` and Three.js's
        # `withCredentials` are behavior flags, not credentials.  Keep these
        # safe literals so sanitizing vendored code does not change runtime.
        if normalized_key in {"credential", "credentials", "auth", "authorization"} and normalized_value in safe_credential_values:
            return match.group(0)
        stats.add("secret_assignment")
        quote = raw[:1] if raw[:1] in {"'", '"'} else ""
        return f"{key}={quote}{SECRET_MARKER}{quote}"
    text = _ASSIGN_SECRET_RE.sub(repl, text)
    return _ASSIGN_SECRET_WORD_RE.sub(repl, text)


def _sanitize_text_impl(text: str, source_name: str, stats: SanitizationStats) -> str:
    if not text:
        return text
    if _looks_like_listing(text, source_name):
        stats.add("home_directory_listing")
        return LISTING_MARKER + "\n"

    text, held_urls = _replace_urls(text, stats, source_name)
    text = _FILE_URL_RE.sub(lambda _: (stats.add("local_file_url") or FILE_URL_MARKER), text)
    text = _DATA_URL_RE.sub(lambda _: (stats.add("binary_payload") or BINARY_MARKER), text)
    text = _BASE64_RUN_RE.sub(lambda _: (stats.add("binary_payload") or BINARY_MARKER), text)
    text = _replace_data_image_urls(text, stats)
    # Opaque hex values are redacted unless they are explicitly labeled as a
    # checksum/hash.  Public provenance hashes are evidence, not credentials.
    def redact_hex(match: re.Match[str]) -> str:
        prefix = text[max(0, match.start() - 64):match.start()].lower()
        if re.search(
            r"(?:sha(?:1|224|256|384|512)?|hash|digest|checksum)\s*[\"']?\s*[:=]\s*[\"']?\s*$",
            prefix,
        ):
            return match.group(0)
        stats.add("opaque_hex")
        return SECRET_MARKER

    text = _UUID_RE.sub(lambda match: _redacted_id(match.group(0), stats), text)
    text = _OPAQUE_TEXT_ID_RE.sub(lambda match: _redacted_id(match.group(0), stats), text)
    text = _HEX_SECRET_RE.sub(redact_hex, text)
    text = _sanitize_secret_assignments(text, stats)
    text = _EMAIL_RE.sub(lambda _: (stats.add("email") or EMAIL_MARKER), text)
    text = _LOCAL_ROOT_RE.sub(lambda _: (stats.add("local_path") or PATH_MARKER), text)
    text = _PRIVATE_IPV4_RE.sub(lambda _: (stats.add("internal_address") or INTERNAL_MARKER), text)
    text = _INTERNAL_HOST_RE.sub(lambda _: (stats.add("internal_address") or INTERNAL_MARKER), text)
    text = _LOCAL_PATH_RE.sub(lambda _: (stats.add("local_path") or PATH_MARKER), text)
    text = _HOME_PATH_RE.sub(lambda _: (stats.add("local_path") or PATH_MARKER), text)
    for token, safe in held_urls.items():
        text = text.replace(token, safe)
    return text


def sanitize_text(text: str, source_name: str = "") -> str:
    """Sanitize markdown/code/log text and return only public-safe text."""
    stats = SanitizationStats()
    return _sanitize_text_impl(text, source_name, stats)


def sanitize_obj(value: Any, source_name: str = "", stats: SanitizationStats | None = None) -> Any:
    """Recursively sanitize JSON-compatible keys and values.

    Sensitive object keys retain a stable redacted label so the record shape and
    ordering remain inspectable without retaining a credential name/value.
    Embedded image/binary objects are replaced by an auditable marker object.
    """
    own_stats = stats if stats is not None else SanitizationStats()

    if isinstance(value, dict):
        # Drop binary image objects as one unit rather than retaining a nested
        # media type/data shape that could be mistaken for complete evidence.
        kind = str(value.get("type", "")).lower()
        media = str(value.get("media_type", value.get("mime_type", ""))).lower()
        if kind in {"image", "input_image", "image_url"} or media.startswith("image/"):
            own_stats.add("embedded_image_object")
            return {"omitted": True, "reason": "embedded binary image omitted"}
        out: dict[str, Any] = {}
        used: set[str] = set()
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            metadata_key = _metadata_key(key)
            if metadata_key in _DROP_METADATA_KEYS:
                own_stats.add("metadata_omission")
                continue
            if _is_sensitive_key(key):
                safe_key = KEY_MARKER
                if safe_key in used:
                    n = 2
                    while f"{KEY_MARKER}_{n}" in used:
                        n += 1
                    safe_key = f"{KEY_MARKER}_{n}"
                used.add(safe_key)
                own_stats.add("sensitive_key")
                out[safe_key] = SECRET_MARKER
                continue
            key_stats = SanitizationStats()
            safe_key = _sanitize_text_impl(key, source_name, key_stats)
            own_stats.merge(key_stats)
            if safe_key in used:
                n = 2
                while f"{safe_key}_{n}" in used:
                    n += 1
                safe_key = f"{safe_key}_{n}"
            used.add(safe_key)
            if metadata_key in _OPAQUE_ID_KEYS:
                out[safe_key] = _redacted_id(raw_value, own_stats)
            else:
                out[safe_key] = sanitize_obj(raw_value, source_name, own_stats)
        return out
    if isinstance(value, list):
        return [sanitize_obj(item, source_name, own_stats) for item in value]
    if isinstance(value, str):
        return _sanitize_text_impl(value, source_name, own_stats)
    return value


def _safe_relative_name(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = Path(path.name)
    # Name itself is sanitized; absolute source roots never enter a manifest.
    name = sanitize_text(rel.as_posix())
    return name.lstrip("/") or "unnamed"


def _read_text(path: Path) -> tuple[str | None, str | None]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, f"read error: {type(exc).__name__}"
    if b"\x00" in raw:
        return None, "binary content"
    try:
        return raw.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, "non-UTF-8 content"


def _looks_jsonl(path: Path) -> bool:
    return path.suffix.lower() in {".jsonl", ".ndjson"} or path.name.lower().endswith(".jsonl.gz")


def _sanitize_file(path: Path, out_path: Path, root: Path, stats: SanitizationStats) -> dict[str, Any]:
    stats.files_seen += 1
    label = _safe_relative_name(path, root)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        stats.files_omitted += 1
        stats.add("unreadable_file")
        return {"name": label, "omitted": True, "reason": f"read error: {type(exc).__name__}"}
    source_hash = hashlib.sha256(raw).hexdigest()
    if path.suffix.lower() in _BINARY_SUFFIXES or b"\x00" in raw:
        stats.files_omitted += 1
        stats.add("binary_file")
        return {"name": label, "omitted": True, "reason": "binary file", "source_sha256": source_hash, "bytes": len(raw)}
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        stats.files_omitted += 1
        stats.add("non_utf8_file")
        return {"name": label, "omitted": True, "reason": "non-UTF-8 file", "source_sha256": source_hash, "bytes": len(raw)}

    file_stats = SanitizationStats()
    if _looks_jsonl(path):
        rendered: list[str] = []
        for line in text.splitlines(keepends=True):
            ending = "\n" if line.endswith("\n") else ""
            body = line[:-1] if ending else line
            if body.endswith("\r"):
                body = body[:-1]
                ending = "\r\n"
            if not body.strip():
                rendered.append(ending)
                continue
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                rendered.append(_sanitize_text_impl(body, label, file_stats) + ending)
            else:
                safe = sanitize_obj(parsed, label, file_stats)
                rendered.append(json.dumps(safe, ensure_ascii=False, separators=(",", ":")) + ending)
        output = "".join(rendered)
    elif path.suffix.lower() == ".json" or path.name.lower().endswith(".json"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            output = _sanitize_text_impl(text, label, file_stats)
        else:
            output = json.dumps(sanitize_obj(parsed, label, file_stats), ensure_ascii=False, indent=2) + "\n"
    else:
        output = _sanitize_text_impl(text, label, file_stats)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output, encoding="utf-8", newline="")
    try:
        shutil.copystat(path, out_path, follow_symlinks=False)
    except OSError:
        pass
    stats.files_written += 1
    stats.merge(file_stats)
    return {
        "name": label,
        "source_sha256": source_hash,
        "public_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "bytes": len(raw),
        "public_bytes": len(output.encode("utf-8")),
        "redactions": dict(sorted(file_stats.replacements.items())),
    }


def sanitize_path(input_path: str | os.PathLike[str], output_path: str | os.PathLike[str], manifest_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Sanitize one file/tree into ``output_path`` and return a safe manifest."""
    src = Path(input_path)
    dst = Path(output_path)
    if not src.exists():
        raise FileNotFoundError(str(src))
    root = src if src.is_dir() else src.parent
    stats = SanitizationStats()
    entries: list[dict[str, Any]] = []
    if src.is_file():
        target = dst / src.name if dst.exists() and dst.is_dir() else dst
        entries.append(_sanitize_file(src, target, root, stats))
    else:
        for current, dirnames, filenames in os.walk(src, topdown=True, followlinks=False):
            current_path = Path(current)
            kept_dirs: list[str] = []
            for dirname in sorted(dirnames):
                if dirname in _OMIT_DIRS or dirname.startswith(".") and dirname in {".git", ".hg", ".svn"}:
                    stats.files_omitted += 1
                    stats.add("excluded_directory")
                    continue
                kept_dirs.append(dirname)
            dirnames[:] = kept_dirs
            for filename in sorted(filenames):
                source = current_path / filename
                rel = source.relative_to(src)
                target = dst / rel
                if source.is_symlink():
                    stats.files_omitted += 1
                    stats.add("symlink")
                    entries.append({"name": _safe_relative_name(source, src), "omitted": True, "reason": "symlink"})
                    continue
                entries.append(_sanitize_file(source, target, src, stats))
    manifest: dict[str, Any] = {
        "sanitizer": {"name": "sanitize_public.py", "version": VERSION},
        "input_kind": "directory" if src.is_dir() else "file",
        "files": entries,
        "totals": stats.as_dict(),
    }
    if manifest_path is not None:
        mp = Path(manifest_path)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanitize evidence files for public release")
    parser.add_argument("input", help="source file or directory (never copied into the manifest)")
    parser.add_argument("output", help="public-safe output file or directory")
    parser.add_argument("--manifest", help="optional public manifest JSON path")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        manifest = sanitize_path(args.input, args.output, args.manifest)
    except (OSError, ValueError) as exc:
        print(f"sanitize_public: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.manifest is None:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
