"""Bounded, read-only GitHub REST access for portfolio evidence.

This module deliberately exposes sanitized mappings rather than application
models.  The application layer owns dossier interpretation, ranking, and human
approval; this infrastructure adapter only discovers repositories and retrieves
immutable evidence from ``api.github.com``.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import time
import unicodedata
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from resume_tailor.backend.utils.utilities import (
    InputError,
    ModelError,
    atomic_write_json,
    check_cancelled,
    utc_now_iso,
)


GITHUB_TOKEN_ENV = "GITHUB_TOKEN"
GITHUB_API_BASE = "https://api.github.com"
GITHUB_API_VERSION = "2022-11-28"

MAX_API_RESPONSE_BYTES = 2_000_000
MAX_ERROR_RESPONSE_BYTES = 32_000
MAX_REPOSITORIES = 1_000
MAX_PAGES = 10
PAGE_SIZE = 100
MAX_TREE_ENTRIES = 2_000
MAX_README_BYTES = 16_384
MAX_FILE_BYTES = 65_536
MAX_CACHE_BYTES = 2_000_000
MAX_RETRY_DELAY_SECONDS = 2.0
DEFAULT_RETRY_ATTEMPTS = 2
CACHE_VERSION = 2

_SHA_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")
_OWNER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
_TOKEN_WHITESPACE_RE = re.compile(r"\s")

_TEXT_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cfg",
        ".conf",
        ".cpp",
        ".cs",
        ".css",
        ".csv",
        ".go",
        ".graphql",
        ".h",
        ".hpp",
        ".htm",
        ".html",
        ".ini",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".kt",
        ".kts",
        ".md",
        ".markdown",
        ".mdx",
        ".php",
        ".properties",
        ".proto",
        ".py",
        ".rb",
        ".rs",
        ".rst",
        ".scala",
        ".sh",
        ".sql",
        ".svelte",
        ".swift",
        ".tf",
        ".tfvars.example",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".vue",
        ".xml",
        ".yaml",
        ".yml",
    }
)
_TEXT_BASENAMES = frozenset(
    {
        ".dockerignore",
        ".editorconfig",
        ".gitattributes",
        ".gitignore",
        "brewfile",
        "containerfile",
        "dockerfile",
        "gemfile",
        "justfile",
        "license",
        "makefile",
        "procfile",
        "readme",
        "requirements",
    }
)
_VENDORED_OR_BUILD_COMPONENTS = frozenset(
    {
        ".bundle",
        ".cache",
        ".git",
        ".gradle",
        ".idea",
        ".mypy_cache",
        ".next",
        ".nuxt",
        ".pytest_cache",
        ".tox",
        ".venv",
        ".vscode",
        "__pycache__",
        "artifacts",
        "bin",
        "bower_components",
        "build",
        "coverage",
        "dist",
        "generated",
        "node_modules",
        "obj",
        "packages",
        "target",
        "vendor",
        "venv",
    }
)
_SECRET_DIRECTORY_COMPONENTS = frozenset(
    {
        ".aws",
        ".gnupg",
        ".ssh",
        "credentials",
        "private-keys",
        "secrets",
    }
)
_SECRET_BASENAMES = frozenset(
    {
        ".dockercfg",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "auth.json",
        "credentials",
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "secrets.json",
        "secrets.yaml",
        "secrets.yml",
        "service-account.json",
    }
)
_SECRET_SUFFIXES = frozenset(
    {
        ".der",
        ".jks",
        ".key",
        ".keystore",
        ".p12",
        ".pem",
        ".pfx",
        ".pkcs12",
        ".private",
        ".secret",
    }
)
_BINARY_SUFFIXES = frozenset(
    {
        ".7z",
        ".a",
        ".avi",
        ".bin",
        ".bmp",
        ".class",
        ".dll",
        ".dmg",
        ".doc",
        ".docx",
        ".eot",
        ".exe",
        ".gif",
        ".gz",
        ".ico",
        ".jar",
        ".jpeg",
        ".jpg",
        ".lockb",
        ".mov",
        ".mp3",
        ".mp4",
        ".o",
        ".otf",
        ".pdf",
        ".png",
        ".pyc",
        ".so",
        ".tar",
        ".tif",
        ".tiff",
        ".ttf",
        ".wav",
        ".webm",
        ".webp",
        ".woff",
        ".woff2",
        ".xls",
        ".xlsx",
        ".zip",
    }
)
_CACHE_FORBIDDEN_KEYS = frozenset(
    {
        "access_token",
        "authorization",
        "credential",
        "credentials",
        "github_token",
        "private_key",
        "secret",
        "secrets",
        "token",
    }
)

_CACHE_REDACTION = "[credential-like text omitted]"
_CACHE_PRIVATE_KEY_BLOCK_RE = re.compile(
    r"-----BEGIN ([A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?)-----.*?"
    r"-----END \1-----",
    flags=re.IGNORECASE | re.DOTALL,
)
_CACHE_CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"""(?ix)
    (?<![A-Za-z0-9_])
    (?:[A-Za-z0-9]+[_-])*(?:
        access[_ -]?token
        |api[_ -]?key
        |auth(?:orization)?
        |aws[_ -]?secret[_ -]?access[_ -]?key
        |client[_ -]?secret
        |credential(?:s)?
        |github[_ -]?token
        |password
        |passwd
        |private[_ -]?key
        |secret
        |token
    )\b
    \s*[:=]\s*
    (?:
        "(?:\\.|[^"\r\n])*"
        |'(?:\\.|[^'\r\n])*'
        |[^\r\n,;]+
    )
    """
)
_CACHE_COMMON_TOKEN_RE = re.compile(
    r"(?ix)(?:"
    r"\bgithub_pat_[A-Za-z0-9_]{8,255}\b"
    r"|\bgh[pousr]_[A-Za-z0-9]{8,255}\b"
    r"|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"
    r"|\bAIza[0-9A-Za-z_-]{35}\b"
    r"|\bxox[baprs]-[A-Za-z0-9-]{10,}\b"
    r"|\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{12,}\b"
    r"|\b(?:sk-ant-|sk-proj-|glpat-|npm_)[A-Za-z0-9_-]{16,}\b"
    r"|(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\."
    r"[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"
    r"|\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"
    r")"
)


_ERROR_MESSAGES = {
    "authentication_failure": (
        "GitHub authentication was rejected. Verify GITHUB_TOKEN without "
        "placing it in a command, URL, or form."
    ),
    "forbidden": "GitHub denied this read-only repository request.",
    "rate_limited": (
        "GitHub rate-limited the repository request. Wait for the reported "
        "limit to reset, then retry."
    ),
    "not_found": "The requested GitHub repository resource was not found.",
    "empty_repository": "The GitHub repository has no inspectable default-branch head.",
    "network_error": "Resume Tailor could not complete the bounded GitHub HTTPS request.",
    "timeout": "The bounded GitHub HTTPS request timed out.",
    "malformed_response": "GitHub returned a malformed or oversized API response.",
    "pagination_limit": "GitHub repository discovery exceeded the local pagination limit.",
    "invalid_request": "GitHub rejected the bounded read-only repository request.",
    "provider_failure": "GitHub repository discovery stopped with a provider failure.",
    "unsafe_path": "The requested repository path is not permitted for evidence retrieval.",
    "unknown_path": "The requested repository path was not present in the inspected tree.",
    "non_text_file": "The requested repository file is not bounded UTF-8 text.",
    "file_too_large": "The requested repository file exceeds the local evidence limit.",
}


class GitHubConfigurationError(InputError):
    """Local GitHub configuration is absent or violates private-by-default rules."""

    def __init__(self, classification: str) -> None:
        messages = {
            "missing_identity": (
                "GitHub portfolio selection requires a public username or a "
                "GITHUB_TOKEN for authenticated-user discovery."
            ),
            "invalid_username": "The supplied GitHub username is malformed.",
            "invalid_token": (
                "GITHUB_TOKEN is malformed. Store it only in the process "
                "environment without surrounding whitespace."
            ),
            "invalid_timeout": "The GitHub request timeout must be between 1 and 60 seconds.",
            "invalid_retry_attempts": (
                "The GitHub retry-attempt limit must be an integer from zero to four."
            ),
            "private_requires_token": (
                "Private GitHub repository discovery requires an explicit "
                "include-private option and GITHUB_TOKEN."
            ),
            "private_username_mismatch": (
                "Private repository discovery is limited to the authenticated "
                "GitHub account. Omit the username or use the authenticated login."
            ),
        }
        self.classification = (
            classification if classification in messages else "invalid_username"
        )
        super().__init__(messages[self.classification])


class GitHubAPIError(ModelError):
    """A structured GitHub failure that never retains response or credential text."""

    def __init__(
        self,
        classification: str,
        *,
        operation: str,
        http_status: int | None = None,
        retry_after_seconds: int | None = None,
        rate_limit_reset: int | None = None,
    ) -> None:
        if classification not in _ERROR_MESSAGES:
            classification = "provider_failure"
        self.classification = classification
        self.operation = _safe_operation(operation)
        self.http_status = http_status
        self.retry_after_seconds = retry_after_seconds
        self.rate_limit_reset = rate_limit_reset
        super().__init__(_ERROR_MESSAGES[classification])

    def diagnostic(self) -> dict[str, Any]:
        """Return bounded, credential-free fields suitable for a run diagnostic."""

        return {
            "provider": "github",
            "operation": self.operation,
            "classification": self.classification,
            "http_status": self.http_status,
            "retry_after_seconds": self.retry_after_seconds,
            "rate_limit_reset": self.rate_limit_reset,
            "response_body_omitted": True,
            "request_url_omitted": True,
            "github_token_omitted": True,
        }


class GitHubRepositoryClient(Protocol):
    """Infrastructure contract consumed by the application portfolio service."""

    @property
    def token_configured(self) -> bool: ...

    def authenticated_user(self) -> dict[str, Any]: ...

    def discover_repositories(
        self,
        *,
        username: str | None = None,
        include_private: bool = False,
    ) -> list[dict[str, Any]]: ...

    def fetch_repository_head(
        self,
        repository: Mapping[str, Any],
    ) -> dict[str, Any]: ...

    def fetch_repository_snapshot(
        self,
        repository: Mapping[str, Any],
        *,
        head: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def fetch_file(
        self,
        *,
        full_name: str,
        path: str,
        head_sha: str,
        known_paths: Mapping[str, str] | Collection[str],
        max_bytes: int = MAX_FILE_BYTES,
    ) -> dict[str, Any]: ...


Transport = Callable[[Request, float], Any]


class _GitHubRedirectHandler(HTTPRedirectHandler):
    """Allow GitHub rename redirects without permitting a credential hop."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        del fp, msg, headers
        if req.get_method() != "GET" or code not in {301, 302, 307, 308}:
            return None
        if not _is_github_api_url(newurl):
            return None
        return Request(newurl, method="GET", headers=dict(req.header_items()))


def _open_api_request(request: Request, timeout: float) -> Any:
    return build_opener(_GitHubRedirectHandler()).open(request, timeout=timeout)


def _safe_operation(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_-]+", "_", str(value).casefold()).strip("_")
    return normalized[:80] or "github_request"


def _validated_token(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    if (
        not isinstance(value, str)
        or value != value.strip()
        or len(value) > 4_096
        or _TOKEN_WHITESPACE_RE.search(value)
        or any(ord(character) < 33 or ord(character) == 127 for character in value)
    ):
        raise GitHubConfigurationError("invalid_token")
    return value


def _validated_username(value: str) -> str:
    if not isinstance(value, str):
        raise GitHubConfigurationError("invalid_username")
    username = value.strip()
    if username != value or not _OWNER_RE.fullmatch(username) or username.endswith("-"):
        raise GitHubConfigurationError("invalid_username")
    return username


def _validated_full_name(value: str) -> tuple[str, str]:
    if not isinstance(value, str) or value.count("/") != 1:
        raise GitHubAPIError("malformed_response", operation="repository_identity")
    owner, name = value.split("/", 1)
    try:
        owner = _validated_username(owner)
    except GitHubConfigurationError as exc:
        raise GitHubAPIError(
            "malformed_response", operation="repository_identity"
        ) from exc
    if not _REPOSITORY_RE.fullmatch(name) or name in {".", ".."}:
        raise GitHubAPIError("malformed_response", operation="repository_identity")
    return owner, name


def _validated_sha(value: Any, *, operation: str) -> str:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        raise GitHubAPIError("malformed_response", operation=operation)
    return value.casefold()


def _is_github_api_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme.casefold() == "https"
        and parsed.hostname == "api.github.com"
        and port in {None, 443}
        and parsed.username is None
        and parsed.password is None
    )


def _repository_path(full_name: str) -> str:
    owner, name = _validated_full_name(full_name)
    return f"/repos/{quote(owner, safe='')}/{quote(name, safe='')}"


def _bounded_untrusted_text(value: Any, *, maximum: int) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace(
        "\r", "\n"
    )
    if any(
        (ord(character) < 32 or 0x7F <= ord(character) <= 0x9F)
        and character not in {"\n", "\t"}
        for character in normalized
    ):
        return None
    stripped = normalized.strip()
    return stripped[:maximum] if stripped else None


def _safe_timestamp(value: Any) -> str | None:
    text = _bounded_untrusted_text(value, maximum=40)
    if text is None or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.+-]+Z?", text
    ):
        return None
    return text


def _safe_homepage(value: Any) -> str | None:
    text = _bounded_untrusted_text(value, maximum=2_048)
    if text is None:
        return None
    try:
        parsed = urlsplit(text)
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme.casefold() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
    ):
        return None
    return urlunsplit(("https", parsed.netloc, parsed.path or "/", parsed.query, ""))


def _strict_json_loads(value: bytes, *, operation: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = item
        return result

    def reject_nonfinite(_: str) -> None:
        raise ValueError("non-finite JSON number")

    try:
        text = value.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise GitHubAPIError("malformed_response", operation=operation) from exc


def _header_integer(headers: Any, name: str) -> int | None:
    if headers is None:
        return None
    try:
        value = headers.get(name)
    except Exception:
        return None
    if not isinstance(value, str) or not value.isdigit():
        return None
    return int(value)


def _http_classification(error: HTTPError) -> str:
    if error.code == 401:
        return "authentication_failure"
    if error.code == 403:
        if _header_integer(error.headers, "X-RateLimit-Remaining") == 0:
            return "rate_limited"
        return "forbidden"
    if error.code == 404:
        return "not_found"
    if error.code == 409:
        return "empty_repository"
    if error.code in {422}:
        return "invalid_request"
    if error.code == 429:
        return "rate_limited"
    return "provider_failure"


def _network_classification(error: BaseException) -> str:
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, URLError) and isinstance(error.reason, TimeoutError):
        return "timeout"
    return "network_error"


def _retry_delay(
    *, attempt: int, headers: Any = None, rate_limited: bool = False
) -> float | None:
    retry_after = _header_integer(headers, "Retry-After")
    if retry_after is not None:
        if retry_after > MAX_RETRY_DELAY_SECONDS:
            return None
        return float(retry_after)
    if rate_limited:
        return None
    return min(MAX_RETRY_DELAY_SECONDS, 0.1 * (2**attempt))


def _sleep_with_cancellation(sleep: Callable[[float], None], delay: float) -> None:
    check_cancelled()
    sleep(delay)
    check_cancelled()


def _sanitize_topics(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    topics: list[str] = []
    seen: set[str] = set()
    for item in value[:100]:
        topic = _bounded_untrusted_text(item, maximum=100)
        if topic is None:
            continue
        normalized = topic.casefold()
        if normalized in seen:
            continue
        topics.append(topic)
        seen.add(normalized)
    return topics


def _sanitize_license(value: Any) -> dict[str, str | None] | None:
    if not isinstance(value, Mapping):
        return None
    result = {
        "key": _bounded_untrusted_text(value.get("key"), maximum=80),
        "name": _bounded_untrusted_text(value.get("name"), maximum=200),
        "spdx_id": _bounded_untrusted_text(value.get("spdx_id"), maximum=80),
    }
    return result if any(item is not None for item in result.values()) else None


def _sanitize_repository(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise GitHubAPIError("malformed_response", operation="repository_metadata")
    repository_id = payload.get("id")
    if (
        not isinstance(repository_id, int)
        or isinstance(repository_id, bool)
        or repository_id <= 0
    ):
        raise GitHubAPIError("malformed_response", operation="repository_metadata")
    full_name = payload.get("full_name")
    if not isinstance(full_name, str):
        raise GitHubAPIError("malformed_response", operation="repository_metadata")
    owner, name = _validated_full_name(full_name)
    payload_name = payload.get("name")
    if isinstance(payload_name, str) and payload_name.casefold() != name.casefold():
        raise GitHubAPIError("malformed_response", operation="repository_metadata")
    raw_owner = payload.get("owner")
    if isinstance(raw_owner, Mapping):
        owner_login = raw_owner.get("login")
        if isinstance(owner_login, str) and owner_login.casefold() != owner.casefold():
            raise GitHubAPIError("malformed_response", operation="repository_metadata")
    raw_visibility = payload.get("visibility")
    internal = raw_visibility == "internal"
    private = payload.get("private") is True or raw_visibility in {
        "private",
        "internal",
    }
    visibility = "internal" if internal else ("private" if private else "public")
    default_branch = _bounded_untrusted_text(
        payload.get("default_branch"), maximum=255
    )
    if default_branch is not None and (
        default_branch.startswith("/")
        or "\\" in default_branch
        or ".." in PurePosixPath(default_branch).parts
    ):
        default_branch = None
    size = payload.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        size = None
    return {
        "repository_id": repository_id,
        "full_name": f"{owner}/{name}",
        "owner": owner,
        "name": name,
        "visibility": visibility,
        "private": private,
        "description": _bounded_untrusted_text(
            payload.get("description"), maximum=1_000
        ),
        "topics": _sanitize_topics(payload.get("topics")),
        "fork": payload.get("fork") is True,
        "archived": payload.get("archived") is True,
        "disabled": payload.get("disabled") is True,
        "size": size,
        "created_at": _safe_timestamp(payload.get("created_at")),
        "updated_at": _safe_timestamp(payload.get("updated_at")),
        "pushed_at": _safe_timestamp(payload.get("pushed_at")),
        "default_branch": default_branch,
        "homepage": _safe_homepage(payload.get("homepage")),
        "license": _sanitize_license(payload.get("license")),
        "html_url": f"https://github.com/{quote(owner, safe='')}/{quote(name, safe='')}",
    }


def _safe_path_structure(path: str) -> tuple[PurePosixPath, str | None]:
    if not isinstance(path, str) or not path or len(path.encode("utf-8")) > 1_024:
        return PurePosixPath("."), "invalid_path"
    if path != path.strip() or path.startswith("/") or "\\" in path:
        return PurePosixPath("."), "invalid_path"
    if any(ord(character) < 32 or ord(character) == 127 for character in path):
        return PurePosixPath("."), "invalid_path"
    parsed = PurePosixPath(path)
    if not parsed.parts or any(part in {"", ".", ".."} for part in parsed.parts):
        return parsed, "invalid_path"
    components = [part.casefold() for part in parsed.parts]
    if any(component in _VENDORED_OR_BUILD_COMPONENTS for component in components):
        return parsed, "vendored_or_build_path"
    if any(component in _SECRET_DIRECTORY_COMPONENTS for component in components[:-1]):
        return parsed, "secret_path"
    return parsed, None


def repository_path_exclusion_reason(path: str) -> str | None:
    """Return why a repository file cannot be retrieved, or ``None`` if safe."""

    parsed, structural_reason = _safe_path_structure(path)
    if structural_reason is not None:
        return structural_reason
    basename = parsed.name.casefold()
    suffixes = [suffix.casefold() for suffix in parsed.suffixes]
    final_suffix = suffixes[-1] if suffixes else ""
    if basename == ".env" or basename.startswith(".env."):
        return "secret_path"
    if basename in _SECRET_BASENAMES:
        return "secret_path"
    if (
        basename.startswith(("credentials.", "secret.", "secrets."))
        or ".secret." in basename
        or ".credentials." in basename
        or ".private." in basename
        or basename.startswith(("config.local.", "settings.local."))
        or basename.endswith((".local.json", ".local.yaml", ".local.yml", ".local.toml"))
    ):
        return "secret_path"
    if final_suffix in _SECRET_SUFFIXES:
        return "secret_path"
    if final_suffix in _BINARY_SUFFIXES:
        return "non_text_path"
    if basename in _TEXT_BASENAMES:
        return None
    if final_suffix in _TEXT_SUFFIXES:
        return None
    return "unsupported_text_path"


def is_safe_repository_path(path: str) -> bool:
    return repository_path_exclusion_reason(path) is None


def _source_url(full_name: str, head_sha: str, path: str) -> str:
    owner, name = _validated_full_name(full_name)
    sha = _validated_sha(head_sha, operation="source_url")
    safe_path = "/".join(quote(part, safe="") for part in PurePosixPath(path).parts)
    return (
        f"https://github.com/{quote(owner, safe='')}/{quote(name, safe='')}"
        f"/blob/{sha}/{safe_path}"
    )


def _decode_content_payload(
    payload: Any,
    *,
    full_name: str,
    expected_path: str | None,
    head_sha: str,
    maximum: int,
    allow_truncate: bool,
    operation: str,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("type") != "file":
        raise GitHubAPIError("malformed_response", operation=operation)
    path = payload.get("path")
    if not isinstance(path, str) or not is_safe_repository_path(path):
        raise GitHubAPIError("unsafe_path", operation=operation)
    if expected_path is not None and path != expected_path:
        raise GitHubAPIError("malformed_response", operation=operation)
    file_sha = _validated_sha(payload.get("sha"), operation=operation)
    size = payload.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise GitHubAPIError("malformed_response", operation=operation)
    if size > maximum and not allow_truncate:
        raise GitHubAPIError("file_too_large", operation=operation)
    if payload.get("encoding") != "base64" or not isinstance(payload.get("content"), str):
        raise GitHubAPIError("non_text_file", operation=operation)
    try:
        compact_content = "".join(payload["content"].split())
        raw = base64.b64decode(compact_content.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise GitHubAPIError("malformed_response", operation=operation) from exc
    if len(raw) != size:
        raise GitHubAPIError("malformed_response", operation=operation)
    truncated = len(raw) > maximum
    if truncated:
        raw = raw[:maximum]
        while raw:
            try:
                text = raw.decode("utf-8", errors="strict")
                break
            except UnicodeDecodeError as exc:
                if exc.start < len(raw) - 4:
                    raise GitHubAPIError("non_text_file", operation=operation) from exc
                raw = raw[: exc.start]
        else:
            text = ""
    else:
        try:
            text = raw.decode("utf-8-sig", errors="strict")
        except UnicodeDecodeError as exc:
            raise GitHubAPIError("non_text_file", operation=operation) from exc
    if "\x00" in text:
        raise GitHubAPIError("non_text_file", operation=operation)
    if any(
        (ord(character) < 32 or 0x7F <= ord(character) <= 0x9F)
        and character not in {"\n", "\r", "\t"}
        for character in text
    ):
        raise GitHubAPIError("non_text_file", operation=operation)
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace(
        "\r", "\n"
    )
    return {
        "path": path,
        "sha": file_sha,
        "size_bytes": size,
        "text": normalized,
        "truncated": truncated,
        "source_url": _source_url(full_name, head_sha, path),
    }


@dataclass
class GitHubRESTClient:
    """Small read-only GitHub REST client with bounded, injectable transport."""

    token: str | None = field(default=None, repr=False)
    request_timeout_seconds: float = 15.0
    transport: Transport = field(default=_open_api_request, repr=False)
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    _authenticated_login: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.token = _validated_token(self.token)
        if not isinstance(self.request_timeout_seconds, (int, float)) or not (
            0 < float(self.request_timeout_seconds) <= 60
        ):
            raise GitHubConfigurationError("invalid_timeout")
        if (
            not isinstance(self.retry_attempts, int)
            or isinstance(self.retry_attempts, bool)
            or not 0 <= self.retry_attempts <= 4
        ):
            raise GitHubConfigurationError("invalid_retry_attempts")

    @classmethod
    def from_environment(
        cls,
        *,
        environment: Mapping[str, str] | None = None,
        token: str | None = None,
        request_timeout_seconds: float = 15.0,
        transport: Transport = _open_api_request,
        retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> "GitHubRESTClient":
        values = os.environ if environment is None else environment
        selected = token if token is not None else values.get(GITHUB_TOKEN_ENV)
        return cls(
            token=selected,
            request_timeout_seconds=request_timeout_seconds,
            transport=transport,
            retry_attempts=retry_attempts,
            sleep=sleep,
        )

    @property
    def token_configured(self) -> bool:
        return self.token is not None

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
            "User-Agent": "resume-tailor",
        }
        if self.token is not None:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request_json(
        self,
        path: str,
        *,
        operation: str,
        query: Mapping[str, str | int] | None = None,
    ) -> Any:
        if not isinstance(path, str) or not path.startswith("/") or "?" in path:
            raise GitHubAPIError("invalid_request", operation=operation)
        query_text = f"?{urlencode(query)}" if query else ""
        url = f"{GITHUB_API_BASE}{path}{query_text}"
        if not _is_github_api_url(url):
            raise GitHubAPIError("invalid_request", operation=operation)
        for attempt in range(self.retry_attempts + 1):
            check_cancelled()
            request = Request(url, method="GET", headers=self._headers())
            try:
                with self.transport(request, float(self.request_timeout_seconds)) as response:
                    raw = response.read(MAX_API_RESPONSE_BYTES + 1)
            except HTTPError as exc:
                classification = _http_classification(exc)
                transient = exc.code in {429, 500, 502, 503, 504}
                if transient and attempt < self.retry_attempts:
                    delay = _retry_delay(
                        attempt=attempt,
                        headers=exc.headers,
                        rate_limited=classification == "rate_limited",
                    )
                    if delay is not None:
                        _sleep_with_cancellation(self.sleep, delay)
                        continue
                retry_after = _header_integer(exc.headers, "Retry-After")
                reset = _header_integer(exc.headers, "X-RateLimit-Reset")
                try:
                    exc.read(MAX_ERROR_RESPONSE_BYTES + 1)
                except Exception:
                    pass
                raise GitHubAPIError(
                    classification,
                    operation=operation,
                    http_status=exc.code,
                    retry_after_seconds=retry_after,
                    rate_limit_reset=reset,
                ) from None
            except (URLError, TimeoutError, OSError) as exc:
                if attempt < self.retry_attempts:
                    delay = _retry_delay(attempt=attempt)
                    assert delay is not None
                    _sleep_with_cancellation(self.sleep, delay)
                    continue
                raise GitHubAPIError(
                    _network_classification(exc), operation=operation
                ) from None
            if len(raw) > MAX_API_RESPONSE_BYTES:
                raise GitHubAPIError("malformed_response", operation=operation)
            return _strict_json_loads(raw, operation=operation)
        raise AssertionError("unreachable GitHub request retry state")

    def authenticated_user(self) -> dict[str, Any]:
        if self.token is None:
            raise GitHubConfigurationError("private_requires_token")
        payload = self._request_json("/user", operation="authenticated_user")
        if not isinstance(payload, Mapping):
            raise GitHubAPIError(
                "malformed_response", operation="authenticated_user"
            )
        login = payload.get("login")
        user_id = payload.get("id")
        if (
            not isinstance(login, str)
            or not isinstance(user_id, int)
            or isinstance(user_id, bool)
            or user_id <= 0
        ):
            raise GitHubAPIError(
                "malformed_response", operation="authenticated_user"
            )
        try:
            login = _validated_username(login)
        except GitHubConfigurationError as exc:
            raise GitHubAPIError(
                "malformed_response", operation="authenticated_user"
            ) from exc
        self._authenticated_login = login
        return {"login": login, "user_id": user_id}

    def _paginated_repositories(
        self,
        path: str,
        *,
        query: Mapping[str, str | int],
        operation: str,
    ) -> list[dict[str, Any]]:
        repositories: list[dict[str, Any]] = []
        seen: set[int] = set()
        for page in range(1, MAX_PAGES + 1):
            payload = self._request_json(
                path,
                operation=operation,
                query={**query, "per_page": PAGE_SIZE, "page": page},
            )
            if not isinstance(payload, list):
                raise GitHubAPIError("malformed_response", operation=operation)
            for raw in payload:
                repository = _sanitize_repository(raw)
                repository_id = repository["repository_id"]
                if repository_id in seen:
                    continue
                seen.add(repository_id)
                repositories.append(repository)
                if len(repositories) > MAX_REPOSITORIES:
                    raise GitHubAPIError("pagination_limit", operation=operation)
            if len(payload) < PAGE_SIZE:
                break
            if page == MAX_PAGES:
                raise GitHubAPIError("pagination_limit", operation=operation)
        repositories.sort(key=lambda item: (item["full_name"].casefold(), item["repository_id"]))
        return repositories

    def discover_repositories(
        self,
        *,
        username: str | None = None,
        include_private: bool = False,
    ) -> list[dict[str, Any]]:
        selected_username = (
            _validated_username(username) if username is not None else None
        )
        if include_private:
            if self.token is None:
                raise GitHubConfigurationError("private_requires_token")
            user = self.authenticated_user()
            if (
                selected_username is not None
                and selected_username.casefold() != user["login"].casefold()
            ):
                raise GitHubConfigurationError("private_username_mismatch")
            repositories = self._paginated_repositories(
                "/user/repos",
                query={
                    "visibility": "all",
                    "affiliation": "owner",
                    "sort": "full_name",
                    "direction": "asc",
                },
                operation="discover_authenticated_repositories",
            )
            return [
                repository
                for repository in repositories
                if repository["owner"].casefold() == user["login"].casefold()
            ]
        if selected_username is not None:
            repositories = self._paginated_repositories(
                f"/users/{quote(selected_username, safe='')}/repos",
                query={
                    "type": "owner",
                    "sort": "full_name",
                    "direction": "asc",
                },
                operation="discover_public_repositories",
            )
            return [repository for repository in repositories if not repository["private"]]
        if self.token is None:
            raise GitHubConfigurationError("missing_identity")
        repositories = self._paginated_repositories(
            "/user/repos",
            query={
                "visibility": "public",
                "affiliation": "owner",
                "sort": "full_name",
                "direction": "asc",
            },
            operation="discover_authenticated_repositories",
        )
        return [repository for repository in repositories if not repository["private"]]

    def _repository_metadata(self, full_name: str) -> dict[str, Any]:
        return _sanitize_repository(
            self._request_json(
                _repository_path(full_name), operation="repository_metadata"
            )
        )

    def fetch_repository_head(
        self,
        repository: Mapping[str, Any],
    ) -> dict[str, Any]:
        current = dict(repository)
        full_name = current.get("full_name")
        if not isinstance(full_name, str):
            raise GitHubAPIError("malformed_response", operation="repository_head")
        _validated_full_name(full_name)
        branch = current.get("default_branch")
        warnings: list[str] = []
        if not isinstance(branch, str) or not branch:
            return {
                "repository": current,
                "head_sha": None,
                "tree_sha": None,
                "empty": True,
                "warnings": ["missing_default_branch"],
            }

        def read_commit(name: str, ref: str) -> Mapping[str, Any]:
            payload = self._request_json(
                f"{_repository_path(name)}/commits/{quote(ref, safe='')}",
                operation="repository_head",
            )
            if not isinstance(payload, Mapping):
                raise GitHubAPIError(
                    "malformed_response", operation="repository_head"
                )
            return payload

        try:
            commit = read_commit(full_name, branch)
        except GitHubAPIError as exc:
            if exc.classification == "empty_repository":
                return {
                    "repository": current,
                    "head_sha": None,
                    "tree_sha": None,
                    "empty": True,
                    "warnings": ["empty_repository"],
                }
            if exc.classification != "not_found":
                raise
            refreshed = self._repository_metadata(full_name)
            refreshed_branch = refreshed.get("default_branch")
            if not isinstance(refreshed_branch, str) or not refreshed_branch:
                return {
                    "repository": refreshed,
                    "head_sha": None,
                    "tree_sha": None,
                    "empty": True,
                    "warnings": ["missing_default_branch"],
                }
            if refreshed_branch == branch and refreshed.get("size") not in {0, None}:
                raise
            if refreshed.get("size") == 0 and refreshed_branch == branch:
                return {
                    "repository": refreshed,
                    "head_sha": None,
                    "tree_sha": None,
                    "empty": True,
                    "warnings": ["empty_repository"],
                }
            current = refreshed
            full_name = refreshed["full_name"]
            branch = refreshed_branch
            warnings.append("default_branch_refreshed")
            commit = read_commit(full_name, branch)

        head_sha = _validated_sha(commit.get("sha"), operation="repository_head")
        nested_commit = commit.get("commit")
        tree = nested_commit.get("tree") if isinstance(nested_commit, Mapping) else None
        tree_sha = _validated_sha(
            tree.get("sha") if isinstance(tree, Mapping) else None,
            operation="repository_head",
        )
        return {
            "repository": current,
            "head_sha": head_sha,
            "tree_sha": tree_sha,
            "empty": False,
            "warnings": warnings,
        }

    def _topics(self, full_name: str) -> list[str]:
        payload = self._request_json(
            f"{_repository_path(full_name)}/topics", operation="repository_topics"
        )
        if not isinstance(payload, Mapping):
            raise GitHubAPIError("malformed_response", operation="repository_topics")
        return _sanitize_topics(payload.get("names"))

    def _languages(self, full_name: str) -> dict[str, int]:
        payload = self._request_json(
            f"{_repository_path(full_name)}/languages",
            operation="repository_languages",
        )
        if not isinstance(payload, Mapping):
            raise GitHubAPIError(
                "malformed_response", operation="repository_languages"
            )
        languages: dict[str, int] = {}
        for raw_name, raw_bytes in payload.items():
            name = _bounded_untrusted_text(raw_name, maximum=100)
            if (
                name is None
                or not isinstance(raw_bytes, int)
                or isinstance(raw_bytes, bool)
                or raw_bytes < 0
            ):
                raise GitHubAPIError(
                    "malformed_response", operation="repository_languages"
                )
            languages[name] = raw_bytes
        return dict(sorted(languages.items(), key=lambda item: item[0].casefold()))

    def _readme(self, full_name: str, head_sha: str) -> dict[str, Any]:
        payload = self._request_json(
            f"{_repository_path(full_name)}/readme",
            operation="repository_readme",
            query={"ref": head_sha},
        )
        return _decode_content_payload(
            payload,
            full_name=full_name,
            expected_path=None,
            head_sha=head_sha,
            maximum=MAX_README_BYTES,
            allow_truncate=True,
            operation="repository_readme",
        )

    def _tree(self, full_name: str, tree_sha: str) -> tuple[list[dict[str, Any]], int, list[str]]:
        payload = self._request_json(
            f"{_repository_path(full_name)}/git/trees/{tree_sha}",
            operation="repository_tree",
            query={"recursive": "1"},
        )
        if not isinstance(payload, Mapping) or not isinstance(payload.get("tree"), list):
            raise GitHubAPIError("malformed_response", operation="repository_tree")
        manifest: list[dict[str, Any]] = []
        excluded = 0
        warnings: list[str] = []
        for entry in payload["tree"]:
            if not isinstance(entry, Mapping):
                raise GitHubAPIError("malformed_response", operation="repository_tree")
            path = entry.get("path")
            kind = entry.get("type")
            if not isinstance(path, str) or kind not in {"blob", "tree"}:
                excluded += 1
                continue
            _, structural_reason = _safe_path_structure(path)
            if structural_reason is not None:
                excluded += 1
                continue
            if kind == "blob" and not is_safe_repository_path(path):
                excluded += 1
                continue
            sha = _validated_sha(entry.get("sha"), operation="repository_tree")
            item: dict[str, Any] = {"path": path, "type": kind, "sha": sha}
            if kind == "blob":
                size = entry.get("size")
                if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                    raise GitHubAPIError(
                        "malformed_response", operation="repository_tree"
                    )
                item["size_bytes"] = size
            manifest.append(item)
            if len(manifest) == MAX_TREE_ENTRIES:
                warnings.append("tree_entry_limit_reached")
                break
        if payload.get("truncated") is True:
            warnings.append("github_tree_truncated")
        manifest.sort(key=lambda item: (item["path"].casefold(), item["type"]))
        return manifest, excluded, warnings

    def fetch_repository_snapshot(
        self,
        repository: Mapping[str, Any],
        *,
        head: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_head = dict(head) if head is not None else self.fetch_repository_head(repository)
        current = resolved_head.get("repository")
        if not isinstance(current, Mapping):
            raise GitHubAPIError("malformed_response", operation="repository_snapshot")
        current = dict(current)
        full_name = current.get("full_name")
        if not isinstance(full_name, str):
            raise GitHubAPIError("malformed_response", operation="repository_snapshot")
        warnings = [
            str(item)[:120]
            for item in resolved_head.get("warnings", [])
            if isinstance(item, str)
        ]
        snapshot: dict[str, Any] = {
            "repository": current,
            "head_sha": resolved_head.get("head_sha"),
            "tree_sha": resolved_head.get("tree_sha"),
            "empty": resolved_head.get("empty") is True,
            "languages": {},
            "readme": None,
            "tree": [],
            "excluded_path_count": 0,
            "warnings": warnings,
            "partial": False,
            "cached": False,
        }
        if snapshot["empty"]:
            return snapshot
        head_sha = _validated_sha(snapshot["head_sha"], operation="repository_snapshot")
        tree_sha = _validated_sha(snapshot["tree_sha"], operation="repository_snapshot")

        try:
            current["topics"] = self._topics(full_name)
        except GitHubAPIError as exc:
            warnings.append(f"topics_{exc.classification}")
            snapshot["partial"] = True
        try:
            snapshot["languages"] = self._languages(full_name)
        except GitHubAPIError as exc:
            warnings.append(f"languages_{exc.classification}")
            snapshot["partial"] = True
        try:
            snapshot["readme"] = self._readme(full_name, head_sha)
        except GitHubAPIError as exc:
            if exc.classification == "not_found":
                warnings.append("missing_readme")
            else:
                warnings.append(f"readme_{exc.classification}")
                snapshot["partial"] = True
        try:
            tree, excluded, tree_warnings = self._tree(full_name, tree_sha)
            snapshot["tree"] = tree
            snapshot["excluded_path_count"] = excluded
            warnings.extend(tree_warnings)
            if tree_warnings:
                snapshot["partial"] = True
        except GitHubAPIError as exc:
            warnings.append(f"tree_{exc.classification}")
            snapshot["partial"] = True
        snapshot["warnings"] = list(dict.fromkeys(warnings))
        return snapshot

    def fetch_file(
        self,
        *,
        full_name: str,
        path: str,
        head_sha: str,
        known_paths: Mapping[str, str] | Collection[str],
        max_bytes: int = MAX_FILE_BYTES,
    ) -> dict[str, Any]:
        if (
            not isinstance(max_bytes, int)
            or isinstance(max_bytes, bool)
            or not 1 <= max_bytes <= MAX_FILE_BYTES
        ):
            raise GitHubAPIError("invalid_request", operation="repository_file")
        _validated_full_name(full_name)
        sha = _validated_sha(head_sha, operation="repository_file")
        reason = repository_path_exclusion_reason(path)
        if reason is not None:
            raise GitHubAPIError("unsafe_path", operation="repository_file")
        if isinstance(known_paths, Mapping):
            expected_blob_sha = known_paths.get(path)
            known = path in known_paths
        else:
            expected_blob_sha = None
            known = path in known_paths
        if not known:
            raise GitHubAPIError("unknown_path", operation="repository_file")
        encoded_path = "/".join(
            quote(part, safe="") for part in PurePosixPath(path).parts
        )
        payload = self._request_json(
            f"{_repository_path(full_name)}/contents/{encoded_path}",
            operation="repository_file",
            query={"ref": sha},
        )
        result = _decode_content_payload(
            payload,
            full_name=full_name,
            expected_path=path,
            head_sha=sha,
            maximum=max_bytes,
            allow_truncate=False,
            operation="repository_file",
        )
        if expected_blob_sha is not None:
            expected = _validated_sha(expected_blob_sha, operation="repository_file")
            if result["sha"] != expected:
                raise GitHubAPIError("malformed_response", operation="repository_file")
        return result


def default_github_cache_directory(
    *, environment: Mapping[str, str] | None = None
) -> Path:
    values = os.environ if environment is None else environment
    configured = values.get("XDG_CACHE_HOME", "").strip()
    if configured and Path(configured).is_absolute():
        root = Path(configured)
    else:
        root = Path.home() / ".cache"
    return root / "resume-tailor" / "github"


def _contains_forbidden_cache_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and key.casefold() in _CACHE_FORBIDDEN_KEYS:
                return True
            if _contains_forbidden_cache_key(item):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_cache_key(item) for item in value)
    return False


def _sanitize_cache_text(value: str) -> tuple[str, bool]:
    """Remove recognizable credentials from untrusted repository text."""

    sanitized = value
    redacted = False
    for pattern in (
        _CACHE_PRIVATE_KEY_BLOCK_RE,
        _CACHE_CREDENTIAL_ASSIGNMENT_RE,
        _CACHE_COMMON_TOKEN_RE,
    ):
        sanitized, count = pattern.subn(_CACHE_REDACTION, sanitized)
        redacted = redacted or count > 0
    return sanitized, redacted


def _sanitize_cache_value(value: Any) -> tuple[Any, bool]:
    """Copy a JSON-compatible value while redacting credential-like strings."""

    if isinstance(value, str):
        return _sanitize_cache_text(value)
    if isinstance(value, Mapping):
        result: dict[Any, Any] = {}
        redacted = False
        for key, item in value.items():
            sanitized, item_redacted = _sanitize_cache_value(item)
            result[key] = sanitized
            redacted = redacted or item_redacted
        return result, redacted
    if isinstance(value, list):
        result_items: list[Any] = []
        redacted = False
        for item in value:
            sanitized, item_redacted = _sanitize_cache_value(item)
            result_items.append(sanitized)
            redacted = redacted or item_redacted
        return result_items, redacted
    return value, False


def _sanitized_cache_snapshot(snapshot: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    sanitized, redacted = _sanitize_cache_value(snapshot)
    if not isinstance(sanitized, dict):  # defensive; mappings normalize to dictionaries.
        raise InputError("The GitHub dossier cache snapshot is invalid.")
    if redacted and isinstance(sanitized.get("warnings"), list):
        warnings = sanitized["warnings"]
        if "credential_like_text_redacted" not in warnings:
            warnings.append("credential_like_text_redacted")
    return sanitized, redacted


@dataclass
class GitHubDossierCache:
    """Private local cache keyed only by repository identity and inspected head."""

    directory: Path = field(default_factory=default_github_cache_directory)
    maximum_bytes: int = MAX_CACHE_BYTES

    def __post_init__(self) -> None:
        self.directory = self.directory.expanduser()
        if not self.directory.is_absolute():
            raise InputError("The GitHub dossier cache directory must be absolute.")
        if (
            not isinstance(self.maximum_bytes, int)
            or isinstance(self.maximum_bytes, bool)
            or not 1_024 <= self.maximum_bytes <= MAX_CACHE_BYTES
        ):
            raise InputError("The GitHub dossier cache size limit is invalid.")

    def _path(self, repository_id: int, head_sha: str) -> Path:
        if (
            not isinstance(repository_id, int)
            or isinstance(repository_id, bool)
            or repository_id <= 0
        ):
            raise InputError("The GitHub cache repository identity is invalid.")
        sha = _validated_sha(head_sha, operation="dossier_cache")
        return self.directory / f"v{CACHE_VERSION}-{repository_id}-{sha}.json"

    def _publish(self, path: Path, payload: Mapping[str, Any]) -> None:
        encoded = (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        if len(encoded) > self.maximum_bytes:
            raise InputError("The GitHub dossier cache entry exceeds its limit.")
        try:
            if self.directory.exists() and (
                self.directory.is_symlink() or not self.directory.is_dir()
            ):
                raise InputError("The GitHub dossier cache directory is unsafe.")
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.directory.chmod(0o700)
            atomic_write_json(path, dict(payload))
            path.chmod(0o600)
        except InputError:
            raise
        except OSError as exc:
            raise InputError("The GitHub dossier cache entry could not be written.") from exc

    def load(self, *, repository_id: int, head_sha: str) -> dict[str, Any] | None:
        path = self._path(repository_id, head_sha)
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file():
            raise InputError("The GitHub dossier cache entry is not a safe file.")
        try:
            if path.stat().st_size > self.maximum_bytes:
                raise InputError("The GitHub dossier cache entry exceeds its limit.")
            raw = path.read_bytes()
        except OSError as exc:
            raise InputError("The GitHub dossier cache entry could not be read.") from exc
        try:
            payload = _strict_json_loads(raw, operation="dossier_cache")
        except GitHubAPIError as exc:
            raise InputError(
                "The GitHub dossier cache entry is not valid UTF-8 JSON."
            ) from exc
        if (
            not isinstance(payload, Mapping)
            or payload.get("version") != CACHE_VERSION
            or payload.get("repository_id") != repository_id
            or payload.get("head_sha") != head_sha.casefold()
            or not isinstance(payload.get("snapshot"), Mapping)
            or _contains_forbidden_cache_key(payload)
        ):
            raise InputError("The GitHub dossier cache entry failed local validation.")
        snapshot, redacted = _sanitized_cache_snapshot(payload["snapshot"])
        if redacted:
            sanitized_payload = dict(payload)
            sanitized_payload["snapshot"] = snapshot
            self._publish(path, sanitized_payload)
        return snapshot

    def store(
        self,
        *,
        repository_id: int,
        head_sha: str,
        snapshot: Mapping[str, Any],
    ) -> Path:
        path = self._path(repository_id, head_sha)
        if _contains_forbidden_cache_key(snapshot):
            raise InputError("Credential-like fields are forbidden in the GitHub cache.")
        sanitized_snapshot, _ = _sanitized_cache_snapshot(snapshot)
        payload = {
            "version": CACHE_VERSION,
            "repository_id": repository_id,
            "head_sha": head_sha.casefold(),
            "cached_at": utc_now_iso(),
            "snapshot": sanitized_snapshot,
        }
        self._publish(path, payload)
        return path


def _partial_snapshot(
    repository: Mapping[str, Any], error: GitHubAPIError
) -> dict[str, Any]:
    return {
        "repository": dict(repository),
        "head_sha": None,
        "tree_sha": None,
        "empty": False,
        "languages": {},
        "readme": None,
        "tree": [],
        "excluded_path_count": 0,
        "warnings": [f"head_{error.classification}"],
        "partial": True,
        "cached": False,
    }


def discover_repository_snapshots(
    client: GitHubRepositoryClient,
    *,
    username: str | None = None,
    include_private: bool = False,
    cache: GitHubDossierCache | None = None,
) -> dict[str, Any]:
    """Discover and inspect repositories without letting one repo erase others.

    Inventory failure remains fatal.  Once an inventory is available, per-repo
    failures are represented as partial snapshots so the application can record
    deterministic eligibility reasons rather than silently dropping candidates.
    """

    repositories = client.discover_repositories(
        username=username, include_private=include_private
    )
    snapshots: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for repository in repositories:
        check_cancelled()
        try:
            head = client.fetch_repository_head(repository)
            head_sha = head.get("head_sha")
            cached = None
            if cache is not None and isinstance(head_sha, str):
                cached = cache.load(
                    repository_id=int(repository["repository_id"]),
                    head_sha=head_sha,
                )
            if cached is not None:
                snapshot = cached
                snapshot["repository"] = dict(head.get("repository") or repository)
                snapshot["head_sha"] = head_sha
                snapshot["tree_sha"] = head.get("tree_sha")
                snapshot["empty"] = head.get("empty") is True
                snapshot["cached"] = True
            else:
                snapshot = client.fetch_repository_snapshot(repository, head=head)
                if (
                    cache is not None
                    and isinstance(head_sha, str)
                    and snapshot.get("partial") is False
                ):
                    cache.store(
                        repository_id=int(repository["repository_id"]),
                        head_sha=head_sha,
                        snapshot=snapshot,
                    )
        except GitHubAPIError as exc:
            snapshot = _partial_snapshot(repository, exc)
            failures.append(
                {
                    "repository_id": repository["repository_id"],
                    "classification": exc.classification,
                    "operation": exc.operation,
                }
            )
        snapshots.append(snapshot)
    return {
        "version": 1,
        "discovery": {
            "requested_username": username,
            "authenticated": client.token_configured,
            "include_private": include_private,
            "repository_count": len(snapshots),
            "partial_failure_count": len(failures),
        },
        "repositories": snapshots,
        "partial_failures": failures,
        "credentials_excluded": True,
    }
