from __future__ import annotations

import errno
import http.client
import json
import socket
import sys
from pathlib import Path
from typing import Any, Callable

from .schemas import parse_json_text
from .utilities import ModelError, OllamaConnectionError, OllamaRequestError, run_command


OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

MAX_OLLAMA_REQUEST_BYTES = 1_500_000
MAX_OLLAMA_RESPONSE_BYTES = 2_000_000
_ALLOWED_PATHS = {"/api/chat", "/api/show", "/api/version"}


def run_ollama_request(
    *,
    path: str,
    body: dict[str, Any] | None,
    cwd: Path,
    timeout_seconds: int,
    heartbeat_handler: Callable[[float, bool], None] | None = None,
    connect_timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Call the fixed localhost Ollama API through a cancellable child process.

    The complete request travels over UTF-8 stdin. Résumé and job-derived content
    therefore never appears in argv, environment variables, or a prompt file.

    On failure, raises ``OllamaRequestError`` with a sanitized ``classification``
    so callers can distinguish timeout, connection refused, and HTTP errors.
    """
    if path not in _ALLOWED_PATHS:
        raise OllamaConnectionError("Refusing an unsupported Ollama API path.")
    if timeout_seconds < 1:
        raise OllamaRequestError(
            "Ollama request timeout must be at least 1 second.",
            classification="transport_failure",
        )
    connect_timeout = (
        connect_timeout_seconds
        if isinstance(connect_timeout_seconds, int) and connect_timeout_seconds > 0
        else min(30, timeout_seconds)
    )
    request = {
        "path": path,
        "body": body,
        "socket_timeout_seconds": timeout_seconds,
        "connect_timeout_seconds": connect_timeout,
    }
    encoded = json.dumps(
        request,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > MAX_OLLAMA_REQUEST_BYTES:
        raise OllamaRequestError(
            "The localhost Ollama request exceeds the bounded transport limit.",
            classification="transport_failure",
        )
    # Parent kills the worker slightly after the socket deadline so the worker
    # can report a structured timeout when the socket fires first.
    parent_timeout = timeout_seconds + max(5, connect_timeout)
    try:
        result = run_command(
            [sys.executable, "-m", "resume_tailor.ollama_transport"],
            cwd=cwd,
            timeout_seconds=parent_timeout,
            input_text=encoded.decode("utf-8"),
            heartbeat_handler=heartbeat_handler,
        )
    except ModelError as exc:
        raise OllamaRequestError(
            "The localhost Ollama request exceeded its bounded timeout. The full "
            "worker process group was stopped and provider content was omitted.",
            classification="timeout",
        ) from exc

    if result.returncode != 0:
        # Prefer structured stderr envelope when the worker emitted one.
        stderr = (result.stderr or "").strip()
        if stderr.startswith("{"):
            try:
                failure = json.loads(stderr)
            except json.JSONDecodeError:
                failure = None
            if isinstance(failure, dict) and failure.get("transport_error"):
                raise _error_from_worker_failure(failure)

        raise OllamaRequestError(
            "The localhost Ollama API request failed. Provider output was omitted; "
            "confirm that Ollama is running on 127.0.0.1:11434.",
            classification="transport_failure",
        )

    try:
        envelope = parse_json_text(result.stdout, label="Ollama transport")
    except ModelError as exc:
        raise OllamaRequestError(
            "The localhost Ollama transport returned an invalid response envelope.",
            classification="malformed_envelope",
        ) from exc
    if not isinstance(envelope, dict):
        raise OllamaRequestError(
            "The localhost Ollama transport response was not an object.",
            classification="malformed_envelope",
        )

    # Structured failure envelopes written to stdout (worker success path for
    # classified HTTP/timeout failures).
    if envelope.get("transport_error"):
        raise _error_from_worker_failure(envelope)

    status_code = envelope.get("status_code")
    response_body = envelope.get("body")
    if not isinstance(status_code, int):
        raise OllamaRequestError(
            "The localhost Ollama transport omitted the HTTP status code.",
            classification="malformed_envelope",
        )
    if status_code != 200:
        raise _http_status_error(status_code)
    if not isinstance(response_body, dict):
        raise OllamaRequestError(
            "The localhost Ollama API response body was not an object.",
            classification="malformed_envelope",
            http_status=status_code,
        )
    return response_body


def _error_from_worker_failure(failure: dict[str, Any]) -> OllamaRequestError:
    error_class = failure.get("error_class")
    http_status = failure.get("status_code")
    if not isinstance(http_status, int):
        http_status = None
    if error_class == "timeout":
        return OllamaRequestError(
            "The localhost Ollama request exceeded its bounded timeout. The full "
            "worker process group was stopped and provider content was omitted.",
            classification="timeout",
            http_status=http_status,
            transport_error="timeout",
        )
    if error_class == "connection_refused":
        return OllamaRequestError(
            "The localhost Ollama server refused the connection. Confirm Ollama "
            "is running on 127.0.0.1:11434.",
            classification="connection_refused",
            transport_error="connection_refused",
        )
    if error_class == "response_too_large":
        return OllamaRequestError(
            "The localhost Ollama response exceeds the bounded transport limit.",
            classification="response_too_large",
            http_status=http_status,
            transport_error="response_too_large",
        )
    if error_class == "http_error" and http_status is not None:
        return _http_status_error(http_status)
    return OllamaRequestError(
        "The localhost Ollama API request failed. Provider output was omitted; "
        "confirm that Ollama is running on 127.0.0.1:11434.",
        classification="transport_failure",
        http_status=http_status,
        transport_error=str(error_class) if error_class else "transport_failure",
    )


def _http_status_error(status_code: int) -> OllamaRequestError:
    if status_code in {404, 400}:
        return OllamaRequestError(
            "The localhost Ollama API rejected the request. Response content was "
            "omitted; confirm the configured model name and server status.",
            classification="http_error",
            http_status=status_code,
            transport_error="http_error",
        )
    if status_code >= 500:
        return OllamaRequestError(
            "The localhost Ollama API returned an internal server error. Provider "
            "content was omitted.",
            classification="http_error",
            http_status=status_code,
            transport_error="http_error",
        )
    return OllamaRequestError(
        "The localhost Ollama API rejected the request. Response content was "
        "omitted; confirm the configured model name and server status.",
        classification="http_error",
        http_status=status_code,
        transport_error="http_error",
    )


def ollama_dependency_versions(
    *,
    model: str,
    cwd: Path,
    timeout_seconds: int = 10,
) -> dict[str, str]:
    version_payload = run_ollama_request(
        path="/api/version",
        body=None,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
    )
    version = version_payload.get("version")
    if not isinstance(version, str) or not version.strip():
        raise OllamaConnectionError("Ollama did not report a usable server version.")
    show_payload = run_ollama_request(
        path="/api/show",
        body={"model": model},
        cwd=cwd,
        timeout_seconds=timeout_seconds,
    )
    details = show_payload.get("details")
    family = details.get("family") if isinstance(details, dict) else None
    return {
        "ollama": version.strip()[:200],
        "ollama_model": model,
        "ollama_model_family": (
            family.strip()[:200]
            if isinstance(family, str) and family.strip()
            else "unavailable"
        ),
        "ollama_endpoint": OLLAMA_BASE_URL,
    }


def _read_worker_request() -> dict[str, Any]:
    payload = sys.stdin.buffer.read(MAX_OLLAMA_REQUEST_BYTES + 1)
    if len(payload) > MAX_OLLAMA_REQUEST_BYTES:
        raise ValueError("request exceeds the worker safety limit")
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("request must be an object")
    return value


def _write_failure(error_class: str, *, status_code: int | None = None) -> int:
    """Emit a content-free structured failure for the parent process."""
    payload: dict[str, Any] = {
        "transport_error": True,
        "error_class": error_class,
        "provider_output_omitted": True,
    }
    if status_code is not None:
        payload["status_code"] = status_code
    # Prefer stdout so parents that only parse stdout still see structure when
    # returncode is 0; also write stderr for returncode != 0 paths.
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write(encoded)
    sys.stderr.write(encoded + "\n")
    return 1


def _worker_main() -> int:
    connection: http.client.HTTPConnection | None = None
    try:
        request = _read_worker_request()
        path = request.get("path")
        if path not in _ALLOWED_PATHS:
            raise ValueError("unsupported API path")
        timeout = request.get("socket_timeout_seconds")
        if not isinstance(timeout, int) or not 1 <= timeout <= 86_400:
            raise ValueError("invalid socket timeout")
        connect_timeout = request.get("connect_timeout_seconds")
        if not isinstance(connect_timeout, int) or connect_timeout < 1:
            connect_timeout = min(30, timeout)
        body = request.get("body")
        if path == "/api/version":
            method = "GET"
            encoded_body = None
        else:
            if not isinstance(body, dict):
                raise ValueError("POST request body must be an object")
            method = "POST"
            encoded_body = json.dumps(
                body,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        # Connect uses a short bound; the overall socket timeout covers generation.
        connection = http.client.HTTPConnection(
            OLLAMA_HOST,
            OLLAMA_PORT,
            timeout=timeout,
        )
        try:
            connection.connect()
        except OSError as exc:
            if exc.errno in {errno.ECONNREFUSED, errno.ENETUNREACH, errno.EHOSTUNREACH}:
                return _write_failure("connection_refused")
            if isinstance(exc, TimeoutError) or exc.errno in {
                errno.ETIMEDOUT,
            }:
                return _write_failure("timeout")
            return _write_failure("connection_refused")
        # After connect, enforce the generation deadline on the socket.
        connection.sock.settimeout(timeout)  # type: ignore[union-attr]
        connection.request(
            method,
            path,
            body=encoded_body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        response_bytes = response.read(MAX_OLLAMA_RESPONSE_BYTES + 1)
        if len(response_bytes) > MAX_OLLAMA_RESPONSE_BYTES:
            return _write_failure("response_too_large", status_code=response.status)
        if response.status != 200:
            # Do not forward Ollama error bodies (may include prompt fragments).
            return _write_failure("http_error", status_code=response.status)
        response_body = json.loads(response_bytes.decode("utf-8"))
        envelope = {
            "status_code": response.status,
            "body": response_body,
        }
        sys.stdout.write(
            json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
        )
        return 0
    except TimeoutError:
        return _write_failure("timeout")
    except socket.timeout:
        return _write_failure("timeout")
    except ConnectionRefusedError:
        return _write_failure("connection_refused")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, http.client.HTTPException):
        # Never echo a request, model response, prompt, résumé, or job-derived text.
        return _write_failure("transport_failure")
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    raise SystemExit(_worker_main())
