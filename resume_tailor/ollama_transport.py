from __future__ import annotations

import http.client
import json
import sys
from pathlib import Path
from typing import Any, Callable

from .schemas import parse_json_text
from .utilities import ModelError, OllamaConnectionError, run_command


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
) -> dict[str, Any]:
    """Call the fixed localhost Ollama API through a cancellable child process.

    The complete request travels over UTF-8 stdin. Résumé and job-derived content
    therefore never appears in argv, environment variables, or a prompt file.
    """
    if path not in _ALLOWED_PATHS:
        raise OllamaConnectionError("Refusing an unsupported Ollama API path.")
    request = {
        "path": path,
        "body": body,
        "socket_timeout_seconds": timeout_seconds,
    }
    encoded = json.dumps(
        request,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > MAX_OLLAMA_REQUEST_BYTES:
        raise OllamaConnectionError(
            "The localhost Ollama request exceeds the bounded transport limit."
        )
    try:
        result = run_command(
            [sys.executable, "-m", "resume_tailor.ollama_transport"],
            cwd=cwd,
            timeout_seconds=timeout_seconds + 5,
            input_text=encoded.decode("utf-8"),
            heartbeat_handler=heartbeat_handler,
        )
    except ModelError as exc:
        raise OllamaConnectionError(
            "The localhost Ollama request exceeded its bounded timeout. The full "
            "worker process group was stopped and provider content was omitted."
        ) from exc
    if result.returncode != 0:
        raise OllamaConnectionError(
            "The localhost Ollama API request failed. Provider output was omitted; "
            "confirm that Ollama is running on 127.0.0.1:11434."
        )
    try:
        envelope = parse_json_text(result.stdout, label="Ollama transport")
    except ModelError as exc:
        raise OllamaConnectionError(
            "The localhost Ollama transport returned an invalid response envelope."
        ) from exc
    if not isinstance(envelope, dict):
        raise OllamaConnectionError(
            "The localhost Ollama transport response was not an object."
        )
    status_code = envelope.get("status_code")
    response_body = envelope.get("body")
    if status_code != 200:
        raise OllamaConnectionError(
            "The localhost Ollama API rejected the request. Response content was "
            "omitted; confirm the configured model name and server status."
        )
    if not isinstance(response_body, dict):
        raise OllamaConnectionError(
            "The localhost Ollama API response body was not an object."
        )
    return response_body


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
        connection = http.client.HTTPConnection(
            OLLAMA_HOST,
            OLLAMA_PORT,
            timeout=timeout,
        )
        connection.request(
            method,
            path,
            body=encoded_body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        response_bytes = response.read(MAX_OLLAMA_RESPONSE_BYTES + 1)
        if len(response_bytes) > MAX_OLLAMA_RESPONSE_BYTES:
            raise ValueError("response exceeds the worker safety limit")
        response_body = json.loads(response_bytes.decode("utf-8"))
        envelope = {
            "status_code": response.status,
            "body": response_body,
        }
        sys.stdout.write(
            json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
        )
        return 0
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, http.client.HTTPException):
        # Never echo a request, model response, prompt, résumé, or job-derived text.
        sys.stderr.write("localhost Ollama transport failure\n")
        return 1
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    raise SystemExit(_worker_main())
