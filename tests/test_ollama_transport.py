from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import resume_tailor.backend.providers.ollama_transport as transport
from resume_tailor.backend.utils.utilities import CommandResult, OllamaConnectionError


def test_local_ollama_request_uses_stdin_and_fixed_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "SYNTHETIC_PRIVATE_RESUME_MARKER"
    observed: dict[str, object] = {}

    def fake_run_command(args: list[str], **kwargs: object) -> CommandResult:
        observed["args"] = args
        observed["input_text"] = kwargs["input_text"]
        return CommandResult(
            tuple(args),
            json.dumps(
                {
                    "status_code": 200,
                    "body": {"done": True, "message": {"content": "{}"}},
                }
            ),
            "",
            0,
        )

    monkeypatch.setattr(transport, "run_command", fake_run_command)
    result = transport.run_ollama_request(
        path="/api/chat",
        body={"messages": [{"role": "user", "content": marker}]},
        cwd=tmp_path,
        timeout_seconds=30,
    )

    assert result["done"] is True
    assert observed["args"] == [
        sys.executable,
        "-m",
        "resume_tailor.ollama_transport",
    ]
    assert marker not in "\0".join(observed["args"])  # type: ignore[arg-type]
    request = json.loads(str(observed["input_text"]))
    assert request["path"] == "/api/chat"
    assert request["body"]["messages"][0]["content"] == marker
    assert transport.OLLAMA_BASE_URL == "http://127.0.0.1:11434"


def test_local_ollama_transport_rejects_remote_or_failed_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(OllamaConnectionError, match="unsupported"):
        transport.run_ollama_request(
            path="https://example.com/api/chat",
            body={},
            cwd=tmp_path,
            timeout_seconds=30,
        )

    monkeypatch.setattr(
        transport,
        "run_command",
        lambda *args, **kwargs: CommandResult(tuple(), "", "private detail", 1),
    )
    with pytest.raises(OllamaConnectionError, match="127.0.0.1:11434") as raised:
        transport.run_ollama_request(
            path="/api/chat",
            body={},
            cwd=tmp_path,
            timeout_seconds=30,
        )
    assert "private detail" not in str(raised.value)

    # Structured worker timeout must surface as classification=timeout.
    monkeypatch.setattr(
        transport,
        "run_command",
        lambda *args, **kwargs: CommandResult(
            tuple(),
            "",
            json.dumps(
                {
                    "transport_error": True,
                    "error_class": "timeout",
                    "provider_output_omitted": True,
                }
            ),
            1,
        ),
    )
    with pytest.raises(OllamaConnectionError) as timed:
        transport.run_ollama_request(
            path="/api/chat",
            body={},
            cwd=tmp_path,
            timeout_seconds=30,
        )
    assert getattr(timed.value, "classification", None) == "timeout"

    monkeypatch.setattr(
        transport,
        "run_command",
        lambda *args, **kwargs: CommandResult(
            tuple(),
            "",
            json.dumps(
                {
                    "transport_error": True,
                    "error_class": "http_error",
                    "status_code": 500,
                    "provider_output_omitted": True,
                }
            ),
            1,
        ),
    )
    with pytest.raises(OllamaConnectionError) as internal:
        transport.run_ollama_request(
            path="/api/chat",
            body={},
            cwd=tmp_path,
            timeout_seconds=30,
        )
    assert getattr(internal.value, "classification", None) == "http_error"
    assert getattr(internal.value, "http_status", None) == 500


def test_ollama_dependency_versions_verify_server_and_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    def fake_request(**kwargs: object) -> dict[str, object]:
        calls.append((str(kwargs["path"]), kwargs["body"]))
        if kwargs["path"] == "/api/version":
            return {"version": "0.12.3"}
        return {"details": {"family": "gemma4"}}

    monkeypatch.setattr(transport, "run_ollama_request", fake_request)
    versions = transport.ollama_dependency_versions(
        model="resume-tailor-gemma",
        cwd=tmp_path,
    )

    assert calls == [
        ("/api/version", None),
        ("/api/show", {"model": "resume-tailor-gemma"}),
    ]
    assert versions == {
        "ollama": "0.12.3",
        "ollama_model": "resume-tailor-gemma",
        "ollama_model_family": "gemma4",
        "ollama_endpoint": "http://127.0.0.1:11434",
    }
