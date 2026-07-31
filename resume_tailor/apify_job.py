from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.parse import unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .linkedin_job import (
    ValidatedLinkedInURL,
    diagnostic_job_payload,
    normalize_job_description,
    validate_job_source,
    validate_linkedin_url,
)
from .schemas import validate_payload
from .utilities import (
    ApifyConfigurationError,
    ApifyProviderError,
    CancellationError,
    InputError,
    atomic_write_json,
    atomic_write_text,
    check_cancelled,
)


APIFY_API_TOKEN_ENV = "APIFY_API_TOKEN"
APIFY_ACTOR_ENV = "RESUME_TAILOR_APIFY_ACTOR"
APIFY_TOKEN_FILE = Path("~/.config/resume-tailor/apify-token")
DEFAULT_APIFY_ACTOR = "piotrv1001/linkedin-job-details-scraper"
APIFY_RESPONSE_METADATA_FILENAME = "apify-job-response.json"
_APIFY_API_BASE = "https://api.apify.com/v2"
_ACTOR_ID_RE = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$")
_RESOURCE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_TERMINAL_RUN_STATUSES = frozenset(
    {"SUCCEEDED", "FAILED", "TIMED-OUT", "ABORTED"}
)
_RUN_STATUSES = _TERMINAL_RUN_STATUSES | frozenset(
    {"READY", "RUNNING", "ABORTING", "TIMING-OUT"}
)
_MAX_API_RESPONSE_BYTES = 2_000_000
_MAX_DATASET_ITEMS = 5
_RECOGNIZED_ITEM_FIELDS = frozenset(
    {
        "aiFocusAreas",
        "ai_focus_areas",
        "company",
        "companyName",
        "company_name",
        "compensation",
        "criteria",
        "description",
        "descriptionText",
        "employmentType",
        "employment_type",
        "jobDescription",
        "jobLink",
        "jobLocation",
        "jobName",
        "jobResponsibilities",
        "jobTitle",
        "jobType",
        "jobURL",
        "jobUrl",
        "job_description",
        "job_location",
        "job_title",
        "link",
        "location",
        "preferredQualifications",
        "preferred_qualifications",
        "requiredQualifications",
        "required_qualifications",
        "requirements",
        "responsibilities",
        "salary",
        "salaryRange",
        "salaryText",
        "skills",
        "technologies",
        "technologiesAndSkills",
        "title",
        "url",
        "workType",
        "workplaceType",
        "workplace_type",
    }
)


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        return None


def _open_api_request(request: Request, timeout: float) -> Any:
    return build_opener(_RejectRedirects()).open(request, timeout=timeout)


def resolve_linkedin_provider(
    requested: str,
    *,
    environment: Mapping[str, str] | None = None,
    token_file: Path | None = None,
) -> str:
    normalized = requested.strip().casefold()
    if normalized not in {"auto", "apify", "antigravity"}:
        raise InputError(
            "LinkedIn provider must be auto, apify, or antigravity."
        )
    if normalized != "auto":
        return normalized
    values = environment if environment is not None else os.environ
    if values.get(APIFY_API_TOKEN_ENV, "").strip():
        return "apify"
    if environment is None and _token_file_exists(token_file):
        return "apify"
    return "antigravity"


def _token_file_path(token_file: Path | None) -> Path:
    return (token_file or APIFY_TOKEN_FILE).expanduser()


def _token_file_exists(token_file: Path | None) -> bool:
    path = _token_file_path(token_file)
    try:
        return path.is_file() and not path.is_symlink()
    except OSError:
        return False


def _validated_token(value: str) -> str:
    token = value.strip()
    if not token or len(token) > 1_024:
        raise ApifyConfigurationError(
            "The Apify API token is empty or exceeds 1,024 characters."
        )
    if any(character.isspace() or ord(character) < 33 for character in token):
        raise ApifyConfigurationError(
            "The Apify API token contains whitespace or control characters."
        )
    return token


def load_apify_token(
    *,
    explicit: str | None = None,
    environment: Mapping[str, str] | None = None,
    token_file: Path | None = None,
) -> str:
    if explicit is not None:
        return _validated_token(explicit)
    values = environment if environment is not None else os.environ
    configured = values.get(APIFY_API_TOKEN_ENV, "")
    if configured.strip():
        return _validated_token(configured)

    path = _token_file_path(token_file)
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ApifyConfigurationError(
            "Apify retrieval requires APIFY_API_TOKEN or the private token file "
            "~/.config/resume-tailor/apify-token. Configure one locally or "
            "select Antigravity, pasted text, or a UTF-8 job file."
        ) from exc
    except OSError as exc:
        raise ApifyConfigurationError(
            "The configured Apify token file could not be inspected safely."
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ApifyConfigurationError(
            "The Apify token path must be a regular, non-symlink file."
        )
    if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
        raise ApifyConfigurationError(
            "The Apify token file must be owned by the current user."
        )
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ApifyConfigurationError(
            "The Apify token file must use mode 0600 with no group or other access."
        )
    if metadata.st_size > 4_096:
        raise ApifyConfigurationError(
            "The Apify token file exceeds the 4,096-byte safety limit."
        )
    try:
        token = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ApifyConfigurationError(
            "The Apify token file must contain valid UTF-8 text."
        ) from exc
    return _validated_token(token)


class ApifyRunClient(Protocol):
    def start_run(
        self,
        *,
        actor_id: str,
        actor_input: Mapping[str, Any],
    ) -> dict[str, Any]: ...

    def get_run(self, run_id: str) -> dict[str, Any]: ...

    def abort_run(self, run_id: str) -> None: ...

    def get_dataset_items(
        self,
        dataset_id: str,
        *,
        limit: int,
    ) -> list[Any]: ...


@dataclass
class ApifyHTTPClient:
    token: str
    request_timeout_seconds: float = 20.0

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        query: Mapping[str, str | int] | None = None,
    ) -> Any:
        query_text = f"?{urlencode(query)}" if query else ""
        url = f"{_APIFY_API_BASE}{path}{query_text}"
        encoded = (
            json.dumps(body, ensure_ascii=False).encode("utf-8")
            if body is not None
            else None
        )
        request = Request(
            url,
            data=encoded,
            method=method,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.token}",
                **(
                    {"Content-Type": "application/json"}
                    if encoded is not None
                    else {}
                ),
            },
        )
        try:
            with _open_api_request(
                request,
                timeout=self.request_timeout_seconds,
            ) as response:
                payload = response.read(_MAX_API_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            raise ApifyProviderError(
                f"Apify API returned HTTP {exc.code}; provider response content "
                "was not exposed."
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ApifyProviderError(
                f"Apify API request failed ({type(exc).__name__}); no provider "
                "response content was exposed."
            ) from exc
        if len(payload) > _MAX_API_RESPONSE_BYTES:
            raise ApifyProviderError(
                "Apify API response exceeded the 2,000,000-byte safety limit."
            )
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApifyProviderError(
                "Apify API returned malformed UTF-8 JSON."
            ) from exc

    @staticmethod
    def _data(payload: Any, *, label: str) -> dict[str, Any]:
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
            raise ApifyProviderError(
                f"Apify {label} response did not contain an object data envelope."
            )
        return dict(payload["data"])

    def start_run(
        self,
        *,
        actor_id: str,
        actor_input: Mapping[str, Any],
    ) -> dict[str, Any]:
        encoded_actor = quote(actor_id.replace("/", "~"), safe="~_-")
        payload = self._request_json(
            "POST",
            f"/actors/{encoded_actor}/runs",
            body=actor_input,
        )
        return self._data(payload, label="start-run")

    def get_run(self, run_id: str) -> dict[str, Any]:
        payload = self._request_json(
            "GET",
            f"/actor-runs/{quote(run_id, safe='_-')}",
        )
        return self._data(payload, label="run-status")

    def abort_run(self, run_id: str) -> None:
        self._request_json(
            "POST",
            f"/actor-runs/{quote(run_id, safe='_-')}/abort",
        )

    def get_dataset_items(
        self,
        dataset_id: str,
        *,
        limit: int,
    ) -> list[Any]:
        payload = self._request_json(
            "GET",
            f"/datasets/{quote(dataset_id, safe='_-')}/items",
            query={"clean": "true", "format": "json", "limit": limit},
        )
        if not isinstance(payload, list):
            raise ApifyProviderError(
                "Apify dataset response was not a JSON array."
            )
        return payload


class _HTMLToText(HTMLParser):
    _BREAK_TAGS = frozenset(
        {
            "br",
            "div",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "li",
            "p",
            "section",
            "tr",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        normalized = tag.casefold()
        if normalized in {"script", "style"}:
            self._ignored_depth += 1
        elif not self._ignored_depth and normalized in self._BREAK_TAGS:
            self.parts.append("\n")
            if normalized == "li":
                self.parts.append("- ")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif not self._ignored_depth and normalized in self._BREAK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)


def _plain_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    parser = _HTMLToText()
    try:
        parser.feed(value)
        parser.close()
    except Exception as exc:
        raise ApifyProviderError(
            "Apify job description could not be normalized safely."
        ) from exc
    return normalize_job_description("".join(parser.parts))


def _first_text(payload: Mapping[str, Any], *names: str) -> str | None:
    for name in names:
        value = payload.get(name)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return None


def _string_list(payload: Mapping[str, Any], *names: str) -> list[str]:
    for name in names:
        value = payload.get(name)
        if isinstance(value, list):
            normalized: list[str] = []
            seen: set[str] = set()
            for item in value:
                if not isinstance(item, str):
                    continue
                text = " ".join(item.split())
                if text and text not in seen:
                    normalized.append(text)
                    seen.add(text)
            return normalized
    return []


def _criteria(payload: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    raw = payload.get("criteria")
    if not isinstance(raw, list):
        return result
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        name = _first_text(item, "name", "label", "title")
        value = _first_text(item, "value", "text")
        if name and value:
            result[name.casefold()] = value
    return result


def _workplace_type(value: str | None) -> str:
    if value is None:
        return "unspecified"
    normalized = value.casefold().replace("_", "-").strip()
    if normalized in {"remote", "hybrid"}:
        return normalized
    if normalized in {"on-site", "onsite", "on site"}:
        return "on-site"
    return "unspecified"


def _candidate_url(item: Mapping[str, Any]) -> str | None:
    return _first_text(
        item,
        "jobLink",
        "jobUrl",
        "jobURL",
        "url",
        "link",
    )


def _provider_job_id(value: str) -> str | None:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    hostname = (parsed.hostname or "").casefold()
    if (
        parsed.scheme.casefold() != "https"
        or (
            hostname != "linkedin.com"
            and not hostname.endswith(".linkedin.com")
        )
    ):
        return None
    path = unquote(parsed.path)
    if not path.startswith("/jobs/view/"):
        return None
    final_segment = path.rstrip("/").rsplit("/", 1)[-1]
    match = re.search(r"(?:^|-)([0-9]{5,20})$", final_segment)
    return match.group(1) if match else None


def _select_item(
    items: list[Any],
    *,
    requested: ValidatedLinkedInURL,
) -> tuple[dict[str, Any], int, str]:
    objects = [
        (index, dict(item))
        for index, item in enumerate(items)
        if isinstance(item, Mapping)
    ]
    if not objects:
        raise ApifyProviderError("Apify returned no job-detail object.")

    exact: list[tuple[int, dict[str, Any], str]] = []
    for index, item in objects:
        raw_url = _candidate_url(item)
        if raw_url is None:
            continue
        try:
            candidate = validate_linkedin_url(raw_url)
        except InputError:
            provider_job_id = _provider_job_id(raw_url)
            if (
                requested.job_id is not None
                and provider_job_id == requested.job_id
            ):
                exact.append((index, item, requested.normalized))
            continue
        if requested.job_id is not None and candidate.job_id == requested.job_id:
            exact.append((index, item, candidate.normalized))
        elif (
            requested.job_id is None
            and candidate.path.rstrip("/") == requested.path.rstrip("/")
        ):
            exact.append((index, item, candidate.normalized))

    if len(exact) == 1:
        index, item, final_url = exact[0]
        return item, index, final_url
    if len(exact) > 1:
        raise ApifyProviderError(
            "Apify returned multiple objects for the requested LinkedIn job ID."
        )
    if len(objects) == 1 and _candidate_url(objects[0][1]) is None:
        index, item = objects[0]
        return item, index, requested.normalized
    raise ApifyProviderError(
        "Apify did not return exactly one object matching the requested LinkedIn job."
    )


def map_apify_job_item(
    item: Mapping[str, Any],
    *,
    requested: ValidatedLinkedInURL,
    final_url: str,
    actor_id: str,
) -> dict[str, Any]:
    criteria = _criteria(item)
    description = _plain_text(
        item.get("description")
        or item.get("jobDescription")
        or item.get("job_description")
        or item.get("descriptionText")
    )
    workplace = _first_text(
        item,
        "workplaceType",
        "workplace_type",
        "workType",
    )
    if workplace is None:
        workplace = criteria.get("workplace type") or criteria.get("work type")
    employment = _first_text(
        item,
        "employmentType",
        "employment_type",
        "jobType",
    )
    if employment is None:
        employment = criteria.get("employment type")
    salary = _first_text(
        item,
        "salary",
        "salaryRange",
        "salaryText",
        "compensation",
    )
    canonical = {
        "fetch_status": "success",
        "requested_url": requested.normalized,
        "final_resolved_url": final_url,
        "linkedin_job_id": requested.job_id
        or validate_linkedin_url(final_url).job_id,
        "job_title": _first_text(item, "jobTitle", "job_title", "title", "jobName"),
        "company": _first_text(
            item,
            "companyName",
            "company_name",
            "company",
        ),
        "location": _first_text(
            item,
            "jobLocation",
            "job_location",
            "location",
        ),
        "workplace_type": _workplace_type(workplace),
        "employment_type": employment,
        "salary": salary,
        "normalized_job_description": description,
        "responsibilities": _string_list(
            item,
            "responsibilities",
            "jobResponsibilities",
        ),
        "required_qualifications": _string_list(
            item,
            "requiredQualifications",
            "required_qualifications",
            "requirements",
        ),
        "preferred_qualifications": _string_list(
            item,
            "preferredQualifications",
            "preferred_qualifications",
        ),
        "technologies_and_skills": _string_list(
            item,
            "skills",
            "technologies",
            "technologiesAndSkills",
        ),
        "ai_focus_areas": _string_list(
            item,
            "aiFocusAreas",
            "ai_focus_areas",
        ),
        "warnings": [
            (
                f"Retrieved through Apify actor {actor_id}; provider output was "
                "normalized and validated locally."
            )
        ],
    }
    validate_payload(
        canonical,
        "linkedin_job.schema.json",
        label="Apify LinkedIn extraction",
    )
    return validate_job_source(canonical, requested=requested)


def _validate_actor_id(value: str) -> str:
    actor_id = value.strip()
    if not _ACTOR_ID_RE.fullmatch(actor_id):
        raise ApifyConfigurationError(
            "RESUME_TAILOR_APIFY_ACTOR must use owner/actor-name syntax."
        )
    return actor_id


def _run_value(payload: Mapping[str, Any], name: str) -> str | None:
    value = payload.get(name)
    return value if isinstance(value, str) and value else None


def _resource_id(payload: Mapping[str, Any], name: str) -> str | None:
    value = _run_value(payload, name)
    if value is None:
        return None
    return value if _RESOURCE_ID_RE.fullmatch(value) else None


def _run_status(payload: Mapping[str, Any]) -> str:
    value = (_run_value(payload, "status") or "").upper()
    return value if value in _RUN_STATUSES else "UNKNOWN"


def _metadata(
    *,
    actor_id: str,
    run: Mapping[str, Any],
    items: list[Any],
    selected_index: int | None,
    validation_result: str,
) -> dict[str, Any]:
    serialized = json.dumps(
        items,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    selected = (
        items[selected_index]
        if selected_index is not None and 0 <= selected_index < len(items)
        else None
    )
    selected_keys = (
        sorted(set(selected) & _RECOGNIZED_ITEM_FIELDS)
        if isinstance(selected, Mapping)
        else []
    )
    return {
        "version": 1,
        "provider": "apify",
        "actor_id": actor_id,
        "authorization_transport": "bearer-header",
        "run_id": _resource_id(run, "id"),
        "build_id": _resource_id(run, "buildId"),
        "dataset_id": _resource_id(run, "defaultDatasetId"),
        "terminal_status": _run_status(run),
        "item_count": len(items),
        "selected_index": selected_index,
        "selected_keys": selected_keys,
        "unrecognized_selected_key_count": (
            len(set(selected) - _RECOGNIZED_ITEM_FIELDS)
            if isinstance(selected, Mapping)
            else 0
        ),
        "selected_field_types": (
            {
                key: type(selected[key]).__name__
                for key in selected_keys
            }
            if isinstance(selected, Mapping)
            else {}
        ),
        "provider_output_bytes": len(serialized),
        "provider_output_sha256": hashlib.sha256(serialized).hexdigest(),
        "provider_output_omitted": True,
        "validation_result": validation_result,
    }


def invoke_apify_job_extraction(
    *,
    requested_url: ValidatedLinkedInURL,
    run_directory: Path,
    timeout_seconds: int,
    token: str | None = None,
    token_file: Path | None = None,
    actor_id: str | None = None,
    client: ApifyRunClient | None = None,
    poll_interval_seconds: float = 2.0,
    progress_handler: Callable[[float, str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    api_token = load_apify_token(
        explicit=token,
        token_file=token_file,
    )
    selected_actor = _validate_actor_id(
        actor_id
        or os.environ.get(APIFY_ACTOR_ENV, "")
        or DEFAULT_APIFY_ACTOR
    )
    if timeout_seconds <= 0:
        raise InputError("Apify timeout must be positive.")
    if poll_interval_seconds <= 0:
        raise InputError("Apify polling interval must be positive.")

    active_client = client or ApifyHTTPClient(api_token)
    artifact_path = run_directory / "job-source.json"
    metadata_path = run_directory / APIFY_RESPONSE_METADATA_FILENAME
    run: dict[str, Any] = {}
    items: list[Any] = []
    selected_index: int | None = None
    run_id: str | None = None
    started = time.monotonic()

    try:
        check_cancelled()
        run = active_client.start_run(
            actor_id=selected_actor,
            actor_input={"searchUrls": [requested_url.normalized]},
        )
        run_id = _resource_id(run, "id")
        if run_id is None:
            raise ApifyProviderError("Apify start-run response omitted the run ID.")

        while True:
            check_cancelled()
            status = _run_status(run)
            elapsed = time.monotonic() - started
            if progress_handler is not None:
                progress_handler(elapsed, status or "UNKNOWN")
            if status in _TERMINAL_RUN_STATUSES:
                break
            if elapsed >= timeout_seconds:
                raise ApifyProviderError(
                    f"Apify actor exceeded the bounded {timeout_seconds}s timeout."
                )
            sleep(min(poll_interval_seconds, max(0.0, timeout_seconds - elapsed)))
            run = active_client.get_run(run_id)

        if status != "SUCCEEDED":
            raise ApifyProviderError(
                f"Apify actor stopped with status {status or 'UNKNOWN'}."
            )
        dataset_id = _resource_id(run, "defaultDatasetId")
        if dataset_id is None:
            raise ApifyProviderError(
                "Successful Apify run omitted its default dataset ID."
            )
        items = active_client.get_dataset_items(
            dataset_id,
            limit=_MAX_DATASET_ITEMS,
        )
        item, selected_index, final_url = _select_item(
            items,
            requested=requested_url,
        )
        try:
            canonical = map_apify_job_item(
                item,
                requested=requested_url,
                final_url=final_url,
                actor_id=selected_actor,
            )
        except InputError as exc:
            raise ApifyProviderError(
                "Apify job data failed the local canonical job-source contract."
            ) from exc
        atomic_write_json(artifact_path, canonical)
        atomic_write_text(
            run_directory / "job-description.txt",
            canonical["normalized_job_description"].rstrip() + "\n",
        )
        atomic_write_json(
            metadata_path,
            _metadata(
                actor_id=selected_actor,
                run=run,
                items=items,
                selected_index=selected_index,
                validation_result="PASS",
            ),
        )
        return canonical
    except (CancellationError, ApifyProviderError, InputError) as exc:
        if run_id is not None:
            try:
                active_client.abort_run(run_id)
            except Exception:
                pass
        atomic_write_json(
            artifact_path,
            diagnostic_job_payload(requested_url.normalized, str(exc)),
        )
        atomic_write_json(
            metadata_path,
            _metadata(
                actor_id=selected_actor,
                run=run,
                items=items,
                selected_index=selected_index,
                validation_result="REJECTED",
            ),
        )
        raise
