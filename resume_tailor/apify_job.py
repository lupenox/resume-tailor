from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .linkedin_job import (
    ValidatedLinkedInURL,
    normalize_job_description,
    validate_job_source,
    validate_linkedin_url,
)
from .schemas import validate_payload
from .utilities import (
    ApifyConfigurationError,
    ApifyLinkedInRetrievalError,
    CancellationError,
    InputError,
    ModelError,
    atomic_write_json,
    atomic_write_text,
    check_cancelled,
)


APIFY_API_TOKEN_ENV = "APIFY_API_TOKEN"
APIFY_ACTOR_ID_ENV = "APIFY_ACTOR_ID"
APIFY_DIAGNOSTIC_FILENAME = "apify-linkedin-retrieval-diagnostic.json"
APIFY_ACTOR_INPUT_FORMAT = "searchUrls"

_APIFY_API_BASE = "https://api.apify.com/v2"
_ACTOR_NAME_RE = re.compile(
    r"^[A-Za-z0-9_-]{1,128}/[A-Za-z0-9_-]{1,128}$"
)
_RESOURCE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_API_TOKEN_RE = re.compile(r"^apify_api_[A-Za-z0-9_-]+$")
_TOKEN_RE = re.compile(r"apify_api_[A-Za-z0-9_-]+", re.I)
_LOCATION_APPLICANT_SUFFIX_RE = re.compile(
    r"^(?P<location>.+?)\s+"
    r"(?P<count>[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)\s+applicants?$",
    re.I,
)
_TERMINAL_RUN_STATUSES = frozenset(
    {"SUCCEEDED", "FAILED", "TIMED-OUT", "ABORTED"}
)
_ACTIVE_RUN_STATUSES = frozenset(
    {"READY", "RUNNING", "ABORTING", "TIMING-OUT"}
)
_KNOWN_RUN_STATUSES = _TERMINAL_RUN_STATUSES | _ACTIVE_RUN_STATUSES
_MAX_API_RESPONSE_BYTES = 2_000_000
_MAX_ERROR_RESPONSE_BYTES = 64_000
_MAX_DATASET_ITEMS = 20
_MAX_PROVIDER_MESSAGE_CHARACTERS = 500
_RECOGNIZED_ITEM_FIELDS = frozenset(
    {
        "aiFocusAreas",
        "ai_focus_areas",
        "applicantCount",
        "applicantsCount",
        "applicationCount",
        "company",
        "companyDetails",
        "companyName",
        "company_name",
        "compensation",
        "criteria",
        "datePosted",
        "date_posted",
        "description",
        "descriptionHtml",
        "descriptionText",
        "employmentType",
        "employment_type",
        "experienceLevel",
        "id",
        "jobDescription",
        "jobDescriptionHtml",
        "jobDescriptionText",
        "jobId",
        "jobLink",
        "jobLocation",
        "jobName",
        "jobResponsibilities",
        "jobTitle",
        "jobType",
        "jobURL",
        "jobUrl",
        "job_description",
        "job_id",
        "job_location",
        "job_title",
        "link",
        "linkedInJobId",
        "linkedinJobId",
        "linkedin_job_id",
        "location",
        "numApplicants",
        "postedAt",
        "postedDate",
        "preferredQualifications",
        "preferred_qualifications",
        "publishedAt",
        "requiredQualifications",
        "required_qualifications",
        "requirements",
        "responsibilities",
        "salary",
        "salaryRange",
        "salaryText",
        "seniority",
        "seniorityLevel",
        "seniority_level",
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


ProgressHandler = Callable[[str, float, str | None], None]


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


def _validated_token(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ApifyConfigurationError("missing_token")
    if value != value.strip() or len(value) > 1_024:
        raise ApifyConfigurationError("invalid_token")
    if (
        any(character.isspace() or ord(character) < 33 for character in value)
        or not _API_TOKEN_RE.fullmatch(value)
    ):
        raise ApifyConfigurationError("invalid_token")
    return value


def _validated_actor_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ApifyConfigurationError("missing_actor_id")
    actor_id = value.strip()
    if not (
        _RESOURCE_ID_RE.fullmatch(actor_id)
        or _ACTOR_NAME_RE.fullmatch(actor_id)
    ):
        raise ApifyConfigurationError("invalid_actor_id")
    return actor_id


def load_apify_configuration(
    *,
    environment: Mapping[str, str] | None = None,
    token: str | None = None,
    actor_id: str | None = None,
) -> tuple[str, str]:
    """Read the existing process environment without loading or rewriting files."""

    values = os.environ if environment is None else environment
    configured_token = token if token is not None else values.get(APIFY_API_TOKEN_ENV, "")
    configured_actor = (
        actor_id if actor_id is not None else values.get(APIFY_ACTOR_ID_ENV, "")
    )
    return _validated_token(configured_token), _validated_actor_id(configured_actor)


def actor_api_identifier(actor_id: str) -> str:
    """Convert local owner/name syntax to Apify's tilde-separated API path form."""

    validated = _validated_actor_id(actor_id)
    return validated.replace("/", "~")


def build_apify_actor_input(requested_url: ValidatedLinkedInURL) -> dict[str, Any]:
    """Build the single verified input shape used by the configured job Actor."""

    return {APIFY_ACTOR_INPUT_FORMAT: [requested_url.normalized]}


def _sanitized_provider_message(value: str, *, token: str) -> str | None:
    if not value:
        return None
    sanitized = value.replace(token, "[credential omitted]")
    sanitized = _TOKEN_RE.sub("[credential omitted]", sanitized)
    sanitized = re.sub(
        r"(?i)\b(?:authorization|token)\s*[:=]\s*[^\s,;]+",
        "[credential omitted]",
        sanitized,
    )
    sanitized = re.sub(
        r"https://[^\s?#]+\?[^\s]+",
        "[signed URL omitted]",
        sanitized,
    )
    sanitized = " ".join(sanitized.split())
    if not sanitized:
        return None
    return sanitized[:_MAX_PROVIDER_MESSAGE_CHARACTERS]


def _diagnostic_provider_message(value: str | None) -> str | None:
    if value is None:
        return None
    sanitized = _TOKEN_RE.sub("[credential omitted]", value)
    sanitized = re.sub(
        r"(?i)\b(?:authorization|token)\s*[:=]\s*[^\s,;]+",
        "[credential omitted]",
        sanitized,
    )
    sanitized = re.sub(
        r"https://[^\s?#]+\?[^\s]+",
        "[signed URL omitted]",
        sanitized,
    )
    sanitized = " ".join(sanitized.split())
    return sanitized[:_MAX_PROVIDER_MESSAGE_CHARACTERS] or None


def _http_error_message(error: HTTPError, *, token: str) -> str | None:
    try:
        raw = error.read(_MAX_ERROR_RESPONSE_BYTES + 1)
    except Exception:
        return None
    if len(raw) > _MAX_ERROR_RESPONSE_BYTES:
        return None
    try:
        decoded = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None
    message = decoded
    try:
        payload = json.loads(decoded)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(payload, Mapping):
            error_value = payload.get("error")
            if isinstance(error_value, Mapping):
                candidate = error_value.get("message") or error_value.get("type")
            else:
                candidate = payload.get("message") or error_value
            if isinstance(candidate, str):
                message = candidate
    return _sanitized_provider_message(message, token=token)


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
        operation: str,
        body: Mapping[str, Any] | None = None,
        query: Mapping[str, str | int] | None = None,
    ) -> Any:
        query_text = f"?{urlencode(query)}" if query else ""
        url = f"{_APIFY_API_BASE}{path}{query_text}"
        encoded = (
            json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
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
            classification = "provider_failure"
            if exc.code in {401, 403}:
                classification = "authentication_failure"
            elif exc.code == 404 and operation == "start_run":
                classification = "actor_not_found"
            elif exc.code == 429:
                classification = "rate_limited"
            raise ApifyLinkedInRetrievalError(
                classification,
                http_status=exc.code,
                provider_message=_http_error_message(exc, token=self.token),
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ApifyLinkedInRetrievalError("network_error") from exc
        if len(payload) > _MAX_API_RESPONSE_BYTES:
            raise ApifyLinkedInRetrievalError("malformed_output")
        try:
            return json.loads(payload.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApifyLinkedInRetrievalError("malformed_output") from exc

    @staticmethod
    def _data(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("data"), Mapping
        ):
            raise ApifyLinkedInRetrievalError("malformed_output")
        return dict(payload["data"])

    def start_run(
        self,
        *,
        actor_id: str,
        actor_input: Mapping[str, Any],
    ) -> dict[str, Any]:
        encoded_actor = quote(actor_api_identifier(actor_id), safe="~_-")
        payload = self._request_json(
            "POST",
            f"/actors/{encoded_actor}/runs",
            operation="start_run",
            body=actor_input,
        )
        return self._data(payload)

    def get_run(self, run_id: str) -> dict[str, Any]:
        payload = self._request_json(
            "GET",
            f"/actor-runs/{quote(run_id, safe='_-')}",
            operation="get_run",
        )
        return self._data(payload)

    def abort_run(self, run_id: str) -> None:
        self._request_json(
            "POST",
            f"/actor-runs/{quote(run_id, safe='_-')}/abort",
            operation="abort_run",
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
            operation="get_dataset_items",
            query={"clean": "true", "format": "json", "limit": limit},
        )
        if not isinstance(payload, list):
            raise ApifyLinkedInRetrievalError("malformed_output")
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
            "ol",
            "p",
            "section",
            "tr",
            "ul",
        }
    )
    _IGNORED_TAGS = frozenset({"script", "style", "noscript", "svg"})

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
        if normalized in self._IGNORED_TAGS:
            self._ignored_depth += 1
        elif not self._ignored_depth and normalized in self._BREAK_TAGS:
            self.parts.append("\n")
            if normalized == "li":
                self.parts.append("- ")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized in self._IGNORED_TAGS and self._ignored_depth:
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
        raise ApifyLinkedInRetrievalError("malformed_output") from exc
    return normalize_job_description("".join(parser.parts))


def _path_value(payload: Mapping[str, Any], name: str) -> Any:
    value: Any = payload
    for component in name.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(component)
    return value


def _first_text(payload: Mapping[str, Any], *names: str) -> str | None:
    for name in names:
        value = _path_value(payload, name)
        if isinstance(value, str):
            normalized = _plain_text(value)
            if normalized:
                return " ".join(normalized.split())
    return None


def _string_list(payload: Mapping[str, Any], *names: str) -> list[str]:
    for name in names:
        value = _path_value(payload, name)
        if isinstance(value, str):
            text = _plain_text(value)
            candidates = [
                re.sub(r"^(?:[-*•‣▪◦]|\d{1,3}[.)])\s*", "", line).strip()
                for line in text.splitlines()
                if line.strip()
            ]
        elif isinstance(value, list):
            candidates = []
            for item in value:
                if isinstance(item, str):
                    candidates.append(" ".join(_plain_text(item).split()))
                elif isinstance(item, Mapping):
                    nested = _first_text(item, "text", "name", "title", "value")
                    if nested:
                        candidates.append(nested)
        else:
            continue
        normalized: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            candidate = " ".join(candidate.split())
            if candidate and candidate not in seen:
                normalized.append(candidate)
                seen.add(candidate)
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
    return _first_text(item, "jobLink", "jobUrl", "jobURL", "url", "link")


def _candidate_job_id(item: Mapping[str, Any]) -> str | None:
    for name in (
        "jobId",
        "job_id",
        "linkedinJobId",
        "linkedInJobId",
        "linkedin_job_id",
        "id",
    ):
        value = item.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            candidate = str(value)
        elif isinstance(value, str):
            candidate = value.strip()
        else:
            continue
        if re.fullmatch(r"[0-9]{5,20}", candidate):
            return candidate
    return None


def _linkedin_job_id_from_any_host(value: str) -> str | None:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    hostname = (parsed.hostname or "").casefold()
    if parsed.scheme.casefold() != "https" or not (
        hostname == "linkedin.com" or hostname.endswith(".linkedin.com")
    ):
        return None
    match = re.search(r"(?:^|-)([0-9]{5,20})$", parsed.path.rstrip("/").rsplit("/", 1)[-1])
    return match.group(1) if match else None


def select_apify_item(
    items: list[Any],
    *,
    requested: ValidatedLinkedInURL,
) -> tuple[dict[str, Any], int, str]:
    if not items:
        raise ApifyLinkedInRetrievalError("empty_dataset", item_count=0)
    objects = [
        (index, dict(item))
        for index, item in enumerate(items)
        if isinstance(item, Mapping)
    ]
    if not objects:
        raise ApifyLinkedInRetrievalError(
            "malformed_output", item_count=len(items)
        )

    matches: list[tuple[dict[str, Any], int, str]] = []
    for index, item in objects:
        candidate_id = _candidate_job_id(item)
        raw_url = _candidate_url(item)
        final_url = requested.normalized
        if raw_url is not None:
            try:
                candidate_url = validate_linkedin_url(raw_url)
            except InputError:
                url_job_id = _linkedin_job_id_from_any_host(raw_url)
            else:
                url_job_id = candidate_url.job_id
                final_url = candidate_url.normalized
            if candidate_id is not None and url_job_id is not None and candidate_id != url_job_id:
                continue
            candidate_id = candidate_id or url_job_id
        if candidate_id == requested.job_id:
            matches.append((item, index, final_url))

    if len(matches) != 1:
        raise ApifyLinkedInRetrievalError(
            "no_matching_result", item_count=len(items)
        )
    return matches[0]


def _applicant_count(payload: Mapping[str, Any]) -> int | str | None:
    for name in (
        "applicantCount",
        "applicantsCount",
        "applicationCount",
        "numApplicants",
    ):
        value = payload.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return None


def _location_and_embedded_applicant_count(
    payload: Mapping[str, Any],
) -> tuple[str | None, int | None]:
    location = _first_text(payload, "jobLocation", "job_location", "location")
    if location is None:
        return None, None
    match = _LOCATION_APPLICANT_SUFFIX_RE.fullmatch(location)
    if match is None:
        return location, None
    applicant_count = int(match.group("count").replace(",", ""))
    return match.group("location"), applicant_count


def normalize_apify_job_item(
    item: Mapping[str, Any],
    *,
    requested: ValidatedLinkedInURL,
    final_url: str,
) -> dict[str, Any]:
    criteria = _criteria(item)
    location, embedded_applicant_count = _location_and_embedded_applicant_count(
        item
    )
    structured_applicant_count = _applicant_count(item)
    description = _plain_text(
        next(
            (
                item[name]
                for name in (
                    "description",
                    "descriptionText",
                    "descriptionHtml",
                    "jobDescription",
                    "jobDescriptionText",
                    "jobDescriptionHtml",
                    "job_description",
                )
                if isinstance(item.get(name), str) and item[name].strip()
            ),
            "",
        )
    )
    workplace = _first_text(
        item,
        "workplaceType",
        "workplace_type",
        "workType",
    ) or criteria.get("workplace type") or criteria.get("work type")
    employment = _first_text(
        item,
        "employmentType",
        "employment_type",
        "jobType",
    ) or criteria.get("employment type")
    seniority = _first_text(
        item,
        "seniorityLevel",
        "seniority_level",
        "seniority",
        "experienceLevel",
    ) or criteria.get("seniority level") or criteria.get("experience level")
    canonical: dict[str, Any] = {
        "fetch_status": "success",
        "requested_url": requested.normalized,
        "final_resolved_url": final_url,
        "linkedin_job_id": requested.job_id,
        "job_title": _first_text(
            item, "jobTitle", "job_title", "title", "jobName"
        ),
        "company": _first_text(
            item,
            "companyName",
            "company_name",
            "company",
            "companyDetails.name",
        ),
        "location": location,
        "workplace_type": _workplace_type(workplace),
        "employment_type": employment,
        "salary": _first_text(
            item, "salary", "salaryRange", "salaryText", "compensation"
        ),
        "seniority_level": seniority,
        "date_posted": _first_text(
            item, "datePosted", "date_posted", "postedAt", "postedDate", "publishedAt"
        ),
        "applicant_count": (
            structured_applicant_count
            if structured_applicant_count is not None
            else embedded_applicant_count
        ),
        "retrieval_source": "apify",
        "normalized_job_description": description,
        "responsibilities": _string_list(
            item, "responsibilities", "jobResponsibilities"
        ),
        "required_qualifications": _string_list(
            item,
            "requiredQualifications",
            "required_qualifications",
            "requirements",
        ),
        "preferred_qualifications": _string_list(
            item, "preferredQualifications", "preferred_qualifications"
        ),
        "technologies_and_skills": _string_list(
            item, "skills", "technologies", "technologiesAndSkills"
        ),
        "ai_focus_areas": _string_list(
            item, "aiFocusAreas", "ai_focus_areas"
        ),
        "warnings": [
            "Retrieved by Apify and normalized against the local canonical job schema."
        ],
    }
    try:
        validate_payload(
            canonical,
            "linkedin_job.schema.json",
            label="Apify LinkedIn job normalization",
        )
    except ModelError as exc:
        raise ApifyLinkedInRetrievalError("malformed_output") from exc
    return validate_job_source(canonical, requested=requested)


def _resource_id(payload: Mapping[str, Any], name: str) -> str | None:
    value = payload.get(name)
    if not isinstance(value, str) or not _RESOURCE_ID_RE.fullmatch(value):
        return None
    return value


def _run_status(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("status")
    if not isinstance(value, str):
        return None
    normalized = value.upper()
    return normalized if normalized in _KNOWN_RUN_STATUSES else None


def _diagnostic_payload(
    *,
    actor_id: str | None,
    run: Mapping[str, Any],
    items: list[Any],
    selected_index: int | None,
    classification: str,
    phase: str,
    validation_result: str,
    error: ApifyLinkedInRetrievalError | None = None,
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
    recognized_keys = (
        sorted(set(selected) & _RECOGNIZED_ITEM_FIELDS)
        if isinstance(selected, Mapping)
        else []
    )
    return {
        "version": 1,
        "provider": "apify",
        "operation": "linkedin-job-retrieval",
        "actor_id": actor_id,
        "actor_input_format": APIFY_ACTOR_INPUT_FORMAT,
        "authorization_transport": "bearer-header",
        "phase": phase,
        "classification": classification,
        "validation_result": validation_result,
        "http_status": error.http_status if error is not None else None,
        "sanitized_provider_message": (
            _diagnostic_provider_message(error.provider_message)
            if error is not None
            else None
        ),
        "run_id": _resource_id(run, "id")
        or (error.run_id if error is not None else None),
        "run_status": _run_status(run)
        or (error.run_status if error is not None else None),
        "dataset_id": _resource_id(run, "defaultDatasetId")
        or (error.dataset_id if error is not None else None),
        "item_count": len(items) if items else (
            error.item_count if error is not None else 0
        ),
        "selected_index": selected_index,
        "selected_keys": recognized_keys,
        "unrecognized_selected_key_count": (
            len(set(selected) - _RECOGNIZED_ITEM_FIELDS)
            if isinstance(selected, Mapping)
            else 0
        ),
        "provider_output_bytes": len(serialized),
        "provider_output_sha256": (
            hashlib.sha256(serialized).hexdigest() if items else None
        ),
        "provider_output_omitted": True,
        "api_token_omitted": True,
    }


@dataclass
class ApifyLinkedInJobRetriever:
    client: ApifyRunClient
    actor_id: str
    timeout_seconds: int
    poll_interval_seconds: float = 2.0
    progress_handler: ProgressHandler | None = None
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic

    def _progress(
        self,
        phase: str,
        started: float,
        status: str | None = None,
    ) -> None:
        if self.progress_handler is not None:
            self.progress_handler(phase, max(0.0, self.clock() - started), status)

    def retrieve(
        self,
        *,
        requested_url: ValidatedLinkedInURL,
        run_directory: Path,
    ) -> dict[str, Any]:
        if self.timeout_seconds <= 0:
            raise InputError("Apify retrieval timeout must be positive.")
        if self.poll_interval_seconds <= 0:
            raise InputError("Apify polling interval must be positive.")

        diagnostic_path = run_directory / APIFY_DIAGNOSTIC_FILENAME
        run: dict[str, Any] = {}
        items: list[Any] = []
        selected_index: int | None = None
        run_id: str | None = None
        phase = "starting_actor"
        started = self.clock()
        try:
            check_cancelled()
            self._progress(phase, started)
            run = self.client.start_run(
                actor_id=self.actor_id,
                actor_input=build_apify_actor_input(requested_url),
            )
            run_id = _resource_id(run, "id")
            if run_id is None:
                raise ApifyLinkedInRetrievalError("malformed_output")

            phase = "waiting_for_actor"
            while True:
                check_cancelled()
                status = _run_status(run)
                if status is None:
                    raise ApifyLinkedInRetrievalError(
                        "malformed_output", run_id=run_id
                    )
                self._progress(phase, started, status)
                if status in _TERMINAL_RUN_STATUSES:
                    break
                elapsed = self.clock() - started
                if elapsed >= self.timeout_seconds:
                    raise ApifyLinkedInRetrievalError(
                        "actor_timeout", run_id=run_id, run_status=status
                    )
                self.sleep(
                    min(
                        self.poll_interval_seconds,
                        max(0.0, self.timeout_seconds - elapsed),
                    )
                )
                run = self.client.get_run(run_id)

            if status != "SUCCEEDED":
                raise ApifyLinkedInRetrievalError(
                    "actor_timeout" if status == "TIMED-OUT" else "actor_failure",
                    run_id=run_id,
                    run_status=status,
                )
            dataset_id = _resource_id(run, "defaultDatasetId")
            if dataset_id is None:
                raise ApifyLinkedInRetrievalError(
                    "malformed_output", run_id=run_id, run_status=status
                )

            phase = "reading_job_result"
            self._progress(phase, started, status)
            items = self.client.get_dataset_items(
                dataset_id,
                limit=_MAX_DATASET_ITEMS,
            )
            item, selected_index, final_url = select_apify_item(
                items,
                requested=requested_url,
            )

            phase = "normalizing_job_posting"
            self._progress(phase, started, status)
            canonical = normalize_apify_job_item(
                item,
                requested=requested_url,
                final_url=final_url,
            )
            atomic_write_json(run_directory / "job-source.json", canonical)
            atomic_write_text(
                run_directory / "job-description.txt",
                canonical["normalized_job_description"].rstrip() + "\n",
            )
            atomic_write_json(
                diagnostic_path,
                _diagnostic_payload(
                    actor_id=self.actor_id,
                    run=run,
                    items=items,
                    selected_index=selected_index,
                    classification="success",
                    phase="ready_for_review",
                    validation_result="PASS",
                ),
            )
            self._progress("ready_for_review", started, status)
            return canonical
        except CancellationError:
            if run_id is not None and _run_status(run) not in _TERMINAL_RUN_STATUSES:
                try:
                    self.client.abort_run(run_id)
                except Exception:
                    pass
            atomic_write_json(
                diagnostic_path,
                _diagnostic_payload(
                    actor_id=self.actor_id,
                    run=run,
                    items=items,
                    selected_index=selected_index,
                    classification="cancelled",
                    phase=phase,
                    validation_result="REJECTED",
                ),
            )
            raise
        except ApifyLinkedInRetrievalError as exc:
            if exc.run_id is None:
                exc.run_id = run_id
            if exc.run_status is None:
                exc.run_status = _run_status(run)
            if exc.dataset_id is None:
                exc.dataset_id = _resource_id(run, "defaultDatasetId")
            if exc.item_count is None and items:
                exc.item_count = len(items)
            if run_id is not None and _run_status(run) not in _TERMINAL_RUN_STATUSES:
                try:
                    self.client.abort_run(run_id)
                except Exception:
                    pass
            atomic_write_json(
                diagnostic_path,
                _diagnostic_payload(
                    actor_id=self.actor_id,
                    run=run,
                    items=items,
                    selected_index=selected_index,
                    classification=exc.classification,
                    phase=phase,
                    validation_result="REJECTED",
                    error=exc,
                ),
            )
            raise


def invoke_apify_linkedin_retrieval(
    *,
    requested_url: ValidatedLinkedInURL,
    run_directory: Path,
    timeout_seconds: int,
    token: str | None = None,
    actor_id: str | None = None,
    client: ApifyRunClient | None = None,
    poll_interval_seconds: float = 2.0,
    progress_handler: ProgressHandler | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Retrieve one validated LinkedIn posting through one configured Apify Actor."""

    try:
        api_token, selected_actor = load_apify_configuration(
            token=token,
            actor_id=actor_id,
        )
    except ApifyConfigurationError as exc:
        atomic_write_json(
            run_directory / APIFY_DIAGNOSTIC_FILENAME,
            {
                "version": 1,
                "provider": "apify",
                "operation": "linkedin-job-retrieval",
                "actor_id": None,
                "actor_input_format": APIFY_ACTOR_INPUT_FORMAT,
                "authorization_transport": "bearer-header",
                "classification": exc.classification,
                "validation_result": "REJECTED",
                "provider_output_omitted": True,
                "api_token_omitted": True,
            },
        )
        raise
    active_client = client or ApifyHTTPClient(
        api_token,
        request_timeout_seconds=min(20.0, max(1.0, float(timeout_seconds))),
    )
    retriever = ApifyLinkedInJobRetriever(
        client=active_client,
        actor_id=selected_actor,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        progress_handler=progress_handler,
        sleep=sleep,
        clock=clock,
    )
    return retriever.retrieve(
        requested_url=requested_url,
        run_directory=run_directory,
    )
