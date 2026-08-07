from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import unquote, urlsplit, urlunsplit

from resume_tailor.backend.jobs.job_requirements import job_description_sha256, validate_job_requirement_catalog
from resume_tailor.backend.utils.utilities import InputError, utc_now_iso


ANALYTICS_DATABASE_FILENAME = "job-search-analytics.sqlite3"
ANALYTICS_EXPORT_FILENAME = "job-search-analytics-export.json"
ANALYTICS_SCHEMA_VERSION = 1
SANITIZED_EXPORT_CONTRACT_VERSION = 1
MINIMUM_RATE_SAMPLE_SIZE = 5
APPLICATION_UPDATE_DAYS = 7
SQLITE_BUSY_TIMEOUT_MS = 5_000

APPLICATION_STATUSES: tuple[str, ...] = (
    "viewed",
    "saved",
    "planned",
    "applied",
    "screening",
    "interview",
    "technical_interview",
    "final_interview",
    "rejected",
    "withdrawn",
    "offer",
    "accepted",
    "declined",
)
INTERVIEW_TYPES: tuple[str, ...] = (
    "screening",
    "interview",
    "technical_interview",
    "final_interview",
)

_SUBMITTED_STATUSES = frozenset(
    {
        "applied",
        "screening",
        "interview",
        "technical_interview",
        "final_interview",
        "rejected",
        "withdrawn",
        "offer",
        "accepted",
        "declined",
    }
)
_SCREENING_STATUSES = frozenset(
    {
        "screening",
        "interview",
        "technical_interview",
        "final_interview",
        "offer",
        "accepted",
        "declined",
    }
)
_INTERVIEW_STATUSES = frozenset(
    {
        "interview",
        "technical_interview",
        "final_interview",
        "offer",
        "accepted",
        "declined",
    }
)
_OFFER_STATUSES = frozenset({"offer", "accepted", "declined"})
_ACTIVE_INTERVIEW_STATUSES = frozenset(INTERVIEW_TYPES)
_UPDATE_REQUIRED_STATUSES = frozenset(
    {
        "applied",
        "screening",
        "interview",
        "technical_interview",
        "final_interview",
        "offer",
    }
)
_SKILL_CATEGORIES = frozenset(
    {
        "required_qualification",
        "preferred_qualification",
        "technology_and_skill",
        "ai_focus_area",
    }
)

# This dictionary is deliberately source-controlled and immutable at runtime. An
# LLM is never consulted when grouping requirement wording.
SKILL_ALIASES: Mapping[str, str] = MappingProxyType({
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "ci cd": "CI/CD",
    "continuous integration": "CI/CD",
    "continuous integration continuous delivery": "CI/CD",
    "continuous integration continuous deployment": "CI/CD",
})

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\s().-]*){7,}\d(?!\d)")
_ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9.' -]{2,80}\s+"
    r"(?:street|st\.?|avenue|ave\.?|road|rd\.?|boulevard|blvd\.?|"
    r"lane|ln\.?|drive|dr\.?|court|ct\.?)\b",
    re.I,
)
_CREDENTIAL_RE = re.compile(
    r"(?:\bBearer\s+[A-Za-z0-9._~+/=-]+|\bapify_api_[A-Za-z0-9_-]+|"
    r"\b(?:api[_ -]?key|access[_ -]?token|authorization)\s*[:=])",
    re.I,
)
_TITLE_LEVEL_RE = re.compile(
    r"\b(?:jr\.?|junior|sr\.?|senior|staff|principal|lead|entry[- ]level|"
    r"mid[- ]level|level\s+[ivx]+|[ivx]{1,4})\b",
    re.I,
)


class AnalyticsError(RuntimeError):
    """The isolated local analytics subsystem could not complete an operation."""


@dataclass(frozen=True)
class JobObservation:
    title: str
    company: str
    source: str
    description_sha256: str
    linkedin_job_id: str | None = None
    canonical_url: str | None = None
    location: str | None = None
    workplace_type: str | None = None
    employment_type: str | None = None
    seniority: str | None = None
    compensation_text: str | None = None
    compensation_min: int | None = None
    compensation_max: int | None = None
    compensation_currency: str | None = None
    posting_date: str | None = None
    applicant_count: int | None = None
    captured_at: str | None = None


@dataclass(frozen=True)
class JobRecordResult:
    job_id: int
    application_id: int
    created: bool
    snapshot_created: bool


def _inside_git_repository(path: Path) -> bool:
    directory = path if path.is_dir() else path.parent
    for candidate in (directory, *directory.parents):
        marker = candidate / ".git"
        if marker.is_dir() and (marker / "HEAD").is_file():
            return True
        if marker.is_file():
            try:
                if marker.read_text(encoding="utf-8", errors="replace").startswith(
                    "gitdir:"
                ):
                    return True
            except OSError:
                continue
    return False


def default_application_data_directory() -> Path:
    configured = os.environ.get("XDG_DATA_HOME", "").strip()
    if configured and Path(configured).is_absolute():
        root = Path(configured)
    else:
        root = Path.home() / ".local" / "share"
    return root / "resume-tailor" / "data"


def default_analytics_database_path() -> Path:
    configured = os.environ.get("RESUME_TAILOR_ANALYTICS_DB", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_absolute():
            return candidate
    return default_application_data_directory() / ANALYTICS_DATABASE_FILENAME


def _normalize_timestamp(value: str | datetime | None = None) -> str:
    if value is None:
        return utc_now_iso()
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise InputError("Analytics timestamps must use timezone-aware ISO 8601.") from exc
    else:
        raise InputError("Analytics timestamps must use timezone-aware ISO 8601.")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise InputError("Analytics timestamps must include a timezone offset.")
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds")


def _clean_text(
    value: Any,
    *,
    label: str,
    maximum: int,
    required: bool = False,
) -> str | None:
    if value is None:
        if required:
            raise InputError(f"{label} is required for local analytics.")
        return None
    if not isinstance(value, str):
        raise InputError(f"{label} must be text for local analytics.")
    cleaned = " ".join(value.replace("\r", "\n").split())
    if not cleaned:
        if required:
            raise InputError(f"{label} is required for local analytics.")
        return None
    if len(cleaned) > maximum:
        raise InputError(f"{label} exceeds the {maximum:,}-character analytics limit.")
    if any(ord(character) < 32 or 0x7F <= ord(character) <= 0x9F for character in cleaned):
        raise InputError(f"{label} contains unsafe control text.")
    return cleaned


def _clean_user_text(value: Any, *, label: str, maximum: int) -> str | None:
    cleaned = _clean_text(value, label=label, maximum=maximum)
    if cleaned is None:
        return None
    phone_candidate = re.sub(r"\b[0-9]{4}-[0-9]{2}-[0-9]{2}\b", "", cleaned)
    if (
        _EMAIL_RE.search(cleaned)
        or _PHONE_RE.search(phone_candidate)
        or _ADDRESS_RE.search(cleaned)
        or _CREDENTIAL_RE.search(cleaned)
    ):
        raise InputError(
            f"{label} must not contain an address, email, phone number, or credential."
        )
    return cleaned


def _normalize_source(value: str) -> str:
    source = (_clean_text(value, label="Analytics source", maximum=80, required=True) or "").casefold()
    if "apify" in source or source in {"linkedin-url", "linkedin"}:
        return "apify"
    if "clipboard" in source:
        return "clipboard"
    if "paste" in source:
        return "pasted_text"
    if source in {"job-file", "file", "text-file"}:
        return "file"
    if "retry" in source or "reprocess" in source:
        return "preserved_artifacts"
    return source.replace(" ", "_")


def canonicalize_job_url(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = _clean_text(value, label="Canonical job URL", maximum=2048)
    if candidate is None:
        return None
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError as exc:
        raise InputError("Canonical job URL is malformed.") from exc
    if parsed.scheme.casefold() != "https" or parsed.username or parsed.password:
        raise InputError("Canonical job URLs must be credential-free HTTPS URLs.")
    hostname = (parsed.hostname or "").casefold()
    if hostname not in {"linkedin.com", "www.linkedin.com"} or port not in {None, 443}:
        raise InputError("Analytics accepts only public LinkedIn canonical job URLs.")
    path = unquote(parsed.path or "/")
    if "\\" in path or "/../" in f"{path}/" or any(ord(item) < 32 for item in path):
        raise InputError("Canonical job URL contains an unsafe path.")
    path = re.sub(r"/{2,}", "/", path)
    if path != "/":
        path = path.rstrip("/") + "/"
    # All query parameters are omitted. LinkedIn URL identity comes from the path
    # and, preferentially, the separately validated LinkedIn job ID.
    return urlunsplit(("https", "www.linkedin.com", path, "", ""))


def parse_applicant_count(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if not isinstance(value, str):
        return None
    match = re.search(r"\b([0-9][0-9,]*)\b", value)
    if match is None:
        return None
    try:
        parsed = int(match.group(1).replace(",", ""))
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _parse_compensation(value: Any) -> tuple[str | None, int | None, int | None, str | None]:
    text = _clean_text(value, label="Compensation", maximum=500)
    if text is None:
        return None, None, None, None
    currency = None
    if "$" in text or re.search(r"\bUSD\b", text, re.I):
        currency = "USD"
    elif "€" in text or re.search(r"\bEUR\b", text, re.I):
        currency = "EUR"
    elif "£" in text or re.search(r"\bGBP\b", text, re.I):
        currency = "GBP"
    amounts: list[int] = []
    for raw, suffix in re.findall(r"(?:[$€£]\s*)?([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)([kK]?)", text):
        try:
            amount = float(raw.replace(",", ""))
        except ValueError:
            continue
        if suffix:
            amount *= 1000
        if amount >= 1000:
            amounts.append(int(amount))
    minimum = min(amounts[:2]) if amounts else None
    maximum = max(amounts[:2]) if amounts else None
    return text, minimum, maximum, currency


def title_family(value: str) -> str:
    cleaned = _clean_text(value, label="Job title", maximum=300, required=True) or ""
    family = _TITLE_LEVEL_RE.sub(" ", cleaned)
    family = re.sub(r"\s*[-–—|]\s*", " ", family)
    family = " ".join(family.split()).strip(" ,/()")
    return family or cleaned


def _skill_alias_key(value: str) -> str:
    normalized = value.casefold().replace("&", " and ")
    normalized = re.sub(r"[/+_-]+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def normalize_skill_name(value: str) -> tuple[str, str]:
    original = _clean_text(
        value,
        label="Original skill wording",
        maximum=5_000,
        required=True,
    ) or ""
    alias_key = _skill_alias_key(original)
    canonical = SKILL_ALIASES.get(alias_key, original)
    return canonical, _skill_alias_key(canonical)


def _job_identity(
    *,
    linkedin_job_id: str | None,
    canonical_url: str | None,
    company: str,
    title: str,
    description_sha256: str,
) -> str:
    if linkedin_job_id:
        return f"linkedin:{linkedin_job_id}"
    if canonical_url:
        return "url:" + hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()
    material = "\0".join(
        (
            company.casefold().strip(),
            title.casefold().strip(),
            description_sha256,
        )
    )
    return "local:" + hashlib.sha256(material.encode("utf-8")).hexdigest()


def _material_hash(values: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        values,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def observation_from_canonical_job(payload: Mapping[str, Any]) -> JobObservation:
    description = payload.get("normalized_job_description")
    if not isinstance(description, str) or not description.strip():
        raise InputError("Canonical analytics input is missing its job description.")
    compensation, minimum, maximum, currency = _parse_compensation(payload.get("salary"))
    return JobObservation(
        linkedin_job_id=_clean_text(
            payload.get("linkedin_job_id"),
            label="LinkedIn job ID",
            maximum=20,
        ),
        canonical_url=str(payload.get("final_resolved_url") or payload.get("requested_url") or "") or None,
        title=str(payload.get("job_title") or ""),
        company=str(payload.get("company") or ""),
        location=_clean_text(payload.get("location"), label="Job location", maximum=500),
        workplace_type=_clean_text(
            payload.get("workplace_type"), label="Workplace type", maximum=80
        ),
        employment_type=_clean_text(
            payload.get("employment_type"), label="Employment type", maximum=200
        ),
        seniority=_clean_text(
            payload.get("seniority_level"), label="Seniority", maximum=300
        ),
        compensation_text=compensation,
        compensation_min=minimum,
        compensation_max=maximum,
        compensation_currency=currency,
        posting_date=_clean_text(
            payload.get("date_posted"), label="Posting date", maximum=200
        ),
        applicant_count=parse_applicant_count(payload.get("applicant_count")),
        source=_normalize_source(str(payload.get("retrieval_source") or "apify")),
        description_sha256=job_description_sha256(description),
    )


def observation_from_local_job(
    *,
    company: str,
    title: str,
    description: str,
    source: str,
    linkedin_job_id: str | None = None,
    canonical_url: str | None = None,
) -> JobObservation:
    if not isinstance(description, str) or not description.strip():
        raise InputError("Local analytics input is missing its job description.")
    return JobObservation(
        linkedin_job_id=linkedin_job_id,
        canonical_url=canonical_url,
        title=title,
        company=company,
        source=_normalize_source(source),
        description_sha256=job_description_sha256(description),
    )


_MIGRATION_1 = f"""
CREATE TABLE jobs (
    id INTEGER PRIMARY KEY,
    identity_key TEXT NOT NULL UNIQUE,
    linkedin_job_id TEXT,
    canonical_url TEXT,
    title TEXT NOT NULL,
    title_family TEXT NOT NULL,
    company TEXT NOT NULL,
    location TEXT,
    workplace_type TEXT,
    employment_type TEXT,
    seniority TEXT,
    compensation_text TEXT,
    compensation_min INTEGER,
    compensation_max INTEGER,
    compensation_currency TEXT,
    posting_date TEXT,
    source TEXT NOT NULL,
    description_sha256 TEXT NOT NULL CHECK(length(description_sha256) = 64),
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK(compensation_min IS NULL OR compensation_min >= 0),
    CHECK(compensation_max IS NULL OR compensation_max >= 0)
);
CREATE UNIQUE INDEX idx_jobs_linkedin_job_id
    ON jobs(linkedin_job_id) WHERE linkedin_job_id IS NOT NULL;
CREATE UNIQUE INDEX idx_jobs_canonical_url
    ON jobs(canonical_url) WHERE canonical_url IS NOT NULL;
CREATE INDEX idx_jobs_last_seen_at ON jobs(last_seen_at DESC);
CREATE INDEX idx_jobs_title_family ON jobs(title_family);

CREATE TABLE job_snapshots (
    id INTEGER PRIMARY KEY,
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    captured_at TEXT NOT NULL,
    applicant_count INTEGER,
    posting_date TEXT,
    description_sha256 TEXT NOT NULL CHECK(length(description_sha256) = 64),
    title TEXT NOT NULL,
    company TEXT NOT NULL,
    location TEXT,
    workplace_type TEXT,
    employment_type TEXT,
    seniority TEXT,
    compensation_text TEXT,
    compensation_min INTEGER,
    compensation_max INTEGER,
    compensation_currency TEXT,
    source TEXT NOT NULL,
    material_hash TEXT NOT NULL CHECK(length(material_hash) = 64),
    CHECK(applicant_count IS NULL OR applicant_count >= 0)
);
CREATE INDEX idx_job_snapshots_job_captured
    ON job_snapshots(job_id, captured_at DESC);
CREATE INDEX idx_job_snapshots_material
    ON job_snapshots(job_id, material_hash);

CREATE TABLE skills (
    id INTEGER PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    normalized_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE job_skills (
    id INTEGER PRIMARY KEY,
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    skill_id INTEGER NOT NULL REFERENCES skills(id) ON DELETE RESTRICT,
    requirement_id TEXT NOT NULL,
    original_wording TEXT NOT NULL,
    requirement_level TEXT NOT NULL
        CHECK(requirement_level IN ('required', 'preferred', 'unspecified')),
    requirement_category TEXT NOT NULL,
    evidence_reference TEXT NOT NULL,
    evidence_excerpt TEXT NOT NULL,
    gap_status TEXT CHECK(gap_status IN ('supported', 'missing')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(job_id, requirement_id)
);
CREATE INDEX idx_job_skills_skill ON job_skills(skill_id, job_id);
CREATE INDEX idx_job_skills_gap ON job_skills(gap_status, skill_id);

CREATE TABLE job_events (
    id INTEGER PRIMARY KEY,
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL CHECK(event_type IN ('viewed', 'approved_for_tailoring')),
    timestamp TEXT NOT NULL,
    source TEXT NOT NULL,
    run_identifier TEXT,
    event_key TEXT UNIQUE
);
CREATE INDEX idx_job_events_type_timestamp
    ON job_events(event_type, timestamp DESC);
CREATE INDEX idx_job_events_job_timestamp
    ON job_events(job_id, timestamp DESC);

CREATE TABLE applications (
    id INTEGER PRIMARY KEY,
    job_id INTEGER NOT NULL UNIQUE REFERENCES jobs(id) ON DELETE CASCADE,
    current_status TEXT NOT NULL CHECK(current_status IN {APPLICATION_STATUSES!r}),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX idx_applications_status_updated
    ON applications(current_status, updated_at DESC);
CREATE TRIGGER applications_status_requires_event
BEFORE UPDATE OF current_status ON applications
WHEN NEW.current_status <> OLD.current_status
BEGIN
    SELECT CASE WHEN COALESCE((
        SELECT event.new_status FROM application_status_events event
        WHERE event.application_id = OLD.id
        ORDER BY event.id DESC LIMIT 1
    ), '') <> NEW.current_status
    THEN RAISE(ABORT, 'application status changes require an append-only event') END;
END;

CREATE TABLE application_status_events (
    id INTEGER PRIMARY KEY,
    application_id INTEGER NOT NULL REFERENCES applications(id) ON DELETE CASCADE,
    previous_status TEXT CHECK(previous_status IS NULL OR previous_status IN {APPLICATION_STATUSES!r}),
    new_status TEXT NOT NULL CHECK(new_status IN {APPLICATION_STATUSES!r}),
    timestamp TEXT NOT NULL,
    source TEXT NOT NULL,
    note TEXT,
    event_kind TEXT NOT NULL DEFAULT 'status_change'
        CHECK(event_kind IN ('status_change', 'correction')),
    correction_of_event_id INTEGER REFERENCES application_status_events(id) ON DELETE RESTRICT
);
CREATE INDEX idx_application_status_events_application
    ON application_status_events(application_id, timestamp, id);
CREATE INDEX idx_application_status_events_status_timestamp
    ON application_status_events(new_status, timestamp DESC);
CREATE TRIGGER application_status_events_no_update
BEFORE UPDATE ON application_status_events
BEGIN
    SELECT RAISE(ABORT, 'application status history is append-only');
END;
CREATE TRIGGER application_status_events_no_delete
BEFORE DELETE ON application_status_events
BEGIN
    SELECT RAISE(ABORT, 'application status history is append-only');
END;

CREATE TABLE application_notes (
    id INTEGER PRIMARY KEY,
    application_id INTEGER NOT NULL REFERENCES applications(id) ON DELETE CASCADE,
    timestamp TEXT NOT NULL,
    source TEXT NOT NULL,
    note TEXT NOT NULL
);
CREATE INDEX idx_application_notes_application
    ON application_notes(application_id, timestamp DESC);
CREATE TRIGGER application_notes_no_update
BEFORE UPDATE ON application_notes
BEGIN
    SELECT RAISE(ABORT, 'application notes are append-only');
END;
CREATE TRIGGER application_notes_no_delete
BEFORE DELETE ON application_notes
BEGIN
    SELECT RAISE(ABORT, 'application notes are append-only');
END;

CREATE TABLE resume_versions (
    id INTEGER PRIMARY KEY,
    application_id INTEGER NOT NULL REFERENCES applications(id) ON DELETE CASCADE,
    run_identifier TEXT NOT NULL,
    artifact_reference TEXT NOT NULL,
    created_at TEXT NOT NULL,
    writer_provider TEXT NOT NULL,
    qa_outcome TEXT NOT NULL,
    match_score REAL,
    UNIQUE(application_id, run_identifier, artifact_reference),
    CHECK(match_score IS NULL OR (match_score >= 0 AND match_score <= 100))
);
CREATE INDEX idx_resume_versions_application
    ON resume_versions(application_id, created_at DESC);
CREATE INDEX idx_resume_versions_run_identifier
    ON resume_versions(run_identifier);

CREATE TABLE interviews (
    id INTEGER PRIMARY KEY,
    application_id INTEGER NOT NULL REFERENCES applications(id) ON DELETE CASCADE,
    interview_type TEXT NOT NULL CHECK(interview_type IN {INTERVIEW_TYPES!r}),
    scheduled_at TEXT,
    completed_at TEXT,
    contact_label TEXT,
    result TEXT,
    notes TEXT,
    confirmed_at TEXT NOT NULL,
    source TEXT NOT NULL
);
CREATE INDEX idx_interviews_application_scheduled
    ON interviews(application_id, scheduled_at DESC);
CREATE INDEX idx_interviews_type_scheduled
    ON interviews(interview_type, scheduled_at DESC);
""".replace(str(APPLICATION_STATUSES), "(" + ", ".join(repr(item) for item in APPLICATION_STATUSES) + ")").replace(
    str(INTERVIEW_TYPES), "(" + ", ".join(repr(item) for item in INTERVIEW_TYPES) + ")"
).replace("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS ").replace(
    "CREATE UNIQUE INDEX ", "CREATE UNIQUE INDEX IF NOT EXISTS "
).replace("CREATE INDEX ", "CREATE INDEX IF NOT EXISTS ").replace(
    "CREATE TRIGGER ", "CREATE TRIGGER IF NOT EXISTS "
)


class AnalyticsStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = (path or default_analytics_database_path()).expanduser().resolve()

    def _prepare_directory(self) -> None:
        if _inside_git_repository(self.path):
            raise InputError(
                "The private analytics database must be outside a Git repository."
            )
        created = not self.path.parent.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if created:
            try:
                self.path.parent.chmod(0o700)
            except OSError:
                pass

    def _open(self) -> sqlite3.Connection:
        self._prepare_directory()
        connection = sqlite3.connect(self.path, timeout=5.0)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
            connection.execute("PRAGMA secure_delete = ON")
            self._enable_wal(connection)
        except Exception:
            connection.close()
            raise
        try:
            self.path.chmod(0o600)
        except OSError:
            pass
        return connection

    @staticmethod
    def _enable_wal(connection: sqlite3.Connection) -> None:
        """Negotiate WAL safely when multiple first-time initializers race."""
        deadline = time.monotonic() + (SQLITE_BUSY_TIMEOUT_MS / 1_000)
        while True:
            try:
                row = connection.execute("PRAGMA journal_mode = WAL").fetchone()
                if row is None or str(row[0]).casefold() != "wal":
                    raise sqlite3.OperationalError(
                        "SQLite did not enable the required WAL journal mode."
                    )
                return
            except sqlite3.OperationalError as exc:
                busy = any(
                    marker in str(exc).casefold()
                    for marker in ("database is locked", "database is busy")
                )
                if not busy or time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)

    def initialize(self) -> int:
        try:
            connection = self._open()
        except InputError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise AnalyticsError(
                "Could not open the private local analytics database."
            ) from exc
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
            connection.commit()
            rows = connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
            versions = [int(row["version"]) for row in rows]
            if versions and versions != list(range(1, max(versions) + 1)):
                raise AnalyticsError("Analytics schema migration history is not contiguous.")
            current = versions[-1] if versions else 0
            if current > ANALYTICS_SCHEMA_VERSION:
                raise AnalyticsError(
                    "Analytics database was created by a newer Resume Tailor version."
                )
            migrations = {1: _MIGRATION_1}
            for version in range(current + 1, ANALYTICS_SCHEMA_VERSION + 1):
                sql = migrations.get(version)
                if sql is None:
                    raise AnalyticsError(f"Missing analytics migration {version}.")
                try:
                    connection.executescript("BEGIN IMMEDIATE;\n" + sql)
                    connection.execute(
                        "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                        (version, utc_now_iso()),
                    )
                    connection.execute(f"PRAGMA user_version = {version}")
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise
            pragma_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if pragma_version != ANALYTICS_SCHEMA_VERSION:
                raise AnalyticsError("Analytics schema version markers do not match.")
            return ANALYTICS_SCHEMA_VERSION
        except sqlite3.Error as exc:
            raise AnalyticsError("Could not initialize the local analytics database.") from exc
        finally:
            connection.close()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self.initialize()
        try:
            connection = self._open()
        except InputError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise AnalyticsError(
                "Could not open the private local analytics database."
            ) from exc
        try:
            with connection:
                yield connection
        except sqlite3.Error as exc:
            raise AnalyticsError("The local analytics transaction failed.") from exc
        finally:
            connection.close()

    def schema_version(self) -> int:
        return self.initialize()

    def record_job_viewed(
        self,
        observation: JobObservation,
        *,
        run_identifier: str | None = None,
    ) -> JobRecordResult:
        timestamp = _normalize_timestamp(observation.captured_at)
        title = _clean_text(observation.title, label="Job title", maximum=300, required=True) or ""
        company = _clean_text(observation.company, label="Company", maximum=300, required=True) or ""
        source = _normalize_source(observation.source)
        description_hash = observation.description_sha256.casefold()
        if not _HASH_RE.fullmatch(description_hash):
            raise InputError("Analytics description hash must be a SHA-256 digest.")
        linkedin_id = _clean_text(
            observation.linkedin_job_id, label="LinkedIn job ID", maximum=20
        )
        if linkedin_id is not None and not re.fullmatch(r"[0-9]{5,20}", linkedin_id):
            raise InputError("Analytics LinkedIn job ID is malformed.")
        canonical_url = canonicalize_job_url(observation.canonical_url)
        identity = _job_identity(
            linkedin_job_id=linkedin_id,
            canonical_url=canonical_url,
            company=company,
            title=title,
            description_sha256=description_hash,
        )
        run_id = None
        if run_identifier is not None:
            run_id = _clean_text(
                run_identifier,
                label="Analytics run identifier",
                maximum=200,
                required=True,
            )
        fields = {
            "title": title,
            "company": company,
            "location": _clean_text(
                observation.location, label="Job location", maximum=500
            ),
            "workplace_type": _clean_text(
                observation.workplace_type, label="Workplace type", maximum=80
            ),
            "employment_type": _clean_text(
                observation.employment_type, label="Employment type", maximum=200
            ),
            "seniority": _clean_text(
                observation.seniority, label="Seniority", maximum=300
            ),
            "compensation_text": _clean_text(
                observation.compensation_text, label="Compensation", maximum=500
            ),
            "compensation_min": observation.compensation_min,
            "compensation_max": observation.compensation_max,
            "compensation_currency": _clean_text(
                observation.compensation_currency,
                label="Compensation currency",
                maximum=10,
            ),
            "posting_date": _clean_text(
                observation.posting_date, label="Posting date", maximum=200
            ),
            "source": source,
            "description_sha256": description_hash,
        }
        applicant_count = observation.applicant_count
        if applicant_count is not None and (
            isinstance(applicant_count, bool)
            or not isinstance(applicant_count, int)
            or applicant_count < 0
        ):
            raise InputError("Applicant count must be a non-negative integer or null.")
        for key in ("compensation_min", "compensation_max"):
            value = fields[key]
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise InputError("Compensation amounts must be non-negative integers.")
        material = {**fields, "applicant_count": applicant_count}
        snapshot_hash = _material_hash(material)

        with self._connection() as connection:
            # Serialize identity resolution and latest-snapshot comparison. This
            # prevents concurrent observations from inserting duplicate no-op
            # snapshots while still allowing a material A -> B -> A history.
            connection.execute("BEGIN IMMEDIATE")
            matches: list[sqlite3.Row] = []
            if linkedin_id is not None:
                row = connection.execute(
                    "SELECT * FROM jobs WHERE linkedin_job_id = ?", (linkedin_id,)
                ).fetchone()
                if row is not None:
                    matches.append(row)
            if canonical_url is not None:
                row = connection.execute(
                    "SELECT * FROM jobs WHERE canonical_url = ?", (canonical_url,)
                ).fetchone()
                if row is not None and all(existing["id"] != row["id"] for existing in matches):
                    matches.append(row)
            row = connection.execute(
                "SELECT * FROM jobs WHERE identity_key = ?", (identity,)
            ).fetchone()
            if row is not None and all(existing["id"] != row["id"] for existing in matches):
                matches.append(row)
            if len(matches) > 1:
                raise AnalyticsError("Conflicting analytics job identities require manual review.")

            created = not matches
            if created:
                cursor = connection.execute(
                    """
                    INSERT INTO jobs(
                        identity_key, linkedin_job_id, canonical_url, title,
                        title_family, company, location, workplace_type,
                        employment_type, seniority, compensation_text,
                        compensation_min, compensation_max, compensation_currency,
                        posting_date, source, description_sha256, first_seen_at,
                        last_seen_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identity,
                        linkedin_id,
                        canonical_url,
                        title,
                        title_family(title),
                        company,
                        fields["location"],
                        fields["workplace_type"],
                        fields["employment_type"],
                        fields["seniority"],
                        fields["compensation_text"],
                        fields["compensation_min"],
                        fields["compensation_max"],
                        fields["compensation_currency"],
                        fields["posting_date"],
                        source,
                        description_hash,
                        timestamp,
                        timestamp,
                        timestamp,
                        timestamp,
                    ),
                )
                job_id = int(cursor.lastrowid)
            else:
                existing = matches[0]
                if (
                    linkedin_id is not None
                    and existing["linkedin_job_id"] is not None
                    and existing["linkedin_job_id"] != linkedin_id
                ):
                    raise AnalyticsError(
                        "Conflicting LinkedIn job identities require manual review."
                    )
                job_id = int(existing["id"])
                connection.execute(
                    """
                    UPDATE jobs SET
                        identity_key = ?,
                        linkedin_job_id = COALESCE(?, linkedin_job_id),
                        canonical_url = COALESCE(?, canonical_url),
                        title = ?, title_family = ?, company = ?, location = ?,
                        workplace_type = ?, employment_type = ?, seniority = ?,
                        compensation_text = ?, compensation_min = ?,
                        compensation_max = ?, compensation_currency = ?,
                        posting_date = ?, source = ?, description_sha256 = ?,
                        first_seen_at = CASE WHEN first_seen_at < ? THEN first_seen_at ELSE ? END,
                        last_seen_at = CASE WHEN last_seen_at > ? THEN last_seen_at ELSE ? END,
                        updated_at = CASE WHEN updated_at > ? THEN updated_at ELSE ? END
                    WHERE id = ?
                    """,
                    (
                        identity,
                        linkedin_id,
                        canonical_url,
                        title,
                        title_family(title),
                        company,
                        fields["location"],
                        fields["workplace_type"],
                        fields["employment_type"],
                        fields["seniority"],
                        fields["compensation_text"],
                        fields["compensation_min"],
                        fields["compensation_max"],
                        fields["compensation_currency"],
                        fields["posting_date"],
                        source,
                        description_hash,
                        timestamp,
                        timestamp,
                        timestamp,
                        timestamp,
                        timestamp,
                        timestamp,
                        job_id,
                    ),
                )

            latest_snapshot = connection.execute(
                """
                SELECT material_hash FROM job_snapshots
                WHERE job_id = ? ORDER BY captured_at DESC, id DESC LIMIT 1
                """,
                (job_id,),
            ).fetchone()
            snapshot_created = (
                latest_snapshot is None
                or latest_snapshot["material_hash"] != snapshot_hash
            )
            if snapshot_created:
                connection.execute(
                    """
                    INSERT INTO job_snapshots(
                        job_id, captured_at, applicant_count, posting_date,
                        description_sha256, title, company, location, workplace_type,
                        employment_type, seniority, compensation_text,
                        compensation_min, compensation_max, compensation_currency,
                        source, material_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        timestamp,
                        applicant_count,
                        fields["posting_date"],
                        description_hash,
                        title,
                        company,
                        fields["location"],
                        fields["workplace_type"],
                        fields["employment_type"],
                        fields["seniority"],
                        fields["compensation_text"],
                        fields["compensation_min"],
                        fields["compensation_max"],
                        fields["compensation_currency"],
                        source,
                        snapshot_hash,
                    ),
                )
            application = connection.execute(
                "SELECT id FROM applications WHERE job_id = ?", (job_id,)
            ).fetchone()
            if application is None:
                cursor = connection.execute(
                    """
                    INSERT INTO applications(job_id, current_status, created_at, updated_at)
                    VALUES (?, 'viewed', ?, ?)
                    """,
                    (job_id, timestamp, timestamp),
                )
                application_id = int(cursor.lastrowid)
                connection.execute(
                    """
                    INSERT INTO application_status_events(
                        application_id, previous_status, new_status, timestamp,
                        source, event_kind
                    ) VALUES (?, NULL, 'viewed', ?, ?, 'status_change')
                    """,
                    (application_id, timestamp, source),
                )
            else:
                application_id = int(application["id"])
            event_key = f"view:{run_id}" if run_id else None
            connection.execute(
                """
                INSERT OR IGNORE INTO job_events(
                    job_id, event_type, timestamp, source, run_identifier, event_key
                ) VALUES (?, 'viewed', ?, ?, ?, ?)
                """,
                (job_id, timestamp, source, run_id, event_key),
            )
        return JobRecordResult(
            job_id=job_id,
            application_id=application_id,
            created=created,
            snapshot_created=snapshot_created,
        )

    def record_requirements(
        self,
        job_id: int,
        catalog: Mapping[str, Any],
        *,
        job_description: str | None = None,
    ) -> int:
        validated = validate_job_requirement_catalog(
            dict(catalog),
            job_description=job_description,
        )
        timestamp = utc_now_iso()
        stored = 0
        with self._connection() as connection:
            self._require_job(connection, job_id)
            for requirement in validated:
                category = requirement["category"]
                if category not in _SKILL_CATEGORIES:
                    continue
                original = requirement["exact_text"]
                canonical, normalized = normalize_skill_name(original)
                connection.execute(
                    """
                    INSERT INTO skills(canonical_name, normalized_key, created_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(normalized_key) DO NOTHING
                    """,
                    (canonical, normalized, timestamp),
                )
                skill = connection.execute(
                    "SELECT id FROM skills WHERE normalized_key = ?", (normalized,)
                ).fetchone()
                assert skill is not None
                level = (
                    "required"
                    if category == "required_qualification"
                    else "preferred"
                    if category == "preferred_qualification"
                    else "unspecified"
                )
                cursor = connection.execute(
                    """
                    INSERT INTO job_skills(
                        job_id, skill_id, requirement_id, original_wording,
                        requirement_level, requirement_category,
                        evidence_reference, evidence_excerpt, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(job_id, requirement_id) DO UPDATE SET
                        skill_id = excluded.skill_id,
                        original_wording = excluded.original_wording,
                        requirement_level = excluded.requirement_level,
                        requirement_category = excluded.requirement_category,
                        evidence_reference = excluded.evidence_reference,
                        evidence_excerpt = excluded.evidence_excerpt,
                        updated_at = excluded.updated_at
                    """,
                    (
                        job_id,
                        int(skill["id"]),
                        requirement["requirement_id"],
                        original,
                        level,
                        category,
                        requirement["requirement_id"],
                        original,
                        timestamp,
                        timestamp,
                    ),
                )
                stored += 1 if cursor.rowcount >= 0 else 0
        return stored

    def record_gap_assessments(
        self,
        job_id: int,
        resolved_analysis: Mapping[str, Any],
    ) -> int:
        assessments = resolved_analysis.get("requirement_assessment")
        if not isinstance(assessments, list):
            raise InputError("Validated analysis has no requirement assessment.")
        changed = 0
        with self._connection() as connection:
            self._require_job(connection, job_id)
            for item in assessments:
                if not isinstance(item, Mapping):
                    raise InputError("Validated requirement assessment is malformed.")
                requirement_id = item.get("requirement_id")
                status = item.get("status")
                if not isinstance(requirement_id, str) or status not in {
                    "present_verbatim",
                    "supported_by_source",
                    "unsupported",
                }:
                    raise InputError("Validated requirement assessment is malformed.")
                gap = "missing" if status == "unsupported" else "supported"
                cursor = connection.execute(
                    """
                    UPDATE job_skills SET gap_status = ?, updated_at = ?
                    WHERE job_id = ? AND requirement_id = ?
                    """,
                    (gap, utc_now_iso(), job_id, requirement_id),
                )
                changed += cursor.rowcount
        return changed

    def record_tailoring_approval(
        self,
        job_id: int,
        *,
        run_identifier: str,
        source: str = "pipeline",
        timestamp: str | datetime | None = None,
    ) -> int:
        observed_at = _normalize_timestamp(timestamp)
        run_id = _clean_text(
            run_identifier,
            label="Analytics run identifier",
            maximum=200,
            required=True,
        ) or ""
        source_value = _normalize_source(source)
        with self._connection() as connection:
            self._require_job(connection, job_id)
            connection.execute(
                """
                INSERT OR IGNORE INTO job_events(
                    job_id, event_type, timestamp, source, run_identifier, event_key
                ) VALUES (?, 'approved_for_tailoring', ?, ?, ?, ?)
                """,
                (
                    job_id,
                    observed_at,
                    source_value,
                    run_id,
                    f"approved:{run_id}",
                ),
            )
            application_id = self._application_id(connection, job_id)
            application = self._require_application(connection, application_id)
            if application["current_status"] in {"viewed", "saved"}:
                self._append_status(
                    connection,
                    application_id,
                    "planned",
                    source=source_value,
                    timestamp=observed_at,
                    note=None,
                )
        return application_id

    def application_for_job(self, job_id: int) -> int:
        with self._connection() as connection:
            return self._application_id(connection, job_id)

    def set_application_status(
        self,
        application_id: int,
        new_status: str,
        *,
        source: str = "manual_ui",
        note: str | None = None,
        timestamp: str | datetime | None = None,
    ) -> int:
        status = self._status(new_status)
        event_time = _normalize_timestamp(timestamp)
        safe_note = _clean_user_text(note, label="Application note", maximum=2_000)
        with self._connection() as connection:
            return self._append_status(
                connection,
                application_id,
                status,
                source=_normalize_source(source),
                timestamp=event_time,
                note=safe_note,
            )

    def correct_application_status(
        self,
        application_id: int,
        new_status: str,
        *,
        confirmed: bool,
        source: str = "manual_ui",
        note: str | None = None,
        correction_of_event_id: int | None = None,
        timestamp: str | datetime | None = None,
    ) -> int:
        if not confirmed:
            raise InputError("Status correction requires explicit confirmation.")
        status = self._status(new_status)
        safe_note = _clean_user_text(note, label="Correction note", maximum=2_000)
        event_time = _normalize_timestamp(timestamp)
        with self._connection() as connection:
            application = self._require_application(connection, application_id)
            previous = str(application["current_status"])
            if previous == status:
                raise InputError("The corrected status must differ from the current status.")
            if correction_of_event_id is None:
                latest = connection.execute(
                    """
                    SELECT id FROM application_status_events
                    WHERE application_id = ? ORDER BY timestamp DESC, id DESC LIMIT 1
                    """,
                    (application_id,),
                ).fetchone()
                correction_of_event_id = int(latest["id"]) if latest is not None else None
            if correction_of_event_id is not None:
                target = connection.execute(
                    """
                    SELECT id FROM application_status_events
                    WHERE id = ? AND application_id = ?
                    """,
                    (correction_of_event_id, application_id),
                ).fetchone()
                if target is None:
                    raise InputError("The correction target is not part of this application.")
            cursor = connection.execute(
                """
                INSERT INTO application_status_events(
                    application_id, previous_status, new_status, timestamp,
                    source, note, event_kind, correction_of_event_id
                ) VALUES (?, ?, ?, ?, ?, ?, 'correction', ?)
                """,
                (
                    application_id,
                    previous,
                    status,
                    event_time,
                    _normalize_source(source),
                    safe_note,
                    correction_of_event_id,
                ),
            )
            connection.execute(
                "UPDATE applications SET current_status = ?, updated_at = ? WHERE id = ?",
                (status, event_time, application_id),
            )
            return int(cursor.lastrowid)

    def add_note(
        self,
        application_id: int,
        note: str,
        *,
        source: str = "manual_ui",
        timestamp: str | datetime | None = None,
    ) -> int:
        safe_note = _clean_user_text(note, label="Application note", maximum=2_000)
        if safe_note is None:
            raise InputError("Application note must not be empty.")
        event_time = _normalize_timestamp(timestamp)
        with self._connection() as connection:
            self._require_application(connection, application_id)
            cursor = connection.execute(
                """
                INSERT INTO application_notes(application_id, timestamp, source, note)
                VALUES (?, ?, ?, ?)
                """,
                (application_id, event_time, _normalize_source(source), safe_note),
            )
            return int(cursor.lastrowid)

    def record_interview(
        self,
        application_id: int,
        interview_type: str,
        *,
        confirmed: bool,
        scheduled_at: str | datetime | None = None,
        completed_at: str | datetime | None = None,
        contact_label: str | None = None,
        result: str | None = None,
        notes: str | None = None,
        source: str = "manual_ui",
        confirmed_at: str | datetime | None = None,
    ) -> int:
        if not confirmed:
            raise InputError("Interview recording requires explicit confirmation.")
        kind = interview_type.strip().casefold()
        if kind not in INTERVIEW_TYPES:
            raise InputError("Choose a supported interview type.")
        scheduled = _normalize_timestamp(scheduled_at) if scheduled_at is not None else None
        completed = _normalize_timestamp(completed_at) if completed_at is not None else None
        contact = _clean_user_text(
            contact_label, label="Interviewer/contact label", maximum=300
        )
        safe_result = _clean_user_text(result, label="Interview result", maximum=500)
        safe_notes = _clean_user_text(notes, label="Interview notes", maximum=2_000)
        timestamp = _normalize_timestamp(confirmed_at)
        source_value = _normalize_source(source)
        with self._connection() as connection:
            self._require_application(connection, application_id)
            cursor = connection.execute(
                """
                INSERT INTO interviews(
                    application_id, interview_type, scheduled_at, completed_at,
                    contact_label, result, notes, confirmed_at, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    application_id,
                    kind,
                    scheduled,
                    completed,
                    contact,
                    safe_result,
                    safe_notes,
                    timestamp,
                    source_value,
                ),
            )
            self._append_status(
                connection,
                application_id,
                kind,
                source=source_value,
                timestamp=timestamp,
                note=None,
            )
            return int(cursor.lastrowid)

    def record_resume_version(
        self,
        application_id: int,
        *,
        run_identifier: str,
        artifact_reference: str,
        writer_provider: str,
        qa_outcome: str,
        match_score: float | None = None,
        created_at: str | datetime | None = None,
    ) -> int:
        run_id = _clean_text(
            run_identifier,
            label="Résumé run identifier",
            maximum=200,
            required=True,
        ) or ""
        if not _SAFE_ID_RE.fullmatch(run_id):
            raise InputError("Résumé run identifier contains unsafe characters.")
        reference = _clean_text(
            artifact_reference,
            label="Résumé artifact reference",
            maximum=255,
            required=True,
        ) or ""
        reference_path = Path(reference)
        if reference_path.is_absolute() or reference_path.name != reference or reference in {".", ".."}:
            raise InputError("Résumé artifact reference must be a safe filename.")
        provider = _clean_text(
            writer_provider,
            label="Writer provider",
            maximum=100,
            required=True,
        ) or ""
        outcome = _clean_text(
            qa_outcome,
            label="QA outcome",
            maximum=100,
            required=True,
        ) or ""
        if match_score is not None and (
            isinstance(match_score, bool)
            or not isinstance(match_score, (int, float))
            or not 0 <= float(match_score) <= 100
        ):
            raise InputError("Validated match score must be between 0 and 100.")
        timestamp = _normalize_timestamp(created_at)
        with self._connection() as connection:
            self._require_application(connection, application_id)
            connection.execute(
                """
                INSERT OR IGNORE INTO resume_versions(
                    application_id, run_identifier, artifact_reference,
                    created_at, writer_provider, qa_outcome, match_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    application_id,
                    run_id,
                    reference,
                    timestamp,
                    provider,
                    outcome,
                    float(match_score) if match_score is not None else None,
                ),
            )
            row = connection.execute(
                """
                SELECT id FROM resume_versions
                WHERE application_id = ? AND run_identifier = ? AND artifact_reference = ?
                """,
                (application_id, run_id, reference),
            ).fetchone()
            assert row is not None
            return int(row["id"])

    def application_history(self, application_id: int) -> list[dict[str, Any]]:
        with self._connection() as connection:
            self._require_application(connection, application_id)
            rows = connection.execute(
                """
                SELECT id, previous_status, new_status, timestamp, source, note,
                       event_kind, correction_of_event_id
                FROM application_status_events
                WHERE application_id = ? ORDER BY timestamp, id
                """,
                (application_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def summary(
        self,
        *,
        now: str | datetime | None = None,
        recent_limit: int = 10,
    ) -> dict[str, Any]:
        current = datetime.fromisoformat(_normalize_timestamp(now))
        week_start = (current - timedelta(days=current.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).isoformat(timespec="seconds")
        update_cutoff = (current - timedelta(days=APPLICATION_UPDATE_DAYS)).isoformat(
            timespec="seconds"
        )
        with self._connection() as connection:
            total_jobs = self._scalar(connection, "SELECT COUNT(*) FROM jobs")
            viewed_week = self._scalar(
                connection,
                """
                SELECT COUNT(DISTINCT job_id) FROM job_events
                WHERE event_type = 'viewed' AND timestamp >= ?
                """,
                (week_start,),
            )
            approved = self._scalar(
                connection,
                """
                SELECT COUNT(DISTINCT job_id) FROM job_events
                WHERE event_type = 'approved_for_tailoring'
                """,
            )
            applications = self._count_applications_with_statuses(
                connection, _SUBMITTED_STATUSES
            )
            applications_week = self._scalar(
                connection,
                f"""
                SELECT COUNT(*) FROM (
                    SELECT event.application_id, MIN(event.timestamp) AS submitted_at
                    FROM application_status_events event
                    WHERE event.new_status IN ({self._placeholders(_SUBMITTED_STATUSES)})
                      AND NOT EXISTS (
                          SELECT 1 FROM application_status_events correction
                          WHERE correction.correction_of_event_id = event.id
                      )
                    GROUP BY event.application_id
                    HAVING submitted_at >= ?
                )
                """,
                (*sorted(_SUBMITTED_STATUSES), week_start),
            )
            active_interviews = self._scalar(
                connection,
                f"SELECT COUNT(*) FROM applications WHERE current_status IN ({self._placeholders(_ACTIVE_INTERVIEW_STATUSES)})",
                tuple(sorted(_ACTIVE_INTERVIEW_STATUSES)),
            )
            offers = self._count_applications_with_statuses(connection, _OFFER_STATUSES)
            screenings = self._count_applications_with_statuses(
                connection, _SCREENING_STATUSES
            )
            interviews = self._count_applications_with_statuses(
                connection, _INTERVIEW_STATUSES
            )
            interview_offers = self._count_applications_with_both_status_groups(
                connection,
                _INTERVIEW_STATUSES,
                _OFFER_STATUSES,
            )
            resume_versions = self._scalar(
                connection, "SELECT COUNT(*) FROM resume_versions"
            )
            top_skills = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT s.canonical_name AS name, COUNT(DISTINCT js.job_id) AS jobs
                    FROM job_skills js JOIN skills s ON s.id = js.skill_id
                    GROUP BY s.id, s.canonical_name
                    ORDER BY jobs DESC, s.canonical_name COLLATE NOCASE
                    LIMIT 10
                    """
                )
            ]
            top_missing = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT s.canonical_name AS name, COUNT(DISTINCT js.job_id) AS jobs
                    FROM job_skills js JOIN skills s ON s.id = js.skill_id
                    WHERE js.gap_status = 'missing'
                    GROUP BY s.id, s.canonical_name
                    ORDER BY jobs DESC, s.canonical_name COLLATE NOCASE
                    LIMIT 10
                    """
                )
            ]
            by_title = self._grouped_jobs(connection, "title_family")
            by_seniority = self._grouped_jobs(connection, "seniority")
            by_workplace = self._grouped_jobs(connection, "workplace_type")
            applicant_distribution = self._applicant_distribution(connection)
            recently_viewed = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT j.id, j.title, j.company, j.location, j.workplace_type,
                           j.last_seen_at, j.canonical_url, a.id AS application_id,
                           a.current_status
                    FROM jobs j JOIN applications a ON a.job_id = j.id
                    ORDER BY j.last_seen_at DESC, j.id DESC LIMIT ?
                    """,
                    (max(1, min(recent_limit, 100)),),
                )
            ]
            requiring_update = [
                dict(row)
                for row in connection.execute(
                    f"""
                    SELECT a.id AS application_id, a.current_status, a.updated_at,
                           j.id AS job_id, j.title, j.company
                    FROM applications a JOIN jobs j ON j.id = a.job_id
                    WHERE a.current_status IN ({self._placeholders(_UPDATE_REQUIRED_STATUSES)})
                      AND a.updated_at <= ?
                    ORDER BY a.updated_at, a.id
                    """,
                    (*sorted(_UPDATE_REQUIRED_STATUSES), update_cutoff),
                )
            ]
        return {
            "generated_at": current.isoformat(timespec="seconds"),
            "week_start_utc": week_start,
            "minimum_rate_sample_size": MINIMUM_RATE_SAMPLE_SIZE,
            "totals": {
                "unique_jobs_viewed": total_jobs,
                "jobs_viewed_this_week": viewed_week,
                "jobs_approved_for_tailoring": approved,
                "applications_submitted": applications,
                "applications_this_week": applications_week,
                "active_interviews": active_interviews,
                "offers": offers,
                "resume_versions": resume_versions,
            },
            "rates": {
                "application_to_screening": self._rate(screenings, applications),
                "application_to_interview": self._rate(interviews, applications),
                "interview_to_offer": self._rate(interview_offers, interviews),
            },
            "top_requested_skills": top_skills,
            "top_missing_skills": top_missing,
            "roles_by_title_family": by_title,
            "roles_by_seniority": by_seniority,
            "roles_by_workplace_type": by_workplace,
            "applicant_count_distribution": applicant_distribution,
            "recently_viewed_jobs": recently_viewed,
            "applications_requiring_update": requiring_update,
        }

    def sanitized_export(self, *, now: str | datetime | None = None) -> dict[str, Any]:
        generated_at = _normalize_timestamp(now)
        with self._connection() as connection:
            jobs = [
                {
                    "job_id": f"job-{row['id']}",
                    "title": row["title"],
                    "company": row["company"],
                    "location": row["location"],
                    "workplace_type": row["workplace_type"],
                    "employment_type": row["employment_type"],
                    "seniority": row["seniority"],
                    "compensation": {
                        "text": row["compensation_text"],
                        "minimum": row["compensation_min"],
                        "maximum": row["compensation_max"],
                        "currency": row["compensation_currency"],
                    },
                    "posting_date": row["posting_date"],
                    "source": row["source"],
                    "first_seen_at": row["first_seen_at"],
                    "last_seen_at": row["last_seen_at"],
                }
                for row in connection.execute("SELECT * FROM jobs ORDER BY id")
            ]
            snapshots = [
                {
                    "job_id": f"job-{row['job_id']}",
                    "captured_at": row["captured_at"],
                    "applicant_count": row["applicant_count"],
                    "posting_date": row["posting_date"],
                    "workplace_type": row["workplace_type"],
                    "employment_type": row["employment_type"],
                    "seniority": row["seniority"],
                }
                for row in connection.execute(
                    "SELECT * FROM job_snapshots ORDER BY job_id, captured_at, id"
                )
            ]
            skills = [
                {
                    "job_id": f"job-{row['job_id']}",
                    "canonical_name": row["canonical_name"],
                    "requirement_level": row["requirement_level"],
                    "requirement_category": row["requirement_category"],
                    "gap_status": row["gap_status"],
                }
                for row in connection.execute(
                    """
                    SELECT js.job_id, s.canonical_name, js.requirement_level,
                           js.requirement_category, js.gap_status
                    FROM job_skills js JOIN skills s ON s.id = js.skill_id
                    ORDER BY js.job_id, js.id
                    """
                )
            ]
            status_events = [
                {
                    "event_id": f"event-{row['id']}",
                    "application_id": f"application-{row['application_id']}",
                    "job_id": f"job-{row['job_id']}",
                    "previous_status": row["previous_status"],
                    "new_status": row["new_status"],
                    "timestamp": row["timestamp"],
                    "source": row["source"],
                    "event_kind": row["event_kind"],
                    "correction_of_event_id": (
                        f"event-{row['correction_of_event_id']}"
                        if row["correction_of_event_id"] is not None
                        else None
                    ),
                }
                for row in connection.execute(
                    """
                    SELECT event.*, application.job_id
                    FROM application_status_events event
                    JOIN applications application ON application.id = event.application_id
                    ORDER BY event.application_id, event.timestamp, event.id
                    """
                )
            ]
            resume_versions = [
                {
                    "resume_version_id": f"resume-version-{row['id']}",
                    "application_id": f"application-{row['application_id']}",
                    "job_id": f"job-{row['job_id']}",
                    "run_identifier": row["run_identifier"],
                    "created_at": row["created_at"],
                    "writer_provider": row["writer_provider"],
                    "qa_outcome": row["qa_outcome"],
                    "match_score": row["match_score"],
                }
                for row in connection.execute(
                    """
                    SELECT version.*, application.job_id
                    FROM resume_versions version
                    JOIN applications application ON application.id = version.application_id
                    ORDER BY version.application_id, version.created_at, version.id
                    """
                )
            ]
        statistics = self.summary(now=generated_at)
        aggregate_statistics = {
            key: value
            for key, value in statistics.items()
            if key
            not in {
                "recently_viewed_jobs",
                "applications_requiring_update",
            }
        }
        return {
            "contract": "resume-tailor-sanitized-job-analytics",
            "contract_version": SANITIZED_EXPORT_CONTRACT_VERSION,
            "generated_at": generated_at,
            "local_only": True,
            "jobs": jobs,
            "applicant_snapshots": snapshots,
            "normalized_skills": skills,
            "application_status_events": status_events,
            "resume_versions": resume_versions,
            "aggregate_statistics": aggregate_statistics,
            "excluded": [
                "resume_contents",
                "contact_information",
                "prompts",
                "credentials",
                "raw_actor_output",
                "diagnostics",
                "user_notes",
                "interviewer_contacts",
            ],
        }

    def write_sanitized_export(
        self,
        destination: Path,
        *,
        now: str | datetime | None = None,
    ) -> Path:
        path = destination.expanduser().resolve()
        if path.suffix.casefold() != ".json":
            raise InputError("Sanitized analytics exports must use a .json filename.")
        if _inside_git_repository(path):
            raise InputError("Sanitized analytics exports must be outside a Git repository.")
        if not path.parent.is_dir():
            raise InputError("Sanitized analytics export parent directory does not exist.")
        payload = (
            json.dumps(
                self.sanitized_export(now=now),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.tmp-",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o600)
            temporary.replace(path)
        except Exception:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return path

    @staticmethod
    def _status(value: str) -> str:
        status = value.strip().casefold()
        if status not in APPLICATION_STATUSES:
            raise InputError("Choose a supported application status.")
        return status

    @staticmethod
    def _require_job(connection: sqlite3.Connection, job_id: int) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise InputError("The selected analytics job does not exist.")
        return row

    @staticmethod
    def _require_application(
        connection: sqlite3.Connection, application_id: int
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM applications WHERE id = ?", (application_id,)
        ).fetchone()
        if row is None:
            raise InputError("The selected application does not exist.")
        return row

    @staticmethod
    def _application_id(connection: sqlite3.Connection, job_id: int) -> int:
        row = connection.execute(
            "SELECT id FROM applications WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise InputError("The selected job has no application tracking record.")
        return int(row["id"])

    def _append_status(
        self,
        connection: sqlite3.Connection,
        application_id: int,
        new_status: str,
        *,
        source: str,
        timestamp: str,
        note: str | None,
    ) -> int:
        application = self._require_application(connection, application_id)
        previous = str(application["current_status"])
        if previous == new_status:
            latest = connection.execute(
                """
                SELECT id FROM application_status_events
                WHERE application_id = ? ORDER BY timestamp DESC, id DESC LIMIT 1
                """,
                (application_id,),
            ).fetchone()
            return int(latest["id"]) if latest is not None else 0
        cursor = connection.execute(
            """
            INSERT INTO application_status_events(
                application_id, previous_status, new_status, timestamp,
                source, note, event_kind
            ) VALUES (?, ?, ?, ?, ?, ?, 'status_change')
            """,
            (application_id, previous, new_status, timestamp, source, note),
        )
        connection.execute(
            "UPDATE applications SET current_status = ?, updated_at = ? WHERE id = ?",
            (new_status, timestamp, application_id),
        )
        return int(cursor.lastrowid)

    @staticmethod
    def _scalar(
        connection: sqlite3.Connection,
        query: str,
        parameters: Sequence[Any] = (),
    ) -> int:
        row = connection.execute(query, parameters).fetchone()
        return int(row[0]) if row is not None else 0

    @staticmethod
    def _placeholders(values: Sequence[str] | frozenset[str]) -> str:
        return ", ".join("?" for _ in values)

    def _count_applications_with_statuses(
        self,
        connection: sqlite3.Connection,
        statuses: frozenset[str],
    ) -> int:
        return self._scalar(
            connection,
            f"""
            SELECT COUNT(DISTINCT application_id)
            FROM application_status_events event
            WHERE event.new_status IN ({self._placeholders(statuses)})
              AND NOT EXISTS (
                  SELECT 1 FROM application_status_events correction
                  WHERE correction.correction_of_event_id = event.id
              )
            """,
            tuple(sorted(statuses)),
        )

    def _count_applications_with_both_status_groups(
        self,
        connection: sqlite3.Connection,
        first: frozenset[str],
        second: frozenset[str],
    ) -> int:
        return self._scalar(
            connection,
            f"""
            SELECT COUNT(*) FROM applications application
            WHERE EXISTS (
                SELECT 1 FROM application_status_events first_event
                WHERE first_event.application_id = application.id
                  AND first_event.new_status IN ({self._placeholders(first)})
                  AND NOT EXISTS (
                      SELECT 1 FROM application_status_events correction
                      WHERE correction.correction_of_event_id = first_event.id
                  )
            ) AND EXISTS (
                SELECT 1 FROM application_status_events second_event
                WHERE second_event.application_id = application.id
                  AND second_event.new_status IN ({self._placeholders(second)})
                  AND NOT EXISTS (
                      SELECT 1 FROM application_status_events correction
                      WHERE correction.correction_of_event_id = second_event.id
                  )
            )
            """,
            (*sorted(first), *sorted(second)),
        )

    @staticmethod
    def _rate(numerator: int, denominator: int) -> dict[str, Any]:
        enough = denominator >= MINIMUM_RATE_SAMPLE_SIZE
        percentage = round((numerator / denominator) * 100, 1) if enough else None
        return {
            "numerator": numerator,
            "denominator": denominator,
            "percentage": percentage,
            "enough_data": enough,
            "display": f"{percentage:.1f}%" if enough else "Not enough data",
        }

    @staticmethod
    def _grouped_jobs(
        connection: sqlite3.Connection, field: str
    ) -> list[dict[str, Any]]:
        if field not in {"title_family", "seniority", "workplace_type"}:
            raise AnalyticsError("Unsupported analytics grouping field.")
        rows = connection.execute(
            f"""
            SELECT COALESCE(NULLIF({field}, ''), 'Unspecified') AS label,
                   COUNT(*) AS jobs
            FROM jobs GROUP BY label ORDER BY jobs DESC, label COLLATE NOCASE
            """
        ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _applicant_distribution(
        connection: sqlite3.Connection,
    ) -> list[dict[str, Any]]:
        counts = {label: 0 for label in ("Missing", "0–24", "25–49", "50–99", "100–199", "200+")}
        rows = connection.execute(
            """
            SELECT s.applicant_count
            FROM job_snapshots s
            WHERE s.id = (
                SELECT latest.id FROM job_snapshots latest
                WHERE latest.job_id = s.job_id
                ORDER BY latest.captured_at DESC, latest.id DESC LIMIT 1
            )
            """
        ).fetchall()
        for row in rows:
            value = row["applicant_count"]
            if value is None:
                label = "Missing"
            elif value < 25:
                label = "0–24"
            elif value < 50:
                label = "25–49"
            elif value < 100:
                label = "50–99"
            elif value < 200:
                label = "100–199"
            else:
                label = "200+"
            counts[label] += 1
        return [{"label": label, "jobs": count} for label, count in counts.items()]
