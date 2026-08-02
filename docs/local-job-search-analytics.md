# Local job-search analytics

Phase 1 provides a deterministic, private source of truth for job-search
activity. It uses Python's standard `sqlite3` module and does not contact a
provider, expose a new port, send telemetry, or transmit exports.

## Storage and schema lifecycle

The default database is:

```text
${XDG_DATA_HOME:-~/.local/share}/resume-tailor/data/job-search-analytics.sqlite3
```

`RESUME_TAILOR_ANALYTICS_DB` or the CLI/UI `--analytics-db` option may select an
absolute private path. This override is primarily useful for tests and isolated
local installations. The parent directory and database are restricted to the
current user where the filesystem permits it. SQLite WAL, shared-memory,
journal, database, and sanitized-export filenames are ignored by Git.

Schema version `1` is recorded in both `schema_migrations` and SQLite's
`user_version`. Initialization is idempotent. Migrations move only forward and
are applied under SQLite locking; a database from a newer application version
is rejected rather than downgraded.

The tables are:

- `jobs`: stable job identity and current metadata, including first/last seen.
- `job_snapshots`: materially distinct observations with nullable applicant
  counts and description hashes.
- `skills`: source-controlled alias-normalized names.
- `job_skills`: original validated requirement wording, required/preferred
  level, category, evidence reference, and validated gap status.
- `job_events`: viewed and approved-for-tailoring events.
- `applications`: the current tracking status for one job.
- `application_status_events`: append-only status changes and corrections.
- `application_notes`: append-only, non-sensitive user notes.
- `resume_versions`: safe artifact/run references without résumé content.
- `interviews`: manually confirmed interview records only.

Unique indexes prefer LinkedIn job ID, then a query-free canonical LinkedIn URL.
Local inputs use a company/title/description-hash identity. Tracking query
parameters and fragments never participate in URL identity. Snapshot material
hashes prevent duplicate observations when no tracked field changed.

## Pipeline boundaries

A URL posting is recorded as viewed only after Apify retrieval and canonical
validation succeed and the existing posting-review presentation exists. The
recording callback runs before the user responds, so rejecting or cancelling
the posting does not erase the fact that it was viewed. A malformed or failed
retrieval never reaches this callback.

Pasted, clipboard, and file input is recorded after the existing local input
validation and review/input surface succeeds. The source classification remains
distinct while all modes share the same bounded storage contract.

Skills are copied only from the already validated deterministic requirement
catalog. URL-mode catalog creation remains after the posting approval boundary.
The fixed alias dictionary groups only explicit equivalents such as
Postgres/PostgreSQL and CI/CD/continuous integration; no model can change that
dictionary at runtime. Gap status is added only from the locally resolved,
validated requirement assessment after the existing Codex-analysis approval.

Approval for tailoring is a separate `job_events` fact and moves the tracking
record to `planned`. It is not an application submission. A résumé version is
associated only after final QA passes and final artifacts are published, and it
still does not change the application to `applied`.

Each analytics write is an isolated transaction. A database error adds a
sanitized `analytics.warnings` entry to `run-metadata.json`, identifies the
failed operation as retryable from preserved artifacts, and allows the résumé
pipeline to continue. Exception details, prompts, credentials, and résumé text
are not copied into that warning.

## Status and interview policy

Supported statuses are:

```text
viewed, saved, planned, applied, screening, interview,
technical_interview, final_interview, rejected, withdrawn,
offer, accepted, declined
```

Status events and notes have database triggers that reject update or deletion.
The current application status can change only when the newest appended event
matches it. A correction appends an explicit correction event and points to the
event being corrected; history is never rewritten. Corrected erroneous events
are excluded from conversion statistics.

An interview row requires an explicit `confirmed=True` call or the corresponding
checked UI confirmation. Notes or job text containing the word “interview” do
not create interview records.

## Deterministic statistics

The `/analytics` page calculates every number from SQLite:

- unique jobs viewed and jobs viewed in the current UTC week;
- jobs approved for tailoring and generated résumé versions;
- submitted applications and applications submitted in the UTC week;
- active interviews and offers;
- application-to-screening, application-to-interview, and interview-to-offer
  rates with numerator and denominator;
- top requested and validated missing skills;
- roles by deterministic title family, seniority, and workplace type;
- latest applicant-count distribution, retaining missing as `NULL`;
- recently viewed jobs and active applications not updated for seven days.

A percentage is shown only when its denominator is at least five. Smaller or
zero samples display `Not enough data` while still showing the exact numerator
and denominator.

## Privacy and sanitized export boundary

The database never receives résumé body text, home address, phone number, email
address, model prompts, credentials, authorization headers, raw Actor output, or
private diagnostic bundles. Résumé associations contain only a stable run ID,
safe filename, creation time, writer provider, QA outcome, and an optional
already-validated match score. User-entered notes and contact labels reject
obvious addresses, emails, phone numbers, and credentials.

`AnalyticsStore.sanitized_export()` builds a local in-memory versioned contract;
`write_sanitized_export()` can write it to a user-selected JSON file. Phase 1
does not expose an export UI, transmit the object, or call an external agent.
The contract includes anonymous local IDs, structured job metadata, applicant
snapshots, normalized skills, status events, aggregate statistics, and
non-sensitive résumé-version identifiers. It excludes original requirement
excerpts, description hashes, user notes, interviewer labels, résumé data,
credentials, prompts, raw provider output, and diagnostics.
