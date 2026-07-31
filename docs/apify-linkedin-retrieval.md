# Apify LinkedIn job-detail retrieval

## Boundary

Apify is a retrieval provider for workflow step 2. It is not a résumé authority,
analysis model, or tailoring model.

```text
exact LinkedIn URL
  -> Apify job-detail Actor
  -> local allowlist mapper
  -> canonical linkedin_job schema
  -> human posting approval
  -> Codex evidence analysis
  -> approved edit catalog
  -> Antigravity tailoring
```

Antigravity does not parse Apify output. Step 6 continues to receive only the
authoritative résumé content and locally authenticated approved edit catalog.

## Provider selection

- `auto`: Apify when `APIFY_API_TOKEN` or the private fixed token file is
  configured before the run; otherwise Antigravity.
- `apify`: require Apify configuration and fail before provider launch if it is
  absent.
- `antigravity`: use the existing passive read-only URL adapter.

There is no automatic provider fallback after launch.

## Actor contract

The default Actor is `piotrv1001/linkedin-job-details-scraper`. Its documented
input uses a required `searchUrls` array of LinkedIn job-detail URLs. Resume
Tailor supplies one already validated URL and accepts one matching result.

Actor data is mapped through a bounded field alias allowlist. Title, company,
location, description, employment type, workplace type, salary, and explicitly
provided string arrays may be copied. Missing skill categories are left empty;
Python does not infer them and no model fills them in. The complete normalized
description remains available to the deterministic job-requirement catalog.

## Authentication and diagnostics

The API token is sent only as an `Authorization: Bearer` header. It is never a
URL query parameter, command-line argument, run artifact, exception message, or
UI value.

The desktop-safe token file is fixed at
`~/.config/resume-tailor/apify-token`. It must be a regular non-symlink file,
owned by the current user, with mode `0600`.

`apify-job-response.json` stores only content-free provenance:

- Actor, run, build, and dataset identifiers;
- terminal status and dataset item count;
- selected item index, field names, and field types;
- canonical provider-output byte count and SHA-256;
- local validation result.

The raw provider response is not retained. `job-source.json` is the canonical
posting artifact that the user reviews and approves.

## Cancellation and failure

The integration starts the Actor asynchronously and polls with bounded HTTPS
requests so the existing cancellation event can be checked between calls. A
cancelled or timed-out run triggers a best-effort Actor abort. Provider errors
never expose response bodies.

The adapter fails closed on missing configuration, unsafe Actor identifiers,
API or JSON errors, failed Actor status, missing datasets, zero/multiple/
mismatched jobs, malformed fields, unsafe controls, insufficient descriptions,
job-ID drift, or canonical schema failure.
