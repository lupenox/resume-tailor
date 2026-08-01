# Apify LinkedIn job retrieval contract

## Responsibility boundary

Workflow Step 2 has one web-retrieval provider: Apify. It receives one validated
public LinkedIn job URL, runs the configured job-detail Actor, and returns the
Actor's default-dataset records. Local Python—not Apify—selects and normalizes
the matching posting.

Apify never receives a résumé, résumé hash, extracted résumé, Codex analysis,
approved edit catalog, tailored content, preview, or final-QA input. It performs
no résumé analysis, tailoring, scoring, generation, or application action.
Pasted text, clipboard input, and UTF-8 job files remain local input modes and do
not call Apify.

```text
validated LinkedIn URL
  -> configured Apify job-detail Actor
  -> default dataset
  -> local identity match and allowlisted normalization
  -> canonical linkedin_job.schema.json
  -> explicit posting approval
  -> later analysis and writing stages
```

## Configuration

Configure the existing process environment before starting the CLI or UI:

```bash
export APIFY_API_TOKEN='apify_api_REPLACE_WITH_THE_COMPLETE_TOKEN'
export APIFY_ACTOR_ID='username/actor-name'
```

`APIFY_API_TOKEN` is required and is preserved exactly. Keep the complete
`apify_api_` prefix and do not surround the value with whitespace.
`APIFY_ACTOR_ID` is also required; obtain the exact value from the Actor page in
Apify Console. Resume Tailor accepts either an Apify Actor ID or the normal
`username/actor-name` form and converts the latter to the documented
`username~actor-name` API path form.

`.env.example` contains placeholders only. Populated `.env*`, `*.token`,
`apify-token`, and `.apify-token` files are ignored by Git. Resume Tailor reads
the inherited process environment; it does not parse, print, copy, or overwrite
local secret files.

Start the web interface from the configured shell:

```bash
tailor-resume-ui
```

## Actor input and API flow

Repository history documents the tested job-detail Actor contract as one
`searchUrls` entry. Input creation is isolated in
`build_apify_actor_input()` and sends exactly:

```json
{
  "searchUrls": [
    "https://www.linkedin.com/jobs/view/example-role-1234567890/"
  ]
}
```

The adapter uses the official Apify API v2 over HTTPS and sends the token only
in `Authorization: Bearer ...`. It never puts the token in the URL. It starts an
asynchronous Actor run, polls documented run states within the pipeline timeout,
best-effort aborts a still-active run after cancellation or timeout, reads at
most 20 default-dataset items, and makes no automatic provider fallback.

Official API references:

- [Run Actor](https://docs.apify.com/api/v2/actors-runs-post)
- [Get run](https://docs.apify.com/api/v2/actor-run-get)
- [Get dataset items](https://docs.apify.com/api/v2/dataset-items-get)
- [Apify API authentication](https://docs.apify.com/api/v2)

## Local validation and normalization

Before any API call, local Python requires HTTPS, an allowlisted LinkedIn host,
a supported `/jobs/view/...` path, no embedded credentials or unsafe port, and
one stable 5–20 digit job ID from the path or `currentJobId`. Conflicting IDs are
rejected.

After the Actor succeeds, local code requires exactly one record matching the
requested URL or job ID. A bounded alias map recognizes common Actor fields for:

- title, company, location, workplace type, employment type, and seniority;
- full description, responsibilities, required/preferred qualifications, and
  skills;
- salary/compensation, posting URL, LinkedIn job ID, date posted, and applicant
  count.

HTML descriptions are converted to text while preserving paragraph, heading,
and list boundaries. Script, style, noscript, and SVG content is discarded.
Missing optional values remain `null`, `unspecified`, or empty arrays. Local code
does not infer or fabricate facts. A meaningful title, company, and substantive
description are required by the current downstream run-identity contract.

Only the canonical result reaches posting review and later job-requirement
catalog construction. Raw Actor keys and JSON never reach Codex, Qwen,
Antigravity, or the normal UI.

## Approval boundary and progress

The UI reports these bounded Step 2 states:

1. Validating the LinkedIn URL
2. Starting Apify retrieval
3. Waiting for the Actor
4. Reading the matching job result
5. Normalizing the job posting
6. Ready for review

Success is not reported until identity checks, canonical schema validation, safe
text checks, and substantive-content validation all pass. Resume extraction and
Codex analysis do not begin until the existing explicit LinkedIn-posting
approval gate is approved. The UI displays normalized fields, not raw JSON.

## Errors and diagnostics

User-facing failures distinguish missing token, missing or invalid Actor ID,
authentication failure, Actor not found, timeout, failed run, empty dataset, no
matching result, malformed output, insufficient content, network failure, rate
limit, and other provider failure. Every case stops before résumé analysis and
offers pasted text or a UTF-8 job file as the bounded fallback.

`apify-linkedin-retrieval-diagnostic.json` is sanitized and content-free. It may
record the Actor/run/build/dataset identifiers, HTTP status, sanitized provider
message, run state, dataset item count, selected index, recognized key names,
byte count/hash, classification, phase, and validation result. It omits:

- API tokens and authorization headers;
- signed URLs or credential query parameters;
- raw Actor records and job descriptions;
- unrelated environment variables;
- résumé, prompt, analysis, and writer content.

`job-source.json` is the locally validated canonical posting and
`job-description.txt` is its normalized full description. They are written only
after successful normalization.

## Offline and live verification

Run the focused offline Step 2 tests:

```bash
.venv/bin/python -m pytest -q \
  tests/test_linkedin_job.py \
  tests/test_pipeline.py \
  tests/test_ui.py
```

Run the complete offline suite:

```bash
.venv/bin/python -m pytest -q
```

The tests use synthetic fixtures and injected fake HTTP/run clients. They do not
contact Apify, LinkedIn, Codex, Ollama, Antigravity, or any network service and
do not consume Apify credits.

A live URL-mode run starts the configured Actor and may consume Apify credits
according to that Actor's current pricing and runtime. Before any live check,
show the exact synthetic LinkedIn URL, exact non-secret Actor ID, Actor pricing
or expected credit basis, and the precise CLI/UI trigger; then obtain explicit
approval. Never display the API token while requesting approval.
