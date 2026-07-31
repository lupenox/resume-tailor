# resume-tailor

`resume-tailor` is a local Linux CLI and polished localhost web application that
creates a truthful, job-tailored resume from a structured master DOCX. Both
interfaces share one Python pipeline: OpenAI Codex CLI performs evidence
analysis, Google Antigravity CLI produces schema-constrained wording,
deterministic local code preserves DOCX formatting, LibreOffice and Poppler
validate a one-page PDF, and Codex performs a final read-only visual/content
review.

The application never asks either model to edit the document. Models return
structured content; Python validates it and inserts approved text into a copy of
the master. The source DOCX is hashed before and after every run and is never an
output target.

The web UI is optional. `tailor-resume` remains the complete terminal interface;
`tailor-resume-ui` starts the browser interface on `127.0.0.1`.

## Pipeline architecture

1. Read a job posting from the Linux clipboard or a UTF-8 text file, or retrieve
   one public LinkedIn job-detail URL through Apify (preferred when configured)
   or the explicit Antigravity fallback.
2. For URL mode, validate HTTPS/hostname/path/redirect/job-ID consistency, save
   canonical `job-source.json`, display the locally derived posting identity,
   and require approval. Apify output is allowlist-mapped and schema-validated
   by Python; it is never sent to Antigravity for reparsing.
3. Validate and structurally extract the master DOCX with `python-docx`.
4. Send the extracted resume and untrusted posting to Codex in an ephemeral,
   read-only session with a strict analysis schema.
5. Print the analysis. Stop on unanswered factual questions; otherwise require
   explicit approval.
6. Run a local completeness and hash-authentication preflight, then send the
   original content, immutable résumé source catalog, and approved edit catalog
   to Antigravity over UTF-8 stdin in sandboxed print mode with a strict
   tailored-content schema. Tailoring is an execution task, not generic plan
   mode. Prompt text is never placed in argv or environment variables.
7. Select exactly one documented Antigravity print-mode response candidate,
   validate it strictly against the tailoring schema, and then run immutable-fact,
   approved-edit, technology, metrics, seniority, availability, structure, and
   content-budget checks. Prose, Markdown fences, fragments, and ambiguous
   candidates fail closed.
8. Write and display a section-by-section before/after diff. Refuse questionable
   claims; otherwise require explicit approval.
9. Copy the master, replace only mapped text runs, and verify that paragraph/run
   formatting, hyperlinks, contact information, section geometry, and list
   structure remain unchanged.
10. Export one PDF page through a unique LibreOffice profile, validate text and
   bounding boxes with Poppler, and render `preview.png`.
11. Give Codex the preview and complete evidence bundle for a fresh read-only QA.
    A material QA finding returns a nonzero status and preserves every artifact.

Codex uses two schema layers. The full Draft 2020-12 schemas
`codex_analysis.schema.json` and `final_qa.schema.json` remain canonical for
local validation and retain constraints such as `uniqueItems` and nonempty
strings. The checked-in `codex_analysis.openai.schema.json` and
`final_qa.openai.schema.json` files are compatibility sentinels derived for
OpenAI's Structured Outputs subset. After the posting is confirmed and the résumé
is extracted, local code builds two immutable catalogs: exact résumé source blocks
and exact job requirements with stable IDs and categories. Analysis receives a
fresh run-specific transport schema whose requirement, evidence, and edit-target
enums contain only IDs from those catalogs. Required evidence arrays cannot be
empty. The generated schema is size-bounded, recursively preflighted, hashed into
run metadata, and revalidated immediately before `codex exec`. Final QA uses the
checked-in transport schema. Unsupported requirements, incomplete or conflicting
classifications, object-shape drift, empty or unknown IDs, and schema/catalog
mismatches stop locally without a live request or unconstrained fallback.

Codex classifies every catalog requirement into exactly one of two collections:
a supported mapping with one or more résumé source IDs, or an unsupported ID with
no evidence. Human-facing requirement text and cited résumé text are resolved
locally. A conservative case/dash/whitespace match may be labeled `present_verbatim`;
otherwise a supported mapping is explicitly labeled as a model-assessed semantic
match requiring human review. Local code never claims to prove semantic equivalence.

Exact duplicate values in canonically unique arrays are removed in first-seen
order before canonical validation. When this occurs, the normalized Codex
artifact is saved and a `*-normalization-warnings.json` sidecar records every
affected field. All remaining Draft 2020-12 constraints are enforced locally;
the transport transformation does not weaken factual-integrity validation.

The current template validator expects the inspected design: one section, 32
paragraphs, no tables, US Letter geometry, three skill groups, three projects
with 3/4/3 bullets, one open-source entry, one experience entry, six hyperlinks,
real `List Bullet` styles, and the established direct-run formatting patterns.
This is a deliberate safety boundary. Significant template drift stops the run
instead of risking edits to the wrong paragraphs.

### Local web architecture

The UI uses FastAPI, server-rendered Jinja2 templates, local CSS, and a small
amount of dependency-free JavaScript. There is no Node server, Electron runtime,
external CDN, remote font, tracking, or analytics. A background worker calls the
same `run_pipeline` function as the CLI through reusable progress, approval, and
cancellation hooks.

The workflow page polls a small structured status endpoint. It displays concise
stage messages, never hidden reasoning, raw prompts, environment variables,
provider transcripts, or credentials. During Codex analysis it reports elapsed time and periodic
process-liveness heartbeats, distinguishes “still running” from “no process
detected” during local validation, and never invents an ETA. Cancellation and bounded
timeouts terminate the full subprocess group. Only one run can be active. A
cancel request reaches the shared
subprocess runner, terminates the complete process group, and preserves useful
run artifacts.

Provider failures use a concise public message. Sanitized, length-limited
technical details are available in a collapsed disclosure instead of exposing a
raw prompt or provider response by default.

After the Codex analysis gate, the pipeline writes a hash-only approval record
binding the source résumé, confirmed job text, requirement catalog, generated
transport schema, resolved analysis, company, and role. An Antigravity-only
recovery is offered for authenticated post-approval launch, tailoring-contract,
response-envelope, bounded `cannot_apply`, and technical failures when that
record and every bound artifact still match. Recovery creates a new isolated
run, skips LinkedIn and Codex, and pauses at the normal validated-content-diff
approval gate. When a preserved response itself is one complete schema-valid
result, a separate authenticated offline reprocessing path skips all providers;
it is never offered for prose, fragments, or ambiguous output.

The dashboard includes the protected bundled master or a validated DOCX upload;
LinkedIn URL, pasted-text, and text-file inputs; a ten-stage live workflow; three
approval gates; run history; validated downloads; final QA and factual-integrity
results; and an in-browser PDF preview.

### Synthetic UI preview

All screenshots below were rendered from stubbed synthetic states. They contain
no real résumé, posting, run history, provider transcript, or personal data.

![Synthetic desktop dashboard](docs/screenshots/synthetic-dashboard-desktop.png)

![Synthetic completed-run view](docs/screenshots/synthetic-success-desktop.png)

## Dependencies

The installer does **not** install dependencies. Check and install them yourself:

- Python 3.11 or newer
- `python-docx`
- `jsonschema`
- `fastapi`, `uvicorn`, `jinja2`, and `python-multipart` for the local UI
- OpenAI Codex CLI
- Google Antigravity CLI (`agy`)
- LibreOffice
- Poppler (`pdfinfo`, `pdftotext`, and `pdftoppm`)
- `wl-paste`, `xclip`, or `xsel` only when using `--clipboard`

HTTPX is a test-only dependency for the stubbed UI suite.

On EndeavourOS/Arch, an appropriate system dependency command is:

```bash
sudo pacman -S --needed python uv libreoffice-fresh poppler wl-clipboard
```

`xclip` or `xsel` may replace `wl-clipboard`. Development tools are optional:

```bash
sudo pacman -S --needed shellcheck
```

Install Codex CLI and Antigravity CLI through their official distribution
channels. This project never runs `sudo`, package managers, or global installers.

### Authentication prerequisites

Before the first real pipeline run, authenticate Codex and Antigravity using
their normal vendor login flows. Verify the exact executables used by your shell:

```bash
codex --version
codex exec --help
agy --version
agy --help
```

`resume-tailor` does not read, copy, print, or store credentials. Authentication
must already be available to each CLI in headless mode.

URL mode additionally requires Antigravity permission for
`read_url(linkedin.com)`. Grant only passive URL-read permission. The workflow
does not require `execute_url`, interactive browser control, form submission, or
LinkedIn-account access. A denied or soft-denied permission is handled as
`permission_denied`, even when the Antigravity process exits successfully.

## Local installation

The public repository contains only the synthetic
`template/sample_resume.docx`. It never publishes a real master résumé. Before
installation, place your own compatible file at the ignored path
`template/master_resume.docx`; see [template/README.md](template/README.md).

Then, from the repository:

```bash
./install.sh
```

This copies the application to `~/.local/share/resume-tailor` and installs both
launchers:

```text
~/.local/bin/tailor-resume
~/.local/bin/tailor-resume-ui
```

It refuses an existing installation. `./install.sh --force` replaces one only
after moving the old application and launchers to timestamped backups.

The installer does not modify `.bashrc` or any other shell file. If
`~/.local/bin` is not in `PATH`, it prints a notice.

When the repository already contains a working `.venv`, the installed
application reuses it through
`~/.local/share/resume-tailor/.venv`. This is a symbolic link; dependencies
remain physically inside the repository and are not reinstalled globally. Keep
the checkout in place while using that linked environment.

To explicitly add a KDE/Linux application-menu entry and a clickable desktop
icon:

```bash
./install.sh --desktop
```

This opt-in action creates
`~/.local/share/applications/resume-tailor.desktop` and an executable
`Resume Tailor.desktop` shortcut in the directory returned by
`xdg-user-dir DESKTOP`. A validated `~/Desktop` is used if that utility is
missing or returns an unsafe path. Both entries reuse the bundled Resume Tailor
icon. The installer refuses to overwrite an unrelated desktop file, even with
`--force`, and does not change KDE settings, MIME associations, or browser
preferences. Use `./install.sh --force --desktop` when replacing a managed
installation and its entries.

If the source checkout has no `.venv`, create an isolated dependency
environment manually after installation:

```bash
uv venv "$HOME/.local/share/resume-tailor/.venv"
uv pip install \
  --python "$HOME/.local/share/resume-tailor/.venv/bin/python" \
  -e "$HOME/.local/share/resume-tailor"
```

The installed launcher automatically prefers that environment. No dependency is
installed by `install.sh` itself.

To uninstall, run:

```bash
"$HOME/.local/share/resume-tailor/uninstall.sh"
```

The uninstaller lists every existing application, CLI launcher, UI launcher, and
optional desktop-entry target and requires typing `remove`.
Generated resumes and timestamped installation backups are not deleted.

## Usage

### Local web UI

Launch the server and open a dedicated Google Chrome application window:

```bash
tailor-resume-ui
```

The default address is `http://127.0.0.1:8765/`. The bind host is intentionally
fixed to `127.0.0.1`; no `--host` option is provided. Choose another local port
or artifact directory when needed:

```bash
tailor-resume-ui \
  --port 8876 \
  --output-dir "$HOME/Documents/Resumes/Tailored"
```

Use `tailor-resume-ui --no-browser` when you want to open the address manually.
Press `Ctrl+C` in the launching terminal to cancel active work safely and stop
the server.

Browser selection checks `google-chrome-stable`, `google-chrome`, and `chromium`
in that order. Each is started with an argument array and receives the complete
localhost session URL. If none is available—or launching it fails—the UI prints
an actionable notice and safely uses the existing system-default browser. It
never changes browser defaults or MIME associations.

Only one Resume Tailor server can own a configured port. Reopening the desktop
icon probes the localhost health endpoint and opens the already-running
dashboard instead of starting a conflicting process. A simultaneous-launch
race is handled by reserving the listening socket before application startup.

The bundled master is selected by default. LinkedIn URL mode derives company
and role after extraction. Pasted-text and job-file modes ask for those labels.
A compatible DOCX can be uploaded instead; uploads are limited to 5 MiB,
inspected as DOCX archives, and checked against the safe template structure
before the run starts.

### Terminal CLI

Clipboard mode:

```bash
tailor-resume \
  --resume "/absolute/path/master_resume.docx" \
  --clipboard \
  --company "RG Talent" \
  --role "Agentic AI Developer"
```

Job-file mode:

```bash
tailor-resume \
  --resume "/absolute/path/master_resume.docx" \
  --job-file "/absolute/path/job-description.txt" \
  --company "RG Talent" \
  --role "Agentic AI Developer" \
  --output-dir "$HOME/Documents/Resumes/Tailored"
```

LinkedIn URL mode derives the company and title from the fetched posting, so
`--company` and `--role` must be omitted:

```bash
tailor-resume \
  --resume "/absolute/path/master_resume.docx" \
  --job-url "https://www.linkedin.com/jobs/view/example-ai-role-1234567890/" \
  --linkedin-provider auto \
  --output-dir "$HOME/Documents/Resumes/Tailored"
```

`--linkedin-provider auto` chooses Apify when a local Apify token is configured
and otherwise chooses Antigravity. `apify` and `antigravity` are available as
explicit values. Once a provider starts, Resume Tailor never silently calls the
other provider after a failure; this avoids duplicate disclosure, cost, and
ambiguous provenance.

Apify URL mode uses the community Actor
[`piotrv1001/linkedin-job-details-scraper`](https://apify.com/piotrv1001/linkedin-job-details-scraper),
which accepts an exact job-detail URL in its required `searchUrls` array.
Resume Tailor sends only the normalized URL, retrieves at most five dataset
items, selects exactly one matching job ID, strips HTML locally, and validates
the existing canonical LinkedIn schema. Actor output is not treated as trusted
instructions and is not interpreted by another model.

For desktop-launcher use, store the token in the fixed private configuration
file rather than another repository's `.env`:

```bash
mkdir -p "$HOME/.config/resume-tailor"
chmod 700 "$HOME/.config/resume-tailor"
install -m 600 /dev/null "$HOME/.config/resume-tailor/apify-token"
${EDITOR:-nano} "$HOME/.config/resume-tailor/apify-token"
```

Paste only the token, save, and close the editor. Resume Tailor rejects
symlinks, non-user-owned token files, files with group/other access, whitespace
inside tokens, and files over 4 KiB. `APIFY_API_TOKEN` is an equivalent
environment-only configuration for terminal launches. The application never
copies or reads the Job Source Agent `.env`, never places the token in a URL,
and never writes it to run metadata or diagnostics.

The default Actor can be changed explicitly with
`RESUME_TAILOR_APIFY_ACTOR=owner/actor-name`. Community Actor interfaces can
change, so every response still fails closed against local URL, job-ID, size,
shape, and canonical-schema checks.

Only public `https://linkedin.com/jobs/view/...` and
`https://www.linkedin.com/jobs/view/...` URLs are accepted initially. Tracking
parameters are permitted when valid. Embedded credentials, other schemes,
unrelated hosts, non-job paths, conflicting IDs, and suspicious redirects are
rejected.

For a source checkout with its local environment:

```bash
./tailor-resume \
  --resume "$(pwd)/template/master_resume.docx" \
  --job-file "$(pwd)/job-description.txt" \
  --company "RG Talent" \
  --role "Agentic AI Developer"
```

Complete interface:

```text
tailor-resume --resume PATH
              ((--clipboard | --job-file PATH) --company NAME --role NAME
               | --job-url HTTPS_LINKEDIN_JOB_URL)
              [--linkedin-provider {auto,apify,antigravity}]
              [--output-dir PATH]
              [--yes]
              [--keep-workdir]
              [--timeout DURATION]
```

Durations accept positive seconds, minutes, or hours such as `90s`, `15m`, and
`1h`; the maximum is 24 hours. The default is `15m`.

`--yes` skips interactive approval input. In URL mode the posting confirmation
screen is still displayed and recorded as approved by `--yes`. The flag does
**not** bypass URL/status/redirect validation, schema validation, unanswered
factual questions, immutable-field checks, evidence checks, template validation,
one-page checks, source integrity, or final QA.

`--keep-workdir` retains the run's internal `work/` directory, including the
isolated LibreOffice profile and raw structured final-QA result. Normal output
does not retain it.

The web UI intentionally has no `--yes` equivalent. Every browser approval gate
requires an explicit button click; the terminal CLI retains its existing
`--yes` behavior.

## Human approval stages

Without `--yes`, file and clipboard modes require typing the exact word
`approve` twice:

1. after reading the Codex resume-to-job analysis;
2. after reading the local section-by-section content diff.

URL mode adds an earlier gate—before any resume extraction or Codex analysis.
The screen shows company, job title, location, requested URL, final URL,
description preview, and extraction warnings. Confirm it with `approve`; any
other response stops before a tailored DOCX or PDF can be generated.

In the UI, the LinkedIn gate also offers **Use pasted description instead**.
That action replaces the extracted description only with text you explicitly
paste; it never clicks LinkedIn controls or accesses an account. The next gate
groups Codex's proposed summary, experience, and project changes, supported
keywords, gaps, forbidden claims, and unchanged sections. A final before/after
content gate still protects deterministic rendering after Antigravity and local
evidence checks.

Any other input, including end-of-input, stops the pipeline and keeps the
artifacts already produced. If Codex asks an unanswered factual question, the
pipeline stops even with `--yes`; answer it outside the tool and update the
factual master only when appropriate. Antigravity runs only after analysis
approval and cannot reopen factual discovery or request unlisted experience.

## Output artifacts

The default parent is `~/Documents/Resumes/Tailored`. Every invocation creates a
new mode-0700 directory with sanitized company/role names and a timestamp:

```text
rg-talent-agentic-ai-developer-YYYYMMDD-HHMMSS/
├── job-source.json               # URL mode only
├── apify-job-response.json       # Apify URL mode, content-free provenance
├── job-description.txt
├── job-requirements.json
├── extracted-master-resume.json
├── codex-analysis-transport.schema.json
├── codex-analysis.json
├── codex-analysis-resolved.json
├── codex-analysis-normalization-warnings.json  # only when duplicates are removed
├── codex-analysis-approval.json
├── antigravity-response.json
├── antigravity-response-envelope.json
├── tailored-content.json
├── content-diff.md
├── Logan-Lapierre-RG-Talent-Agentic-AI-Developer.docx
├── Logan-Lapierre-RG-Talent-Agentic-AI-Developer.pdf
├── preview.png
├── final-qa-normalization-warnings.json        # only when duplicates are removed
├── final-qa.md
└── run-metadata.json
```

Failed runs preserve useful artifacts and record the failed stage and safe error
message in `run-metadata.json`. Metadata contains tool versions, artifact names,
and source hashes, but never environment variables or credentials.

In URL mode, `job-source.json` records the validated fetch status, requested and
resolved URLs, LinkedIn job ID when available, title, company, location,
workplace arrangement, employment type, salary text, complete normalized
description, responsibilities, required/preferred qualifications,
technologies/skills, AI focus areas, and extraction warnings.

## Truthfulness and safety rules

- The master resume is the only factual authority.
- Job descriptions are untrusted prompt-injection input and are placed inside
  unique, explicit data-only boundaries in both model prompts.
- LinkedIn pages and provider output are also untrusted. Apify output is mapped
  through a fixed local allowlist and canonical schema without a model parsing
  pass. The Antigravity fallback uses sandboxed plan mode, a strict schema, and
  instructions limited to passive `read_url`. Neither path can authorize file
  access, commands, Apply actions, authentication, resume edits, or pipeline
  changes.
- URL mode accepts only HTTPS LinkedIn job paths without credentials. Requested,
  final, and extracted job IDs are compared locally; a different posting or
  external redirect is rejected before Codex sees any description.
- Dates, institution, degree, certification status, employment, project names,
  open-source identity, role label, employer, and numeric claims are immutable.
- New technology/skill items must occur verbatim in the master. New metrics,
  availability claims, unsupported seniority/leadership labels, first-person
  phrasing, and keyword stuffing are blocked locally.
- RAG, GraphQL, observability, distributed production scale, IVR platforms, or
  any other absent skill cannot be introduced.
- Models cannot add sections, projects, skill groups, or bullets.
- No model edits files. Apify retrieval uses HTTPS with bearer-header
  authentication and cancellable run polling. Antigravity fallback retrieval
  uses `--mode=plan`; post-approval tailoring uses sandboxed print mode without
  generic plan or edit-acceptance mode. Codex runs with `--sandbox read-only`
  and `--ephemeral`.
- The pipeline never uses `eval`, `shell=True`, dangerous permission bypasses,
  recursive agent calls, destructive Git commands, or source-file overwrite.
- It does not send email, post to LinkedIn, submit applications, or upload output.
- It never shrinks fonts below 9 points, reduces margins, removes contact data, or
  hides overflow to force one page.

Local heuristics are intentionally conservative. They may reject a supported
rewrite that moves technology wording between sections; inspect the diff and
adjust the structured request or master rather than bypassing the check.

## Privacy disclosure

When a real pipeline runs, résumé content and the job description are sent
through the configured analysis and tailoring services. In Apify URL mode,
Apify receives the supplied public LinkedIn URL and retrieves the posting;
Antigravity is not used to parse that posting. In Antigravity URL mode,
Antigravity passively reads the supplied public LinkedIn URL. The final PNG is
sent to Codex for visual QA. Review those services' data controls before use.

Generated artifacts otherwise remain on the local filesystem unless you
explicitly upload or share them. No keys or environment variables are logged.
Antigravity 1.1.8 print mode accepts stdin prompts. Resume Tailor keeps
Antigravity argv limited to flags and short schema paths and sends prompt bytes
through UTF-8 stdin, so résumé/job/prompt content is absent from process command
lines. Malformed provider output diagnostics retain only byte counts and hashes,
not raw stdout or stderr.

For valid JSON responses, run metadata records only the Antigravity CLI version,
execution/output mode, response-envelope type, tailoring-schema hash,
response-artifact hash, and validation result. Provider prose remains escaped,
bounded, collapsed, and omitted from exceptions.

### Localhost security model

- The server binds only to `127.0.0.1` and exposes no remote-bind option.
- Every launch generates a fresh random session/CSRF token stored in a
  `SameSite=Strict`, HTTP-only cookie; every state-changing form also carries the
  token.
- Responses set a restrictive Content Security Policy, deny framing, disable
  MIME sniffing, and send no referrer.
- Jinja auto-escaping and DOM `textContent` render webpage, model, and user text
  as text, never trusted HTML.
- Uploaded résumés must be structurally valid DOCX files; job uploads must be
  UTF-8 `.txt`; request and expanded-archive sizes are bounded.
- Download routes resolve only known direct files inside validated run-artifact
  directories. Symlinks, traversal, and arbitrary paths are rejected.
- The UI provides no filesystem browser, LinkedIn login automation, Apply
  interaction, permission bypass, or application submission.

Localhost is not a multi-user authorization boundary. Run the server only from a
trusted local account, stop it when finished, and do not place a reverse proxy
or public tunnel in front of it.

## PDF and layout acceptance

LibreOffice runs headlessly with a unique writable user profile and controlled
temporary directory. The pipeline then requires:

- exactly one US Letter PDF page;
- extractable text containing the name, all contact-link labels, and all section
  headings;
- no replacement glyphs;
- every Poppler text bounding box inside the page;
- no detected overlapping text lines;
- a nonempty PNG preview;
- unchanged DOCX geometry, run/paragraph formatting, list structure, contact
  information, and all six hyperlink targets.

The final Codex review inspects clipping, overflow, readability, grammar,
truthfulness, duplication, ATS alignment, and keyword density. It is read-only
and never repairs the document. A material finding yields a nonzero exit while
preserving DOCX, PDF, preview, diff, and QA report.

## Troubleshooting

### Codex reports `invalid_json_schema`

Resume Tailor preflights the checked-in compatibility schemas and each generated
source-bound analysis schema locally before launching Codex. If a bundled schema
drifts or a generated schema is invalid, too large, or inconsistent with its
source catalog, the run stops with `Codex could not start because its output
schema was incompatible.` No model request is made. Expand **Sanitized technical
details** in the UI or inspect the CLI error, fix the schema/derivation, and run
the stubbed schema tests before retrying. Do not remove canonical
factual-integrity constraints merely to make a provider transport schema pass.

### Antigravity command-line-size launch failure

Older runs placed the complete tailoring prompt in one Antigravity command-line
argument. Linux can reject a single argument with `E2BIG` even when the system's
aggregate `ARG_MAX` is larger. Current code uses Antigravity 1.1.8's documented
stdin print-mode path and keeps argv prompt-free.

The UI identifies this failure without displaying prompt content. **Retry
Antigravity tailoring** appears only when a persisted Codex approval record and
all approved input hashes can be verified. Eligible recovery creates a new run,
does not retrieve the posting or invoke Codex, and pauses at the content-diff
approval gate. Historical runs without the approval record require a new run;
the application never infers approval merely because a later stage was reached.

### No clipboard backend

Install one clipboard tool, or use a text file:

```bash
tailor-resume ... --job-file "/path/to/job.txt"
```

Backend order is `wl-paste`, then `xclip -selection clipboard -o`, then
`xsel --clipboard --output`. An empty clipboard is rejected.

The web UI uses pasted clipboard text instead of invoking a system clipboard
binary. Click **Paste from clipboard** when the browser grants permission, or
focus the field and use `Ctrl+V`.

### LinkedIn login wall

URL mode never automates login or accesses your LinkedIn account. If extraction
returns `login_required`, copy the complete posting text yourself and rerun with:

```bash
tailor-resume ... --job-file "/path/to/job.txt" --company "Company" --role "Role"
```

Clipboard mode is an equivalent fallback.

In the web UI, return to the dashboard and select **Clipboard text** or **Text
file**. The failed URL run remains visible with its safe diagnostic artifacts.

### Apify is not configured

Choose **Apify job details** only after configuring `APIFY_API_TOKEN` or the
private `~/.config/resume-tailor/apify-token` file described above.
**Automatic** uses Apify when either configuration is present and otherwise
selects Antigravity before the run begins. It does not fall through to
Antigravity after an Apify run starts or fails.

Apify diagnostics are content-free: actor/run/build/dataset identifiers,
selected field names and types, byte counts, hashes, and validation status are
recorded in `apify-job-response.json`; the provider response body and token are
omitted. `job-source.json` remains the canonical, human-approved posting
artifact.

### Apify Actor output changed

An Actor can be updated independently because it is community-maintained.
Missing title/company/description, multiple results, a mismatched job ID,
unexpected object shape, unsafe control text, oversized content, or a failed
Actor run stops before Codex. Explicitly choose Antigravity, pasted text, or a
UTF-8 job file while reviewing the Actor change; Resume Tailor never performs
automatic paid-provider failover.

### Expired or unavailable LinkedIn posting

`expired`, `unavailable`, and `insufficient_content` stop before Codex analysis.
Find an active posting or provide a complete saved description through
`--job-file` or `--clipboard`. Search-card snippets are intentionally rejected.

### Incorrect or suspicious LinkedIn redirect

The requested URL, resolved URL, page job ID, scheme, and hostname are checked
locally. Same-job LinkedIn canonicalization is accepted when the stable job ID
matches. A different job ID, unrelated domain, missing verifiable identity, or
embedded credentials stops the run. Confirm the URL in a trusted browser and
use a copied job description if LinkedIn routing remains ambiguous.

### Antigravity URL permission denied

Allow Antigravity the narrow `read_url(linkedin.com)` permission. Do not enable
`execute_url` or a dangerous permission bypass. A process that exits zero but
reports `permission_denied` still fails closed.

### Antigravity returns a planning or `WAITING` response

Post-approval tailoring has no factual-discovery state. A legacy `WAITING`,
generic readiness statement, or request for another task is classified as a
tailoring-contract failure; provider prose is omitted from the public error.
The UI never asks for missing experience or suggests changing the confirmed
posting. **Retry Antigravity tailoring** is available only when the source
résumé, confirmed job, requirement catalog, transport schema, resolved analysis,
and explicit approval record still match their authenticated hashes. Otherwise,
start a new run.

### Antigravity returns an unsupported JSON response envelope

Tailoring accepts only one complete documented structured-output candidate. A
direct-root result, a supported JSON-wrapper field, or one typed terminal
`stream-json` result is decoded once and validated strictly. Resume Tailor never
extracts braces from prose, removes Markdown fences, joins fragments, or chooses
among conflicting candidates. If `structured_output` and `response` contain
canonically identical complete JSON values, they are treated as two documented
representations of the same result and `structured_output` is preferred.

The UI classifies this as **Antigravity returned JSON in an unsupported response
format.** If the preserved bytes already contain one valid complete result and
every source, requirement, schema, approval, response, and ancestry hash
authenticates, **Reprocess preserved Antigravity response** creates a new
provider-free run and pauses at the content-diff gate. Otherwise, offline salvage
is unavailable; authenticated **Retry Antigravity tailoring** remains available
without rerunning LinkedIn or Codex.

LinkedIn URL retrieval uses the documented `stream-json` terminal
`{"event":"result","result":{...}}` envelope. A malformed, missing, conflicting,
or schema-invalid terminal result is shown as **LinkedIn response-format
failure**, writes only content-free hashes/types to
`linkedin-response-envelope.json`, and stops before résumé analysis. Tailoring
retry and offline tailoring-response reprocessing are never offered for this
stage; use a UTF-8 job file or pasted description while correcting provider
compatibility.

### The UI does not open

Run `tailor-resume-ui --no-browser`, read the printed localhost address, and
open it manually. Confirm the chosen port is unused. The default health endpoint
is `http://127.0.0.1:8765/health`. If the browser reports a stale session after a
restart, reload the dashboard so the per-launch cookie is replaced.

If Chrome is not found, install one of `google-chrome-stable`, `google-chrome`,
or `chromium`, or continue with the safely reported system-browser fallback.
Closing the Chrome window does not start another server; clicking the Resume
Tailor icon again reconnects to the existing localhost instance.

### A UI run appears busy

Only one active run is supported. Finish the current approval gate or use
**Cancel run**. The background worker stops its active subprocess group and
retains useful diagnostics. Starting a second run while one is active returns a
clear conflict instead of spawning concurrent model work.

### The PDF is two pages

The DOCX and PDF are preserved. Shorten the approved tailored wording within the
reported content budgets, then rerun. The pipeline will not shrink body text,
collapse margins, or hide overflow.

### Template drift

Use the inspected master at `template/master_resume.docx`, or intentionally
update the semantic mapper and tests for a new design. The renderer never falls
back to blind paragraph indexes.

### LibreOffice fails headlessly

Close stale LibreOffice processes, confirm `libreoffice --version`, ensure the
output filesystem is writable, and retry. Each run already uses its own profile,
which avoids normal profile-lock conflicts.

### A local evidence check blocks a rewrite

Read `content-diff.md`. The check never silently changes content. Add real
evidence to the master only when true, or use wording made solely from existing
facts and technologies.

## Version compatibility

Development and inspection verified these local interfaces:

- Codex CLI `0.146.0`: `codex exec`, stdin prompt (`-`), `--cd`,
  `--sandbox read-only`, `--ephemeral`, `--skip-git-repo-check`,
  `--output-schema`, `--output-last-message`, and `--image`.
- Antigravity CLI `1.1.8`: `--prompt` print mode with UTF-8 stdin, optional
  `--mode=plan` for passive LinkedIn retrieval, `--sandbox`,
  `--output-format json` for tailoring, `--output-format stream-json` with an
  `event=result` terminal envelope for LinkedIn retrieval,
  `--json-schema`, and `--print-timeout`.
- Apify API v2: bearer-header authentication, asynchronous Actor runs,
  run-status polling, best-effort run abort, and default-dataset item retrieval.
  The default community Actor contract uses
  `piotrv1001/linkedin-job-details-scraper` with one `searchUrls` item.
- LibreOffice `26.2.4.2`.
- Poppler `26.07.0`.
- FastAPI `0.119.1`, Starlette `0.48.0`, Uvicorn `0.52.0`, Jinja2 `3.1.6`,
  python-multipart `0.0.32`, and HTTPX `0.28.1`.

Antigravity 1.1.8 does not expose a `--cwd` option, so this project sets the
subprocess working directory directly. At startup it records detected versions.
If a newer CLI removes a verified flag or changes its JSON wrapper, the adapter
fails closed with an actionable error; re-run the four help/version commands and
update the corresponding adapter and tests.

## Development and tests

Create the environment yourself; no script does this automatically:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

Run the required checks:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall resume_tailor
bash -n tailor-resume tailor-resume-ui install.sh uninstall.sh
shellcheck tailor-resume tailor-resume-ui install.sh uninstall.sh
```

The current suite contains **195 synthetic/offline tests**. Tests prepend
`tests/stubs` to `PATH` for Codex and Antigravity. They never make
real model or network calls. URL tests simulate successful and redirected
postings, multiple companies and AI titles, mismatches, login walls, expiry,
missing content, permission denial, malformed output, unsafe URLs, webpage
prompt injection, rejection, and confirmed continuation. Integration tests
exercise `python-docx`, LibreOffice, and Poppler against the fixture copy,
validate one-page output, and confirm the master hash is unchanged.

UI tests use a stub pipeline and HTTPX ASGI transport. They cover
startup/health, localhost configuration, all form modes, approval and rejection,
completed and failed runs, permission denial, cancellation, double submission,
uploads, HTML injection, CSRF/session rejection, path traversal, and artifact
downloads. A separate harmless sleep-process test verifies shared subprocess
cancellation. No test invokes a real model or network fetch.

## Limitations and development disclosure

Resume Tailor is a personal, local-first portfolio project rather than a hosted
multi-user service. Its deterministic template mapper intentionally supports one
inspected DOCX structure; significant template drift fails closed. Provider CLI
formats and authentication behavior can change, model output can still be
incorrect, LinkedIn may block passive retrieval, local schema validation cannot
prove semantic equivalence, and human review remains mandatory. This tool does
not submit applications or replace factual résumé maintenance.

The project was designed and directed through AI-assisted development and prompt
engineering. Codex served as implementation owner; Antigravity was used as a
read-only visual designer/critic during the bounded UI redesign. Subsequent
contract fixes were diagnosed, implemented, and tested locally with synthetic
fixtures and stubbed providers. The human owner specified the architecture,
safety boundaries, approval gates, publication scope, and acceptance criteria.

## License

MIT. See [LICENSE](LICENSE).
