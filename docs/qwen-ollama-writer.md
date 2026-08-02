# Local Qwen writer contract

Resume Tailor uses `resume-tailor-qwen` through Ollama as its default résumé
writer. Codex remains responsible for source-bound job analysis and independent
final QA. Python remains the authority for schemas, evidence, approval state,
formatting, and artifact publication.

## Local-only transport

The adapter has one fixed endpoint: `http://127.0.0.1:11434`. It accepts no
remote host, URL, token, or automatic writer fallback. The parent process sends
the complete Ollama request to a cancellable Python worker over UTF-8 stdin, so
résumé-derived content is absent from argv, environment variables, and prompt
files. The worker permits only `/api/version`, `/api/show`, and `/api/chat`.

Chat requests use:

- `stream: false`;
- `think: false`;
- the selected Ollama model, defaulting to `resume-tailor-qwen`;
- a derived JSON Schema in `format`;
- an 8,192-token context and 4,096-token output bound;
- low-temperature sampling.

The provider schema removes canonical cross-field `allOf` assertions only to
constrain token generation. The returned `message.content` must still parse as
exactly one strict JSON object and pass the complete checked-in Draft 2020-12
schema locally. Duplicate keys, non-finite values, Markdown, prose wrappers,
wrong statuses, unknown edit IDs, and canonical schema violations fail closed.

## Model profile

A compatible local profile can be created with this `Modelfile`:

```text
FROM qwen3.5:9b
PARAMETER num_ctx 8192
PARAMETER temperature 0.2
```

```bash
ollama create resume-tailor-qwen -f Modelfile
ollama show resume-tailor-qwen
```

The API request repeats the important runtime bounds, so a profile with broader
defaults cannot silently weaken the per-run configuration.

## Evidence and authorship boundary

Qwen receives the authenticated master content, approved edit catalog, source
blocks authorized by those edits, immutable facts, forbidden claims, and local
content budgets. The raw job description is authenticated during preflight but
is not repeated in the writer prompt. The approved Codex plan is the writer's
only job-targeting instruction.

Qwen may change wording only where the approved plan permits. It cannot add
sections, projects, skills groups, bullets, technologies, metrics, credentials,
employment facts, citizenship, availability, or other unsupported claims. A
safe inability to apply one edit uses `cannot_apply`; an execution failure uses
`technical_failure`. Neither status triggers an automatic provider fallback.

Qwen never edits a DOCX. Python validates the complete content object, displays
the diff for approval, renders the document, exports the PDF, and verifies all
stable artifacts. A fresh Codex session then reviews the finished résumé. At
most one additional Qwen call can occur, and only after authenticated material
QA findings plus explicit revision authorization.

## Artifacts

Initial Qwen runs preserve:

- `ollama-tailoring-transport.schema.json`;
- `ollama-response.json`;
- `ollama-response-envelope.json`.

The response artifact contains the bounded Ollama response fields needed for
diagnosis. The envelope records only provider/runtime identity, fixed endpoint,
model, prompt byte count and SHA-256, schema hash, response hash, context bound,
and validation result. The prompt itself is not written to disk.

A single revision uses equivalent `ollama-revision-*` artifacts. Antigravity
artifacts and authenticated recovery remain supported only when
`--writer-provider antigravity` is selected or a historical Antigravity run is
being recovered.
