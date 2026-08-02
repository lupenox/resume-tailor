# Local Ollama writer contract

Resume Tailor uses `resume-tailor-gemma` (Gemma 4 12B) through Ollama as its
default résumé writer. Codex remains responsible for source-bound job analysis
and independent final QA. Python remains the authority for schemas, evidence,
approval state, formatting, and artifact publication.

## Local-only transport

The adapter has one fixed endpoint: `http://127.0.0.1:11434`. It accepts no
remote host, URL, token, or automatic writer fallback. The parent process sends
the complete Ollama request to a cancellable Python worker over UTF-8 stdin, so
résumé-derived content is absent from argv, environment variables, and prompt
files. The worker permits only `/api/version`, `/api/show`, and `/api/chat`.

Chat requests use:

- `stream: false`;
- `think: false`;
- the selected Ollama model, defaulting to `resume-tailor-gemma`;
- a derived JSON Schema in `format`;
- a context window and output bound derived from the declared model
  capabilities and the deterministic pre-launch budget (see
  [Context and output budgeting](#context-and-output-budgeting));
- low-temperature sampling.

The provider schema removes canonical cross-field `allOf` assertions only to
constrain token generation. The returned `message.content` must still parse as
exactly one strict JSON object and pass the complete checked-in Draft 2020-12
schema locally. Duplicate keys, non-finite values, Markdown, prose wrappers,
wrong statuses, unknown edit IDs, and canonical schema violations fail closed.

## Model profile

Create the active local profile once after downloading Gemma 4 12B:

```bash
ollama pull gemma4:12b
ollama create resume-tailor-gemma -f Modelfile
ollama show resume-tailor-gemma
```

A compatible `Modelfile`:

```text
FROM gemma4:12b
PARAMETER num_ctx 32768
PARAMETER temperature 0.2
```

The `num_ctx` value must be at least the `context_window` declared for the
model in `resume_tailor/ollama_capabilities.py`. A profile built with a
smaller window silently truncates the prompt inside the server, which is one
way a structurally valid request still produces a wrong-root response.

> **Note — conservative context window.** The operational context is set to
> 32,768 tokens, not the larger maximum the base model supports. The
> application requests only what is needed and tested. Do not raise
> `num_ctx` beyond the declared capability without updating the capability
> registry and re-running the offline test suite.

## Context and output budgeting

Model capabilities are explicit and configurable rather than hardcoded at the
call site. `MODEL_CAPABILITIES` in `resume_tailor/ollama_capabilities.py`
declares `context_window`, `max_output_tokens`, `min_output_tokens`, and
`supports_json_schema` per model, resolved by exact name and then by tag-free
prefix; unknown models fall back to conservative defaults. `invoke_ollama` and
`invoke_ollama_revision` accept a `capability_overrides` mapping that replaces
any of the three token values for one call, so a differently built local
profile does not require a code change. No CLI flag is wired to this yet.

Before any request is launched, `plan_ollama_budget` deterministically
estimates the prompt cost, adds a fixed framing overhead, and reserves the
remaining room for generation. If the prompt plus the minimum useful response
cannot fit inside the context window, the run fails closed with
`OllamaBudgetError` (`failure_class: ollama-budget-preflight`) and no request
is sent. The budget never trims approved content, edits the prompt, or relaxes
a schema: it refuses.

### Historical context: the preserved Step 6 overflow failure

The budgeting system was introduced to close a failure observed with the prior
Qwen writer (no longer the default). A 7,590-token prompt and 1,145 generated
tokens totalled 8,735 against an 8,192-token window. Generation stopped
naturally and stayed under the output cap, so no truncation signal was raised,
but the window had already overflowed. The preserved wrong-root response from
that failure is pinned as a synthetic regression fixture at
`tests/fixtures/ollama_wrong_root_resume_response.json`; the test remains valid
regardless of the active writer model.

## Failure classification

A rejected response records one specific sanitized `failure_class` and a
`validation_path` in `ollama-response-envelope.json`. All of them remain
contract failures for existing handlers and none of them make an Ollama run
eligible for authenticated Antigravity retry or offline reprocessing.

| `failure_class` | `validation_path` | Meaning |
| --- | --- | --- |
| `ollama-malformed-json` | `malformed_json` | `message.content` is not exactly one strict JSON object. |
| `ollama-response-envelope` | `response_envelope` | The Ollama HTTP envelope is missing, incomplete, or not an assistant message. |
| `ollama-transport-schema` | `transport_schema` | Parsed JSON violated the derived transport schema, including a wrong root shape. |
| `ollama-canonical-schema` | `canonical_schema` | Envelope-shaped output violated the checked-in Draft 2020-12 schema or its cross-field rules. |
| `ollama-output-truncation` | `output_truncation` | Generation stopped at the output ceiling or reported a length stop. |
| `ollama-downstream-evidence` | `downstream_evidence` | Schema-valid output failed source-evidence or immutable-fact validation. |

The envelope records `validation_result`, `validation_path`,
`validation_message`, the resolved capabilities, the planned budget, and
generation counters (`done_reason`, `eval_count`, `prompt_eval_count`,
`reported_total_tokens`, `truncated`, `output_ceiling_reached`,
`context_window_exceeded`). `content_logged` is always `false`: paths and
messages describe *where* validation failed and never carry résumé content,
field values, or model prose.

The preserved wrong-root response is pinned as a synthetic regression fixture
at `tests/fixtures/ollama_wrong_root_resume_response.json`. It contains no
real résumé content and asserts the observed root (`header`,
`objective_summary`, `education_certifications`, `technical_skills`,
`ai_engineering_projects`, `open_source_contribution`, `experience`) is
classified as `ollama-transport-schema` and not as malformed JSON or
truncation. This fixture is a historical record of the prior Qwen writer's
failure mode; the test continues to protect the classification logic regardless
of which model is active.

## Structured-output capability probe

`probe_structured_output_support` checks the derived schema offline, with no
provider call, and asserts the constructs this contract depends on: `$ref`
declaration and resolution, `oneOf` branching, `additionalProperties: false`,
the `status` enum, and the required root fields `status`, `message`,
`cannot_apply`, `technical_failure`, and `tailored_resume`. It is a guard
against a silently weakened schema, not a provider health check.

The API request repeats the important runtime bounds, so a profile with broader
defaults cannot silently weaken the per-run configuration.

## Evidence and authorship boundary

The local writer (Gemma 4 12B) receives the authenticated master content,
approved edit catalog, source blocks authorized by those edits, immutable
facts, forbidden claims, and local content budgets. The raw job description is
authenticated during preflight but is not repeated in the writer prompt. The
approved Codex plan is the writer's only job-targeting instruction.

The local writer may change wording only where the approved plan permits. It
cannot add sections, projects, skill groups, bullets, technologies, metrics,
credentials, employment facts, citizenship, availability, or other unsupported
claims. A safe inability to apply one edit uses `cannot_apply`; an execution
failure uses `technical_failure`. Neither status triggers an automatic provider
fallback.

The local writer never edits a DOCX. Python validates the complete content
object, displays the diff for approval, renders the document, exports the PDF,
and verifies all stable artifacts. A fresh Codex session then reviews the
finished résumé. At most one additional local writer call can occur, and only
after authenticated material QA findings plus explicit revision authorization.

## Artifacts

Initial local writer runs preserve:

- `ollama-tailoring-transport.schema.json`;
- `ollama-response.json`;
- `ollama-response-envelope.json`.

The response artifact contains the bounded Ollama response fields needed for
diagnosis. The envelope records only provider/runtime identity, fixed endpoint,
model, prompt byte count and SHA-256, schema hash, response hash, context
bound, and validation result. The prompt itself is not written to disk.

A single revision uses equivalent `ollama-revision-*` artifacts. Antigravity
artifacts and authenticated recovery remain supported only when
`--writer-provider antigravity` is selected or a historical Antigravity run is
being recovered.
