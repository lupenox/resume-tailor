# Antigravity response-envelope contract

## Incident

Antigravity CLI 1.1.8 completed a sandboxed print-mode process and returned its
documented JSON wrapper, but the wrapper had no `structured_output` field. Its
`response` value contained prose followed by several separate JSON-like objects.
None was a complete value accepted by the tailoring schema. This was a response
envelope/structured-output failure, not missing résumé or job information.

The preserved diagnostic was inspected and replayed read-only. This document and
the regression fixtures contain only synthetic structure; they do not contain a
real résumé, posting, prompt, provider transcript, URL, personal path, or hidden
reasoning.

## Conservative parsing

Plan-mode LinkedIn extraction and print-mode tailoring use separate adapters.
For tailoring, local code accepts only one unambiguous whole-document candidate:

- the tailored schema object at the JSON root;
- one documented JSON-wrapper field whose object or complete JSON string is the
  candidate;
- one terminal `result` event in Antigravity 1.1.8 `stream-json`, with the
  documented closed-vocabulary `step_type` discriminator.

A string candidate is decoded exactly once as one JSON document. Local code does
not scan prose for braces, strip Markdown fences, join fragments, or choose among
multiple candidates. Missing, malformed, ambiguous, schema-mismatched, or
schema-invalid output is rejected before content validation or rendering.

Every accepted candidate is validated against `tailored_resume.schema.json` and
then against the local approved-edit and factual-integrity contracts. Run
metadata records the CLI version, print mode, output format, envelope type,
schema hash, response-artifact hash, and validation result. Logs and exceptions
contain no provider response text.

## Recovery

An authenticated post-approval envelope failure retains manual
**Retry Antigravity tailoring** as a fallback. It never retries automatically.

Offline **Reprocess preserved Antigravity response** is shown only when all of
the following succeed locally:

- the preserved response is exactly one complete, schema-valid tailoring result;
- the response and local expected-schema hashes match;
- the source résumé, extraction, confirmed job input, requirement catalog,
  resolved analysis, generated Codex transport schema, and Codex approval record
  authenticate;
- any recovery ancestor and recovery-input hash chain authenticate;
- deterministic factual-integrity validation passes.

Reprocessing creates a new isolated run, copies the authenticated response bytes,
invokes no provider, preserves the failed run, and stops at the normal validated
content-diff approval gate. No DOCX or PDF is rendered before that approval.
When the preserved response contains prose, fragments, multiple candidates, or
no valid candidate, offline salvage is unavailable and only the authenticated
manual provider retry remains.
