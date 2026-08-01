# Codex LinkedIn retrieval contract

## Responsibility boundary

Workflow step 2 has one web-retrieval provider: a dedicated Codex adapter. It
retrieves and structures the exact public LinkedIn job-detail URL supplied by
the user. This adapter is separate from step-4 résumé analysis and from the
fresh final-QA Codex session. Antigravity is not invoked during retrieval; its
only provider role is writing the complete tailored résumé content at step 6.

Pasted text, clipboard input, and UTF-8 job files are local input modes and do
not enable Codex live search.

## Invocation

The locally verified Codex CLI 0.146.0 interface is invoked in this shape:

```text
codex --search exec \
  --ignore-user-config \
  --cd RUN_ARTIFACT_DIRECTORY \
  --sandbox read-only \
  --ephemeral \
  --skip-git-repo-check \
  --output-schema schemas/linkedin_job.openai.schema.json \
  --output-last-message RUN_ARTIFACT_DIRECTORY/.codex-linkedin-last-message-UNIQUE.json \
  -
```

`--search` is a global option and therefore precedes `exec`. The complete UTF-8
prompt is written to stdin through the final `-`; prompt text is absent from
argv and environment variables. The subprocess starts in the isolated run
artifact directory, inherits the shared bounded timeout, cancellation,
process-group termination, and liveness heartbeat behavior, and runs in a
read-only ephemeral session. `--ignore-user-config` avoids unrelated user
configuration while retaining authentication. Repository rules are not bypassed.

The transient last-message file is read as exactly one complete UTF-8 JSON
document and removed. Local code never strips Markdown fences, scans prose for
braces, joins fragments, or selects among candidates.

## Prompt and disclosure boundary

The retrieval prompt contains only:

- the normalized requested public LinkedIn URL;
- the stable numeric job ID extracted from that URL by local Python; and
- retrieval, safety, and structured-output instructions.

It contains no résumé text, master résumé path, résumé hash, extracted résumé,
approved analysis, tailored content, or Antigravity response. The prompt limits
live search to the exact supplied posting and same-job LinkedIn
canonicalization. It treats webpage and search-result content as untrusted data
and forbids obeying embedded instructions, running suggested commands, account
access, sign-in, Apply actions, forms, messages, unrelated postings, inferred
facts, fabricated descriptions, local-file access, file modification, and
invoking another agent. Prompt-injection text present in the actual posting may
only remain inert quoted job-description data.

## Local authentication and validation

Before launch, local Python requires HTTPS, an allowlisted LinkedIn hostname,
no embedded credentials or non-HTTPS port, a `/jobs/view/...` path, and one
stable 5–20 digit job ID obtained from the path or `currentJobId`. Conflicting
IDs are rejected.

After launch, local Python validates the whole result against the canonical
Draft 2020-12 schema and then authenticates its identity against the trusted
request. The returned `requested_url` must equal the normalized supplied URL.
The final URL must still be an allowed LinkedIn job-detail URL, and its locally
extracted ID, the structured `linkedin_job_id`, and the requested ID must all
match. Same-job LinkedIn canonicalization is accepted; another posting or host
is rejected.

A successful result also requires nonempty company and title plus a complete,
substantive normalized description. Existing schema size, array, uniqueness,
safe-text, and control-character limits remain local. Only after every check
succeeds are `job-source.json` and `job-description.txt` written. The user then
sees the posting and must approve it before résumé extraction or analysis.

## Fail-closed outcomes and diagnostics

Retrieval exposes only bounded local classifications:

- `login_required`
- `expired`
- `unavailable`
- `insufficient_content`
- `url_mismatch`
- `job_id_mismatch`
- `search_unavailable`
- `provider_failure`
- `malformed_output`

Free-form provider prose is never used as a public error. Failure stops before
résumé content reaches Codex or Antigravity and never triggers another web
provider or invented-content fallback. The user may instead supply a complete
posting through pasted text, the clipboard, or a UTF-8 file.

`codex-linkedin-retrieval-diagnostic.json` is content-free. It may record the
bounded classification, validation stage and result, process exit status, byte
counts, hashes, JSON root type, field names/types, and schema hashes. It omits
the raw provider response, job description, credentials, hidden reasoning,
environment variables, and résumé data.
