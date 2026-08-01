# Antigravity stdin transport and recovery

## Failure

A preserved run failed before Antigravity started because the complete prompt
was passed as one command-line argument. The synthetic reproduction is larger
than Linux's practical single-argument limit and fails locally with `E2BIG`.
No private résumé, posting, prompt, provider transcript, or hidden reasoning is
stored in this document or its regression fixtures.

## Transport contract

Antigravity CLI 1.1.8 documents stdin prompts for print mode. Resume Tailor now:

- uses a short argument array containing only the executable, verified flags,
  a local schema path, and a bounded timeout;
- sends the complete prompt as strict UTF-8 through stdin;
- uses `shell=False` and never transports prompt text through argv,
  environment variables, interpolation, or a shell;
- retains sandbox restrictions, structured JSON output, process-group
  cancellation, and bounded timeout behavior; step-6 tailoring deliberately
  does not use generic plan mode;
- rejects prompts above a 750,000-byte local resource bound before launch;
- stores only output byte counts and hashes when provider output is malformed.

The bound protects local memory and subprocess resources. It is intentionally
larger than practical argv limits and does not truncate or summarize a prompt.
No prompt file is created.

## Approval record

After local evidence validation and explicit Codex-analysis approval, the
pipeline writes `codex-analysis-approval.json`. It contains hashes and local
identity metadata, not résumé or job text. Run metadata stores the approval
record's own hash. The record is internal and is not exposed by the validated
artifact-download route.

## Antigravity-only recovery

Recovery is limited to authenticated post-approval Antigravity launch-size,
response-envelope, tailoring-contract, bounded `cannot_apply`, and technical
failures. Before a new run is created, local code verifies:

- the current source résumé and a fresh deterministic extraction;
- the confirmed job-description bytes;
- the job-requirement catalog;
- the run-specific Codex transport schema against both local catalogs;
- the resolved Codex analysis and its source references;
- the Codex approval record and every hash it binds.

Eligible recovery creates a separate run, copies only authenticated local
artifacts, reuses the approved analysis, invokes neither Codex retrieval nor
Codex analysis, and pauses at the normal validated-content-diff approval gate.
The source run remains unchanged. A missing approval record or any hash mismatch
requires a new run.
