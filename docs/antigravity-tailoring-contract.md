# Antigravity post-approval tailoring contract

## Incident summary

A preserved post-approval run supplied every required authenticated input but
received a generic planning response instead of tailored content. The launch
combined a generic plan-mode flag, planning language, and a response schema that
allowed a waiting state. This was a contract conflict, not missing résumé or job
information. This document intentionally contains no private content, raw
provider transcript, prompt, or hidden reasoning.

## Local preflight

Before any tailoring provider launch, local code verifies:

- the confirmed company, role, and job-description presence;
- the immutable extracted content and résumé source catalog;
- the deterministic job-requirement catalog;
- the locally resolved approved analysis and absence of unanswered questions;
- each locally assigned approved edit ID, editable target, exact existing text,
  and nonempty evidence-source references;
- the persisted approval record and every bound artifact hash.

A preflight failure is local and explicitly records that no provider request
was launched.

## Response contract

Tailoring is a bounded execution step and has three statuses:

- `complete` returns the full structured content;
- `cannot_apply` identifies one locally assigned approved edit ID and a bounded
  reason code;
- `technical_failure` identifies a bounded execution reason code.

There is no factual-discovery or generic waiting status. Antigravity must apply
only approved edits, omit unsupported requirements, and never request unlisted
skills, experience, metrics, credentials, or accomplishments. Provider-authored
messages are schema-bounded, HTML-escaped when rendered, and omitted from
exceptions and collapsed technical diagnostics.

The local adapter rejects legacy waiting responses, unknown edit IDs, malformed
statuses, and incidental prose without weakening downstream factual-integrity
validation. A valid structured content payload is not reclassified merely
because an outer legacy wrapper contains incidental status text.

## Authenticated step-6 recovery

Eligible recovery creates a new isolated run, reuses the confirmed job,
requirement catalog, extraction, resolved analysis, transport schema, and
explicit approval record, and invokes neither LinkedIn nor Codex. The original
failed run remains unchanged. Recovery stops at the normal validated
content-diff approval gate; it is never automatic. Any missing record, changed
hash, invalid source reference, or unauthenticated input requires a new run.
