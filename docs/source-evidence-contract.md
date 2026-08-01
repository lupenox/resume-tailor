# Source-evidence contract

## Scope

This change removes model-transcribed résumé quotations from the analysis trust
boundary. It does not relax factual-integrity checks, alter approval gates, or
change document rendering rules.

## Source catalog

The extractor's existing deterministic `content_id` values become explicit
`source_id` values. Every extracted paragraph is represented by an immutable
source object containing:

- `source_id`
- section context
- block kind
- exact extracted text
- whether the block may be cited as evidence
- whether the block may be targeted by an edit

Section labels and contact/header blocks remain available for document context
but cannot substantiate claims. Locally generated source objects are the only
authority for existing text and displayed evidence.

## Job-requirement catalog

After the job input is explicitly confirmed, local code creates a deterministic
catalog from the structured posting fields when they are available. Each
responsibility, required qualification, preferred qualification, technology or
skill, and AI focus area receives:

- a stable `requirement_id`;
- a category;
- exact locally retained text.

Confirmed plain-text inputs use a deterministic line and heading parser. The
catalog is saved as `job-requirements.json`, bound to the confirmed job-description
hash, included in the retry manifest, and never reconstructed from model-authored
labels.

## Codex analysis output

Codex returns source references, not copied source prose:

- supported requirement mappings contain `requirement_id` and one or more
  `evidence_source_ids`;
- unsupported requirements contain only `requirement_id` values and cannot carry
  evidence;
- recommended edits contain one `target_source_id`, an operation, proposed
  text, and `evidence_source_ids`;
- content-budget guidance targets a source ID while the local extractor remains
  authoritative for the numeric budget.

Local resolution rejects missing, unknown, duplicate, or contextually
inappropriate IDs before any approval request or Antigravity invocation. The
resolved analysis stores exact text copied from the local source catalog and is
the object passed to approval, tailoring, and final QA.

For each run, the application derives a fresh provider transport schema from the
complete canonical schema and both deterministic catalogs. Requirement fields use
an enum containing only this run's requirement IDs. Evidence arrays use an enum
of evidence-eligible source IDs; edit and budget targets use a separate enum of
editable IDs. Required evidence arrays contain at least one item. The generated
schema is checked against the supported Structured Outputs subset, bounded for
size and complexity, written into the isolated run, hashed in run metadata, and
revalidated immediately before Codex starts. There is no unconstrained fallback.

Local cross-field validation remains authoritative after schema validation. It
requires every requirement ID to appear exactly once across the supported and
unsupported collections, rejects omissions, duplicates, conflicts, unknown IDs,
and evidence for unsupported requirements, and forbids questions that invite
unlisted experience. Questions may still surface genuine contradictions within
supplied source text. Résumé edit targets and evidence are validated separately,
so job-analysis wording cannot weaken factual edit protection.

## ATS status

The model cannot set a support boolean or author the displayed ATS label. Local
code derives ATS rows from exact catalog text and assigns one of three states:

1. `present_verbatim`: the exact keyword occurs in an evidence-eligible source
   block.
2. `supported_by_source`: Codex mapped the requirement to one or more valid local
   source blocks, but local exact matching did not establish verbatim presence.
   This is labeled as model-assessed and requires human review.
3. `unsupported`: neither condition holds.

For `present_verbatim`, local normalization is limited to case, Unicode dash
representation, whitespace, and line breaks. It does not remove punctuation,
reorder tokens, join blocks, or perform fuzzy matching. For
`supported_by_source`, local code validates IDs and displays exact cited blocks;
it does not claim to prove semantic equivalence. Source IDs always resolve to
unchanged exact local text.

## Offline failure replay

A preserved schema-valid analysis exposed a defect in the retired label-based
contract: model-assessed semantic ATS mappings were compared as though their
model-authored labels had to occur verbatim. The read-only replay records only
issue codes, field locations, counts, and hashes. No private posting or résumé
content, provider transcript, prompt, or hidden reasoning is stored in tests or
documentation. The retired response is rejected by the new canonical contract;
synthetic fixtures prove the requirement-ID path reaches the normal approval gate.

## Long-running analysis status

Codex analysis reports elapsed time and periodic subprocess-liveness heartbeats.
The UI distinguishes “still running” from “no process detected; validating output,”
states that strong reasoning may take several minutes, and does not show an ETA.
Cancellation and bounded timeout handling terminate the entire subprocess group.

## Retry boundary

Before Codex analysis, the pipeline stores SHA-256 hashes for the source DOCX,
the extracted-resume JSON, the confirmed job-description bytes, and the generated
job-requirement catalog. A retry:

- is available only for a source-evidence failure at the Codex-analysis stage;
- revalidates every stored hash and the current source DOCX before starting;
- reads the preserved job description and extraction without refetching a URL;
- creates a new isolated run and invokes only Codex until the normal analysis
  approval gate is reached;
- cannot reach Antigravity unless the new analysis passes local resolution and
  a human separately approves it.

Legacy URL-mode runs without the new hash manifest are eligible only when the
metadata and extracted source hashes agree with the unchanged current master
and every stored extraction field matches a fresh deterministic read-only
extraction of that master. The independently stored structured job description
must also exactly reproduce the preserved confirmed text. Retry uses the fresh
local extraction and source catalog after that comparison. Other legacy runs
require a new run.

The source run remains read-only throughout retry preparation.

## Semantic smoke authorization

An earlier semantic smoke execution crossed its synthetic-only input boundary and
included protected local artifacts. No private content, raw prompt, provider
transcript, or hidden reasoning is retained in this documentation.

The smoke harness is dry-run-only and synthetic-only by default. It accepts only
the hash-pinned `template/sample_resume.docx` fixture plus built-in synthetic job
text. Before any provider process can start, it rechecks both hashes and emits
content-free provenance. Custom inputs are refused unless both the explicit
real-input option and a separate authorization reference are supplied. The
harness uses a mode-0700 temporary workspace and stops at the analysis-approval
boundary; it never invokes retrieval, tailoring, rendering, installation, Git,
or GitHub.
