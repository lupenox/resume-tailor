# Local Qwen writer contract (historical — superseded)

> **This document describes the prior Qwen 3.5 9B writer configuration, which
> was the default before the Gemma 4 12B migration. It is retained as a
> historical reference only. The active documentation is
> [docs/ollama-writer.md](ollama-writer.md).**

The preserved Step 6 failure regression fixture (see below) is documented
here for historical context. The classification logic it tests remains active
regardless of the current writer model.

---

Previously, Resume Tailor used `resume-tailor-qwen` through Ollama as its
default résumé writer.

## Historical model profile

```text
FROM qwen3.5:9b
PARAMETER num_ctx 32768
PARAMETER temperature 0.2
```

```bash
ollama create resume-tailor-qwen -f Modelfile
ollama show resume-tailor-qwen
```

## Historical context: the Step 6 overflow failure

The budgeting system was introduced because the prior Qwen writer produced a
wrong-root response when a 7,590-token prompt and 1,145 generated tokens
totalled 8,735 against an 8,192-token window. Generation stopped naturally,
no truncation signal was raised, but the context window had already overflowed.
The preserved wrong-root response is pinned as a synthetic regression fixture at
`tests/fixtures/ollama_wrong_root_resume_response.json`. It contains no real
résumé content and asserts the observed root (`header`, `objective_summary`,
`education_certifications`, `technical_skills`, `ai_engineering_projects`,
`open_source_contribution`, `experience`) is classified as
`ollama-transport-schema` and not as malformed JSON or truncation.

See [docs/ollama-writer.md](ollama-writer.md) for the current writer contract,
capability registry, budgeting rules, and failure classification.
