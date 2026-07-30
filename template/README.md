# Synthetic résumé fixture

`sample_resume.docx` is generated from scratch by
`tools/build_synthetic_resume.py`. Every name, organization, link, date,
credential, and achievement in it is fictional.
The package is created with the project’s MIT-licensed `python-docx`
dependency; generic thumbnail and custom-XML parts are removed after generation.

The fixture preserves the structural constraints used by Resume Tailor:
one page, 32 paragraphs, six hyperlinks, fixed section order, and stable project
bullet counts. It contains no material copied from a real résumé.

Regenerate it after installing the Python dependencies:

```bash
.venv/bin/python tools/build_synthetic_resume.py
```

Do not replace this tracked fixture with a real résumé. Supply private documents
at runtime and keep them outside the repository.
