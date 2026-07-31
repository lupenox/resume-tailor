# Résumé template setup

The public repository intentionally contains only `sample_resume.docx`, a fully
synthetic fixture used by the test suite. It never includes the project owner's
master résumé or any tailored output.

Before a local real installation, place your own compatible document at:

```text
template/master_resume.docx
```

The current mapper is intentionally strict. Start by running the synthetic test
suite, then adapt the extractor/renderer and its tests if your document structure
differs from the documented sample. Do not commit your personal master résumé;
the repository `.gitignore` excludes it and all other DOCX/PDF files except the
synthetic sample.

For a disposable synthetic installation check only:

```bash
cp template/sample_resume.docx template/master_resume.docx
./install.sh
```

Remove that copied synthetic master afterward if it is no longer needed.
