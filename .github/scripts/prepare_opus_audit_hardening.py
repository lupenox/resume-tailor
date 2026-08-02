from pathlib import Path

path = Path(".github/scripts/apply_opus_audit_hardening.py")
text = path.read_text(encoding="utf-8")
old = "    catalog_sha256 = canonical_digest(catalog)''',\n)\n\n\ntests = Path(\"tests/test_gemma_patch_architecture.py\")\n"
new = "    catalog_sha256 = canonical_digest(catalog)''',\n    expected=2,\n)\n\n\ntests = Path(\"tests/test_gemma_patch_architecture.py\")\n"
if text.count(old) != 1:
    raise SystemExit("Could not locate the unique Ollama catalog replacement tail")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
