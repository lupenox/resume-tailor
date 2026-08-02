from pathlib import Path

path = Path(".github/scripts/apply_opus_audit_hardening.py")
text = path.read_text(encoding="utf-8")
old = '''    catalog_sha256 = canonical_digest(catalog)''',
)


tests = Path("tests/test_gemma_patch_architecture.py")
'''
new = '''    catalog_sha256 = canonical_digest(catalog)''',
    expected=2,
)


tests = Path("tests/test_gemma_patch_architecture.py")
'''
if text.count(old) != 1:
    raise SystemExit("Could not locate the unique Ollama catalog replacement tail")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
