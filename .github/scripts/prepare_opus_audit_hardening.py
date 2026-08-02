from pathlib import Path

path = Path('.github/scripts/apply_opus_audit_hardening.py')
text = path.read_text(encoding='utf-8')
old = '''    ''' + "'''" + '''    catalog = approved_edit_catalog(approved_analysis)\n    catalog_sha256 = canonical_digest(catalog)''' + "'''" + ''',
    ''' + "'''" + '''    catalog = approved_edit_catalog(approved_analysis)\n    duplicate_targets = duplicate_catalog_target_ids(catalog)\n    if duplicate_targets:\n        raise TailoringPreflightError(\n            "Local Ollama tailoring preflight failed: the approved edit catalog "\n            f"repeats target source IDs {duplicate_targets}. No writer request was launched."\n        )\n    catalog_sha256 = canonical_digest(catalog)''' + "'''" + ''',
)
'''
new = old[:-2] + '    expected=2,\n)\n'
if text.count(old) != 1:
    raise SystemExit('Could not locate the unique Ollama catalog replacement call')
path.write_text(text.replace(old, new, 1), encoding='utf-8')
