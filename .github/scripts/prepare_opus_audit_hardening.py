from pathlib import Path

path = Path(".github/scripts/apply_opus_audit_hardening.py")
text = path.read_text(encoding="utf-8")
section_start = text.index('ollama_writer = Path("resume_tailor/ollama_writer.py")')
needle = "    catalog_sha256 = canonical_digest(catalog)''',"
first = text.index(needle, section_start)
second = text.index(needle, first + len(needle))
call_close = text.index("\n)\n", second)
text = text[: call_close + 1] + "    expected=2,\n" + text[call_close + 1 :]
path.write_text(text, encoding="utf-8")
