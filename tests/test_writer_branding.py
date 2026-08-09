from pathlib import Path


ACTIVE_WRITER_FILES = (
    "resume_tailor/backend/utils/utilities.py",
    "resume_tailor/ui/ui.py",
    "resume_tailor/backend/documents/headless_render.py",
    "resume_tailor/templates/run.html",
    "resume_tailor/templates/dashboard.html",
)


def test_active_runtime_files_present_gemma_not_qwen() -> None:
    root = Path(__file__).resolve().parents[1]
    combined = "\n".join(
        (root / relative).read_text(encoding="utf-8")
        for relative in ACTIVE_WRITER_FILES
    )
    assert "Gemma 4 12B" in combined
    assert "Qwen" not in combined
    # Preserve the existing free-form Ollama model compatibility suggestion;
    # it is not active writer branding or an automatic provider selection.
    active_branding = combined.replace('<option value="qwen2.5:7b"></option>', "")
    assert "qwen" not in active_branding.casefold()
