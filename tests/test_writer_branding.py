from pathlib import Path


ACTIVE_WRITER_FILES = (
    "resume_tailor/backend/utils/utilities.py",
    "resume_tailor/ui.py",
    "resume_tailor/headless_render.py",
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
    assert "qwen" not in combined.casefold()
