from __future__ import annotations

import zipfile
from pathlib import Path

from resume_tailor.docx_extract import extract_resume
from resume_tailor.docx_render import export_and_validate_pdf, render_tailored_docx
from resume_tailor.utilities import sha256_file


def _relationship_xml(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        return archive.read("word/_rels/document.xml.rels")


def test_docx_source_immutability_formatting_and_hyperlinks(
    master_resume: Path,
    tmp_path: Path,
) -> None:
    before_hash = sha256_file(master_resume)
    extracted, _ = extract_resume(master_resume)
    output = tmp_path / "tailored copy.docx"
    render_tailored_docx(
        source_path=master_resume,
        destination_path=output,
        tailored_content=extracted["content"],
        expected_source_hash=before_hash,
    )
    assert sha256_file(master_resume) == before_hash
    assert output.is_file()
    output_extracted, _ = extract_resume(output)
    assert output_extracted["content"] == extracted["content"]
    assert output_extracted["document"]["hyperlinks"] == extracted["document"]["hyperlinks"]
    assert _relationship_xml(output) == _relationship_xml(master_resume)


def test_one_page_pdf_export_and_preview(master_resume: Path, tmp_path: Path) -> None:
    extracted, _ = extract_resume(master_resume)
    pdf = tmp_path / "resume.pdf"
    preview = tmp_path / "preview.png"
    text = export_and_validate_pdf(
        docx_path=master_resume,
        pdf_path=pdf,
        preview_path=preview,
        working_directory=tmp_path / "work",
        required_text=[
            "SAMPLE CANDIDATE",
            "OBJECTIVE / SUMMARY",
            "EXPERIENCE",
            "sample.candidate@example.com",
        ],
    )
    assert pdf.is_file()
    assert preview.is_file()
    assert "SAMPLE CANDIDATE" in text
