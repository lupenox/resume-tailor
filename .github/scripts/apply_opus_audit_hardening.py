from __future__ import annotations

from pathlib import Path


def replace_exact(path: Path, old: str, new: str, *, expected: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected:
        raise SystemExit(
            f"Expected {expected} occurrence(s) in {path} but found {count}: {old[:120]!r}"
        )
    path.write_text(text.replace(old, new, expected), encoding="utf-8")


patch_engine = Path("resume_tailor/patch_engine.py")
replace_exact(
    patch_engine,
    "import re\nfrom dataclasses import dataclass",
    "import re\nimport unicodedata\nfrom dataclasses import dataclass",
)
replace_exact(
    patch_engine,
    '''def canonical_digest(payload: Any) -> str:\n    """Compute deterministic SHA-256 over canonical JSON representation."""\n    canonical_bytes = json.dumps(\n        payload,\n        ensure_ascii=False,\n        sort_keys=True,\n        separators=(",", ":"),\n    ).encode("utf-8")\n    return hashlib.sha256(canonical_bytes).hexdigest()\n\n\nclass TargetResolutionError(Exception):''',
    '''def canonical_digest(payload: Any) -> str:\n    """Compute deterministic SHA-256 over canonical JSON representation."""\n    canonical_bytes = json.dumps(\n        payload,\n        ensure_ascii=False,\n        sort_keys=True,\n        separators=(",", ":"),\n    ).encode("utf-8")\n    return hashlib.sha256(canonical_bytes).hexdigest()\n\n\ndef duplicate_catalog_target_ids(catalog: list[dict[str, Any]]) -> list[str]:\n    """Return sorted target IDs repeated by more than one approved edit."""\n    counts: dict[str, int] = {}\n    for edit in catalog:\n        target_id = edit.get("target_source_id")\n        if isinstance(target_id, str):\n            counts[target_id] = counts.get(target_id, 0) + 1\n    return sorted(target_id for target_id, count in counts.items() if count > 1)\n\n\ndef _canonical_unicode(value: str) -> str:\n    """Normalize canonically equivalent Unicode sequences for identity checks."""\n    return unicodedata.normalize("NFC", value)\n\n\ndef _normalized_claim_text(value: str) -> str:\n    """Normalize compatibility forms, case, and whitespace for claim matching."""\n    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())\n\n\ndef _contains_forbidden_claim(text: str, claim: str) -> bool:\n    normalized_text_value = _normalized_claim_text(text)\n    normalized_claim = _normalized_claim_text(claim)\n    if not normalized_claim:\n        return False\n    return (\n        re.search(\n            rf"(?<!\\w){re.escape(normalized_claim)}(?!\\w)",\n            normalized_text_value,\n        )\n        is not None\n    )\n\n\nclass TargetResolutionError(Exception):''',
)
replace_exact(
    patch_engine,
    '''    if replacement_text == descriptor.current_mutable_text:\n        raise OllamaTailoringContractError(\n            f"Patch for {edit_id} is a no-op replacement."\n        )''',
    '''    if _canonical_unicode(replacement_text) == _canonical_unicode(\n        descriptor.current_mutable_text\n    ):\n        raise OllamaTailoringContractError(\n            f"Patch for {edit_id} is a no-op replacement."\n        )''',
)
replace_exact(
    patch_engine,
    '''    normalized_replacement = normalized_text(replacement_text)\n    for forbidden in forbidden_claims:\n        if not isinstance(forbidden, str):\n            continue\n        normalized_forbidden = normalized_text(forbidden)\n        if len(normalized_forbidden) >= 8 and normalized_forbidden in normalized_replacement:\n            raise OllamaTailoringContractError(\n                f"Patch for {edit_id} contains a forbidden claim."\n            )''',
    '''    for forbidden in forbidden_claims:\n        if not isinstance(forbidden, str):\n            continue\n        if _contains_forbidden_claim(replacement_text, forbidden):\n            raise OllamaTailoringContractError(\n                f"Patch for {edit_id} contains a forbidden claim."\n            )''',
)
replace_exact(
    patch_engine,
    '''    catalog = approved_edit_catalog(approved_analysis)\n    expected_sha256 = canonical_digest(catalog)''',
    '''    catalog = approved_edit_catalog(approved_analysis)\n    duplicate_targets = duplicate_catalog_target_ids(catalog)\n    if duplicate_targets:\n        raise OllamaTailoringContractError(\n            "The approved edit catalog repeats target source IDs: "\n            f"{duplicate_targets}."\n        )\n    expected_sha256 = canonical_digest(catalog)''',
    expected=1,
)
replace_exact(
    patch_engine,
    '''    catalog_by_target = {\n        edit["target_source_id"]: edit\n        for edit in approved_edit_catalog(approved_analysis)\n    }''',
    '''    approved_catalog = approved_edit_catalog(approved_analysis)\n    duplicate_targets = duplicate_catalog_target_ids(approved_catalog)\n    if duplicate_targets:\n        raise OllamaRevisionContractError(\n            "The approved edit catalog repeats revision target source IDs: "\n            f"{duplicate_targets}."\n        )\n    catalog_by_target = {\n        edit["target_source_id"]: edit for edit in approved_catalog\n    }''',
)

ollama_writer = Path("resume_tailor/ollama_writer.py")
replace_exact(
    ollama_writer,
    '''    TargetResolutionError,\n    authenticated_metrics_for_edit,''',
    '''    TargetResolutionError,\n    authenticated_metrics_for_edit,\n    duplicate_catalog_target_ids,''',
)
replace_exact(
    ollama_writer,
    '''    catalog = approved_edit_catalog(approved_analysis)\n    catalog_sha256 = canonical_digest(catalog)''',
    '''    catalog = approved_edit_catalog(approved_analysis)\n    duplicate_targets = duplicate_catalog_target_ids(catalog)\n    if duplicate_targets:\n        raise TailoringPreflightError(\n            "Local Ollama tailoring preflight failed: the approved edit catalog "\n            f"repeats target source IDs {duplicate_targets}. No writer request was launched."\n        )\n    catalog_sha256 = canonical_digest(catalog)''',
)

tests = Path("tests/test_gemma_patch_architecture.py")
replace_exact(
    tests,
    '''    valid_digest = writer.canonical_digest(catalog)\n\n    # Patch 1 valid, Patch 2 invalid (numeric claim '9999')''',
    '''    valid_digest = writer.canonical_digest(catalog)\n    master_before = copy.deepcopy(extracted["content"])\n\n    # Patch 1 valid, Patch 2 invalid (numeric claim '9999')''',
    expected=1,
)
replace_exact(
    tests,
    '''            extracted_resume=extracted,\n            approved_analysis=analysis,\n        )\n\n\n# 28, 29, 30. Changed-ID set equals approved target set; labels/names/dates/counts unchanged; passes Step 7.''',
    '''            extracted_resume=extracted,\n            approved_analysis=analysis,\n        )\n    assert extracted["content"] == master_before\n\n\n# 28, 29, 30. Changed-ID set equals approved target set; labels/names/dates/counts unchanged; passes Step 7.''',
    expected=1,
)
replace_exact(
    tests,
    '''    assert set(changed) == approved_targets\n\n    # Step 7 validation''',
    '''    assert set(changed) == approved_targets\n\n    original = extracted["content"]\n    assert tailored["education"]["institution"] == original["education"]["institution"]\n    assert tailored["education"]["degree_details"] == original["education"]["degree_details"]\n    assert tailored["education"]["coursework"]["label"] == original["education"]["coursework"]["label"]\n    assert tailored["education"]["certifications"]["label"] == original["education"]["certifications"]["label"]\n    assert [group["label"] for group in tailored["skill_groups"]] == [\n        group["label"] for group in original["skill_groups"]\n    ]\n    assert [project["name"] for project in tailored["projects"]] == [\n        project["name"] for project in original["projects"]\n    ]\n    assert [project["technologies"] for project in tailored["projects"]] == [\n        project["technologies"] for project in original["projects"]\n    ]\n    assert tailored["open_source"]["name"] == original["open_source"]["name"]\n    assert tailored["open_source"]["technologies"] == original["open_source"]["technologies"]\n    assert tailored["experience"]["role"] == original["experience"]["role"]\n    assert tailored["experience"]["employer_location"] == original["experience"]["employer_location"]\n    assert tailored["experience"]["dates"] == original["experience"]["dates"]\n    assert len(tailored["skill_groups"]) == len(original["skill_groups"])\n    assert len(tailored["projects"]) == len(original["projects"])\n    assert [len(project["bullets"]) for project in tailored["projects"]] == [\n        len(project["bullets"]) for project in original["projects"]\n    ]\n    assert len(tailored["experience"]["bullets"]) == len(original["experience"]["bullets"])\n\n    # Step 7 validation''',
    expected=1,
)

append_marker = "# Opus independent-audit follow-up regressions."
text = tests.read_text(encoding="utf-8")
if append_marker in text:
    raise SystemExit("Opus follow-up tests already present")
text = text.rstrip() + '''\n\n\n# Opus independent-audit follow-up regressions.\ndef test_44_empty_catalog_is_an_explicit_atomic_noop(master_resume: Path) -> None:\n    extracted, _job_desc, _reqs, analysis = _setup_synthetic_inputs(master_resume)\n    empty_analysis = copy.deepcopy(analysis)\n    empty_analysis["recommended_edits"] = []\n    payload = {\n        "status": "complete",\n        "message": "No approved edits.",\n        "catalog_sha256": writer.canonical_digest([]),\n        "cannot_apply": None,\n        "technical_failure": None,\n        "patches": [],\n    }\n    tailored = patch_engine.validate_and_apply_patches(\n        payload=payload,\n        master_content=extracted["content"],\n        extracted_resume=extracted,\n        approved_analysis=empty_analysis,\n    )\n    assert tailored == extracted["content"]\n    assert tailored is not extracted["content"]\n\n\ndef _analysis_with_duplicate_summary_target(analysis: dict) -> dict:\n    duplicated = copy.deepcopy(analysis)\n    duplicated["recommended_edits"].append(\n        copy.deepcopy(duplicated["recommended_edits"][0])\n    )\n    return duplicated\n\n\ndef test_45_duplicate_catalog_target_fails_before_writer(\n    master_resume: Path,\n) -> None:\n    extracted, job_desc, reqs, analysis = _setup_synthetic_inputs(master_resume)\n    duplicated = _analysis_with_duplicate_summary_target(analysis)\n    with pytest.raises(writer.TailoringPreflightError, match="repeats target source IDs"):\n        writer.build_ollama_tailoring_prompt(\n            master_content=extracted["content"],\n            extracted_resume=extracted,\n            job_description=job_desc,\n            job_requirements=reqs,\n            approved_analysis=duplicated,\n            company="Synthetic Corp",\n            role="AI Engineer",\n        )\n\n\ndef test_46_duplicate_catalog_target_fails_closed_in_applicator(\n    master_resume: Path,\n) -> None:\n    extracted, _job_desc, _reqs, analysis = _setup_synthetic_inputs(master_resume)\n    duplicated = _analysis_with_duplicate_summary_target(analysis)\n    catalog = writer.approved_edit_catalog(duplicated)\n    payload = _valid_patch_payload(extracted, analysis)\n    payload["catalog_sha256"] = writer.canonical_digest(catalog)\n    payload["patches"].append(\n        {\n            "edit_id": "edit.003",\n            "target_source_id": "professional_summary",\n            "operation": "replace",\n            "replacement_text": "Another summary replacement.",\n        }\n    )\n    with pytest.raises(OllamaTailoringContractError, match="repeats target source IDs"):\n        patch_engine.validate_and_apply_patches(\n            payload=payload,\n            master_content=extracted["content"],\n            extracted_resume=extracted,\n            approved_analysis=duplicated,\n        )\n\n\ndef test_47_short_forbidden_claims_use_token_boundaries(\n    master_resume: Path,\n) -> None:\n    extracted, _job_desc, _reqs, analysis = _setup_synthetic_inputs(master_resume)\n    edit = writer.approved_edit_catalog(analysis)[0]\n    descriptor = patch_engine.resolve_target_descriptor(\n        edit, extracted["content"], extracted\n    )\n    evidence_texts = patch_engine.authorized_evidence_texts_for_edit(\n        edit, descriptor, extracted\n    )\n    with pytest.raises(OllamaTailoringContractError, match="forbidden claim"):\n        patch_engine._validate_replacement_text(\n            edit_id=descriptor.edit_id,\n            descriptor=descriptor,\n            replacement_text="AI engineer building Python workflows.",\n            evidence_texts=evidence_texts,\n            forbidden_claims=["AI"],\n        )\n\n    accepted = patch_engine._validate_replacement_text(\n        edit_id=descriptor.edit_id,\n        descriptor=descriptor,\n        replacement_text="Python training engineer building workflows.",\n        evidence_texts=evidence_texts,\n        forbidden_claims=["AI"],\n    )\n    assert accepted == "Python training engineer building workflows."\n\n\ndef test_48_unicode_equivalent_replacement_is_rejected_as_noop() -> None:\n    descriptor = patch_engine.TargetDescriptor(\n        edit_id="edit.001",\n        target_source_id="professional_summary",\n        operation="replace",\n        kind="plain",\n        label=None,\n        current_mutable_text="Cafe\\u0301 engineer",\n        exact_rendered_existing_text="Cafe\\u0301 engineer",\n        maximum_rendered_characters=100,\n        proposed_text="Café engineer",\n        alignment_rationale="Synthetic",\n        evidence_source_ids=["professional_summary"],\n    )\n    with pytest.raises(OllamaTailoringContractError, match="no-op replacement"):\n        patch_engine._validate_replacement_text(\n            edit_id=descriptor.edit_id,\n            descriptor=descriptor,\n            replacement_text="Café engineer",\n            evidence_texts=[descriptor.exact_rendered_existing_text],\n            forbidden_claims=[],\n        )\n''' + "\n"
tests.write_text(text, encoding="utf-8")
