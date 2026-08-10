from __future__ import annotations

import os
import site
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest


SCHEMA_NAMES = (
    "antigravity_revision.schema.json",
    "codex_analysis.openai.schema.json",
    "codex_analysis.schema.json",
    "final_qa.schema.json",
    "final_qa_provider.openai.schema.json",
    "final_qa_provider.schema.json",
    "github_repository_catalog.schema.json",
    "github_repository_evidence_requests.schema.json",
    "github_repository_ranking.schema.json",
    "github_repository_selection.schema.json",
    "linkedin_job.schema.json",
    "ollama_revision_patch.schema.json",
    "ollama_tailoring_patch.schema.json",
    "tailored_resume.schema.json",
)


@pytest.fixture(scope="session")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    repository_root = Path(__file__).resolve().parents[1]
    output_directory = tmp_path_factory.mktemp("wheel-build")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(output_directory),
        ],
        cwd=repository_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = tuple(output_directory.glob("resume_tailor-*.whl"))
    assert len(wheels) == 1
    return repository_root, wheels[0]


def test_wheel_contains_every_python_module_and_schema(
    built_wheel: tuple[Path, Path],
) -> None:
    repository_root, wheel = built_wheel
    expected_python_files = {
        path.relative_to(repository_root).as_posix()
        for path in (repository_root / "resume_tailor").rglob("*.py")
    }
    expected_schema_files = {
        f"resume_tailor/schemas/{name}" for name in SCHEMA_NAMES
    }

    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())

    assert expected_python_files <= members
    assert expected_schema_files <= members


def test_schemas_load_directly_from_built_wheel_without_checkout(
    built_wheel: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    repository_root, wheel = built_wheel
    script = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path

        wheel = Path(sys.argv[1]).resolve()
        repository = Path(sys.argv[2]).resolve()
        sys.path.insert(0, str(wheel))

        import resume_tailor
        from resume_tailor.backend.utils.schemas import (
            codex_transport_schema_path,
            load_schema,
            schema_path,
        )

        package_file = Path(resume_tailor.__file__).resolve()
        assert repository not in package_file.parents
        for name in {SCHEMA_NAMES!r}:
            assert isinstance(load_schema(name), dict)
            assert schema_path(name).is_file()
        assert codex_transport_schema_path(
            "codex_analysis.schema.json"
        ).name == "codex_analysis.openai.schema.json"
        assert codex_transport_schema_path(
            "final_qa_provider.schema.json"
        ).name == "final_qa_provider.openai.schema.json"
        """
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(wheel),
            str(repository_root),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_wheel_installs_with_resources_and_console_scripts_outside_checkout(
    built_wheel: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    repository_root, wheel = built_wheel
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    virtual_environment = tmp_path / "wheel-venv"
    create = subprocess.run(
        [
            sys.executable,
            "-m",
            "venv",
            "--system-site-packages",
            str(virtual_environment),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert create.returncode == 0, create.stdout + create.stderr

    scripts = virtual_environment / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    install = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            str(wheel),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert install.returncode == 0, install.stdout + install.stderr

    child_site_result = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import site; print(site.getsitepackages()[0])",
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert child_site_result.returncode == 0, (
        child_site_result.stdout + child_site_result.stderr
    )
    child_site = Path(child_site_result.stdout.strip())
    dependency_sites = [
        Path(path).resolve() for path in site.getsitepackages() if Path(path).is_dir()
    ]
    assert dependency_sites
    (child_site / "resume-tailor-test-dependencies.pth").write_text(
        "".join(f"{path}\n" for path in dependency_sites),
        encoding="utf-8",
    )

    outside_checkout = tmp_path / "outside-checkout"
    outside_checkout.mkdir()
    import_script = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path

        repository = Path(sys.argv[1]).resolve()
        environment = Path(sys.prefix).resolve()

        import resume_tailor
        import resume_tailor.application as application
        from resume_tailor.backend.utils.schemas import load_schema, schema_path

        package_file = Path(resume_tailor.__file__).resolve()
        application_file = Path(application.__file__).resolve()
        assert environment in package_file.parents
        assert environment in application_file.parents
        assert repository not in package_file.parents
        assert repository not in application_file.parents
        for name in {SCHEMA_NAMES!r}:
            assert isinstance(load_schema(name), dict)
            assert schema_path(name).is_file()
        """
    )
    imported = subprocess.run(
        [str(python), "-I", "-c", import_script, str(repository_root)],
        cwd=outside_checkout,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert imported.returncode == 0, imported.stdout + imported.stderr

    for command in ("tailor-resume", "tailor-resume-ui"):
        result = subprocess.run(
            [str(scripts / command), "--help"],
            cwd=outside_checkout,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert f"usage: {command}" in result.stdout
