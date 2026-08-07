import os
import shutil
import re
from pathlib import Path

# Paths
root_dir = Path("/home/lupenox/Desktop/Main_Folder_For_Everything/Coding Projects/Portfolio_Projects/resume-tailor")
pkg_dir = root_dir / "resume_tailor"
tests_dir = root_dir / "tests"

# Move plan
move_plan = {
    # providers
    "antigravity_response.py": "backend/providers/antigravity_response.py",
    "antigravity_transport.py": "backend/providers/antigravity_transport.py",
    "antigravity_writer.py": "backend/providers/antigravity_writer.py",
    "codex_analysis.py": "backend/providers/codex_analysis.py",
    "gemma_analysis.py": "backend/providers/gemma_analysis.py",
    "grok_analysis.py": "backend/providers/grok_analysis.py",
    "ollama_capabilities.py": "backend/providers/ollama_capabilities.py",
    "ollama_probe.py": "backend/providers/ollama_probe.py",
    "ollama_transport.py": "backend/providers/ollama_transport.py",
    "ollama_writer.py": "backend/providers/ollama_writer.py",

    # jobs
    "apify_job.py": "backend/jobs/apify_job.py",
    "linkedin_job.py": "backend/jobs/linkedin_job.py",
    "job_requirements.py": "backend/jobs/job_requirements.py",
    "job_text.py": "backend/jobs/job_text.py",

    # documents
    "docx_extract.py": "backend/documents/docx_extract.py",
    "docx_render.py": "backend/documents/docx_render.py",
    "headless_render.py": "backend/documents/headless_render.py",

    # engine
    "analysis.py": "backend/engine/analysis.py",
    "character_budget.py": "backend/engine/character_budget.py",
    "evidence.py": "backend/engine/evidence.py",
    "orchestration.py": "backend/engine/orchestration.py",
    "patch_engine.py": "backend/engine/patch_engine.py",
    "qa.py": "backend/engine/qa.py",
    "retry.py": "backend/engine/retry.py",
    "revision.py": "backend/engine/revision.py",
    "structured_patch_compiler.py": "backend/engine/structured_patch_compiler.py",

    # utils
    "analytics.py": "backend/utils/analytics.py",
    "clipboard.py": "backend/utils/clipboard.py",
    "schemas.py": "backend/utils/schemas.py",
    "smoke.py": "backend/utils/smoke.py",
    "utilities.py": "backend/utils/utilities.py",

    # ui
    "cli.py": "ui/cli.py",
    "desktop.py": "ui/desktop.py",
    "ui.py": "ui/ui.py",
    "ui_cli.py": "ui/ui_cli.py",
}

# module mapping maps from old module name to the new parent path
# e.g., "analysis" -> "resume_tailor.backend.engine"
module_parent = {}
for old, new in move_plan.items():
    old_mod = old[:-3]
    new_mod = "resume_tailor." + new[:-3].replace("/", ".")
    # parent path
    parent_path = "resume_tailor." + str(Path(new).parent).replace("/", ".")
    module_parent[old_mod] = parent_path

directories = set(Path(new).parent for new in move_plan.values())

for d in directories:
    dir_path = pkg_dir / d
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "__init__.py").touch()

file_contents = {}
for old, new in move_plan.items():
    with open(pkg_dir / old, "r") as f:
        file_contents[old] = f.read()

def replace_imports(content):
    # from . import __version__ -> from resume_tailor import __version__
    content = re.sub(
        r'^(\s*)from\s+\.\s+import\s+__version__\b',
        r'\1from resume_tailor import __version__',
        content,
        flags=re.MULTILINE
    )

    # from .MODULE import ...
    def repl_from_dot(m):
        mod = m.group(2)
        if mod in module_parent:
            return f"{m.group(1)}from {module_parent[mod]}.{mod} import"
        return m.group(0)
    
    content = re.sub(
        r'^(\s*)from\s+\.([a-zA-Z0-9_]+)\s+import',
        repl_from_dot,
        content,
        flags=re.MULTILINE
    )

    # from resume_tailor.MODULE import ...
    def repl_from_pkg(m):
        mod = m.group(2)
        if mod in module_parent:
            return f"{m.group(1)}from {module_parent[mod]}.{mod} import"
        return m.group(0)

    content = re.sub(
        r'^(\s*)from\s+resume_tailor\.([a-zA-Z0-9_]+)\s+import',
        repl_from_pkg,
        content,
        flags=re.MULTILINE
    )

    # import resume_tailor.MODULE
    def repl_import_pkg(m):
        mod = m.group(2)
        if mod in module_parent:
            return f"{m.group(1)}import {module_parent[mod]}.{mod}"
        return m.group(0)

    content = re.sub(
        r'^(\s*)import\s+resume_tailor\.([a-zA-Z0-9_]+)\b',
        repl_import_pkg,
        content,
        flags=re.MULTILINE
    )
    
    # from resume_tailor import MODULE
    def repl_from_root(m):
        mod = m.group(2)
        if mod in module_parent:
            return f"{m.group(1)}from {module_parent[mod]} import {mod}"
        return m.group(0)
        
    content = re.sub(
        r'^(\s*)from\s+resume_tailor\s+import\s+([a-zA-Z0-9_]+)\b',
        repl_from_root,
        content,
        flags=re.MULTILINE
    )

    return content

for old, new in move_plan.items():
    content = file_contents[old]
    new_content = replace_imports(content)
    
    # Special fixes for __file__ relative paths
    if old == "schemas.py":
        new_content = new_content.replace(
            'Path(__file__).resolve().parent.parent / "schemas"',
            'Path(__file__).resolve().parents[3] / "schemas"'
        )
    if old == "ui.py":
        new_content = new_content.replace(
            'Path(__file__).resolve().parents[1]',
            'Path(__file__).resolve().parents[2]'
        )
        
    with open(pkg_dir / new, "w") as f:
        f.write(new_content)
    (pkg_dir / old).unlink()

for test_file in tests_dir.rglob("*.py"):
    with open(test_file, "r") as f:
        content = f.read()
    new_content = replace_imports(content)
    if new_content != content:
        with open(test_file, "w") as f:
            f.write(new_content)

print("Refactoring complete.")
