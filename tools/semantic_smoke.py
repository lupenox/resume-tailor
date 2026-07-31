#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from resume_tailor.smoke import prepare_smoke_inputs, run_semantic_smoke
from resume_tailor.utilities import ResumeTailorError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a hash-reported semantic smoke test. It is synthetic-only "
            "and dry-run-only unless explicit options authorize more."
        )
    )
    parser.add_argument(
        "--execute-provider",
        action="store_true",
        help="launch exactly one Codex analysis after provenance reporting",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--job-file", type=Path)
    parser.add_argument(
        "--allow-real-inputs",
        action="store_true",
        help="permit explicitly authorized non-synthetic inputs",
    )
    parser.add_argument(
        "--authorization-reference",
        help="non-sensitive reference confirming separate real-input authorization",
    )
    parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inputs = prepare_smoke_inputs(
            repository_root=REPOSITORY_ROOT,
            resume_path=args.resume,
            job_file=args.job_file,
            allow_real_inputs=args.allow_real_inputs,
            authorization_reference=args.authorization_reference,
        )
        print(
            json.dumps(
                {"event": "input-provenance-preflight", **inputs.provenance()},
                sort_keys=True,
            ),
            flush=True,
        )
        if not args.execute_provider:
            print(
                json.dumps(
                    {
                        "provider_launch": "disabled",
                        "next_step": "requires explicit --execute-provider authorization",
                    },
                    sort_keys=True,
                )
            )
            return 0

        with tempfile.TemporaryDirectory(
            prefix="resume-tailor-semantic-smoke-"
        ) as temporary:
            workspace = Path(temporary)
            workspace.chmod(0o700)
            result = run_semantic_smoke(
                inputs,
                run_directory=workspace,
                timeout_seconds=args.timeout_seconds,
                provenance_reporter=lambda report: print(
                    json.dumps(
                        {"event": "provider-launch-provenance", **report},
                        sort_keys=True,
                    ),
                    flush=True,
                ),
            )
            print(json.dumps(result, sort_keys=True))
        return 0
    except ResumeTailorError as exc:
        print(f"semantic-smoke: {exc}", file=sys.stderr)
        return int(exc.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
