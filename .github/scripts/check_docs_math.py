#!/usr/bin/env python3
"""Reject Markdown math macros that the repository viewer cannot render."""

from __future__ import annotations

import subprocess
from pathlib import Path


DISALLOWED_MACROS = (r"\operatorname",)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def tracked_markdown(root: Path) -> list[Path]:
    output = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.md"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return [
        root / encoded.decode("utf-8")
        for encoded in output.split(b"\0")
        if encoded
    ]


def main() -> int:
    root = project_root()
    violations: list[str] = []
    for path in tracked_markdown(root):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            for macro in DISALLOWED_MACROS:
                if macro in line:
                    relative_path = path.relative_to(root)
                    violations.append(f"{relative_path}:{line_number}: {macro}")

    if violations:
        details = "\n".join(violations)
        raise SystemExit(
            "unsupported Markdown math macros found:\n"
            f"{details}\n"
            "Use a renderer-safe form such as \\mathrm{...}."
        )

    print("documentation math uses renderer-safe macros")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
