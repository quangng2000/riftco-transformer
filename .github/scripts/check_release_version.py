#!/usr/bin/env python3
"""Validate package, CMake, and optional release-tag versions."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from typing import NoReturn


SEMANTIC_VERSION = re.compile(
    r"(?P<version>0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)"
)


def fail(message: str) -> NoReturn:
    raise SystemExit(message)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cmake_version(cmake_lists: str) -> str:
    match = re.search(
        r"project\(\s*transformer_lab\s+VERSION\s+"
        r"([0-9]+\.[0-9]+\.[0-9]+)",
        cmake_lists,
        flags=re.MULTILINE,
    )
    if match is None:
        fail("could not read the transformer_lab version from CMakeLists.txt")
    return match.group(1)


def main(arguments: list[str]) -> int:
    if len(arguments) > 1:
        fail("usage: check_release_version.py [vMAJOR.MINOR.PATCH]")

    root = project_root()
    with (root / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)
    package_version = metadata["project"]["version"]

    if SEMANTIC_VERSION.fullmatch(package_version) is None:
        fail(f"package version is not MAJOR.MINOR.PATCH: {package_version}")

    native_version = cmake_version(
        (root / "CMakeLists.txt").read_text(encoding="utf-8")
    )
    if native_version != package_version:
        fail(
            "CMake and Python versions differ: "
            f"{native_version} != {package_version}"
        )

    if arguments:
        tag = arguments[0]
        match = re.fullmatch(r"v([0-9]+\.[0-9]+\.[0-9]+)", tag)
        if match is None:
            fail(f"release tag must have vMAJOR.MINOR.PATCH form: {tag}")
        if match.group(1) != package_version:
            fail(
                f"release tag {tag} does not match package "
                f"version {package_version}"
            )

        license_path = root / "LICENSE"
        project_metadata = metadata["project"]
        if not license_path.is_file():
            fail("a LICENSE file is required before publishing a release")
        if not (
            project_metadata.get("license")
            or project_metadata.get("license-files")
        ):
            fail(
                "pyproject.toml license metadata is required before "
                "publishing a release"
            )

    print(f"release metadata is consistent at version {package_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
