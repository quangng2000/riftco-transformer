#!/usr/bin/env python3
"""Verify binary-wheel coverage and source-distribution completeness."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from typing import NoReturn


EXPECTED_PLATFORMS = {
    "linux-manylinux-x86_64",
    "linux-manylinux-aarch64",
    "linux-musllinux-x86_64",
    "linux-musllinux-aarch64",
    "macos-x86_64",
    "macos-arm64",
    "windows-amd64",
}
EXPECTED_LICENSE = "Apache-2.0"
EXPECTED_LICENSE_FILE = "LICENSE"
FORBIDDEN_SDIST_PARTS = (
    "/data/external/",
    "/data/pretraining/huggingface/",
)


def fail(message: str) -> NoReturn:
    raise SystemExit(message)


def normalize_text(contents: bytes) -> bytes:
    """Normalize platform checkout newlines before comparing text assets."""

    return contents.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def platform_family(platform_tag: str) -> str:
    if "manylinux" in platform_tag and platform_tag.endswith("_x86_64"):
        return "linux-manylinux-x86_64"
    if "manylinux" in platform_tag and platform_tag.endswith("_aarch64"):
        return "linux-manylinux-aarch64"
    if platform_tag == "musllinux_1_2_x86_64":
        return "linux-musllinux-x86_64"
    if platform_tag == "musllinux_1_2_aarch64":
        return "linux-musllinux-aarch64"
    if platform_tag == "macosx_13_0_x86_64":
        return "macos-x86_64"
    if platform_tag == "macosx_13_0_arm64":
        return "macos-arm64"
    if platform_tag == "win_amd64":
        return "windows-amd64"
    fail(f"unexpected wheel platform tag: {platform_tag}")


def expected_library(family: str) -> str:
    if family.startswith("linux-"):
        filename = "libtransformer_lab_c.so"
    elif family.startswith("macos-"):
        filename = "libtransformer_lab_c.dylib"
    else:
        filename = "transformer_lab_c.dll"
    return f"transformer_lab/.libs/{filename}"


def verify_wheel(wheel: Path, version: str, license_contents: bytes) -> str:
    prefix = f"transformer_lab-{version}-py3-none-"
    if not wheel.name.startswith(prefix) or not wheel.name.endswith(".whl"):
        fail(f"wheel has an unexpected name or Python ABI tag: {wheel.name}")

    platform_tag = wheel.name[len(prefix) : -len(".whl")]
    family = platform_family(platform_tag)

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        native_entries = [
            name
            for name in names
            if name.startswith("transformer_lab/.libs/")
            and not name.endswith("/")
        ]
        required_library = expected_library(family)
        if native_entries != [required_library]:
            fail(
                f"{wheel.name} must contain exactly {required_library}; "
                f"found {native_entries}"
            )
        if any("/__pycache__/" in name or name.endswith(".pyc") for name in names):
            fail(f"{wheel.name} contains generated Python bytecode")

        wheel_metadata_names = [
            name for name in names if name.endswith(".dist-info/WHEEL")
        ]
        package_metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        if len(wheel_metadata_names) != 1 or len(package_metadata_names) != 1:
            fail(f"{wheel.name} has malformed distribution metadata")

        license_names = [
            name
            for name in names
            if name.endswith(f".dist-info/licenses/{EXPECTED_LICENSE_FILE}")
        ]
        if len(license_names) != 1:
            fail(f"{wheel.name} must contain exactly one packaged license")
        if normalize_text(archive.read(license_names[0])) != normalize_text(
            license_contents
        ):
            fail(f"{wheel.name} contains a license that differs from the source")

        wheel_metadata = archive.read(wheel_metadata_names[0]).decode("utf-8")
        actual_tags = {
            line.removeprefix("Tag: ")
            for line in wheel_metadata.splitlines()
            if line.startswith("Tag: ")
        }
        expected_tags = {
            f"py3-none-{tag}" for tag in platform_tag.split(".")
        }
        if actual_tags != expected_tags:
            fail(f"{wheel.name} metadata does not match its filename tag")
        package_metadata = BytesParser(policy=default).parsebytes(
            archive.read(package_metadata_names[0])
        )
        if package_metadata["License-Expression"] != EXPECTED_LICENSE:
            fail(
                f"{wheel.name} does not declare License-Expression: "
                f"{EXPECTED_LICENSE}"
            )
        if EXPECTED_LICENSE_FILE not in package_metadata.get_all(
            "License-File", []
        ):
            fail(f"{wheel.name} does not declare its packaged license file")
        if package_metadata.get_all("Requires-Dist"):
            fail(f"{wheel.name} unexpectedly declares a runtime dependency")

    return family


def verify_sdist(sdist: Path, version: str, license_contents: bytes) -> None:
    expected_name = f"transformer_lab-{version}.tar.gz"
    if sdist.name != expected_name:
        fail(f"unexpected source-distribution name: {sdist.name}")

    required_suffixes = {
        "/CMakeLists.txt",
        f"/{EXPECTED_LICENSE_FILE}",
        "/pyproject.toml",
        "/src/c_api.cpp",
        "/python/transformer_lab/native/bindings.py",
    }
    with tarfile.open(sdist, mode="r:gz") as archive:
        names = archive.getnames()
        license_members = [
            member
            for member in archive.getmembers()
            if member.name.endswith(f"/{EXPECTED_LICENSE_FILE}")
        ]
        if len(license_members) != 1:
            fail("source distribution must contain exactly one top-level license")
        license_stream = archive.extractfile(license_members[0])
        if license_stream is None or normalize_text(
            license_stream.read()
        ) != normalize_text(license_contents):
            fail("source distribution license differs from the source")
    missing = {
        suffix
        for suffix in required_suffixes
        if not any(name.endswith(suffix) for name in names)
    }
    if missing:
        fail(f"source distribution is missing required files: {sorted(missing)}")
    if any("/__pycache__/" in name or name.endswith(".pyc") for name in names):
        fail("source distribution contains generated Python bytecode")
    forbidden = [
        name
        for name in names
        if any(part in name for part in FORBIDDEN_SDIST_PARTS)
    ]
    if forbidden:
        fail(f"source distribution contains external training data: {forbidden}")


def write_checksums(directory: Path, artifacts: list[Path]) -> None:
    lines = []
    for artifact in sorted(artifacts, key=lambda path: path.name):
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        lines.append(f"{digest}  {artifact.name}")
    (directory / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--write-checksums", action="store_true")
    arguments = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as stream:
        version = tomllib.load(stream)["project"]["version"]
    license_contents = (root / EXPECTED_LICENSE_FILE).read_bytes()

    wheels = sorted(arguments.directory.glob("*.whl"))
    sdists = sorted(arguments.directory.glob("*.tar.gz"))
    if len(wheels) != len(EXPECTED_PLATFORMS):
        fail(f"expected seven platform wheels, found {len(wheels)}")
    if len(sdists) != 1:
        fail(f"expected one source distribution, found {len(sdists)}")

    families = {
        verify_wheel(wheel, version, license_contents) for wheel in wheels
    }
    if families != EXPECTED_PLATFORMS:
        fail(
            "wheel platform coverage differs: "
            f"expected {sorted(EXPECTED_PLATFORMS)}, found {sorted(families)}"
        )
    verify_sdist(sdists[0], version, license_contents)

    artifacts = [*wheels, *sdists]
    if arguments.write_checksums:
        write_checksums(arguments.directory, artifacts)
    print(f"verified {len(wheels)} wheels and one source distribution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
