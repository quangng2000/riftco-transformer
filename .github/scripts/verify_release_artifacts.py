#!/usr/bin/env python3
"""Verify binary-wheel coverage and source-distribution completeness."""

from __future__ import annotations

import argparse
import hashlib
import re
import tarfile
import tomllib
import zipfile
from email.message import Message
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


def normalized_distribution_name(project_name: str) -> str:
    return re.sub(r"[-_.]+", "_", project_name).lower()


def verify_package_metadata(
    metadata: Message,
    artifact_name: str,
    project_name: str,
    version: str,
) -> None:
    if metadata["Name"] != project_name:
        fail(
            f"{artifact_name} declares an unexpected project name: "
            f"{metadata['Name']}"
        )
    if metadata["Version"] != version:
        fail(
            f"{artifact_name} declares an unexpected version: "
            f"{metadata['Version']}"
        )
    if metadata["License-Expression"] != EXPECTED_LICENSE:
        fail(
            f"{artifact_name} does not declare License-Expression: "
            f"{EXPECTED_LICENSE}"
        )
    if EXPECTED_LICENSE_FILE not in metadata.get_all("License-File", []):
        fail(f"{artifact_name} does not declare its packaged license file")
    if metadata.get_all("Requires-Dist"):
        fail(f"{artifact_name} unexpectedly declares a runtime dependency")


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
        filename = "libriftco_transformer_c.so"
    elif family.startswith("macos-"):
        filename = "libriftco_transformer_c.dylib"
    else:
        filename = "riftco_transformer_c.dll"
    return f"riftco_transformer/.libs/{filename}"


def verify_wheel(
    wheel: Path,
    project_name: str,
    version: str,
    license_contents: bytes,
) -> str:
    distribution_name = normalized_distribution_name(project_name)
    prefix = f"{distribution_name}-{version}-py3-none-"
    if not wheel.name.startswith(prefix) or not wheel.name.endswith(".whl"):
        fail(f"wheel has an unexpected name or Python ABI tag: {wheel.name}")

    platform_tag = wheel.name[len(prefix) : -len(".whl")]
    family = platform_family(platform_tag)
    dist_info = f"{distribution_name}-{version}.dist-info"
    wheel_metadata_name = f"{dist_info}/WHEEL"
    package_metadata_name = f"{dist_info}/METADATA"
    license_name = f"{dist_info}/licenses/{EXPECTED_LICENSE_FILE}"

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        legacy_package = ("transformer" + "_" + "lab") + "/"
        if any(name.startswith(legacy_package) for name in names):
            fail(f"{wheel.name} contains the removed legacy Python package")
        if "riftco_transformer/__init__.py" not in names:
            fail(f"{wheel.name} is missing the canonical Python package")
        native_entries = [
            name
            for name in names
            if name.startswith("riftco_transformer/.libs/")
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
        if wheel_metadata_names != [wheel_metadata_name] or (
            package_metadata_names != [package_metadata_name]
        ):
            fail(f"{wheel.name} has malformed distribution metadata")

        license_names = [
            name
            for name in names
            if name.endswith(f".dist-info/licenses/{EXPECTED_LICENSE_FILE}")
        ]
        if license_names != [license_name]:
            fail(f"{wheel.name} must contain exactly one packaged license")
        if normalize_text(archive.read(license_name)) != normalize_text(
            license_contents
        ):
            fail(f"{wheel.name} contains a license that differs from the source")

        wheel_metadata = archive.read(wheel_metadata_name).decode("utf-8")
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
            archive.read(package_metadata_name)
        )
        verify_package_metadata(
            package_metadata,
            wheel.name,
            project_name,
            version,
        )

    return family


def verify_sdist(
    sdist: Path,
    project_name: str,
    version: str,
    license_contents: bytes,
) -> None:
    distribution_name = normalized_distribution_name(project_name)
    expected_name = f"{distribution_name}-{version}.tar.gz"
    if sdist.name != expected_name:
        fail(f"unexpected source-distribution name: {sdist.name}")

    required_suffixes = {
        "/CMakeLists.txt",
        f"/{EXPECTED_LICENSE_FILE}",
        "/PKG-INFO",
        "/pyproject.toml",
        "/include/riftco_transformer/core/quantized_weight.hpp",
        "/include/riftco_transformer/nn/quantized_linear.hpp",
        "/src/c_api.cpp",
        "/src/core/autograd/checkpoint.cpp",
        "/src/core/autograd/custom_gradient.cpp",
        "/src/core/autograd/detail/node.hpp",
        "/src/core/autograd/graph.cpp",
        "/src/core/autograd/operations.cpp",
        "/src/core/backend/adapters/cpu/adapter.cpp",
        "/src/core/backend/adapters/cuda/adapter.cu",
        "/src/core/backend/adapters/cuda/stub.cpp",
        "/src/core/backend/adapters/metal/adapter.mm",
        "/src/core/backend/adapters/metal/runtime.mm",
        "/src/core/backend/adapters/metal/stub.cpp",
        "/src/core/backend/attention/cuda/common.cuh",
        "/src/core/backend/attention/cuda/flash_causal.cu",
        "/src/core/backend/attention/cuda/launch.hpp",
        "/src/core/backend/attention/cuda/materialized_causal.cu",
        "/src/core/backend/attention/cuda/paged_decode.cu",
        "/src/core/backend/nn/capability.hpp",
        "/src/core/backend/nn/contracts.hpp",
        "/src/core/backend/nn/dispatch.cpp",
        "/src/core/backend/nn/dispatch.hpp",
        "/src/core/backend/nn/quantized_linear/capability.hpp",
        "/src/core/backend/nn/quantized_linear/contracts.hpp",
        "/src/core/backend/nn/quantized_linear/dispatch.cpp",
        "/src/core/backend/nn/quantized_linear/dispatch.hpp",
        "/src/core/backend/nn/quantized_linear/storage.hpp",
        "/src/core/backend/nn/quantized_linear/cuda/launch.hpp",
        "/src/core/backend/nn/quantized_linear/cuda/operations.cu",
        "/src/core/backend/nn/quantized_linear/cuda/storage.cu",
        "/src/core/backend/nn/quantized_linear/reference/operations.cpp",
        "/src/core/backend/nn/quantized_linear/reference/operations.hpp",
        "/src/core/backend/nn/quantized_linear/metal/kernels.hpp",
        "/src/core/backend/nn/quantized_linear/metal/launch.hpp",
        "/src/core/backend/nn/quantized_linear/metal/runtime.mm",
        "/src/core/backend/nn/quantized_linear/tpu/launch.hpp",
        "/src/core/backend/nn/quantized_linear/tpu/runtime.cpp",
        "/src/core/backend/nn/reference/operations.cpp",
        "/src/core/backend/nn/reference/operations.hpp",
        "/src/core/backend/nn/cuda/common.cuh",
        "/src/core/backend/nn/cuda/elementwise.cu",
        "/src/core/backend/nn/cuda/indexing.cu",
        "/src/core/backend/nn/cuda/launch.hpp",
        "/src/core/backend/nn/cuda/layout.cu",
        "/src/core/backend/nn/cuda/loss.cu",
        "/src/core/backend/nn/cuda/normalization.cu",
        "/src/core/backend/nn/cuda/reduction.cu",
        "/src/core/backend/nn/cuda/softmax.cu",
        "/src/core/backend/optim/adam/capability.hpp",
        "/src/core/backend/optim/adam/contracts.hpp",
        "/src/core/backend/optim/adam/dispatch.cpp",
        "/src/core/backend/optim/adam/dispatch.hpp",
        "/src/core/backend/optim/adam/reference/update.cpp",
        "/src/core/backend/optim/adam/reference/update.hpp",
        "/src/core/quantization/nf4.cpp",
        "/src/core/quantization/nf4.hpp",
        "/src/core/quantization/quantized_weight.cpp",
        "/src/core/backend/optim/adam/cuda/launch.hpp",
        "/src/core/backend/optim/adam/cuda/update.cu",
        "/src/core/backend/attention/metal/flash_causal_kernels.hpp",
        "/src/core/backend/attention/metal/launch.hpp",
        "/src/core/backend/attention/metal/materialized_causal_kernels.hpp",
        "/src/core/backend/attention/metal/paged_decode_kernels.hpp",
        "/src/core/backend/nn/metal/kernels.hpp",
        "/src/core/backend/nn/metal/launch.hpp",
        "/src/core/backend/optim/adam/metal/diagnostics.hpp",
        "/src/core/backend/optim/adam/metal/kernels.hpp",
        "/src/core/backend/optim/adam/metal/launch.hpp",
        "/src/core/backend/adapters/tpu/adapter.cpp",
        "/src/core/backend/adapters/tpu/compile_options.hpp",
        "/src/core/backend/adapters/tpu/runtime.cpp",
        "/src/core/backend/adapters/tpu/runtime.hpp",
        "/src/core/backend/adapters/tpu/stub.cpp",
        "/src/core/backend/attention/tpu/common.hpp",
        "/src/core/backend/attention/tpu/materialized_causal.cpp",
        "/src/core/backend/attention/tpu/materialized_causal.hpp",
        "/src/core/backend/attention/tpu/paged_decode.cpp",
        "/src/core/backend/attention/tpu/paged_decode.hpp",
        "/src/core/backend/unavailable_adapter.hpp",
        "/src/core/tensor/detail/validation.cpp",
        "/src/core/tensor/detail/validation.hpp",
        "/src/core/tensor/elementwise.cpp",
        "/src/core/tensor/indexing.cpp",
        "/src/core/tensor/layout.cpp",
        "/src/core/tensor/layout_ops.cpp",
        "/src/core/tensor/matmul.cpp",
        "/src/core/tensor/reductions.cpp",
        "/src/core/tensor/softmax.cpp",
        "/src/core/tensor/storage.cpp",
        "/src/nn/quantized_linear.cpp",
        "/third_party/pjrt/include/xla/pjrt/c/pjrt_c_api.h",
        "/third_party/pjrt/LICENSE.openxla",
        "/python/riftco_transformer/native/bindings.py",
    }
    archive_root = f"{distribution_name}-{version}"
    license_name = f"{archive_root}/{EXPECTED_LICENSE_FILE}"
    package_metadata_name = f"{archive_root}/PKG-INFO"
    with tarfile.open(sdist, mode="r:gz") as archive:
        names = archive.getnames()
        legacy_source = "/python/" + ("transformer" + "_" + "lab") + "/"
        if any(legacy_source in name for name in names):
            fail("source distribution contains the removed legacy package")
        members = {member.name: member for member in archive.getmembers()}
        license_names = [
            name
            for name in names
            if name.endswith(f"/{EXPECTED_LICENSE_FILE}")
        ]
        if license_names != [license_name]:
            fail("source distribution must contain exactly one top-level license")
        license_stream = archive.extractfile(members[license_name])
        if license_stream is None or normalize_text(
            license_stream.read()
        ) != normalize_text(license_contents):
            fail("source distribution license differs from the source")
        package_metadata_names = [
            name for name in names if name.endswith("/PKG-INFO")
        ]
        if package_metadata_names != [package_metadata_name]:
            fail("source distribution does not contain top-level PKG-INFO")
        package_metadata_stream = archive.extractfile(
            members[package_metadata_name]
        )
        if package_metadata_stream is None:
            fail("source distribution PKG-INFO is unreadable")
        package_metadata = BytesParser(policy=default).parsebytes(
            package_metadata_stream.read()
        )
        verify_package_metadata(
            package_metadata,
            sdist.name,
            project_name,
            version,
        )
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
        project = tomllib.load(stream)["project"]
    project_name = project["name"]
    version = project["version"]
    license_contents = (root / EXPECTED_LICENSE_FILE).read_bytes()

    wheels = sorted(arguments.directory.glob("*.whl"))
    sdists = sorted(arguments.directory.glob("*.tar.gz"))
    if len(wheels) != len(EXPECTED_PLATFORMS):
        fail(f"expected seven platform wheels, found {len(wheels)}")
    if len(sdists) != 1:
        fail(f"expected one source distribution, found {len(sdists)}")

    families = {
        verify_wheel(wheel, project_name, version, license_contents)
        for wheel in wheels
    }
    if families != EXPECTED_PLATFORMS:
        fail(
            "wheel platform coverage differs: "
            f"expected {sorted(EXPECTED_PLATFORMS)}, found {sorted(families)}"
        )
    verify_sdist(sdists[0], project_name, version, license_contents)

    artifacts = [*wheels, *sdists]
    if arguments.write_checksums:
        write_checksums(arguments.directory, artifacts)
    print(f"verified {len(wheels)} wheels and one source distribution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
