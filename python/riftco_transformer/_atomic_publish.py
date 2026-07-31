"""Dependency-free atomic publication helpers shared by pipeline stages."""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path
import sys


def publish_directory_no_replace(
    source: str | Path,
    destination: str | Path,
) -> None:
    """Atomically rename ``source`` only when ``destination`` is absent.

    Plain POSIX ``rename`` may replace an existing empty directory, so it is
    not a safe publication primitive. This function uses the platform's
    exclusive rename operation when Python does not expose one directly and
    fails closed on platforms without a known atomic no-replace contract.
    """

    source_path = Path(source)
    destination_path = Path(destination)
    if os.name == "nt":
        try:
            os.rename(source_path, destination_path)
        except OSError as error:
            if destination_path.exists():
                raise FileExistsError(
                    errno.EEXIST,
                    "publication destination already exists",
                    destination_path,
                ) from error
            raise
        return

    if sys.platform == "darwin":
        _darwin_rename_no_replace(source_path, destination_path)
        return

    if sys.platform.startswith("linux"):
        _linux_rename_no_replace(source_path, destination_path)
        return

    raise OSError(
        errno.ENOTSUP,
        "atomic no-replace directory publication is unavailable "
        f"on {sys.platform}",
        destination_path,
    )


def _darwin_rename_no_replace(
    source: Path,
    destination: Path,
) -> None:
    at_fdcwd = -2
    rename_exclusive = 0x00000004
    library = ctypes.CDLL(None, use_errno=True)
    try:
        rename = library.renameatx_np
    except AttributeError as error:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory publication is unavailable",
            destination,
        ) from error
    rename.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    rename.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = rename(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_exclusive,
    )
    if result != 0:
        _raise_publish_error(ctypes.get_errno(), destination)


def _linux_rename_no_replace(
    source: Path,
    destination: Path,
) -> None:
    at_fdcwd = -100
    rename_no_replace = 0x00000001
    library = ctypes.CDLL(None, use_errno=True)
    try:
        rename = library.renameat2
    except AttributeError as error:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory publication is unavailable",
            destination,
        ) from error
    rename.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    rename.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = rename(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_no_replace,
    )
    if result != 0:
        _raise_publish_error(ctypes.get_errno(), destination)


def _raise_publish_error(
    error_number: int,
    destination: Path,
) -> None:
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            "publication destination already exists",
            destination,
        )
    unsupported_errors = {
        errno.EINVAL,
        errno.ENOSYS,
        errno.ENOTSUP,
        errno.EOPNOTSUPP,
    }
    if error_number in unsupported_errors:
        raise OSError(
            error_number,
            "filesystem does not support atomic no-replace "
            "directory publication",
            destination,
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        destination,
    )


__all__ = ["publish_directory_no_replace"]
