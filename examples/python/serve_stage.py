"""Stage 3: serve a post-trained artifact over a local JSON API."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from transformer_lab.serving import (
    ServingConfig,
    create_http_server,
)


DEFAULT_BUNDLE = Path("results/stages/tiny_post_trained.tlab")
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})


def positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def port_number(value: str) -> int:
    result = positive_integer(value)
    if result > 65535:
        raise argparse.ArgumentTypeError("must be at most 65535")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3: serve a .tlab artifact over dependency-free local HTTP."
        )
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=DEFAULT_BUNDLE,
        help=f"Model artifact to serve (default: {DEFAULT_BUNDLE}).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1, local machine only).",
    )
    parser.add_argument(
        "--port",
        type=port_number,
        default=8000,
        help="TCP port (default: 8000).",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "metal"),
        default="auto",
        help="Execution backend; auto prefers Metal when available.",
    )
    parser.add_argument(
        "--maximum-new-tokens",
        type=positive_integer,
        default=64,
        help="Per-request generation limit (default: 64).",
    )
    parser.add_argument(
        "--maximum-request-bytes",
        type=positive_integer,
        default=1 << 20,
        help="Maximum JSON request size (default: 1048576).",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Required when binding to a non-loopback host.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    if (
        arguments.host.lower() not in LOOPBACK_HOSTS
        and not arguments.allow_remote
    ):
        parser.error(
            "a non-loopback --host requires explicit --allow-remote"
        )

    server = None
    try:
        server = create_http_server(
            arguments.bundle,
            host=arguments.host,
            port=arguments.port,
            config=ServingConfig(
                backend=arguments.backend,
                maximum_new_tokens=arguments.maximum_new_tokens,
                maximum_request_bytes=arguments.maximum_request_bytes,
            ),
        )
        bound_host, bound_port = server.server_address[:2]
        service = server.model_service
        print(
            "[serving] "
            f"bundle={arguments.bundle} "
            f"stage={service.stage} "
            f"artifact_id={service.artifact_id} "
            f"backend={service.backend}"
        )
        print(f"[serving] chat=http://{bound_host}:{bound_port}/")
        print(
            "[serving] GET / | GET /health | POST /v1/generate "
            f"| kv_cache={service.config.kv_cache}"
        )
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serving] stopping")
    except Exception as error:
        print(f"serving failed: {error}", file=sys.stderr)
        return 1
    finally:
        if server is not None:
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
