"""Convert a complete Riftco model between explicit interchange formats."""

from __future__ import annotations

import argparse
from pathlib import Path

from riftco_transformer.interchange import (
    EXPORT_FORMATS,
    IMPORT_FORMATS,
    convert_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a complete Riftco decoder without guessing formats from "
            "file extensions. ONNX import accepts only canonical Riftco "
            "exports with their adjacent .riftco.json sidecar."
        )
    )
    parser.add_argument("source", type=Path, help="Input file or directory.")
    parser.add_argument(
        "destination",
        type=Path,
        help="Output file or Hugging Face-style directory.",
    )
    parser.add_argument(
        "--from",
        dest="source_format",
        choices=IMPORT_FORMATS,
        required=True,
        help="Input model format.",
    )
    parser.add_argument(
        "--to",
        dest="destination_format",
        choices=EXPORT_FORMATS,
        required=True,
        help="Output model format.",
    )
    parser.add_argument(
        "--gguf-model-name",
        default="Riftco Decoder",
        help="GGUF general.name value (default: %(default)s).",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    output = convert_model(
        arguments.source,
        arguments.destination,
        source_format=arguments.source_format,
        destination_format=arguments.destination_format,
        gguf_model_name=arguments.gguf_model_name,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
