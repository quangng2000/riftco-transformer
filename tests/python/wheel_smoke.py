#!/usr/bin/env python3
"""Exercise an installed self-contained wheel outside the source tree."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

from riftco_transformer import (
    Adam,
    Context,
    DecoderOnlyTransformer,
    LoraConfig,
    Tensor,
    TransformerConfig,
    cross_entropy,
)
from riftco_transformer.native import bindings


def native_filename() -> str:
    if sys.platform == "darwin":
        return "libriftco_transformer_c.dylib"
    if os.name == "nt":
        return "riftco_transformer_c.dll"
    return "libriftco_transformer_c.so"


def exercise_qlora() -> None:
    """Exercise the public packed-base, adapter-only training lifecycle."""

    config = TransformerConfig(
        vocabulary_size=5,
        maximum_context=3,
        model_width=4,
        head_count=2,
        block_count=1,
        feed_forward_width=8,
        random_seed=431,
    )
    tokens = [[0, 1]]
    targets = [[1, 2]]

    with DecoderOnlyTransformer(config).to("cpu") as model:
        model.quantize_nf4()
        if not model.quantized_linear_weights:
            raise RuntimeError("NF4 quantization did not remain active")

        packed = model.quantized_memory
        if packed.weight_count <= 0:
            raise RuntimeError("NF4 quantization packed no weights")
        if packed.logical_payload_bytes != (
            packed.packed_code_bytes + packed.scale_bytes
        ):
            raise RuntimeError(f"inconsistent NF4 memory stats: {packed}")
        if packed.resident_payload_bytes < packed.logical_payload_bytes:
            raise RuntimeError(f"incomplete resident NF4 payload: {packed}")
        if packed.fp32_equivalent_bytes <= packed.logical_payload_bytes:
            raise RuntimeError(f"NF4 payload did not reduce memory: {packed}")
        if (
            packed.double_quantized_weight_count != packed.weight_count
            or packed.fp32_scale_bytes != 0
            or packed.scale_code_bytes <= 0
            or packed.second_level_scale_bytes <= 0
        ):
            raise RuntimeError(
                f"default QLoRA scales are not double-quantized: {packed}"
            )

        model.attach_lora(
            LoraConfig(rank=2, alpha=4.0, random_seed=433)
        )
        with model.adapter_parameters() as adapters:
            adapter_values_before = adapters.flat_values()
            with Adam(
                adapters,
                learning_rate=1.0e-2,
                state_storage="paged",
                page_size=3,
            ) as optimizer:
                if (
                    optimizer.state_storage != "paged"
                    or optimizer.state_page_size != 3
                    or optimizer.state_page_count <= 0
                    or optimizer.state_payload_bytes <= 0
                ):
                    raise RuntimeError(
                        "wheel Adam did not retain bounded-page state"
                    )
                with model(tokens) as logits:
                    with cross_entropy(logits, targets) as loss:
                        loss.backward()
                        step = optimizer.step()
                if step.step != 1:
                    raise RuntimeError(f"unexpected Adam step: {step}")

            if adapters.flat_values() == adapter_values_before:
                raise RuntimeError("Adam did not update the LoRA adapters")

        if not model.quantized_linear_weights:
            raise RuntimeError("NF4 base weights expanded during training")
        if model.quantized_memory != packed:
            raise RuntimeError("packed NF4 memory changed during training")

        model.merge_lora()
        if model.lora_attached:
            raise RuntimeError("LoRA merge left the adapter attached")
        if model.quantized_linear_weights:
            raise RuntimeError("LoRA merge left quantization active")
        cleared = model.quantized_memory
        if any(
            (
                cleared.weight_count,
                cleared.packed_code_bytes,
                cleared.scale_bytes,
                cleared.logical_payload_bytes,
                cleared.resident_payload_bytes,
                cleared.fp32_equivalent_bytes,
                cleared.fp32_scale_bytes,
                cleared.scale_code_bytes,
                cleared.second_level_scale_bytes,
                cleared.scale_offset_bytes,
                cleared.double_quantized_weight_count,
            )
        ):
            raise RuntimeError(f"LoRA merge left quantization state: {cleared}")


def main() -> int:
    if os.environ.get("RIFTCO_TRANSFORMER_LIBRARY"):
        raise RuntimeError(
            "wheel smoke test must not use RIFTCO_TRANSFORMER_LIBRARY"
        )

    package_directory = Path(bindings.__file__).resolve().parents[1]
    if (package_directory / "experiments").exists():
        raise RuntimeError(
            "repository labs must not be included in the framework wheel"
        )
    bundled_library = package_directory / ".libs" / native_filename()
    if not bundled_library.is_file():
        raise RuntimeError(f"bundled native library is missing: {bundled_library}")

    candidates = bindings._candidate_libraries()
    if not candidates:
        raise RuntimeError("native loader found no candidate libraries")
    if Path(candidates[0]).resolve() != bundled_library.resolve():
        raise RuntimeError(
            "bundled library is not the loader's first candidate: "
            f"{candidates[0]}"
        )

    with Context("cpu") as context:
        if context.backend != "cpu":
            raise RuntimeError(f"unexpected backend: {context.backend}")
        with Tensor.from_data(context, (1, 2), (1.0, 2.0)) as left:
            with Tensor.from_data(context, (2, 1), (3.0, 4.0)) as right:
                with left.matmul(right) as product:
                    if product.shape != (1, 1):
                        raise RuntimeError(f"unexpected shape: {product.shape}")
                    values = product.tolist()
                    if len(values) != 1 or not math.isclose(values[0], 11.0):
                        raise RuntimeError(f"unexpected matmul result: {values}")

    exercise_qlora()

    library = bindings._native()
    loaded_path = Path(str(library._name)).resolve()
    if loaded_path != bundled_library.resolve():
        raise RuntimeError(
            f"loaded {loaded_path}, expected bundled {bundled_library.resolve()}"
        )
    if int(library.rt_abi_version()) != bindings.ABI_VERSION:
        raise RuntimeError("bundled native ABI does not match the Python client")

    print(f"wheel smoke test passed with {loaded_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
