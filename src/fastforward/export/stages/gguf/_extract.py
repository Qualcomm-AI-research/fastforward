# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Extract tensors from a quantized FastForward model for GGUF export.

Walks the model's parameters and classifies each into:

- **Quantized** — the parameter has an initialized weight quantizer discovered
  via ``ff.find_quantizers``. Integer codes and per-block scales are read
  directly from the quantizer, preserving the user's learned quantization.
- **Float** — everything else (norms, biases, unquantized weights). Passed
  through as-is for downstream F32 writing.

No calibration or fallback quantization is performed. If the user did not
quantize a parameter, it arrives as float.
"""

import logging

from dataclasses import dataclass

import torch

import fastforward as ff

from fastforward.exceptions import ExportError
from fastforward.export.stages.gguf._config import GgufSourceConfig
from fastforward.export.stages.gguf.adapter import ArchAdapter, GgufQuantFormat
from fastforward.nn import LinearQuantizer
from fastforward.quantization import granularity as granularities

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExtractedTensor:
    """A single exportable tensor extracted from the model.

    Quantized tensors carry ``int_codes`` (integer) and ``scales`` (float32).
    Float tensors carry ``float_data``.
    """

    hf_name: str
    kind: str
    rows: int
    cols: int | None = None
    int_codes: torch.Tensor | None = None
    scales: torch.Tensor | None = None
    float_data: torch.Tensor | None = None
    gguf_name: str = ""


def _build_quantizer_map(model: torch.nn.Module) -> dict[str, LinearQuantizer]:
    """Map parameter paths to their initialized weight quantizers.

    Uses ``ff.find_quantizers`` with the ``[quantizer:parameter/weight]`` tag to
    discover all weight quantizers in the model. Returns a dict keyed by the
    weight parameter's full name (e.g. ``"model.layers.0.self_attn.q_proj.weight"``)
    mapped to the corresponding initialized ``LinearQuantizer``.
    """
    quantizer_map: dict[str, LinearQuantizer] = {}
    for result in ff.find_quantizers(model, "**/[quantizer:parameter/weight]"):
        quantizer = result.module
        if not isinstance(quantizer, LinearQuantizer):
            continue
        if quantizer.has_uninitialized_params:
            continue
        # The quantizer lives at e.g. "layers.0.q_proj.weight_quantizer".
        # The corresponding parameter is at "layers.0.q_proj.weight".
        parent_path = result.full_name.rsplit(".", 1)[0]
        param_name = f"{parent_path}.weight"
        quantizer_map[param_name] = quantizer
    return quantizer_map


def _validate_quantizer(
    quantizer: LinearQuantizer,
    quant_format: GgufQuantFormat,
    hf_name: str,
) -> None:
    """Validate that a quantizer's configuration is compatible with the target GGUF format.

    Raises ExportError if the quantizer is asymmetric, has mismatched bit width,
    or (for PerBlock granularity) has a block size that doesn't match the format.
    """
    if quant_format.symmetric and not quantizer.symmetric:
        msg = (
            f"Parameter '{hf_name}': quantizer is asymmetric but GGUF format "
            f"'{quant_format.name}' requires symmetric quantization"
        )
        raise ExportError(msg)

    if not quant_format.symmetric and quantizer.symmetric:
        msg = (
            f"Parameter '{hf_name}': quantizer is symmetric but GGUF format "
            f"'{quant_format.name}' requires asymmetric quantization"
        )
        raise ExportError(msg)

    if quantizer.num_bits != quant_format.num_bits:
        msg = (
            f"Parameter '{hf_name}': quantizer num_bits={quantizer.num_bits} does not match "
            f"GGUF format '{quant_format.name}' which expects num_bits={quant_format.num_bits}"
        )
        raise ExportError(msg)

    if not isinstance(quantizer.granularity, granularities.PerBlock):
        msg = (
            f"Parameter '{hf_name}': quantizer granularity is "
            f"{type(quantizer.granularity).__name__} but GGUF format "
            f"'{quant_format.name}' requires PerBlock granularity"
        )
        raise ExportError(msg)

    gran = quantizer.granularity
    if gran.block_dims != (1,) or gran.per_channel_dims != (0,):
        msg = (
            f"Parameter '{hf_name}': quantizer has block_dims={gran.block_dims}, "
            f"per_channel_dims={gran.per_channel_dims} but GGUF format "
            f"'{quant_format.name}' requires block_dims=(1,), per_channel_dims=(0,)"
        )
        raise ExportError(msg)

    if quant_format.block_size not in gran.block_sizes:
        msg = (
            f"Parameter '{hf_name}': quantizer block_sizes={gran.block_sizes} does not contain "
            f"GGUF format '{quant_format.name}' block_size={quant_format.block_size}"
        )
        raise ExportError(msg)


def extract_module_tensors(
    model: torch.nn.Module,
    *,
    adapter: ArchAdapter,
    config: GgufSourceConfig,
    quant_format: GgufQuantFormat,
) -> list[ExtractedTensor]:
    """Extract every GGUF-exportable tensor from ``model``.

    Parameters with an initialized weight quantizer (discovered via
    ``ff.find_quantizers``) are extracted as quantized (integer codes + scales).
    Everything else passes through as float. No fallback quantization is applied.

    Each quantizer is validated against ``quant_format`` before extraction:
    it must be symmetric, match the format's ``num_bits``, and (for PerBlock
    granularity) have a block size present in the format's ``block_size``.

    Args:
        model: The quantized FastForward module to export.
        adapter: Architecture adapter. Its ``name_map`` decides which parameters
            are exportable, and its ``is_tied`` predicate decides which
            parameters to skip because they share storage with another exported
            parameter.
        config: Source model config, passed to the adapter's ``is_tied``
            predicate for architecture-specific tied-weight logic.
        quant_format: Target GGUF quantization format. Used to validate that
            quantizers are compatible before extracting codes and scales.

    Returns:
        The extracted tensors, in ``named_parameters`` order.

    Raises:
        ExportError: If a quantizer is incompatible with ``quant_format``.
    """
    quantizer_map = _build_quantizer_map(model)
    extracted: list[ExtractedTensor] = []
    skipped: list[str] = []

    for hf_name, param in model.named_parameters():
        if adapter.name_map(hf_name) is None:
            skipped.append(hf_name)
            continue
        if isinstance(param, torch.nn.UninitializedParameter):
            continue
        if adapter.is_tied(hf_name, config):
            continue

        quantizer = quantizer_map.get(hf_name)
        if quantizer is not None:
            _validate_quantizer(quantizer, quant_format, hf_name)
            weight = param.data.float()
            quantized = quantizer(weight)
            int_codes = quantized.int_repr().detach().cpu()
            scales = quantizer.scale.detach().cpu().float()
            rows, cols = weight.shape
            # `_validate_quantizer` pins granularity so scales form a (rows, blocks) grid.
            assert scales.numel() % rows == 0
            extracted.append(
                ExtractedTensor(
                    hf_name=hf_name,
                    kind="quantized",
                    rows=rows,
                    cols=cols,
                    int_codes=int_codes,
                    scales=scales.reshape(rows, -1),
                )
            )
        else:
            data = param.data.detach().cpu().float()
            extracted.append(
                ExtractedTensor(
                    hf_name=hf_name,
                    kind="float",
                    rows=data.shape[0],
                    float_data=data,
                )
            )

    if skipped:
        user_skipped = [s for s in skipped if "_quantizer." not in s]
        internal_skipped = [s for s in skipped if "_quantizer." in s]
        if internal_skipped:
            logger.debug(
                "GGUF export: %d quantizer-internal parameter(s) skipped (expected): %s",
                len(internal_skipped),
                ", ".join(internal_skipped),
            )
        if user_skipped:
            logger.warning(
                "GGUF export: %d parameter(s) had no name_map entry and were skipped: %s",
                len(user_skipped),
                ", ".join(user_skipped),
            )

    return extracted
