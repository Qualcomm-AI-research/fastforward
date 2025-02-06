# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch

from fastforward.common import ensure_tensor


def integer_minimum(num_bits: float) -> float:
    """Return the minimum integer value given num_bits.

    Args:
        num_bits: Number of bits
    Returns:
        float: Minimum integer value supporting the quantization range
    """
    return -(2 ** (num_bits - 1))


def integer_maximum(num_bits: float) -> float:
    """Return the maximum integer value given num_bits.

    Args:
        num_bits: Number of bits
    Returns:
        float: Maximum integer value supporting the quantization range
    """
    return -integer_minimum(num_bits) - 1


def quantization_range(
    scale: torch.Tensor | float, offset: torch.Tensor | float | None, num_bits: float
) -> tuple[torch.Tensor | float, torch.Tensor | float]:
    """Compute quantization range for a set of quantization parameters.

    If both scale and offset are tensors, their dimensions must match.

    Args:
        scale: Scale for quantization
        offset: Offset for quantization
        num_bits: Number of bits used for quantization

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            `Tuple` of tensor representing the minimum and maximum thresholds of the
            quantization range, respectively.
    """
    offset = 0.0 if offset is None else offset
    range_min = (integer_minimum(num_bits) + offset) * scale
    range_max = (integer_maximum(num_bits) + offset) * scale
    return range_min, range_max


def parameters_for_range(
    min_range: torch.Tensor,
    max_range: torch.Tensor,
    num_bits: float,
    symmetric: bool,
    allow_one_sided: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute affine quantization parameters for a range.

    Given a range or ranges (if min_range and max_range are multidimensional),
    compute the scale and offset parameters that best represent that the given
    range(s)

    Args:
        min_range: `Tensor` representing the minimum range threshold
        max_range: `Tensor` representing the maximum range threshold

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]:
            scale and offset values that best represent the given range. Offset
            may be None in the non-onesided symmetric case.

    Notes::
        - The dimensionality of min_range and max_range must match.
        - If symmetric == True, not every range can be represented exactly,
            in that case, the scale and offset parameters are selected such
            that the entire given range is within bounds, i.e., the range
            used by `LinearQuantizer` may be wider and no assumptions on using
            the given range exactly must be made.
    """
    min_range, max_range = ensure_tensor(min_range), ensure_tensor(max_range)

    # Here we check if the minimum range threshold is all zero. In that case,
    # we assume a one-sided (or unsigned following Nagel et al., 2021). LinearQuantizerOp
    # still uses a signed integer representation, hence, we treat the unsigned symmetric case
    # from Nagel et al. as the asymmetric case where the offset is equal to minimum integer
    # value.
    #
    # NB: Theoretically, it is possible for non positive data to be assigned a zero minimum
    # threshold (e.g., when using an l2 quantization error minimization).
    one_sided = min_range.min() >= 0 and allow_one_sided

    int_min = integer_minimum(num_bits)

    if symmetric and one_sided:
        min_range = torch.zeros_like(min_range)

    if symmetric and not one_sided:
        # Choose scale such that entire range falls within bounds
        neg_scale = torch.abs(min_range) / abs(int_min)
        pos_scale = torch.abs(max_range) / abs(integer_maximum(num_bits))
        return torch.max(neg_scale, pos_scale), None
    else:
        # Choose scale and offset such that integer_minimum and integer_maximum
        # map to min and max, respectively.
        # NB: this results in a quantization grid in which min and max lie exactly
        # half a bin from the maximum/minimum values.
        num_steps = 2**num_bits - 1
        interval_length = max_range - min_range
        scale = interval_length / num_steps
        scale = scale.clamp(torch.finfo(scale.dtype).eps)
        offset = min_range / scale - int_min
        return scale, offset
