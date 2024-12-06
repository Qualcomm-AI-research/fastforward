# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import operator
import warnings

import pytest
import torch

from fastforward._gen import fallback
from fastforward.exceptions import QuantizationError
from fastforward.nn.linear_quantizer import LinearQuantizer
from fastforward.quantization import affine
from fastforward.quantized_tensor import QuantizedTensor


def assert_quantized_tensor(x):
    assert isinstance(x, torch.Tensor)
    assert isinstance(x, QuantizedTensor)


def assert_non_quantized_tensor(x):
    assert isinstance(x, torch.Tensor)
    assert not isinstance(x, QuantizedTensor)


def create_per_tensor_linear_quantizer(num_bits: int, scale: float, offset: float | None):
    with warnings.catch_warnings():
        # Suppress warning related to creating quantizers. Not tested here.
        warnings.simplefilter("ignore")
        quantizer = LinearQuantizer(num_bits=num_bits, symmetric=offset is None)
    data = torch.randn(10)
    quantizer.quantization_range = (torch.min(data), torch.max(data))
    with torch.no_grad():
        quantizer.scale.fill_(scale)
        if offset is not None and quantizer.offset is not None:
            quantizer.offset.fill_(offset)
    return quantizer


@pytest.mark.parametrize(
    "function_name,fallback_func,input_shapes,inputs_min_max,output_min_max,kwargs",
    [
        # Convolution functions
        (
            "conv1d",
            "torch.nn.functional.conv1d",
            [(4, 32, 10), (16, 32, 3), (16,)],  # Input, weight, bias
            [(-1, 1), (-1, 1), (-1, 1)],
            (-1, 1),
            {},
        ),
        (
            "conv2d",
            "nn.functional.conv2d",
            [(4, 32, 10, 10), (16, 32, 3, 3), (16,)],  # Input, weight, bias
            [(-1, 1), (-1, 1), (-1, 1)],
            (-1, 1),
            {},
        ),
        (
            "conv3d",
            "torch.nn.functional.conv3d",
            [(4, 32, 10, 10, 10), (16, 32, 3, 3, 3), (16,)],  # Input, weight, bias
            [(-1, 1), (-1, 1), (-1, 1)],
            (-1, 1),
            {},
        ),
        (
            "conv_transpose1d",
            "torch.nn.functional.conv_transpose1d",
            [(4, 32, 10), (32, 16, 3), (16,)],  # Input, weight, bias
            [(-1, 1), (-1, 1), (-1, 1)],
            (-1, 1),
            {},
        ),
        (
            "conv_transpose2d",
            "torch.nn.functional.conv_transpose2d",
            [(4, 32, 10, 10), (32, 16, 3, 3), (16,)],  # Input, weight, bias
            [(-1, 1), (-1, 1), (-1, 1)],
            (-1, 1),
            {},
        ),
        (
            "conv_transpose3d",
            "torch.nn.functional.conv_transpose3d",
            [(4, 32, 10, 10, 10), (32, 16, 3, 3, 3), (16,)],  # Input, weight, bias
            [(-1, 1), (-1, 1), (-1, 1)],
            (-1, 1),
            {},
        ),
        # ("unfold", [(4, 32, 4, 4)], [(-1, 1)], (-1, 1), dict(kernel_size=2)),
        # (
        #     "fold",
        #     [(1, 3 * 2 * 2, 12)],
        #     [(-1, 1)],
        #     (-1, 1),
        #     dict(kernel_size=(2, 2), output_size=(4, 5)),
        # ),
        #
        # Pooling functions
        (
            "avg_pool1d",
            "torch.nn.functional.avg_pool1d",
            [(1, 1, 7)],
            [(-1, 1)],
            (-1, 1),
            dict(kernel_size=3, stride=2),
        ),
        (
            "avg_pool2d",
            "torch.nn.functional.avg_pool2d",
            [(1, 1, 7, 7)],
            [(-1, 1)],
            (-1, 1),
            dict(kernel_size=3, stride=2),
        ),
        (
            "avg_pool3d",
            "torch.nn.functional.avg_pool3d",
            [(1, 1, 7, 7, 7)],  # Input
            [(-1, 1)],
            (-1, 1),
            dict(kernel_size=3, stride=2),
        ),
        # ("max_pool1d", [(1, 1, 7)], [(-1, 1)], (-1, 1), dict(kernel_size=3, stride=2)),
        # ("max_pool2d", [(1, 1, 7, 7)], [(-1, 1)], (-1, 1), dict(kernel_size=3, stride=2)),
        # ("max_pool3d", [(1, 1, 7, 7, 7)], [(-1, 1)], (-1, 1), dict(kernel_size=3, stride=2)),
        # (
        #     "lp_pool1d",
        #     [(20, 16, 10)],
        #     [(-1, 1)],
        #     (-1, 1),
        #     dict(kernel_size=3, stride=2, norm_type=2),
        # ),
        # (
        #     "lp_pool2d",
        #     [(20, 16, 10, 10)],
        #     [(-1, 1)],
        #     (-1, 1),
        #     dict(kernel_size=3, stride=2, norm_type=2),
        # ),
        # ("adaptive_max_pool1d", [(1, 10)], [(-1, 1)], (-1, 1), dict(output_size=5)),
        # ("adaptive_max_pool2d", [(1, 10, 10)], [(-1, 1)], (-1, 1), dict(output_size=5)),
        # ("adaptive_max_pool3d", [(1, 10, 10, 10)], [(-1, 1)], (-1, 1), dict(output_size=5)),
        # ("adaptive_avg_pool1d", [(1, 10)], [(-1, 1)], (-1, 1), dict(output_size=5)),
        # ("adaptive_avg_pool2d", [(1, 10, 10)], [(-1, 1)], (-1, 1), dict(output_size=5)),
        # ("adaptive_avg_pool3d", [(1, 10, 10, 10)], [(-1, 1)], (-1, 1), dict(output_size=5)),
        # #
        # # # Attention Mechanisms
        # # scaled_dot_product_attention  # See test_not_implemented_quantized_functions
        # #
        # # # Non-linear activation functions
        # ("threshold", [(1, 10)], [(-1, 1)], (-1, 1), dict(threshold=0, value=0)),
        # # threshold_  # See test_not_implemented_quantized_functions
        ("relu", "torch.relu", [(1, 10)], [(-1, 1)], (0, 1), {}),
        # # relu_  # See test_not_implemented_quantized_functions
        # ("hardtanh", [(1, 10)], [(-1, 1)], (-1, 1), {}),
        # # hardtanh_  # See test_not_implemented_quantized_functions
        # ("hardswish", [(1, 10)], [(-1, 1)], (-1, 1), {}),
        # ("relu6", [(1, 10)], [(-1, 1)], (-1, 1), {}),
        # ("elu", [(1, 10)], [(-1, 1)], (-1, 1), {}),
        # # elu_  # See test_not_implemented_quantized_functions
        # ("selu", [(1, 10)], [(-1, 1)], (-1, 1), {}),
        # ("celu", [(1, 10)], [(-1, 1)], (-1, 1), {}),
        # ("leaky_relu", [(1, 10)], [(-1, 1)], (-1, 1), {}),
        # # leaky_relu_  # See test_not_implemented_quantized_functions
        # ("prelu", [(1, 10), (10,)], [(-1, 1), (0, 1)], (-1, 1), {}),  # Input, weight
        # ("rrelu", [(1, 10)], [(-1, 1)], (-1, 1), {}),
        # ("leaky_relu", [(1, 10)], [(-1, 1)], (-1, 1), {}),
        # # rrelu_  # See test_not_implemented_quantized_functions
        # ("glu", [(1, 10)], [(-1, 1)], (-1, 1), {}),
        # ("gelu", [(1, 10)], [(-1, 1)], (-1, 1), {}),
        # ("logsigmoid", [(1, 10)], [(-1, 1)], (-1, 1), {}),
        # ("hardshrink", [(1, 10)], [(-1, 1)], (-1, 1), {}),
        # ("tanhshrink", [(1, 10)], [(-1, 1)], (-1, 1), {}),
        # ("softsign", [(1, 10)], [(-1, 1)], (-1, 1), {}),
        # ("softplus", [(1, 10)], [(-1, 1)], (-1, 1), {}),
        # ("softmin", [(1, 10)], [(-1, 1)], (-1, 1), {"dim": 1}),
        ("softmax", "softmax", [(1, 10)], [(-1, 1)], (-1, 1), {"dim": 1}),
        # ("softshrink", [(1, 10)], [(-1, 1)], (-1, 1), {}),
        # # ("gumbel_softmax", [(1, 10)], [(-1, 1)], (-1, 1), {}),  # Disabled because stochastic
        # ("log_softmax", [(1, 10)], [(-1, 1)], (-1, 1), {"dim": 1}),
        # ("tanh", [(1, 10)], [(-1, 1)], (-1, 1), {}),
        # ("sigmoid", [(1, 10)], [(-1, 1)], (-1, 1), {}),
        # ("hardsigmoid", [(1, 10)], [(-1, 1)], (-1, 1), {}),
        # ("silu", [(1, 10)], [(-1, 1)], (-1, 1), {}),
        # ("mish", [(1, 10)], [(-1, 1)], (-1, 1), {}),
        # #
        # # # Normalization
        # # (
        # #     "batch_norm",
        # #     [(2, 32, 10), (32,), (32,)],  # Input, running_mean, running_var
        # #     [(-1, 1), (-1, 1), (0.1, 1)],
        # #     (0, 1),
        # #     {},
        # # ),  # Disabled because it raises issue with gradient of running mean
        # ("group_norm", [(2, 32, 10)], [(-1, 1)], (-1, 1), dict(num_groups=4)),
        # ("instance_norm", [(2, 32, 10)], [(-1, 1)], (-1, 1), {}),
        (
            "layer_norm",
            "nn.functional.layer_norm",
            [(2, 32, 10)],
            [(-1, 1)],
            (-1, 1),
            dict(normalized_shape=(10,)),
        ),
        # ("local_response_norm", [(2, 32, 10)], [(-1, 1)], (-1, 1), dict(size=2)),
        # ("normalize", [(2, 32, 10)], [(-1, 1)], (-1, 1), {}),
        # #
        # # # Linear functions
        (
            "linear",
            "nn.functional.linear",
            [(10,), (20, 10), (20,)],  # Input, weight, bias
            [(-1, 1), (-1, 1), (-1, 1)],
            (-1, 1),
            {},
        ),
        # (
        #     "bilinear",
        #     [(16, 3, 20), (16, 3, 30), (40, 20, 30), (40,)],  # Input1, Input2, weight, bias
        #     [(-1, 1), (-1, 1), (-1, 1), (-1, 1)],
        #     (-1, 1),
        #     {},
        # ),
        # #
        # # # Dropout
        # # Disabled because stochastic
        # # ("dropout", [(2, 5, 10)], [(-1, 1)], (-1, 1), {}),
        # # ("alpha_dropout", [(2, 5, 10)], [(-1, 1)], (-1, 1), {}),
        # # ("feature_alpha_dropout", [(1, 10)], [(-1, 1)], (-1, 1), {}),
        # # ("dropout1d", [(2, 5, 10)], [(-1, 1)], (-1, 1), {}),
        # # ("dropout2d", [(2, 5, 8, 10)], [(-1, 1)], (-1, 1), {}),
        # # ("dropout3d", [(2, 3, 5, 8, 10)], [(-1, 1)], (-1, 1), {}),
        # #
        # # # Sparse functions
        # # embedding # See test_not_implemented_quantized_functions
        # # embedding_bag # See test_not_implemented_quantized_functions
        # # one_hot # See test_not_implemented_quantized_functions
        # #
        # # # Distance functions
        # ("pairwise_distance", [(5, 10), (5, 10)], [(-1, 1), (-1, 1)], (0, 1), {}),
        # ("cosine_similarity", [(5, 10), (5, 10)], [(-1, 1), (-1, 1)], (0, 1), {}),
        # ("pdist", [(5, 10)], [(-1, 1)], (0, 1), {}),
        # #
        # # # Loss function
        # # binary_cross_entropy
        # # binary_cross_entropy_with_logits
        # # poisson_nll_loss
        # # cosine_embedding_loss
        # # cross_entropy
        # # ctc_loss
        # # gaussian_nll_loss
        # # hinge_embedding_loss
        # # kl_div
        # # l1_loss
        # # mse_loss
        # # margin_ranking_loss
        # # multilabel_margin_loss
        # # multilabel_soft_margin_loss
        # # multi_margin_loss
        # # nll_loss
        # # huber_loss
        # # smooth_l1_loss
        # # soft_margin_loss
        # # triplet_margin_loss
        # # triplet_margin_with_distance_loss
        # #
        # # # Vision functions
        # ("pixel_shuffle", [(5, 4, 3 * 4, 10, 10)], [(-1, 1)], (-1, 1), dict(upscale_factor=2)),
        # (
        #     "pixel_unshuffle",
        #     [(5, 4, 3, 10 * 4, 10 * 4)],
        #     [(-1, 1)],
        #     (-1, 1),
        #     dict(downscale_factor=2),
        # ),
        # ("pad", [(10, 10)], [(-1, 1)], (-1, 1), dict(pad=(2, 2, 2, 2), value=0)),
        # ("interpolate", [(1, 4, 3, 10, 10)], [(-1, 1)], (-1, 1), dict(scale_factor=2)),
        # # ("upsample", [(4, 3, 10, 10)], [(-1, 1)], (-1, 1), dict(scale_factor=2)),  # Deprecated
        # # ("upsample_nearest", [(4, 3, 10, 10)], [(-1, 1)], (-1, 1), dict(scale_factor=2)),  # Deprecated
        # # ("upsample_bilinear", [(4, 3, 10, 10)], [(-1, 1)], (-1, 1), dict(scale_factor=2)),  # Deprecated
        # (
        #     "grid_sample",
        #     [(4, 3, 10, 10), (4, 8, 8, 2)],  # Input, flow
        #     [(-1, 1), (-1, 1)],
        #     (-1, 1),
        #     {"align_corners": False},
        # ),
        # ("affine_grid", [(4, 2, 3)], [(-1, 1)], (-1, 1), dict(size=(4, 3, 10, 10), align_corners=False),),
        # Element-wise functions
        (
            "add",
            "torch.add",
            ((10, 10), (10, 10)),
            ((-1, 1), (-1, 1)),
            (-1, 1),
            {},
        ),
        (
            "mm",
            "torch.mm",
            ((10, 10), (10, 10)),
            ((-1, 1), (-1, 1)),
            (-1, 1),
            {},
        ),
        (
            "bmm",
            "torch.bmm",
            ((4, 10, 10), (4, 10, 10)),
            ((-1, 1), (-1, 1)),
            (-1, 1),
            {},
        ),
    ],
)
def test_quantized_functionals(
    function_name,
    fallback_func,
    input_shapes,
    inputs_min_max,
    output_min_max,
    kwargs,
    num_bits=8,
):
    print(function_name)

    inputs_float = []
    inputs_quantized = []
    input_quantizers = []

    assert len(input_shapes) == len(inputs_min_max)

    for i, (input_shape, (input_min, input_max)) in enumerate(zip(input_shapes, inputs_min_max)):
        scale = (input_max - input_min) / (2**num_bits - 1)
        offset = -input_min

        input_quantizer = create_per_tensor_linear_quantizer(
            num_bits,
            scale,
            offset,
        )

        input_float = torch.rand(*input_shape) * (input_max - input_min) + input_min
        input_quantized = input_quantizer(input_float)

        input_quantizers.append(input_quantizer)
        inputs_float.append(input_float)
        inputs_quantized.append(input_quantized)

        print(
            "input",
            i,
            ":",
            input_quantized.shape,
            "[",
            input_quantized.float().min().item(),
            "-",
            input_quantized.float().max().item(),
            "]",
        )

        assert isinstance(input_quantized, QuantizedTensor)

    output_min, output_max = output_min_max
    scale = (output_max - output_min) / (2**num_bits - 1)
    offset = -output_min
    output_quantizer = create_per_tensor_linear_quantizer(
        num_bits,
        scale,
        offset,
    )

    torch_functional = operator.attrgetter(fallback_func)(torch)
    ff_functional = getattr(fallback, function_name)

    with pytest.raises(QuantizationError):
        ff_functional(
            *inputs_float, **kwargs, output_quantizer=output_quantizer, strict_quantization=True
        )

    output_quantized = ff_functional(*inputs_quantized, **kwargs, output_quantizer=output_quantizer)

    print(
        "output",
        ":",
        output_quantized.shape,
        "[",
        output_quantized.float().min().item(),
        "-",
        output_quantized.float().max().item(),
        "]",
    )

    assert isinstance(output_quantized, QuantizedTensor)
    assert torch.isfinite(output_quantized.dequantize()).all()

    torch.testing.assert_close(
        output_quantized.float(),
        output_quantizer(
            torch_functional(*[input.float() for input in inputs_quantized], **kwargs)
        ).float(),
    )


# The following were tested as part of the original functional implementation. Keeping
# this for reference until we have a more complete coverage of operators
# @pytest.mark.parametrize(
#     "function_name",
#     [
#         # Not implemented because one of the input arguments is integer.
#         "max_unpool1d",
#         "max_unpool2d",
#         "max_unpool3d",
#         #
#         # Not implemented because attention involves multiple operations and thus quantization simulation is perhaps not appropriate
#         "scaled_dot_product_attention",
#         #
#         # Not implemented because using in-place operations.
#         "threshold_",
#         "relu_",
#         "hardtanh_",
#         "threshold_",
#         "elu_",
#         "leaky_relu_",
#         "rrelu_",
#         #
#         # Not implemented because input argument is integer.
#         "embedding",
#         "embedding_bag",
#         "one_hot",
#         # # Loss functions are not unit-tested yet. Not all of them will work.
#         "binary_cross_entropy",
#         "binary_cross_entropy_with_logits",
#         "poisson_nll_loss",
#         "cosine_embedding_loss",
#         "cross_entropy",
#         "ctc_loss",
#         "gaussian_nll_loss",
#         "hinge_embedding_loss",
#         "kl_div",
#         "l1_loss",
#         "mse_loss",
#         "margin_ranking_loss",
#         "multilabel_margin_loss",
#         "multilabel_soft_margin_loss",
#         "multi_margin_loss",
#         "nll_loss",
#         "huber_loss",
#         "smooth_l1_loss",
#         "soft_margin_loss",
#         "triplet_margin_loss",
#         "triplet_margin_with_distance_loss",
#     ],
# )
# def test_not_implemented_quantized_functions(
#     function_name,
# ):
#     print(function_name)

#     ff_functional = getattr(ffF, function_name)

#     with pytest.raises(NotImplementedError):
#         ff_functional(None, ff_output_quantizer=None)


def test_dequantization_fallback_grad():
    num_bits = 3

    scale = torch.tensor(0.1)
    offset = torch.tensor(1.0)

    a_float = torch.randn(3, 2, 1)
    b_float = torch.randn(3, 2, 1)

    a_quant = affine.quantize_per_tensor(a_float, scale, offset, num_bits)
    b_quant = affine.quantize_per_tensor(b_float, scale, offset, num_bits)

    a_quant.requires_grad_()
    b_quant.requires_grad_()

    assert_quantized_tensor(a_quant)
    assert_quantized_tensor(b_quant)

    c = a_quant.dequantize() + b_quant.dequantize()
    assert_non_quantized_tensor(c)

    c_grad = torch.ones_like(c)
    c.backward(c_grad)

    assert a_quant.grad is not None
    assert b_quant.grad is not None

    torch.testing.assert_close(a_quant.grad, c_grad)
    torch.testing.assert_close(b_quant.grad, c_grad)
