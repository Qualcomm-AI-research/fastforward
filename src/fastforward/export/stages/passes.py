# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import itertools
import logging

from collections import deque
from typing import Any, Iterator

import torch

from torch.fx.passes.infra.pass_base import PassResult

from fastforward.export.stages.annotations import _ff_quantizer_spec

FF_QUANTIZATION_SPEC = "__FF_QUANTIZATION_SPEC"
QUANTIZE_OPERATIONS = (torch.ops.fastforward.quantize_by_tile.default,)
DEQUANTIZE_OPERATIONS = (torch.ops.fastforward.dequantize_by_tile.default,)

logger = logging.getLogger(__name__)


class AnnotateFFQuantSpecs:
    """Annotate nodes that are quantized with FF quantizers.

    Annotate the node with appropriate quantization encodings/spec. Remove the
    FF quantization nodes from the graph.
    """

    def __call__(self, module: torch.fx.GraphModule) -> PassResult:
        """Apply FF quantization spec annotation pass to the graph module.

        Processes the graph to annotate nodes with FastForward quantization
        specifications and removes quantization/dequantization operations,
        storing quantization metadata in relevant nodes in the graph.

        Args:
            module: The torch.fx.GraphModule to process

        Returns:
            PassResult containing the modified module and success status
        """
        graph = module.graph
        quant_params = _ff_quantization_parameters(module)

        for node in list(_ff_quantization_nodes(graph)):
            if node.target in QUANTIZE_OPERATIONS:
                # Node is a quantize operations. Annotate the appropriate input
                # node with quantization encodings. And remove node from graph.
                spec = _ff_quantizer_spec(node=node, quant_params=quant_params)
                input = node.args[0]
                assert isinstance(input, torch.fx.Node)

                if FF_QUANTIZATION_SPEC in input.meta:
                    logger.warning("Detected re-quantization for %s", input)

                input.meta[FF_QUANTIZATION_SPEC] = spec

                node.replace_all_uses_with(input)
                graph.erase_node(node)

            elif node.target in DEQUANTIZE_OPERATIONS:
                # Node is a dequantize operation. Annotations will occur with
                # the associated quantize operatiorns. Only remove dequantize
                # node from graph.
                input = node.args[0]
                assert isinstance(input, torch.fx.Node)
                node.replace_all_uses_with(input)
                graph.erase_node(node)

        return PassResult(module, True)


class PropagateFFQuantSpecs:
    """Propagate FastForward quantization specifications through compatible operations.

    This pass propagates quantization specifications from inputs to outputs for
    operations that preserve quantization properties, such as reshape operations
    (unsqueeze, squeeze, permute).
    """

    def __call__(self, module: torch.fx.GraphModule) -> PassResult:
        """Apply FF quantization spec propagation pass to the graph module.

        Args:
            module: The torch.fx.GraphModule to process

        Returns:
            PassResult containing the modified module and success status
        """
        graph = module.graph

        queue = deque(graph.nodes)
        while queue:
            node = queue.popleft()
            if isinstance(node.target, torch._ops.OpOverload) and node.target.is_view:
                self._annotate_same_input_output_quant_spec(node, node.args[0], queue)

        return PassResult(module, True)

    def _annotate_same_input_output_quant_spec(
        self, output: torch.fx.Node, input: torch.fx.Node, queue: deque[torch.fx.Node]
    ) -> None:
        if FF_QUANTIZATION_SPEC in input.meta and FF_QUANTIZATION_SPEC in output.meta:
            return
        if FF_QUANTIZATION_SPEC not in input.meta and FF_QUANTIZATION_SPEC not in output.meta:
            return

        if FF_QUANTIZATION_SPEC in input.meta:
            src, tgt = input, output
        else:
            src, tgt = output, input

        tgt.meta[FF_QUANTIZATION_SPEC] = src.meta[FF_QUANTIZATION_SPEC]
        queue.append(tgt)


def _ff_quantization_parameters(
    module: torch.fx.GraphModule,
) -> dict[str, torch.Tensor]:
    """Helper to obtain all quantization parameter tensors from module."""
    all_params = dict(itertools.chain(module.named_parameters(), module.named_buffers()))
    quant_params: dict[str, torch.Tensor] = {}
    scale: Any = None
    offset: Any = None
    for node in _ff_quantization_nodes(module.graph):
        match node.target:
            case torch.ops.fastforward.quantize_by_tile.default:
                _data, scale, _tile_size, _num_bits, _output_dtype, offset = node.args
            case torch.ops.fastforward.dequantize_by_tile.default:
                _data, scale, _tile_size, offset, _output_dtype = node.args
            case _:
                assert False, "unreachable"

        for param_node in (scale, offset):
            if param_node is None:
                continue
            assert isinstance(param_node, torch.fx.Node)
            assert isinstance(param_node.target, str)
            quant_params[param_node.target] = all_params[param_node.target]

    return quant_params


def _ff_quantization_nodes(graph: torch.fx.Graph) -> Iterator[torch.fx.Node]:
    """Returns iterator over all quantize and dequantize op nodes in `graph`."""
    for node in graph.nodes:
        if not node.op == "call_function":
            continue

        if node.target in QUANTIZE_OPERATIONS or node.target in DEQUANTIZE_OPERATIONS:
            yield node
