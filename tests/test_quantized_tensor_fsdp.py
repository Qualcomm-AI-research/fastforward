# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""FSDP2 (`fully_shard`) integration tests for `QuantizedTensor`.

These run real FSDP2 on CPU across two spawned processes using the gloo
backend; no GPU is required. They cover the module-level query path: FSDP2's
`_cast_fp_tensor` calls `torch.is_floating_point(x)` on forward inputs, which
is a distinct callable from the `Tensor.is_floating_point` method descriptor
and so must be on the no-dispatch allowlist in its own right (see
`test_quantized_tensor.py` for the fast, non-distributed coverage of the same
mechanism).
"""

import os
import socket

import fastforward as ff
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fastforward.quantization.affine import quantize_per_tensor

# FSDP2 (`torch.distributed.fsdp.fully_shard`) landed in torch 2.4. Skip the
# whole module cleanly on older/unsupported builds.
fsdp = pytest.importorskip("torch.distributed.fsdp", reason="FSDP2 (fully_shard) not available")
fully_shard = getattr(fsdp, "fully_shard", None)
MixedPrecisionPolicy = getattr(fsdp, "MixedPrecisionPolicy", None)


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `x` arrives as a QuantizedTensor. FSDP's pre-forward hook runs
        # `_cast_fp_tensor` -> `torch.is_floating_point(x)` on this input
        # BEFORE this body executes. The explicit dequantize below is allowed
        # even under strict quantization; only an implicit one would raise.
        if isinstance(x, ff.QuantizedTensor):
            x = x.dequantize()
        return self.lin(x)  # type: ignore[no-any-return]


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _Block()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)  # type: ignore[no-any-return]


def _fsdp_worker(rank: int, world_size: int, port: int) -> None:
    """Entry point for each spawned process; must be importable/picklable."""
    # Force the FSDP DeviceMesh onto gloo/CPU. If CUDA is visible, both spawned
    # processes can land on the same device and NCCL fails with "Duplicate GPU
    # detected".
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("cpu", (world_size,))
        model = _Model()
        mp_policy = fsdp.MixedPrecisionPolicy(param_dtype=torch.bfloat16, cast_forward_inputs=True)
        fsdp.fully_shard(model.block, mesh=mesh, mp_policy=mp_policy)
        fsdp.fully_shard(model, mesh=mesh, mp_policy=mp_policy)

        data = torch.randn(4, 4)
        quantized = quantize_per_tensor(
            data, scale=torch.tensor(0.1), offset=torch.tensor(0.0), num_bits=8
        )

        # Under strict quantization an implicit dequantization raises
        # QuantizationError, so this forward only succeeds if FSDP's
        # `torch.is_floating_point(quantized)` is a pure query.
        with ff.strict_quantization(True):
            out = model(quantized)
        assert out.shape == (4, 4)
    finally:
        dist.destroy_process_group()


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.slow
@pytest.mark.skipif(
    fully_shard is None or MixedPrecisionPolicy is None,
    reason="FSDP2 fully_shard / MixedPrecisionPolicy not available",
)
@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend not available")
def test_fsdp2_is_floating_point_does_not_dequantize() -> None:
    """FSDP2 forward-input casting must not implicitly dequantize a QuantizedTensor.

    FSDP2's `_cast_fp_tensor` calls the module-level `torch.is_floating_point`
    on every forward input. For a `QuantizedTensor` this must resolve as a pure
    query, on every rank, including ranks that do not own the shard.
    """
    # GIVEN a 2-process gloo world running FSDP2 with forward-input casting
    world_size = 2
    port = _free_tcp_port()

    # WHEN a QuantizedTensor is passed as a forward input under strict quantization
    # THEN every rank completes the forward without an implicit dequantization
    # (a child-process failure surfaces here as ProcessRaisedException).
    mp.spawn(_fsdp_worker, args=(world_size, port), nprocs=world_size, join=True)
