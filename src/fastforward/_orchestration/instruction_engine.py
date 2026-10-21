# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import abc
import dataclasses
import functools
import itertools

from collections import defaultdict
from typing import (
    Any,
    Callable,
    Collection,
    ContextManager,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    TypeAlias,
)

import attrs
import torch
import torch.utils._pytree as pytree

from torch.utils.data import DataLoader

from fastforward._orchestration.graph_module import (
    AttributePath,
    Const,
    GradientRef,
    GraphModule,
    InputRef,
    LossGradRef,
    NamedRef,
    Op,
    VJPRef,
    _BaseRef,
)
from fastforward._orchestration.location import (
    DeviceLocation,
    Location,
    LocationLike,
    as_location,
)

# Distinguishes data produced under different execution conditions for the same node.
# A node without a module (a free function or a method call) passes None.
StreamKey: TypeAlias = Callable[[torch.nn.Module | None], ContextManager[None]]

# Ordered sequence of context managers that an instruction executes under.
Contexts: TypeAlias = Sequence[Callable[[torch.nn.Module | None], ContextManager[None]]]

# Selects a location for data by name.
LocationPolicy: TypeAlias = Callable[[str], Location]
LocationPolicyLike: TypeAlias = Callable[[str], LocationLike]


def _fmt_module(module: torch.nn.Module | Callable[..., Any]) -> str:
    """Format a module/callable target without dumping parameters or a memory address."""
    if isinstance(module, torch.nn.Module):
        return type(module).__name__
    return _fmt_callable(module)


def _fmt_callable(fn: Callable[..., Any]) -> str:
    """Format a callable by name, without its memory address."""
    while isinstance(fn, functools.partial):
        fn = fn.func
    return getattr(fn, "__name__", type(fn).__name__)


def _fmt_contexts(contexts: Contexts) -> str:
    """Format execution contexts by callable name, without their memory addresses."""
    return "[" + ", ".join(_fmt_callable(context) for context in contexts) + "]"


def _fmt_value(value: Any) -> str:
    """Format a register value, showing shape/dtype instead of tensor contents."""
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
    return repr(value)


@dataclasses.dataclass(frozen=True)
class ActivationDataset(Collection[Any]):
    """A batched collection that wraps all data flowing through the InstructionEngine."""

    batches: list[Any]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)

    def __contains__(self, item: Any) -> bool:
        return item in self.batches

    @classmethod
    def from_value(cls, value: Any) -> "ActivationDataset":
        """Create an ActivationDataset from 'value'.

        If the value itself is an ActivationDataset we return it unchanged.

        Args:
            value: anything you want wrapped in an ActivationDataset.

        Returns:
            ActivationDataset with value(s) as batches.
        """
        if isinstance(value, cls):
            return value

        batches = list(value) if isinstance(value, (list, DataLoader)) else [value]
        return cls(batches)

    @classmethod
    def merge(cls, datasets: Sequence["ActivationDataset"]) -> "ActivationDataset":
        """Zip multiple ActivationDatasets.

        All datasets must have the same length.

        Args:
            datasets: Non-empty sequence of ActivationDatasets to merge.

        Returns:
            ActivationDataset where each batch is a tuple of corresponding elements.

        Raises:
            ValueError: If datasets have different lengths.
        """
        if len(datasets) == 0:
            msg = "ActivationDataset.merge expects at least one dataset"
            raise ValueError(msg)

        if len(datasets) == 1:
            return datasets[0]

        lengths = [len(ds) for ds in datasets]
        if len(set(lengths)) > 1:
            msg = f"Cannot merge datasets of different sizes: {lengths}"
            raise ValueError(msg)

        batches: list[tuple[Any, ...]] = []
        for i in range(lengths[0]):
            tpl = tuple(ds.batches[i] for ds in datasets)
            batches.append(tpl)

        return ActivationDataset(batches)

    @staticmethod
    def broadcast(datasets: Sequence["ActivationDataset"]) -> list["ActivationDataset"]:
        """Broadcast datasets to a common length.

        Datasets of length 1 are repeated to match N-length datasets.

        Args:
            datasets: Datasets to broadcast.

        Returns:
            List of datasets all at the same length.

        Raises:
            ValueError: If there are multiple non-1 lengths.
        """
        if not datasets:
            return []
        lengths = {len(ds) for ds in datasets}
        if len(lengths) <= 1:
            return list(datasets)
        lengths.discard(1)
        if len(lengths) != 1:
            msg = (
                "Dataset length mismatch: broadcasting supports 1->N but not "
                f"arbitrary mismatches. Lengths: {[len(ds) for ds in datasets]}."
            )
            raise ValueError(msg)
        target = lengths.pop()
        return [ActivationDataset(ds.batches * target) if len(ds) == 1 else ds for ds in datasets]


class ActivationRegister:
    """Stores per-node, per-stream activation data for the instruction engine."""

    def __init__(self) -> None:
        self._data: dict[_BaseRef, dict[StreamKey, ActivationDataset]] = {}

    def store(self, ref: _BaseRef, context: StreamKey, data: ActivationDataset) -> None:
        """Store activation data for a ref under a specific context."""
        if ref not in self._data:
            self._data[ref] = {}
        self._data[ref][context] = data

    def store_all(self, ref: _BaseRef, mapping: dict[StreamKey, ActivationDataset]) -> None:
        """Merge activation data for a ref, so multiple producers accumulate streams."""
        self._data.setdefault(ref, {}).update(mapping)

    def load(self, ref: _BaseRef, context: StreamKey) -> ActivationDataset:
        """Load activation data for a ref under a specific context."""
        return self._data[ref][context]

    def items_for(self, ref: _BaseRef) -> Iterator[tuple[StreamKey, ActivationDataset]]:
        """Iterate over (context, dataset) pairs for a ref."""
        return iter(self._data[ref].items())

    def contexts_for(self, ref: _BaseRef) -> Iterator[StreamKey]:
        """Iterate over contexts that have data stored for a ref."""
        return iter(self._data[ref].keys())

    def delete(self, ref: _BaseRef, context: StreamKey | None = None) -> None:
        """Remove data for `ref`.

        With `context` omitted, drops every stream stored for `ref`. With
        `context` given, drops only that stream; if it was the last, `ref`
        itself is dropped so `ref in register` becomes False.
        """
        if context is None:
            self._data.pop(ref, None)
        elif ref in self._data:
            self._data[ref].pop(context, None)
            if not self._data[ref]:
                self._data.pop(ref)

    def __contains__(self, ref: _BaseRef) -> bool:
        return ref in self._data


@dataclasses.dataclass(frozen=True)
class ActivationBundle:
    """Per-context view of all activations needed to reproduce one call.

    An `ActivationBundle` packages the positional and keyword `ActivationDataset`s
    that together describe a single node's input under one execution context.
    Iteration yields `(args_tuple, kwargs_dict)` per batch — the exact shape needed
    to invoke the underlying callable as `module(*args, **kwargs)`.

    Constructed via `gather`, which resolves a node's `_BaseRef` args and kwargs
    against the register, broadcasts singletons against N-length streams, and
    bundles them. `Const` refs are wrapped on the fly.
    """

    args: tuple[ActivationDataset, ...]
    kwargs: Mapping[str, ActivationDataset]

    @classmethod
    def gather(
        cls,
        register: ActivationRegister,
        context: StreamKey,
        args: Sequence[_BaseRef],
        kwargs: Mapping[str, _BaseRef],
    ) -> "ActivationBundle":
        """Resolve refs from the register under one context key, broadcast, and bundle.

        Args:
            register: The activation register mapping refs to per-context datasets.
            context: The context key under which to resolve each ref.
            args: Positional input refs in declaration order.
            kwargs: Keyword input refs keyed by parameter name.

        Returns:
            An `ActivationBundle` whose positional and keyword streams are broadcast
            to a common batch length and ready for iteration as `(args, kwargs)` pairs.
        """

        def get_dataset(ref: _BaseRef) -> ActivationDataset:
            if isinstance(ref, Const):
                return ActivationDataset([ref.value])
            return register.load(ref, context)

        arg_datasets = [get_dataset(ref) for ref in args]
        kwarg_datasets = {key: get_dataset(ref) for key, ref in kwargs.items()}

        broadcasted = ActivationDataset.broadcast([*arg_datasets, *kwarg_datasets.values()])
        n_args = len(arg_datasets)
        return cls(
            args=tuple(broadcasted[:n_args]),
            kwargs=dict(zip(kwarg_datasets.keys(), broadcasted[n_args:])),
        )

    def __len__(self) -> int:
        """Return the number of batches in this bundle.

        Returns:
            The length of the first positional or keyword stream, or 0 if empty.
        """
        for ds in self.args:
            return len(ds)
        for ds in self.kwargs.values():
            return len(ds)
        return 0

    def __iter__(self) -> Iterator[tuple[tuple[Any, ...], dict[str, Any]]]:
        """Iterate over batches, yielding one `(args, kwargs)` pair per batch.

        Each yielded tuple contains the positional activations as a tuple and
        the keyword activations as a dict, aligned at the same batch index across
        all streams. The result can be passed directly to a module as
        ``module(*args, **kwargs)``.

        Yields:
            Tuples of ``(args_tuple, kwargs_dict)`` for each batch index.
        """
        length = len(self)
        for i in range(length):
            args = tuple(ds.batches[i] for ds in self.args)
            kwargs = {name: ds.batches[i] for name, ds in self.kwargs.items()}
            yield args, kwargs


@dataclasses.dataclass(frozen=True)
class Instruction(abc.ABC):
    """Base class for all instructions in the execution engine.

    Each instruction must implement the execute method to define its behavior
    when executed by the InstructionEngine.
    """

    @abc.abstractmethod
    def execute(self, register: ActivationRegister) -> Any:
        """Execute this instruction.

        Args:
            register: The execution register mapping references to values.

        Returns:
            The result of executing this instruction.
        """

    def uses(self) -> Iterator[_BaseRef]:
        """Return references this instruction uses from the register."""
        return iter(())

    def produces(self) -> Iterator[_BaseRef]:
        """Return references this instruction writes to the register."""
        return iter(())


@dataclasses.dataclass(frozen=True)
class StoreValue(Instruction):
    """Store a value in the register."""

    target: _BaseRef
    value: Any
    context: StreamKey

    def __repr__(self) -> str:
        return (
            f"StoreValue(target={self.target!r}, value={_fmt_value(self.value)}, "
            f"context={_fmt_callable(self.context)})"
        )

    def execute(self, register: ActivationRegister) -> None:  # noqa: D102
        register.store(self.target, self.context, self.value)

    def uses(self) -> Iterator[_BaseRef]:  # noqa: D102
        return iter([self.target])


def get_attribute(value: Any, attribute: str | int) -> Any:
    """Return the item or attribute `attribute` of `value`.

    An int is an index. A str is a key of a mapping (dict, etc.), or else an
    attribute of a structured object (dataclass, namedtuple, etc.).
    """
    if isinstance(attribute, int):
        return value[attribute]
    try:
        return value[attribute]
    except (KeyError, TypeError):
        return getattr(value, attribute)


@dataclasses.dataclass(frozen=True)
class LoadAttribute(Instruction):
    """Load an attribute/index from a register value and store result."""

    source: _BaseRef
    target: _BaseRef
    attribute: str | int

    def execute(self, register: ActivationRegister) -> None:  # noqa: D102
        for context, dataset in register.items_for(self.source):
            register.store(self.target, context, self._extract_attribute(dataset))

    def _extract_attribute(self, dataset: ActivationDataset) -> ActivationDataset:
        """Extract attribute/item from each batch in the dataset."""
        return ActivationDataset([get_attribute(batch, self.attribute) for batch in dataset])

    def uses(self) -> Iterator[_BaseRef]:  # noqa: D102
        return iter([self.source])

    def produces(self) -> Iterator[_BaseRef]:  # noqa: D102
        return iter([self.target])


@dataclasses.dataclass(frozen=True)
class Call(Instruction):
    """Execute one graph node on batched data from the register.

    Handles bundle gathering and result storage.

    Args:
        args: Positional input refs in declaration order.
        kwargs: Keyword input refs keyed by parameter name.
        target: Ref the outputs are stored under.
        context: Execution context to run under.
        caller: Callable invoked for each batch.
        cache: Whether the outputs may be reused by a later reader.
        input_location: Where to place each batch's arguments before invocation.
        output_location: Where to place each result before collecting the next batch.
    """

    args: Sequence[_BaseRef]
    kwargs: dict[str, _BaseRef]
    target: _BaseRef
    context: StreamKey
    caller: Callable[..., Any]
    cache: bool = dataclasses.field(default=False, kw_only=True)
    input_location: Location | None = dataclasses.field(default=None, kw_only=True)
    output_location: Location | None = dataclasses.field(default=None, kw_only=True)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(caller={_fmt_module(self.caller)}, args={list(self.args)!r}, "
            + f"kwargs={dict(self.kwargs)!r}, target={self.target!r}, "
            + f"context={_fmt_callable(self.context)}, cache={self.cache}, "
            + f"input_location={self.input_location!r}, output_location={self.output_location!r})"
        )

    def _batches(
        self, bundle: ActivationBundle
    ) -> Iterator[tuple[tuple[Any, ...], dict[str, Any]]]:
        batches: Iterable[tuple[tuple[Any, ...], dict[str, Any]]] = bundle
        if not bundle:
            batches = [((), {})]
        for args, kwargs in batches:
            if self.input_location is not None:
                args, kwargs = self.input_location.place((args, kwargs))
            yield args, kwargs
            del args, kwargs

    def execute(self, register: ActivationRegister) -> None:  # noqa: D102
        bundle = ActivationBundle.gather(register, self.context, self.args, self.kwargs)
        outputs = []
        module = self.caller if isinstance(self.caller, torch.nn.Module) else None
        with self.context(module):
            for args, kwargs in self._batches(bundle):
                output = self._single_batch(args, kwargs)
                if self.output_location is not None:
                    output = self.output_location.place(output)
                outputs.append(output)
                del args, kwargs, output
        register.store(self.target, self.context, ActivationDataset(outputs))

    def _single_batch(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Return the output of one batch."""
        return self.caller(*args, **kwargs)

    def uses(self) -> Iterator[_BaseRef]:  # noqa: D102
        yield from self.args
        yield from self.kwargs.values()

    def produces(self) -> Iterator[_BaseRef]:  # noqa: D102
        return iter([self.target])


@dataclasses.dataclass(frozen=True, repr=False)
class CallModule(Call):
    """Call a module on batched data, entering the context around each invocation."""

    caller: torch.nn.Module


@dataclasses.dataclass(frozen=True, repr=False)
class CallFunction(Call):
    """Call a free function on batched data."""


@dataclasses.dataclass(frozen=True, repr=False)
class CallMethod(Call):
    """Call a bound method on batched data."""


@dataclasses.dataclass(frozen=True)
class BundleSpec:
    """One bundle handed to a delegate: which refs to gather, under which context.

    A data-flow declaration names both halves, and they vary together: one flow
    may want the region's inputs under an unquantized context while the next wants
    the region's output under a quantized one. Pairing them here keeps a ref from
    being gathered under a context that never produced it.

    Args:
        context: The context to resolve the refs under.
        args: Positional refs to gather.
        kwargs: Keyword refs to gather.
        unwrap: If `args` holds a single value, return that value instead of a bundle.
    """

    context: StreamKey
    args: Sequence[_BaseRef]
    kwargs: Mapping[str, _BaseRef] = dataclasses.field(default_factory=dict)
    unwrap: bool = False


def _gather(spec: BundleSpec, register: ActivationRegister) -> Any:
    """Return what a delegate receives for `spec`: a bundle, or the value of an unwrapped spec."""
    if not spec.unwrap:
        return ActivationBundle.gather(register, spec.context, spec.args, spec.kwargs)

    (ref,) = spec.args

    # The result of an op is always a single batch (even if that batch is an iterable).
    (result,) = register.load(ref, spec.context).batches
    return result


@dataclasses.dataclass(frozen=True)
class RunDelegate(Instruction):
    """Run a function once on a set of ordered data flows and optionally store the result.

    The function `fn` can be any callable that expects an optional module as first argument
    and then a sequence of data flows (in order) that are converted by the scheduler
    to `bundles`. This instruction can be used to optimize the module with an algorithm,
    calculate statistics, etc.

    If you want to store data to the register, make sure the function outputs
    exactly as much items as there are `targets`.

    Args:
        fn: The function to call.
        bundles: The bundles to gather for `fn`.
        targets: The ref and context that each returned dataset is stored under.
        module: The module whose weights must be on the compute location during the
            call, if any.
    """

    fn: Callable[..., Sequence[ActivationDataset] | None]
    bundles: Sequence[BundleSpec]
    targets: Sequence[tuple[_BaseRef, StreamKey]] = ()
    module: torch.nn.Module | None = None

    def __repr__(self) -> str:
        args = [ref for bundle in self.bundles for ref in bundle.args]
        kwargs = {key: ref for bundle in self.bundles for key, ref in bundle.kwargs.items()}
        contexts = [bundle.context for bundle in self.bundles]
        module = None if self.module is None else _fmt_module(self.module)
        targets = [f"{ref!r}@{_fmt_callable(context)}" for ref, context in self.targets]
        return (
            f"CallDelegate(fn={_fmt_callable(self.fn)}, module={module}, args={args!r}, "
            f"kwargs={kwargs!r}, contexts={_fmt_contexts(contexts)}, "
            f"targets=[{', '.join(targets)}])"
        )

    def execute(self, register: ActivationRegister) -> None:  # noqa: D102
        fn_inputs: list[Any] = [_gather(spec, register) for spec in self.bundles]
        if self.module is not None:
            fn_inputs = [self.module, *fn_inputs]

        results = self.fn(*fn_inputs)
        datasets = () if results is None else results
        if len(datasets) != len(self.targets):
            msg = (
                f"Delegate {_fmt_callable(self.fn)} returned {len(datasets)} datasets for "
                f"{len(self.targets)} targets."
            )
            raise ValueError(msg)

        for (ref, context), dataset in zip(self.targets, datasets):
            register.store(ref, context, dataset)

    def uses(self) -> Iterator[_BaseRef]:  # noqa: D102
        for spec in self.bundles:
            yield from spec.args
            yield from spec.kwargs.values()

    def produces(self) -> Iterator[_BaseRef]:  # noqa: D102
        for ref, _ in self.targets:
            yield ref


def call_flow_op(
    fn: Callable[..., Any], names: Sequence[str], *inputs: Any
) -> list[ActivationDataset]:
    """Call `fn(**inputs)` with each input under its name, and store the result as one batch.

    Args:
        fn: The function of the op. It can return any object.
        names: The name that `fn` receives each item of `inputs` under.
        inputs: The data that `fn` reads, one item per name: a bundle for a flow, and the
            result of the op for an op.
    """
    return [ActivationDataset([fn(**dict(zip(names, inputs, strict=True)))])]


def loss_grad_outputs(
    objective: Callable[..., Iterable[torch.Tensor]],
    names: Sequence[str],
    values: Mapping[str, Any],
    *outputs: ActivationBundle,
) -> list[ActivationDataset]:
    """Call an objective once and return the gradient of each output stream it reads.

    Invokes `objective(**outputs, **values)` under `torch.enable_grad()`. Each item of
    `outputs` goes to `objective` under the name at the same position in `names`, and
    its floating-point leaves are made differentiable once per call. `values` go to
    `objective` unchanged. `objective` iterates the model outputs itself and yields one
    real scalar loss per batch. The gradient of an output batch is the gradient of the
    loss of that batch, with `None` for leaves that are not floating-point tensors or
    that have no path to the loss. `objective` places its data itself, so each
    gradient is where its output leaf is.

    Each item of `outputs` holds the graph outputs as positional streams.

    Args:
        objective: The objective.
        names: The name that `objective` receives each item of `outputs` under.
        values: Other values that `objective` receives unchanged, keyed by name.
        outputs: The model outputs that `objective` reads, one item per name.

    Returns:
        One gradient dataset per output stream, in the order of `outputs`.

    Raises:
        ValueError: If the streams have different batch counts, if `objective` does not
            yield one real scalar loss per batch, or if the loss of a batch reads
            a different batch.
    """
    name = _fmt_callable(objective)

    # The flattened batches of each output stream, with differentiable leaves.
    streams: list[list[tuple[list[Any], pytree.TreeSpec]]] = []
    differentiable: dict[str, ActivationBundle] = {}
    for key, output in zip(names, outputs, strict=True):
        datasets = []
        for dataset in output.args:
            stream = [_differentiable_leaves(batch) for batch in dataset]
            streams.append(stream)
            datasets.append(ActivationDataset([pytree.tree_unflatten(*batch) for batch in stream]))
        differentiable[key] = ActivationBundle(args=tuple(datasets), kwargs={})

    n_batches = len(streams[0]) if streams else 0
    if any(len(stream) != n_batches for stream in streams):
        msg = f"The sources of objective {name} have different batch counts."
        raise ValueError(msg)

    # Every differentiable leaf, with the index of the batch it belongs to.
    owned = [
        (idx, leaf)
        for stream in streams
        for idx, (leaves, _) in enumerate(stream)
        for leaf in leaves
        if _is_differentiable(leaf)
    ]
    grad_outputs: list[list[Any]] = [[] for _ in streams]

    n_losses = 0
    with torch.enable_grad():
        for idx, loss in enumerate(objective(**differentiable, **values)):
            n_losses = idx + 1
            _check_loss(name, idx, loss, n_batches)
            grads = _batch_grads(name, idx, loss, owned)
            for stream, stream_grad_outputs in zip(streams, grad_outputs):
                leaves, spec = stream[idx]
                stream_grad_outputs.append(
                    pytree.tree_unflatten([grads.get(id(leaf)) for leaf in leaves], spec)
                )
            del loss, grads

    if n_losses != n_batches:
        msg = f"Objective {name} yielded {n_losses} losses for {n_batches} batches."
        raise ValueError(msg)

    return [ActivationDataset(stream_grad_outputs) for stream_grad_outputs in grad_outputs]


def _check_loss(name: str, idx: int, loss: Any, n_batches: int) -> None:
    """Raise if `loss` is not a real scalar for batch `idx` of `n_batches`."""
    if idx >= n_batches:
        msg = f"Objective {name} yielded more losses than its {n_batches} batches."
        raise ValueError(msg)
    if not (isinstance(loss, torch.Tensor) and loss.ndim == 0 and loss.is_floating_point()):
        description = (
            f"a tensor of shape {tuple(loss.shape)} and dtype {loss.dtype}"
            if isinstance(loss, torch.Tensor)
            else type(loss).__name__
        )
        msg = (
            f"Objective {name} must yield one real scalar loss per batch, "
            f"but batch {idx} gave {description}."
        )
        raise ValueError(msg)


def _batch_grads(
    name: str, idx: int, loss: torch.Tensor, owned: list[tuple[int, torch.Tensor]]
) -> dict[int, torch.Tensor]:
    """Return the gradient of `loss` for each leaf it reaches, keyed by leaf `id`.

    Raises:
        ValueError: If `loss` reaches a leaf of a batch other than `idx`.
    """
    if not loss.requires_grad or not owned:
        return {}
    # With `allow_unused`, a leaf the loss does not reach gets None.
    results: Sequence[torch.Tensor | None] = torch.autograd.grad(
        loss, [leaf for _, leaf in owned], allow_unused=True
    )
    grads: dict[int, torch.Tensor] = {}
    for (owner, leaf), grad in zip(owned, results):
        if grad is None:
            continue
        if owner != idx:
            msg = f"The loss of batch {idx} of objective {name} depends on batch {owner}."
            raise ValueError(msg)
        grads[id(leaf)] = grad
    return grads


@dataclasses.dataclass(frozen=True)
class ReturnOutputs(Instruction):
    """Return output values from the register.

    Returns a dict mapping each execution context to its output values.
    Output values are automatically unpacked based on batch/output count.
    """

    outputs: Sequence[_BaseRef]

    def execute(self, register: ActivationRegister) -> Any:  # noqa: D102
        # Invert register[output_ref][context] to contexts[context] = [ds1, ds2, ...],
        # and use this to create a single ActivationDataset per context.
        context_outputs: dict[StreamKey, list[ActivationDataset]] = defaultdict(list)
        for output_ref in self.outputs:
            for context, dataset in register.items_for(output_ref):
                context_outputs[context].append(dataset)

        context_datasets = {
            context: tuple(ActivationDataset.merge(datasets).batches)
            for context, datasets in context_outputs.items()
        }

        return context_datasets

    def uses(self) -> Iterator[_BaseRef]:  # noqa: D102
        return iter(self.outputs)


@dataclasses.dataclass(frozen=True)
class DeleteRegisterEntries(Instruction):
    """Delete specified register entries to free memory."""

    targets: Sequence[_BaseRef]

    def execute(self, register: ActivationRegister) -> None:  # noqa: D102
        for target_id in self.targets:
            register.delete(target_id)


@dataclasses.dataclass(frozen=True)
class MoveModule(Instruction):
    """Move module parameters and buffers to a target location.

    Args:
        location: Target location for all parameters and buffers, or a mapping from name
            to location. When a mapping is provided, each named parameter/buffer is moved
            to its corresponding location.
        module: Module whose parameters and buffers will be moved.
    """

    location: Location | Mapping[str, Location]
    module: torch.nn.Module

    def __repr__(self) -> str:
        return f"MoveModule(location={self.location!r}, module={_fmt_module(self.module)})"

    def execute(self, register: ActivationRegister) -> None:  # noqa: D102, ARG002
        if isinstance(self.location, Mapping):
            for name, parameter in self.module.named_parameters():
                if (location := self.location.get(name)) is not None:
                    parameter.data = location.receive(parameter.data)
            for name, buffer in self.module.named_buffers():
                if (location := self.location.get(name)) is not None:
                    buffer.data = location.receive(buffer.data)
        else:
            for parameter in self.module.parameters():
                parameter.data = self.location.receive(parameter.data)
            for buffer in self.module.buffers():
                buffer.data = self.location.receive(buffer.data)


def _is_differentiable(leaf: Any) -> bool:
    """Return whether `leaf` gets a gradient: only floating-point tensors do."""
    return isinstance(leaf, torch.Tensor) and leaf.is_floating_point()


def _differentiable_leaves(value: Any) -> tuple[list[Any], pytree.TreeSpec]:
    """Flatten `value`, with its floating-point leaves as fresh autograd leaves."""
    leaves, spec = pytree.tree_flatten(value)
    leaves = [
        leaf.detach().requires_grad_() if _is_differentiable(leaf) else leaf for leaf in leaves
    ]
    return leaves, spec


def sum_grad_outputs(*grad_outputs: Any) -> Any:
    """Add the gradient contributions of one value, leaf by leaf.

    Each contribution has the structure of the value, or is `None` as a whole. A `None`
    leaf is no contribution, and a leaf with no contribution at all stays `None`.

    Args:
        grad_outputs: The contributions of one batch.

    Returns:
        The total gradient of the batch.
    """

    def add(*leaves: torch.Tensor | None) -> torch.Tensor | None:
        present = [leaf for leaf in leaves if leaf is not None]
        return functools.reduce(torch.add, present) if present else None

    present = [grad_output for grad_output in grad_outputs if grad_output is not None]
    if not present:
        return None
    return pytree.tree_map(add, *present, is_leaf=lambda leaf: leaf is None)


def _key_part(key: Any) -> str | int:
    """Return the item or attribute that one pytree key path entry names, as `get_attribute` reads it."""
    match key:
        case pytree.SequenceKey(idx=idx):
            return idx
        case pytree.MappingKey(key=str() | int() as item):
            return item
        case pytree.GetAttrKey(name=name):
            return name
    msg = f"Unsupported pytree key {key!r}."
    raise TypeError(msg)


def _non_negative(path: AttributePath, output: Any) -> AttributePath:
    """Return `path` with each negative index replaced by the index it names in `output`."""
    parts: list[str | int] = []
    for part in path:
        if isinstance(part, int) and part < 0:
            part += len(output)
        parts.append(part)
        output = get_attribute(output, part)
    return tuple(parts)


def assemble_grad_output(paths: Sequence[AttributePath], output: Any, *grad_outputs: Any) -> Any:
    """Build the gradient of `output` from the gradients of its parts.

    Each gradient belongs to the part of `output` at the path at the same position in
    `paths`; `()` is the full output. Contributions to one leaf are added, and a leaf
    with no contribution gets `None`.

    Args:
        paths: The path from `output` to the part that each gradient belongs to.
        output: One batch of the value the gradient is for.
        grad_outputs: The gradient of each part, or `None` as a whole for no contribution.

    Returns:
        The gradient of the batch, with the structure of `output`, or `None` if no
        part has a gradient.
    """
    keyed, spec = pytree.tree_flatten_with_path(output)
    keys = [tuple(_key_part(key) for key in keypath) for keypath, _ in keyed]

    def place(path: AttributePath, grad_output: Any) -> Any:
        path = _non_negative(path, output)
        leaves = [
            functools.reduce(get_attribute, key[len(path) :], grad_output)
            if grad_output is not None and key[: len(path)] == path
            else None
            for key in keys
        ]
        return pytree.tree_unflatten(leaves, spec)

    return sum_grad_outputs(
        *(place(path, grad_output) for path, grad_output in zip(paths, grad_outputs, strict=True))
    )


@dataclasses.dataclass(frozen=True, repr=False)
class CallVJP(Call):
    """Run one node backward: store the gradients of its inputs for the gradient of its output.

    For each batch, the node runs again under `torch.enable_grad()` on detached copies
    of its floating-point inputs, so the Vector-Jacobian Product (VJP) never follows an
    autograd graph a stored input already has, and only one local graph exists at a time.
    The stored value of each batch maps each argument of the node, by position or by keyword,
    to its gradient, with the structure of the argument. Leaves that are not floating-point
    tensors, and leaves with no path to the gradient of the output, get `None`.

    The last ref in `args` is the gradient of the output, not an input of the node. It has
    the structure of the output, and a `None` leaf, or `None` as a whole, gives no
    contribution.
    """

    def _single_batch(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Return the gradient of each argument of one batch, keyed by position or keyword."""
        args, grad_output = args[:-1], args[-1]
        arguments: dict[int | str, Any] = dict(enumerate(args))
        arguments.update(kwargs)
        leaves, spec = pytree.tree_flatten(arguments)
        grads: list[torch.Tensor | None] = [None] * len(leaves)
        differentiable = [idx for idx, leaf in enumerate(leaves) if _is_differentiable(leaf)]
        if not differentiable or all(leaf is None for leaf in pytree.tree_leaves(grad_output)):
            return pytree.tree_unflatten(grads, spec)

        inputs = list(leaves)
        for idx in differentiable:
            inputs[idx] = leaves[idx].detach().requires_grad_()
        with torch.enable_grad():
            # The node gets copies, so an in-place operation changes neither a leaf that
            # requires grad nor the stored activation. The copies are made with grad
            # enabled, so that they stay connected to the leaves under an ambient `no_grad`.
            copies = [
                inputs[idx].clone() if idx in differentiable else leaf
                for idx, leaf in enumerate(inputs)
            ]
            call = pytree.tree_unflatten(copies, spec)
            output = self.caller(
                *(call[idx] for idx in range(len(args))), **{key: call[key] for key in kwargs}
            )

        outputs: list[torch.Tensor] = []
        grad_outputs: list[torch.Tensor] = []

        def pair(leaf: torch.Tensor | None, value: Any) -> None:
            if leaf is not None and isinstance(value, torch.Tensor) and value.requires_grad:
                outputs.append(value)
                grad_outputs.append(leaf)

        pytree.tree_map(pair, grad_output, output, is_leaf=lambda leaf: leaf is None)
        del output

        if outputs:
            results = torch.autograd.grad(
                outputs, [inputs[idx] for idx in differentiable], grad_outputs, allow_unused=True
            )
            for idx, grad in zip(differentiable, results):
                grads[idx] = grad
        return pytree.tree_unflatten(grads, spec)


Instructions: TypeAlias = Sequence[Instruction]
InstructionPass: TypeAlias = Callable[[Instructions], Instructions]


@dataclasses.dataclass(frozen=True)
class InstructionProgram:
    """A scheduled program consisting of instructions and input metadata.

    Args:
        instructions: Sequence of instructions to execute.
        input_refs: Mapping from input names to InputRef objects.
    """

    instructions: Instructions
    input_refs: dict[str, InputRef]

    @property
    def contexts(self) -> Contexts:
        """All contexts used in program."""
        all_contexts: set[StreamKey] = set()

        for instruction in self.instructions:
            match instruction:
                case Call(context=context) | StoreValue(context=context):
                    all_contexts.add(context)
                case RunDelegate(bundles=bundles, targets=targets):
                    all_contexts.update(spec.context for spec in bundles)
                    all_contexts.update(context for _, context in targets)
                case _:
                    pass

        return list(all_contexts)


@dataclasses.dataclass(frozen=True)
class InstructionEngine:
    """Executes pre-scheduled instruction sequences.

    This engine runs instructions in order using a register-based approach
    to track intermediate values and activations. Instructions are generated
    by the scheduler during the scheduling phase.
    """

    @staticmethod
    def prepare_input_register(
        input_refs: dict[str, InputRef],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        contexts: Contexts,
    ) -> ActivationRegister:
        """Prepare execution register from user inputs.

        Args:
            input_refs: Mapping from input names to InputRef objects.
            args: Positional arguments from user.
            kwargs: Keyword arguments from user.
            contexts: Each input will be replicated across all provided contexts.

        Returns:
            register mapping InputRefs to ActivationDataset values.

        Raises:
            TypeError: If arguments don't match graph inputs.
        """
        input_names = list(input_refs.keys())

        if len(args) > len(input_names):
            msg = f"Expected {len(input_names)} positional arguments, got {len(args)}"
            raise TypeError(msg)

        inputs: dict[str, Any] = dict(zip(input_names[: len(args)], args))

        for name, value in kwargs.items():
            if name not in input_refs:
                msg = f"Unexpected keyword argument: '{name}'"
                raise TypeError(msg)
            if name in inputs:
                msg = f"Multiple values for argument: '{name}'"
                raise TypeError(msg)
            inputs[name] = value

        if missing := set(input_names) - inputs.keys():
            msg = f"Missing required inputs: {sorted(missing)}"
            raise TypeError(msg)

        register = ActivationRegister()
        for input_name, value in inputs.items():
            ref = input_refs[input_name]
            dataset = ActivationDataset.from_value(value)
            for context in contexts:
                register.store(ref, context, dataset)
        return register

    @staticmethod
    def run_instructions(instructions: Instructions, register: ActivationRegister) -> Any:
        """Run a sequence of instructions with the given register.

        Args:
            instructions: Sequence of instructions to execute.
            register: The execution register.

        Returns:
            Result from RETURN instruction, or None if no RETURN is executed.
        """
        for instruction in instructions:
            if (result := instruction.execute(register)) is not None:
                return result

    def run(self, program: InstructionProgram, *args: Any, **kwargs: Any) -> Any:
        """Run instructions with provided inputs.

        Args:
            program: InstructionProgram containing instructions and input metadata.
            *args: Positional inputs for graph.
            **kwargs: Keyword inputs for graph.

        Returns:
            Result from instruction execution.
        """
        register = self.prepare_input_register(program.input_refs, args, kwargs, program.contexts)
        return self.run_instructions(program.instructions, register)


class InstructionPasses:
    """Applies a sequence of instruction passes to a program.

    Each pass is a pure function transforming an instruction sequence into a new one.
    Passes are applied in order.
    """

    @staticmethod
    def apply(
        program: InstructionProgram, passes: Sequence[InstructionPass] | None = None
    ) -> InstructionProgram:
        """Apply the given passes to the program's instruction sequence.

        Args:
            program: The instruction program to transform.
            passes: Passes to apply in order. If None, the program is returned unchanged.

        Returns:
            A new program with the transformed instruction sequence.
        """
        instructions = program.instructions
        for pass_fn in passes or []:
            instructions = pass_fn(instructions)
        return dataclasses.replace(program, instructions=instructions)


def lifetime_management_pass(instructions: Instructions) -> Instructions:
    """Insert DeleteRegisterEntries instructions to free memory when values are no longer needed.

    Args:
        instructions: Sequence of instructions to analyze.

    Returns:
        New instruction sequence with DeleteRegisterEntries instructions inserted.
    """
    keep_alive: set[_BaseRef] = set()
    for instruction in instructions:
        if isinstance(instruction, ReturnOutputs):
            keep_alive.update(instruction.uses())

    last_use = {}

    # Iterate in reverse to find the last instruction that depends on each register slot
    for idx in range(len(instructions) - 1, -1, -1):
        instruction = instructions[idx]
        for uuid_id in itertools.chain(instruction.uses(), instruction.produces()):
            if uuid_id not in last_use:
                last_use[uuid_id] = idx

    new_instructions: list[Instruction] = []
    for idx, instruction in enumerate(instructions):
        new_instructions.append(instruction)

        # Delete any register entry that has no dependent instructions after this point
        to_delete = [
            uuid_id
            for uuid_id, last_idx in last_use.items()
            if last_idx == idx and uuid_id not in keep_alive
        ]

        if to_delete:
            new_instructions.append(DeleteRegisterEntries(targets=to_delete))

    return tuple(new_instructions)


def _weight_offloading_pass(
    instructions: Instructions,
    compute: Location,
    storage: Location,
    graph: GraphModule,
) -> Instructions:
    """Insert `MoveModule` instructions to move module weights between locations.

    First, record the original device placement of all weights so the model can be restored
    to its initial state after the pass (`post_restore`). Next, move each weight to `storage`
    to perform the actual offload. Finally, wrap each `CallModule`, `CallVJP` and `CallDelegate` with the appropriate
    placement: `compute` for execution and `storage` for storage.

    If we have a instruction stream that goes through two linear layers L1 -> L2, the pass would add
    offload(L1), Offload(L2), Load(L1), Call(L1), Offload(L1), Load(L2), Call(L2), offload(L2), Load(L1), Load(L2).

    NB: We need access to `GraphModule` because during optimization not all parameters have
    to be present in the instruction stream even if they are still possibly on `compute`.

    Args:
        instructions: Sequence of instructions to analyze.
        compute: Where `CallModule`, `CallVJP` and `CallDelegate` execution happens.
        storage: Where we 'offload' to.
        graph: Original GraphModule — all node modules are pre- and post-offloaded.

    Returns:
        New instruction sequence with `MoveModule` instructions inserted.
    """
    all_modules = list(
        dict.fromkeys(
            node.target
            for node in graph._nodes.values()
            if node.op is Op.torch_module and isinstance(node.target, torch.nn.Module)
        )
    )

    # Ensure `post_restore` maps each parameter back to its individual original location.
    original_locations: dict[torch.nn.Module, dict[str, Location]] = {}
    for m in all_modules:
        locations: dict[str, Location] = {
            name: DeviceLocation(param.device) for name, param in m.named_parameters()
        }
        locations.update({name: DeviceLocation(buf.device) for name, buf in m.named_buffers()})
        if locations:
            original_locations[m] = locations

    post_restore = [
        MoveModule(location=locations, module=m) for m, locations in original_locations.items()
    ]

    pre_offload = [MoveModule(location=storage, module=m) for m in all_modules]

    new_instructions: list[Instruction] = [*pre_offload]

    for instruction in instructions:
        match instruction:
            case (
                CallModule(caller=module) | CallVJP(caller=module) | RunDelegate(module=module)
            ) if isinstance(module, torch.nn.Module):
                new_instructions.append(MoveModule(location=compute, module=module))
                new_instructions.append(instruction)
                new_instructions.append(MoveModule(location=storage, module=module))
            case _:
                new_instructions.append(instruction)

    new_instructions.extend(post_restore)
    return tuple(new_instructions)


def _activation_name(ref: _BaseRef) -> str:
    """Return the name of the activation that the value under `ref` belongs to.

    A gradient is stored like the activation it is taken at, so the activation policy
    treats both the same way.
    """
    match ref:
        case NamedRef(name=name) | LossGradRef(name=name):
            return name
        case GradientRef(reference=reference) | VJPRef(reference=reference):
            return reference.name
    msg = f"{ref!r} belongs to no activation."
    raise TypeError(msg)


def _activation_offloading_pass(
    instructions: Instructions, compute: Location, storage: LocationPolicy
) -> Instructions:
    """Configure input and output locations on each `Call`.

    Before each invocation, moves input activations to `compute`. After each
    invocation, moves the output activation to the location selected by
    `storage`, one batch at a time.
    Input datasets remain in the register at their existing locations.

    Args:
        instructions: Sequence of instructions to analyze.
        compute: Where activations are moved to before execution.
        storage: A callable returning the storage location for each activation name.

    Returns:
        Instructions with per-batch placement configured on calls.
    """
    new_instructions: list[Instruction] = []

    for instruction in instructions:
        if isinstance(instruction, Call):
            name = _activation_name(instruction.target.unwrap_ref())
            new_instructions.append(
                dataclasses.replace(
                    instruction,
                    input_location=compute,
                    output_location=storage(name),
                )
            )
        else:
            new_instructions.append(instruction)

    return tuple(new_instructions)


def _offloading_pass(
    instructions: Instructions,
    compute: Location,
    weights: Location,
    activations: LocationPolicy,
    graph: GraphModule,
) -> Instructions:
    """Configure module placement and per-batch activation offloading.

    Composes `_weight_offloading_pass` and `_activation_offloading_pass` to handle both
    module weight movement and per-batch activation movement.

    Args:
        instructions: Original instruction sequence.
        compute: Where data is moved to before execution.
        weights: Where idle module weights rest.
        activations: A callable returning the storage location for each activation name.
        graph: Original GraphModule — all node modules are pre- and post-offloaded.

    Returns:
        New instruction sequence with module and activation placement configured.
    """
    instructions = _weight_offloading_pass(instructions, compute, weights, graph)
    instructions = _activation_offloading_pass(instructions, compute, activations)

    # Cancel out compute(M1) -> storage(M1) -> compute(M1) placements for any module M1.
    instructions = _cancel_module_round_trips(instructions, compute, weights)
    return instructions


def _cancel_module_round_trips(
    instructions: Instructions, compute: Location, storage: Location
) -> Instructions:
    """Cancel adjacent MoveModule round-trips."""
    location_pair = {compute, storage}

    def _is_round_trip(left: Instruction, right: Instruction) -> bool:
        return (
            isinstance(left, MoveModule)
            and isinstance(right, MoveModule)
            and left.module is right.module
            and isinstance(left.location, Location)
            and isinstance(right.location, Location)
            and {left.location, right.location} == location_pair
        )

    result: list[Instruction] = []
    i = 0
    while i < len(instructions):
        if i + 1 < len(instructions) and _is_round_trip(instructions[i], instructions[i + 1]):
            i += 2
        else:
            result.append(instructions[i])
            i += 1

    return tuple(result)


class OffloadingStrategy(abc.ABC):
    """Abstract base for offloading strategies.

    An offloading strategy controls how module weights and activations are moved
    between locations during graph execution. Implement `create_instruction_pass` to
    configure module and activation placement.
    """

    @abc.abstractmethod
    def create_instruction_pass(self, graph: GraphModule) -> InstructionPass:
        """Return an instruction pass that configures placement.

        Args:
            graph: The GraphModule being scheduled.

        Returns:
            An `InstructionPass` that configures placement in the instruction sequence.
        """


def _as_location_rule(policy: LocationLike | LocationPolicyLike) -> LocationPolicy:
    """Convert a location or policy to a callable returning concrete locations.

    Args:
        policy: A fixed location, or a callable selecting a location by name.

    Returns:
        A callable returning a `Location` for a given name.
    """
    if callable(policy):
        return lambda name: as_location(policy(name))
    location = as_location(policy)
    return lambda _name: location


@attrs.define(frozen=True)
class Offload(OffloadingStrategy):
    """Offload model weights and activations between computation steps.

    `compute` specifies the device used for computation. `weights` and `activations`
    specify where their respective tensors are stored between uses. Both accept a
    `Location`, a device name or `torch.device`, or a directory path for disk storage.
    Weights are restored to their original devices when execution finishes.

    `activations` also accepts a location policy: a callable that takes an activation
    name and returns its storage location. This allows you to be selective in which
    activations you offload to, for example, disk or ram.

    Example:
        ```python
        scratch = pathlib.Path("/scratch/activations")

        def where(name: str) -> LocationLike:
            # We calculate gradients at start but only need them later - store to disk.
            return scratch if "grad" in name else "cpu"

        ff.layerwise_optimize(
            model,
            data,
            gptq,
            sample_args=(data[0],),
            offloading=Offload(compute="cuda:0", weights="cpu", activations=where),
        )
        ```

    Args:
        compute: Device where computation happens, e.g. `"cuda:0"`.
        weights: Storage location for weights. Strings name devices; use a path object
            such as `pathlib.Path` to specify a directory for disk storage.
        activations: Storage location for activations, using the same forms as `weights`,
            or a callable selecting a storage location by activation name.

    Raises:
        TypeError: If `compute` names something other than a device.
        ValueError: If a string names no device at all.
    """

    compute: DeviceLocation = attrs.field(
        converter=as_location, validator=attrs.validators.instance_of(DeviceLocation)
    )
    weights: Location = attrs.field(
        default=DeviceLocation(torch.device("cpu")), converter=as_location
    )
    activations: LocationPolicy = attrs.field(
        default=_as_location_rule(DeviceLocation(torch.device("cpu"))),
        converter=_as_location_rule,
    )

    def create_instruction_pass(self, graph: GraphModule) -> InstructionPass:  # noqa: D102
        def _pass(instructions: Instructions) -> Instructions:
            return _offloading_pass(
                instructions, self.compute, self.weights, self.activations, graph
            )

        return _pass
