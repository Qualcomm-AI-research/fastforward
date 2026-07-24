# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Persist and restore quantization state and quantized-model artifacts.

This module owns the on-disk representation of a quantized model. Two related
formats are provided:

- **Quantization state** (`save_quantization_state` / `load_quantization_state`):
  the quantizer parameters and configuration only, written into the FastForward
  asset cache under `quantization-state/{name_or_path}/{tag}/`. Used to restore
  the result of a PTQ run onto a freshly quantized model without re-running it.
- **Quantized-model artifact** (`save_quantized_model` / `load_quantized_model`):
  a self-contained directory holding the quantizer state *plus* the model weights
  and a manifest. A superset of the quantization state, suitable for shipping.

Both formats share a single writer (`_write_quantizer_state`) and reader
(`_load_quantizer_state_from_files`), so the quantizer config/params layout is
identical between them and cannot drift.

The functions here take the model as their first argument and are mirrored by
thin delegating methods on `QuantizedModule`, so either calling style works:

    model.save_quantized_model(path)
    ff.quantization.save_quantized_model(model, path)

`save_quantized_model` snapshots the model exactly as it is; it does not
transform the weights. Callers who want grid-snapped (QDQ) or stubbed weights
run `fastforward.quantization.fuse.fuse_qdq_weights` themselves before saving,
which owns that transform.
"""

from __future__ import annotations

import json
import logging

from collections import defaultdict
from operator import attrgetter
from pathlib import Path

import torch
import yaml

from safetensors import safe_open
from safetensors.torch import save_file

import fastforward as ff

from fastforward.cache import get_assets_path
from fastforward.exceptions import QuantizationError
from fastforward.nn.quantizer import Quantizer, QuantizerStub
from fastforward.quantization.quant_init import _OverwriteOptions

logger = logging.getLogger(__name__)

# Metadata key marking a quantizer-state file that holds no tensors. safetensors
# writes an unparseable header for an empty tensor dict with empty metadata, so a
# marker entry is written instead to keep the file readable.
_EMPTY_STATE_MARKER = "__ff_empty_quantizer_state__"


def _deduplicate_shared_tensors(
    tensors: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Split `tensors` into storage-unique entries plus an alias map.

    SafeTensors refuses to serialize tensors that share memory, but tied
    parameters (e.g. a language model's `lm_head.weight` tied to
    `embed_tokens.weight`) are common. This groups entries by their underlying
    storage and keeps a single representative per group, recording the dropped
    names so they can be restored on load.

    The representative is the lexicographically first name in the group, matching
    the convention `_collect_quantizer_state` uses for shared quantizers.

    Args:
        tensors: Mapping of name to tensor, e.g. a (partial) state dict.

    Returns:
        Tuple of `(unique, aliases)`, where `unique` maps each representative
        name to its tensor and `aliases` maps each dropped name to the
        representative name holding its values.
    """
    groups: dict[tuple[int, torch.Size, tuple[int, ...], torch.device], list[str]] = {}
    for name, tensor in tensors.items():
        key = (tensor.data_ptr(), tensor.shape, tensor.stride(), tensor.device)
        groups.setdefault(key, []).append(name)

    unique: dict[str, torch.Tensor] = {}
    aliases: dict[str, str] = {}
    for names in groups.values():
        representative = min(names)
        unique[representative] = tensors[representative]
        for name in names:
            if name != representative:
                aliases[name] = representative
    return unique, aliases


def _expand_shared_tensors(
    tensors: dict[str, torch.Tensor], aliases: dict[str, str]
) -> dict[str, torch.Tensor]:
    """Restore alias entries dropped by `_deduplicate_shared_tensors`.

    Args:
        tensors: Mapping of representative name to tensor, as read from disk.
        aliases: Mapping of dropped name to representative name.

    Returns:
        `tensors` extended with an entry for every alias.

    Raises:
        RuntimeError: If an alias references a name that is not present.
    """
    expanded = dict(tensors)
    for name, representative in aliases.items():
        if representative not in tensors:
            msg = (
                f"Tied weight '{name}' references '{representative}', which is "
                "missing from the saved weights."
            )
            raise RuntimeError(msg)
        expanded[name] = tensors[representative]
    return expanded


def _collect_quantizer_state(
    model: torch.nn.Module, *, allow_lazy_params: bool = False
) -> tuple[dict[str, torch.Tensor], dict[str, str], dict[str, Quantizer]]:
    """Collect serializable quantizer parameters, metadata, and config.

    Walks all (non-stub) quantizers and produces the three structures used
    to persist quantization state:

    - `state`: quantizer parameters keyed by "first_quantizer_name.param".
      For shared quantizers (same instance used in multiple locations),
      parameters are stored only once under the lexicographically first
      quantizer name to avoid duplication. Example:
      `{"layer1.weight_quantizer.scale": tensor([1.0]), ...}`
    - `metadata`: maps each quantizer name to its parameter keys in the
      format "param=tensor_key", enabling reconstruction of individual
      quantizer state dicts during loading. Uninitialized parameters are
      decorated with a `::lazy` suffix. Example:
      `{"layer1.weight_quantizer": "scale=layer1.weight_quantizer.scale,..."}`
    - `config_quantizers`: maps each quantizer name to its quantizer object,
      for serialization into the YAML config.

    Args:
        model: The module whose quantizer state should be collected.
        allow_lazy_params: If False, a ValueError is raised when a quantizer
            has uninitialized (lazy) parameters. If True, a warning is
            emitted and those parameters are decorated as `::lazy`.

    Returns:
        Tuple of `(state, metadata, config_quantizers)`.
    """
    quantizers = defaultdict(list)
    for name, quantizer in ff.nn.quantized_module.named_quantizers(model, remove_duplicate=False):
        quantizers[quantizer].append(name)

    state: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] = {}

    for quantizer, names in quantizers.items():
        first_name = min(names)
        state_dict = quantizer.state_dict(keep_vars=True)
        lazy_param = set()
        for key, param in state_dict.items():
            if torch.nn.parameter.is_lazy(param):
                lazy_param.add(key)
            else:
                state[f"{first_name}.{key}"] = param.detach()

        if len(lazy_param) > 0:
            msg = (
                "A quantizer having lazy parameters (UninitializedParameter "
                f"or UninitializedBuffer) was found. Parameters: {lazy_param}.\n"
                "Tip: quantizers normally materialize the uninitialized "
                "parameters during range estimation."
            )
            if allow_lazy_params:
                logger.warning(msg)
            else:
                logger.error(msg)
                raise ValueError(msg)

        for name in names:
            params_metadata = [
                f"{param}={first_name}.{param}" + ("::lazy" if param in lazy_param else "")
                for param in state_dict.keys()
            ]
            metadata[name] = ",".join(params_metadata)

    config_quantizers = {
        name: quantizer for quantizer, names in quantizers.items() for name in names
    }
    return state, metadata, config_quantizers


def _resolve_name_or_path(model: torch.nn.Module, name_or_path: str | Path | None) -> str | Path:
    """Resolve the model identifier, falling back to `config.name_or_path`.

    Args:
        model: The model whose `config.name_or_path` is consulted as a fallback.
        name_or_path: Explicit identifier, or None to auto-detect.

    Returns:
        The resolved model identifier.

    Raises:
        RuntimeError: If no identifier could be determined.
    """
    if name_or_path is None:
        name_or_path = getattr(getattr(model, "config", None), "name_or_path", None)
    if name_or_path is None:
        msg = (
            "Unable to detect the model identifier. Please provide it manually "
            "if there is no `config.name_or_path` property in the model"
        )
        raise RuntimeError(msg)
    return name_or_path


def _write_quantizer_state(
    model: torch.nn.Module,
    target_dir: Path,
    name_or_path: str | Path,
    *,
    state_filename: str,
    allow_lazy_params: bool = False,
) -> Path:
    """Write the quantizer state (params + config) into `target_dir`.

    Shared by `save_quantization_state` and `save_quantized_model`, which
    differ only in the name of the SafeTensors file they write. Produces a
    `config.yaml` describing the quantizers plus a SafeTensors file holding
    their parameters, with the per-quantizer parameter map in its metadata.

    Args:
        model: The model whose quantizer state should be written.
        target_dir: Existing directory to write the two files into.
        name_or_path: Resolved model identifier, recorded in the config so it
            can be validated at load time.
        state_filename: File name for the quantizer parameters (e.g.
            'model.safetensors' or 'quantizer_state.safetensors').
        allow_lazy_params: If False, uninitialized quantizer parameters raise
            a ValueError; if True, a warning is emitted instead.

    Returns:
        Path to the written `config.yaml`.
    """
    transformers_version = getattr(getattr(model, "config", None), "transformers_version", None)
    state, metadata, config_quantizers = _collect_quantizer_state(
        model, allow_lazy_params=allow_lazy_params
    )

    config = {
        "version": "1.0",
        "name_or_path": str(name_or_path),
        "transformers_version": str(transformers_version),
        "fastforward_version": str(ff.__version__),
        "quantizers": config_quantizers,
    }
    # safetensors mis-serializes an empty tensor dict paired with empty
    # metadata, producing a file whose header fails to parse on read. A model
    # with no non-stub quantizers is legitimate (e.g. a QDQ_STUBBED artifact
    # whose weight quantizers were all replaced by stubs), so record an
    # explicit marker to keep the file readable by any consumer.
    if not state and not metadata:
        metadata = {_EMPTY_STATE_MARKER: "true"}
    save_file(state, target_dir / state_filename, metadata=metadata)
    config_path = target_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    return config_path


def save_quantization_state(
    model: torch.nn.Module,
    *,
    tag: str = "main",
    name_or_path: str | Path | None = None,
    cache_dir: Path | None = None,
    allow_lazy_params: bool = False,
) -> Path:
    """Save quantization state to disk for later restoration.

    Saves the quantization state of all quantizers in `model` to disk,
    including quantizer parameters, metadata, and configuration information.
    The state is saved as a SafeTensors file with accompanying YAML configuration.

    Args:
        model: The model whose quantization state should be saved.
        tag: Tag to identify this particular save. Used to organize multiple
            saves of the same model. Defaults to "main".
        name_or_path: Model identifier or path. If None, attempts to extract
            from the model's config.name_or_path attribute. Used to determine
            the save location and validate consistency during loading.
        cache_dir: Directory where quantization state should be cached. If None,
            uses the default cache directory from get_assets_path().
        allow_lazy_params:  If False, a ValueError will be raised when trying
            to save uninitialized parameters in the quantization state.
            If True, a warning will be raised instead and the uninitialized
            parameters will be decorated as `lazy` in the quantization state
            metadata.

    Returns:
        Path to the saved configuration file (config.yaml).

    Raises:
        RuntimeError: If the model identifier cannot be determined.
        ValueError: If the cache directory cannot be created due to an
            existing file with the same name.

    Note:
        The quantization state is saved in a directory structure:
        {cache_dir}/quantization-state/{name_or_path}/{tag}/
        containing 'model.safetensors' and 'config.yaml' files.
    """
    name_or_path = _resolve_name_or_path(model, name_or_path)
    assets_path = get_assets_path(f"quantization-state/{name_or_path}", tag, cache_dir=cache_dir)
    try:
        assets_path.mkdir(exist_ok=True, parents=True, mode=0o775)
    except (FileExistsError, NotADirectoryError) as e:
        msg = f"Cannot create directory {assets_path} because of an existing file."
        raise ValueError(msg) from e
    return _write_quantizer_state(
        model,
        assets_path,
        name_or_path,
        state_filename="model.safetensors",
        allow_lazy_params=allow_lazy_params,
    )


def load_quantization_state(
    model: torch.nn.Module,
    *,
    tag: str = "main",
    name_or_path: str | Path | None = None,
    cache_dir: Path | None = None,
    overwrite_policy: _OverwriteOptions = "error",
    allow_lazy_params: bool = False,
) -> None:
    """Load quantization state from saved files.

    Args:
        model: The model to load the quantization state into.
        tag: Tag used when saving the quantization state. Defaults to "main".
        name_or_path: Model identifier used when saving. If None, attempts to get from config.
        cache_dir: Directory where the quantization state was cached. If None, uses
            default cache.
        overwrite_policy: The policy to use when a loader quantizer is already present
            in the model. Options are 'skip', 'overwrite' and 'error'.
        allow_lazy_params: If False, uninitialized parameters encountered in the
            quantization state metadata will raise a ValueError.
            if True, uninitialized parameters will just print a warning message instead.


    Raises:
        RuntimeError: If the model identifier cannot be determined.
        FileNotFoundError: If the quantization state files are not found.
        ValueError: If the loaded configuration is incompatible.
    """
    name: str | None = getattr(getattr(model, "config", None), "name_or_path", None)
    if name_or_path is not None and not Path(name_or_path).exists():
        name = str(name_or_path)
    if name is None:
        raise RuntimeError(
            "Unable to detect the model identifier. Please provide it manually "
            "if there is no `config.name_or_path` property in the model"
        )
    if name_or_path is not None and Path(name_or_path).exists():
        config_path = Path(name_or_path)
    else:
        config_path = (
            get_assets_path(f"quantization-state/{name}", tag, cache_dir=cache_dir) / "config.yaml"
        )
    model_path = config_path.parent / "model.safetensors"

    # Check if files exist
    if not config_path.exists():
        msg = f"Quantization state config not found at {config_path}"
        raise FileNotFoundError(msg)
    if not model_path.exists():
        msg = f"Quantization state model not found at {model_path}"
        raise FileNotFoundError(msg)

    _load_quantizer_state_from_files(
        model,
        config_path,
        model_path,
        expected_name=name,
        overwrite_policy=overwrite_policy,
        allow_lazy_params=allow_lazy_params,
    )


def _load_quantizer_state_from_files(
    model: torch.nn.Module,
    config_path: Path,
    model_path: Path,
    *,
    expected_name: str | None = None,
    overwrite_policy: _OverwriteOptions = "error",
    allow_lazy_params: bool = False,
) -> None:
    """Reconstruct and reattach quantizers from a config/safetensors pair.

    Shared by `load_quantization_state` and `load_quantized_model`. Loads the
    YAML config, validates its version, reconstructs each quantizer's
    parameters from the SafeTensors file, and reattaches the quantizers to
    `model` honoring `overwrite_policy`.

    Args:
        model: The model to reattach the reconstructed quantizers to.
        config_path: Path to the `config.yaml` describing the quantizers.
        model_path: Path to the SafeTensors file holding quantizer params.
        expected_name: If provided, the `name_or_path` recorded in the config
            must match this value (guards against loading mismatched state).
        overwrite_policy: Policy applied when a quantizer is already
            initialized: 'error', 'skip', or 'overwrite'.
        allow_lazy_params: If False, lazy parameters recorded in the state
            raise a ValueError; if True, they emit a warning instead.

    Raises:
        ValueError: If the config version is unsupported or lazy params are
            disallowed.
        RuntimeError: If the model identifier mismatches or state keys are
            missing/unexpected.
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.load(f, yaml.Loader)

    # Validate configuration
    if config.get("version") != "1.0":
        msg = f"Unsupported quantization state version: {config.get('version')}"
        raise ValueError(msg)

    # if user provides a full path to the config, we assume he knows what he is doing
    if expected_name is not None and str(config.get("name_or_path")) != str(expected_name):
        msg = (
            f"Model identifier mismatch: expected '{expected_name}', "
            f"found '{config.get('name_or_path')}' in saved state"
        )
        logger.error(msg)
        raise RuntimeError(msg)

    quantizers: dict[str, Quantizer] = config.get("quantizers", {})
    # Reconstruct quantizer state_dict by parsing metadata to map parameter names to tensor keys.
    # The metadata format "param=tensor_key" allows to load the correct tensors for each
    # parameter. For shared quantizers, multiple quantizer names may reference the same tensor
    # keys.
    # A config may legitimately record no (non-stub) quantizers — e.g. a
    # QDQ_STUBBED artifact whose weight quantizers were all replaced by
    # stubs. In that case there is nothing to reconstruct, so skip opening the
    # (tensor-less) SafeTensors file.
    if quantizers:
        with safe_open(model_path, framework="pt") as f:
            metadata = f.metadata()

            for name, quantizer in quantizers.items():
                state_tensor_keys = {}
                lazy_params = []
                for key in metadata[f"{name}"].split(","):
                    state_key, tensor_key = key.split("=")

                    if "::" in tensor_key:
                        tensor_key, decorators = tensor_key.split("::")
                    else:
                        decorators = ""

                    if "lazy" in decorators:
                        lazy_params.append(state_key)
                    else:
                        state_tensor_keys[state_key] = tensor_key

                missing_keys, unexpected_keys = quantizer.load_state_dict(
                    {
                        state_key: f.get_tensor(tensor_key)
                        for state_key, tensor_key in state_tensor_keys.items()
                    },
                    strict=False,
                )

                if len(lazy_params) > 0:
                    msg = (
                        "Lazy parameters were found in quantization state "
                        f"and cannot be loaded. Parameters: {lazy_params}."
                    )
                    if allow_lazy_params:
                        logger.warning(msg)
                    else:
                        logger.error(msg)
                        raise ValueError(msg)

                missing_keys = sorted(set(missing_keys) - set(lazy_params))
                if missing_keys or unexpected_keys:
                    msg = (
                        f"There are some missing ({missing_keys}) or unexpected "
                        f"({unexpected_keys}) keys during loading state_dict"
                    )
                    logger.error(msg)
                    raise RuntimeError(msg)

    for name, quantizer in quantizers.items():
        parts = name.rsplit(".", 1)
        parent = model if len(parts) == 1 else attrgetter(parts[0])(model)
        parent_attribute = parts[-1]
        current_quantizer = getattr(parent, parent_attribute, None)

        is_quantizer = isinstance(current_quantizer, Quantizer)
        is_quantizer_stub = isinstance(current_quantizer, QuantizerStub)
        if is_quantizer and not is_quantizer_stub:
            if overwrite_policy == "error":
                msg = (
                    f"'{name}' is a quantizer, but is already initialized. If "
                    + 'you want to overwrite the existing quantizer, use overwrite_policy="overwrite" '
                    + "or if you want to skip loading existing quantizers use "
                    + 'overwrite_policy="skip"'
                )
                raise QuantizationError(msg)
            elif overwrite_policy == "skip":
                continue
            elif overwrite_policy != "overwrite":
                msg = (  # type: ignore[unreachable]
                    "Encountered a quantizer that was already initialized. Since "
                    + f"overwrite_policy={overwrite_policy} is illegal cannot resolve conflict."
                    + "please use 'error', 'skip', or 'overwrite"
                )
                raise QuantizationError(msg)
        if not is_quantizer:
            msg = f"'{name}' is not a quantizer or was overwritten by a non-quantizer object"
            raise ValueError(msg)

        setattr(parent, parent_attribute, quantizer)


def save_quantized_model(
    model: torch.nn.Module,
    path: str | Path,
    *,
    name_or_path: str | Path | None = None,
    allow_lazy_params: bool = False,
) -> Path:
    """Save a self-contained, reloadable quantized-model artifact.

    Writes both the model weights and the quantizer state into a single
    directory that can be reloaded with `load_quantized_model` without
    re-running PTQ. The artifact is a superset of the `save_quantization_state`
    output: it adds the model weights and a manifest.

    This is a pure snapshot of the model's *current* state: the weights and
    quantizers are written exactly as they are, with no transformation. To save
    grid-snapped (QDQ) weights, or to stub the weight quantizers, run
    `fastforward.quantization.fuse_qdq_weights(model, stub_quantizers=...)`
    yourself before calling this function. That fuse is in-place, so deep-copy
    the model first if the live one must be preserved for further training.

    The written directory contains:

        <path>/
        ├── config.yaml                  # quantizer config
        ├── quantizer_state.safetensors  # scale/offset per quantizer
        ├── weights.safetensors          # model weights
        └── manifest.json                # ff version, format metadata

    Tied parameters (e.g. a tied `lm_head.weight`/`embed_tokens.weight` pair)
    are stored once, with the dropped names recorded in the manifest's
    `tied_weights` map and restored by `load_quantized_model`.

    Args:
        model: The quantized model to save.
        path: Directory to write the artifact to (created if needed).
        name_or_path: Model identifier. If None, attempts to read
            `config.name_or_path`. Recorded in the config for validation.
        allow_lazy_params: If False, uninitialized quantizer parameters raise
            a ValueError; if True, a warning is emitted instead.

    Returns:
        Path to the artifact directory.

    Raises:
        RuntimeError: If the model identifier cannot be determined.
        ValueError: If the artifact directory cannot be created.
    """
    name_or_path = _resolve_name_or_path(model, name_or_path)

    artifact_path = Path(path)
    try:
        artifact_path.mkdir(exist_ok=True, parents=True, mode=0o775)
    except (FileExistsError, NotADirectoryError) as e:
        msg = f"Cannot create directory {artifact_path} because of an existing file."
        raise ValueError(msg) from e

    # Snapshot the model as-is: any grid-snapping/stubbing is the caller's
    # responsibility (via fuse_qdq_weights) before reaching this point.
    save_model = model

    _write_quantizer_state(
        save_model,
        artifact_path,
        name_or_path,
        state_filename="quantizer_state.safetensors",
        allow_lazy_params=allow_lazy_params,
    )

    # Model weights only (bias, norms, embeddings, ...): exclude quantizer
    # parameters/buffers, which are persisted separately in
    # `quantizer_state.safetensors`. Saving them here too would duplicate
    # state and trips safetensors' shared-memory guard for tied quantizers
    # (e.g. a shared symmetric offset buffer).
    quantizer_prefixes = tuple(
        f"{name}."
        for name, _ in ff.nn.quantized_module.named_quantizers(
            save_model, remove_duplicate=False, skip_stubs=False
        )
    )
    weight_state = {
        name: param.detach()
        for name, param in save_model.state_dict().items()
        if not name.startswith(quantizer_prefixes)
    }
    # SafeTensors rejects tensors sharing storage, so tied weights are stored
    # once and re-expanded on load from the manifest's alias map.
    weight_state, tied_weights = _deduplicate_shared_tensors(weight_state)
    save_file(weight_state, artifact_path / "weights.safetensors")

    manifest = {
        "version": "1.0",
        "name_or_path": str(name_or_path),
        "fastforward_version": str(ff.__version__),
        "tied_weights": tied_weights,
    }
    with open(artifact_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return artifact_path


def load_quantized_model(
    model: torch.nn.Module,
    path: str | Path,
    *,
    overwrite_policy: _OverwriteOptions = "error",
    allow_lazy_params: bool = False,
    expected_name: str | None = None,
) -> None:
    """Load a self-contained quantized-model artifact into `model`.

    Reconstructs the quantizers (attaching them to `model`) and loads the
    saved weights, restoring the exact state written by `save_quantized_model`.
    Loading is a pure snapshot restore: whatever quantizers and weights were
    saved are restored as-is, with no further transformation. In particular, it
    never stubs the target's weight quantizers. If an artifact holds already
    grid-snapped weights and you want those weight quantizers stubbed so a
    forward pass does not re-quantize them, either load into a freshly quantized
    model (whose weight quantizers are stubs by default) or call
    `fastforward.quantization.stub_weight_quantizers(model)` afterwards.

    Args:
        model: The model to load the artifact into.
        path: Directory containing the artifact (as written by
            `save_quantized_model`).
        overwrite_policy: Policy applied when a quantizer is already
            initialized: 'error', 'skip', or 'overwrite'.
        allow_lazy_params: If False, lazy parameters recorded in the state
            raise a ValueError; if True, a warning is emitted instead.
        expected_name: If provided, the `name_or_path` recorded in the
            artifact must match this value. Defaults to the model's
            `config.name_or_path` when available, guarding against loading an
            artifact that was saved for a different model. Pass an empty
            string to skip the check.

    Raises:
        FileNotFoundError: If any of the artifact files are missing.
        ValueError: If the manifest/config version is unsupported.
        RuntimeError: If the model identifier mismatches, or if the saved
            weights do not match the model's expected weight keys.
        QuantizationError: If a quantizer is already initialized and
            `overwrite_policy` is 'error'.

    Warning:
        Loading an artifact executes the class references recorded in its
        `config.yaml` in order to reconstruct the quantizers. Only load
        artifacts from trusted sources.
    """
    artifact_path = Path(path)
    config_path = artifact_path / "config.yaml"
    model_path = artifact_path / "quantizer_state.safetensors"
    weights_path = artifact_path / "weights.safetensors"
    manifest_path = artifact_path / "manifest.json"

    for required in (config_path, model_path, weights_path, manifest_path):
        if not required.exists():
            msg = f"Quantized model artifact file not found: {required}"
            raise FileNotFoundError(msg)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    if manifest.get("version") != "1.0":
        msg = f"Unsupported quantized model artifact version: {manifest.get('version')}"
        raise ValueError(msg)

    if expected_name is None:
        expected_name = getattr(getattr(model, "config", None), "name_or_path", None)

    # Reattach quantizers first (replacing stubs), then load weights so their
    # values land on the reconstructed module.
    _load_quantizer_state_from_files(
        model,
        config_path,
        model_path,
        expected_name=expected_name or None,
        overwrite_policy=overwrite_policy,
        allow_lazy_params=allow_lazy_params,
    )

    with safe_open(weights_path, framework="pt") as f:
        weight_state = {key: f.get_tensor(key) for key in f.keys()}
    weight_state = _expand_shared_tensors(weight_state, manifest.get("tied_weights", {}))

    # `weights.safetensors` deliberately excludes quantizer state, so a strict
    # `load_state_dict` would report every quantizer parameter as missing.
    # Validate against the non-quantizer keys instead to get the same
    # guarantee: no silently missing or unexpected weights.
    quantizer_prefixes = tuple(
        f"{name}."
        for name, _ in ff.nn.quantized_module.named_quantizers(
            model, remove_duplicate=False, skip_stubs=False
        )
    )
    expected_keys = {name for name in model.state_dict() if not name.startswith(quantizer_prefixes)}
    missing_keys = sorted(expected_keys - weight_state.keys())
    unexpected_keys = sorted(weight_state.keys() - expected_keys)
    if missing_keys or unexpected_keys:
        msg = (
            f"Saved weights do not match this model: missing {missing_keys}, "
            f"unexpected {unexpected_keys}."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    model.load_state_dict(weight_state, strict=False)
