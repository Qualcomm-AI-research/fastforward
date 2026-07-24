# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import json

from pathlib import Path

import fastforward as ff
import pytest
import safetensors.torch
import torch
import yaml

from fastforward.quantization import save_load
from safetensors import safe_open


def _quantized_model(
    num_bits: int = 4, granularity: ff.granularity.Granularity | None = None
) -> ff.nn.QuantizedSequential:
    """Build a small quantized Sequential with initialized weight quantizers."""
    granularity = ff.PerTensor() if granularity is None else granularity
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.Linear(4, 4),
    )
    ff.quantize_model(model)
    assert isinstance(model, ff.nn.QuantizedSequential)

    ff.find_quantizers(model, "**/[quantizer:parameter/weight]").initialize(
        ff.nn.LinearQuantizer, num_bits=num_bits, granularity=granularity
    )
    with ff.estimate_ranges(model, ff.range_setting.smoothed_minmax), ff.strict_quantization(False):
        model(torch.randn(4, 4))
    return model


def _tied_quantized_model() -> ff.nn.QuantizedSequential:
    """Build a quantized Sequential whose two layers share one weight Parameter.

    Mirrors the tied `lm_head.weight`/`embed_tokens.weight` pattern common in
    language models, which SafeTensors refuses to serialize naively.
    """
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.Linear(4, 4),
    )
    model[1].weight = model[0].weight
    ff.quantize_model(model)
    assert isinstance(model, ff.nn.QuantizedSequential)

    ff.find_quantizers(model, "**/[quantizer:parameter/weight]").initialize(
        ff.nn.LinearQuantizer, num_bits=4, granularity=ff.PerTensor()
    )
    with ff.estimate_ranges(model, ff.range_setting.smoothed_minmax), ff.strict_quantization(False):
        model(torch.randn(4, 4))
    return model


def _fresh_tied_model() -> ff.nn.QuantizedSequential:
    """Build the tied architecture with un-initialized (stub) quantizers."""
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.Linear(4, 4),
    )
    model[1].weight = model[0].weight
    ff.quantize_model(model)
    assert isinstance(model, ff.nn.QuantizedSequential)
    return model


def _fresh_model() -> ff.nn.QuantizedSequential:
    """Build the same architecture with un-initialized (stub) quantizers."""
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.Linear(4, 4),
    )
    ff.quantize_model(model)
    assert isinstance(model, ff.nn.QuantizedSequential)
    return model


def _layers(model: ff.nn.QuantizedSequential) -> list[ff.nn.QuantizedLinear]:
    layers: list[ff.nn.QuantizedLinear] = []
    for layer in model:
        assert isinstance(layer, ff.nn.QuantizedLinear)
        layers.append(layer)
    return layers


def _expected_qdq_weight(layer: ff.nn.QuantizedLinear) -> torch.Tensor:
    with ff.strict_quantization(False):
        return layer.weight_quantizer(layer.weight).dequantize()


# save_quantized_model snapshots the model exactly as it is handed over; it never
# transforms the weights. Callers pick the on-disk weight representation up front
# by fusing (and optionally stubbing) with fuse_qdq_weights before saving. These
# are the three states the old `mode` enum used to select, now produced by the
# caller.
_WEIGHT_STATES = ["latent", "qdq_active", "qdq_stubbed"]


def _prepare_weights(model: ff.nn.QuantizedSequential, state: str) -> None:
    """Put `model` into the requested on-disk weight state before saving."""
    if state == "latent":
        return
    if state == "qdq_active":
        ff.quantization.fuse_qdq_weights(model)
        return
    if state == "qdq_stubbed":
        ff.quantization.fuse_qdq_weights(model, stub_quantizers=True)
        return
    msg = f"unknown weight state: {state}"
    raise ValueError(msg)


@pytest.mark.parametrize("state", _WEIGHT_STATES)
def test_save_quantized_model_writes_expected_artifact(state: str, tmp_path: Path) -> None:
    """save_quantized_model writes the four-file artifact for any weight state."""
    # GIVEN An initialized quantized model prepared into the given weight state
    model = _quantized_model()
    _prepare_weights(model, state)
    path = tmp_path / "artifact"

    # WHEN Saving the model
    result = model.save_quantized_model(path, name_or_path="test-model")

    # THEN The artifact directory contains the four expected files
    assert result == path
    for filename in (
        "config.yaml",
        "quantizer_state.safetensors",
        "weights.safetensors",
        "manifest.json",
    ):
        assert (path / filename).exists(), f"missing {filename}"

    # THEN The manifest records the version and tied-weight map, and no longer
    #      carries a save-time mode (the model is snapshotted as-is)
    with open(path / "manifest.json", "r") as f:
        manifest = json.load(f)
    assert manifest["version"] == "1.0"
    assert "mode" not in manifest
    assert manifest["tied_weights"] == {}


def test_save_quantized_model_does_not_mutate_model(tmp_path: Path) -> None:
    """Saving is a read-only snapshot: it never fuses or stubs the caller's model."""
    # GIVEN An initialized quantized model
    model = _quantized_model()
    layers = _layers(model)
    original_weights = [layer.weight.clone() for layer in layers]

    # WHEN Saving the model
    model.save_quantized_model(tmp_path / "artifact", name_or_path="test-model")

    # THEN The caller's weights and quantizers are untouched
    for layer, original in zip(layers, original_weights):
        torch.testing.assert_close(layer.weight, original)
        assert isinstance(layer.weight_quantizer, ff.nn.LinearQuantizer)
        assert not layer.weight_quantizer.is_stub()


@pytest.mark.parametrize("state", _WEIGHT_STATES)
def test_save_load_round_trip_matches_forward_output(state: str, tmp_path: Path) -> None:
    """Reloading an artifact reproduces the saved model's forward output."""
    # GIVEN An initialized quantized model prepared into the given weight state
    model = _quantized_model()
    _prepare_weights(model, state)
    path = tmp_path / "artifact"
    model.save_quantized_model(path, name_or_path="test-model")

    # WHEN Loading the artifact into a fresh model with stub quantizers
    loaded = _fresh_model()
    loaded.load_quantized_model(path)

    # THEN Forward outputs match the saved model within tolerance
    data = torch.randn(4, 4)
    with ff.strict_quantization(False):
        expected = model(data)
        actual = loaded(data)
    expected = expected.dequantize() if hasattr(expected, "dequantize") else expected
    actual = actual.dequantize() if hasattr(actual, "dequantize") else actual
    torch.testing.assert_close(actual, expected)


def test_load_quantized_model_restores_qdq_weights(tmp_path: Path) -> None:
    """A fused+stubbed artifact reloads grid-snapped weights matching the QDQ values."""
    # GIVEN A model whose expected QDQ weights are captured before fusing
    model = _quantized_model()
    expected = [_expected_qdq_weight(layer) for layer in _layers(model)]
    _prepare_weights(model, "qdq_stubbed")
    path = tmp_path / "artifact"
    model.save_quantized_model(path, name_or_path="test-model")

    # WHEN Loading into a fresh model
    loaded = _fresh_model()
    loaded.load_quantized_model(path)

    # THEN Weights are the grid-snapped values and weight quantizers are stubs
    for layer, expected_weight in zip(_layers(loaded), expected):
        torch.testing.assert_close(layer.weight, expected_weight)
        assert isinstance(layer.weight_quantizer, ff.nn.QuantizerStub)


def test_load_quantized_model_qdq_active_restores_active_quantizers(tmp_path: Path) -> None:
    """A fused (non-stubbed) artifact reloads grid-snapped weights with active quantizers."""
    # GIVEN A model saved with QDQ weights but active quantizers
    model = _quantized_model()
    _prepare_weights(model, "qdq_active")
    path = tmp_path / "artifact"
    model.save_quantized_model(path, name_or_path="test-model")

    # WHEN Loading into a fresh model
    loaded = _fresh_model()
    loaded.load_quantized_model(path)

    # THEN Weight quantizers are active LinearQuantizers again
    for layer in loaded:
        assert isinstance(layer.weight_quantizer, ff.nn.LinearQuantizer)
        assert not layer.weight_quantizer.is_stub()


def test_load_quantized_model_latent_preserves_original_weights(tmp_path: Path) -> None:
    """A latent save reloads the original (non-snapped) weights."""
    # GIVEN A model saved as-is (latent weights)
    model = _quantized_model()
    original_weights = [layer.weight.clone() for layer in _layers(model)]
    path = tmp_path / "artifact"
    model.save_quantized_model(path, name_or_path="test-model")

    # WHEN Loading into a fresh model
    loaded = _fresh_model()
    loaded.load_quantized_model(path)

    # THEN Loaded weights equal the original latent weights
    for layer, original in zip(_layers(loaded), original_weights):
        torch.testing.assert_close(layer.weight, original)


def test_save_quantized_model_saves_bias_and_all_parameters(tmp_path: Path) -> None:
    """The full state_dict (bias, norms) is persisted, not just weights."""
    # GIVEN A model with biases saved after fusing+stubbing its weights
    model = _quantized_model()
    _prepare_weights(model, "qdq_stubbed")
    path = tmp_path / "artifact"
    model.save_quantized_model(path, name_or_path="test-model")

    # WHEN Loading into a fresh model
    loaded = _fresh_model()
    loaded.load_quantized_model(path)

    # THEN Biases are restored (bias is not grid-snapped, stays latent)
    for layer, source in zip(loaded, model):
        assert layer.bias is not None
        torch.testing.assert_close(layer.bias, source.bias)


def test_quantized_model_artifact_is_superset_of_quantization_state(tmp_path: Path) -> None:
    """The artifact's config/quantizer files load via the shared load machinery."""
    # GIVEN A model saved as a quantized-model artifact
    model = _quantized_model()
    path = tmp_path / "artifact"
    model.save_quantized_model(path, name_or_path="test-model")

    # THEN The config/quantizer files match the save_quantization_state schema and
    #       load through the same internal machinery load_quantization_state uses.
    loaded = _fresh_model()
    save_load._load_quantizer_state_from_files(
        loaded, path / "config.yaml", path / "quantizer_state.safetensors"
    )

    # THEN Quantizers are reattached as active LinearQuantizers
    for layer in loaded:
        assert isinstance(layer.weight_quantizer, ff.nn.LinearQuantizer)
        assert not layer.weight_quantizer.is_stub()


def test_load_quantized_model_missing_file_raises(tmp_path: Path) -> None:
    """A missing artifact file raises FileNotFoundError."""
    # GIVEN An artifact with a required file removed
    model = _quantized_model()
    path = tmp_path / "artifact"
    model.save_quantized_model(path, name_or_path="test-model")
    (path / "weights.safetensors").unlink()

    # WHEN Loading the incomplete artifact
    # THEN A FileNotFoundError is raised
    loaded = _fresh_model()
    with pytest.raises(FileNotFoundError):
        loaded.load_quantized_model(path)


def test_load_quantized_model_unsupported_version_raises(tmp_path: Path) -> None:
    """A manifest with an unsupported version raises ValueError."""
    # GIVEN An artifact whose manifest version has been tampered with
    model = _quantized_model()
    path = tmp_path / "artifact"
    model.save_quantized_model(path, name_or_path="test-model")
    manifest_path = path / "manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    manifest["version"] = "2.0"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    # WHEN Loading the artifact
    # THEN A ValueError about the unsupported version is raised
    loaded = _fresh_model()
    with pytest.raises(ValueError) as exc_info:
        loaded.load_quantized_model(path)
    assert "Unsupported quantized model artifact version" in str(exc_info.value)


def test_save_quantized_model_without_name_or_path_raises(tmp_path: Path) -> None:
    """Saving without a resolvable model identifier raises RuntimeError."""
    # GIVEN A quantized model with no config.name_or_path
    model = _quantized_model()

    # WHEN Saving without providing name_or_path
    # THEN A RuntimeError about the model identifier is raised
    with pytest.raises(RuntimeError) as exc_info:
        model.save_quantized_model(tmp_path / "artifact")
    assert "model identifier" in str(exc_info.value)


def test_save_load_round_trip_shared_weight_quantizer(tmp_path: Path) -> None:
    """Tied/shared weight quantizers survive the save/load round trip."""
    # GIVEN A model whose two layers share one weight quantizer instance
    model = _quantized_model()
    shared = model[0].weight_quantizer
    model[1].weight_quantizer = shared
    assert model[0].weight_quantizer is model[1].weight_quantizer

    path = tmp_path / "artifact"
    model.save_quantized_model(path, name_or_path="test-model")

    # WHEN Loading into a fresh model
    loaded = _fresh_model()
    loaded.load_quantized_model(path)

    # THEN The shared quantizer identity is restored
    assert loaded[0].weight_quantizer is loaded[1].weight_quantizer


@pytest.mark.parametrize("state", _WEIGHT_STATES)
def test_save_load_round_trip_tied_weight_parameters(state: str, tmp_path: Path) -> None:
    """Tied weight *parameters* round trip; SafeTensors rejects them naively."""
    # GIVEN A model whose two layers share one weight Parameter
    model = _tied_quantized_model()
    layers = _layers(model)
    assert layers[1].weight is layers[0].weight
    expected = layers[0].weight.clone()
    _prepare_weights(model, state)
    path = tmp_path / "artifact"

    # WHEN Saving (tied storage must be deduplicated, not rejected)
    model.save_quantized_model(path, name_or_path="test-model")

    # THEN The manifest records the dropped alias, saved once
    with open(path / "manifest.json", "r") as f:
        manifest = json.load(f)
    assert manifest["tied_weights"] == {"1.weight": "0.weight"}

    # WHEN Loading into a fresh tied model
    loaded = _fresh_tied_model()
    loaded.load_quantized_model(path, expected_name="test-model")

    # THEN Both names hold the saved values and the target stays tied
    loaded_layers = _layers(loaded)
    if state == "latent":
        torch.testing.assert_close(loaded_layers[0].weight, expected)
    torch.testing.assert_close(loaded_layers[0].weight, loaded_layers[1].weight)
    assert loaded_layers[1].weight is loaded_layers[0].weight


def test_save_quantized_model_tied_weights_stored_once(tmp_path: Path) -> None:
    """A tied parameter is written to disk exactly once."""
    # GIVEN A model with tied weights saved as an artifact
    model = _tied_quantized_model()
    path = tmp_path / "artifact"
    model.save_quantized_model(path, name_or_path="test-model")

    # THEN Only the representative weight key is present on disk
    with safe_open(path / "weights.safetensors", framework="pt") as f:
        keys = set(f.keys())
    assert "0.weight" in keys
    assert "1.weight" not in keys


def test_load_quantized_model_does_not_restub_active_target(tmp_path: Path) -> None:
    """Load is a pure snapshot: it never stubs an active target's weight quantizers.

    A fused+stubbed artifact records no weight quantizers in its config, so
    loading it into a model whose weight quantizers are already active leaves
    them active. Re-stubbing is the caller's responsibility (load into a fresh
    model, or call stub_weight_quantizers) rather than an implicit load effect.
    """
    # GIVEN An artifact saved with QDQ weights and stubbed weight quantizers
    model = _quantized_model()
    _prepare_weights(model, "qdq_stubbed")
    path = tmp_path / "artifact"
    model.save_quantized_model(path, name_or_path="test-model")

    # WHEN Loading it into a model whose weight quantizers are already active
    target = _quantized_model()
    target.load_quantized_model(path, overwrite_policy="overwrite", expected_name="test-model")

    # THEN The target's weight quantizers are left active (not re-stubbed)
    for layer in _layers(target):
        assert isinstance(layer.weight_quantizer, ff.nn.LinearQuantizer)
        assert not layer.weight_quantizer.is_stub()


def test_load_quantized_model_respects_error_policy(tmp_path: Path) -> None:
    """Reattaching over an initialized quantizer is rejected under 'error'."""
    # GIVEN An artifact whose config records active quantizers
    model = _quantized_model()
    path = tmp_path / "artifact"
    model.save_quantized_model(path, name_or_path="test-model")

    # WHEN Loading into a model whose quantizers are already initialized,
    #       with the default overwrite_policy="error"
    # THEN The conflict is reported rather than silently overwriting them
    target = _quantized_model()
    original_weights = [layer.weight.clone() for layer in _layers(target)]
    with pytest.raises(ff.exceptions.QuantizationError) as exc_info:
        target.load_quantized_model(path, expected_name="test-model")
    assert "already initialized" in str(exc_info.value)

    # THEN The target's quantizers are left intact
    for layer in _layers(target):
        assert isinstance(layer.weight_quantizer, ff.nn.LinearQuantizer)
        assert not layer.weight_quantizer.is_stub()

    # THEN The rejected load left the weights untouched too, rather than
    #      applying them and then failing
    for layer, original in zip(_layers(target), original_weights):
        torch.testing.assert_close(layer.weight, original)


def test_load_quantized_model_respects_skip_policy(tmp_path: Path) -> None:
    """overwrite_policy='skip' leaves already-initialized quantizers in place."""
    # GIVEN An artifact whose config records active quantizers
    model = _quantized_model()
    path = tmp_path / "artifact"
    model.save_quantized_model(path, name_or_path="test-model")

    # WHEN Loading into a model with active quantizers using 'skip'
    target = _quantized_model()
    target.load_quantized_model(path, overwrite_policy="skip", expected_name="test-model")

    # THEN The existing quantizers are preserved rather than overwritten
    for layer in _layers(target):
        assert isinstance(layer.weight_quantizer, ff.nn.LinearQuantizer)
        assert not layer.weight_quantizer.is_stub()


def test_load_quantized_model_stubbed_artifact_into_fresh_model(tmp_path: Path) -> None:
    """Loading a stubbed artifact into a fresh model keeps weight quantizers stubbed."""
    # GIVEN An artifact saved with QDQ weights and stubbed weight quantizers
    model = _quantized_model()
    _prepare_weights(model, "qdq_stubbed")
    path = tmp_path / "artifact"
    model.save_quantized_model(path, name_or_path="test-model")

    # WHEN Loading into a fresh model (weight quantizers are stubs by default)
    loaded = _fresh_model()
    loaded.load_quantized_model(path, expected_name="test-model")

    # THEN Weight quantizers are stubs, so the QDQ weights are not re-quantized
    for layer in _layers(loaded):
        assert isinstance(layer.weight_quantizer, ff.nn.QuantizerStub)


def test_load_quantized_model_keeps_active_quantizers_for_qdq_active(tmp_path: Path) -> None:
    """A fused (non-stubbed) artifact reloads active quantizers on a fresh model."""
    # GIVEN An artifact saved with QDQ weights but active quantizers
    model = _quantized_model()
    _prepare_weights(model, "qdq_active")
    path = tmp_path / "artifact"
    model.save_quantized_model(path, name_or_path="test-model")

    # WHEN Loading into a fresh model
    loaded = _fresh_model()
    loaded.load_quantized_model(path, expected_name="test-model")

    # THEN Weight quantizers are active
    for layer in _layers(loaded):
        assert isinstance(layer.weight_quantizer, ff.nn.LinearQuantizer)
        assert not layer.weight_quantizer.is_stub()


def test_save_quantized_model_with_activation_quantizers(tmp_path: Path) -> None:
    """Activation quantizers are persisted alongside weight quantizers."""
    # GIVEN A model with both weight and activation quantizers initialized
    model = _quantized_model()
    ff.find_quantizers(model, "**/[quantizer:activation/output]").initialize(
        ff.nn.LinearQuantizer, num_bits=8, granularity=ff.PerTensor()
    )
    with ff.estimate_ranges(model, ff.range_setting.smoothed_minmax), ff.strict_quantization(False):
        model(torch.randn(4, 4))

    # WHEN Fusing+stubbing the weights and saving
    _prepare_weights(model, "qdq_stubbed")
    path = tmp_path / "artifact"
    model.save_quantized_model(path, name_or_path="test-model")

    # THEN The activation quantizers survive in the config, and reload restores
    #       them while leaving weight quantizers stubbed
    loaded = _fresh_model()
    loaded.load_quantized_model(path, expected_name="test-model")
    for layer in _layers(loaded):
        assert isinstance(layer.output_quantizer, ff.nn.LinearQuantizer)
        assert not layer.output_quantizer.is_stub()
        assert isinstance(layer.weight_quantizer, ff.nn.QuantizerStub)


def test_save_quantized_model_with_no_initialized_quantizers(tmp_path: Path) -> None:
    """A weight-only fused+stubbed save legitimately records zero quantizers."""
    # GIVEN A model whose only quantizers are weight quantizers, fused+stubbed
    model = _quantized_model()
    expected = [_expected_qdq_weight(layer) for layer in _layers(model)]
    _prepare_weights(model, "qdq_stubbed")
    path = tmp_path / "artifact"
    model.save_quantized_model(path, name_or_path="test-model")

    # THEN The config records no quantizers (all were replaced by stubs)
    with open(path / "config.yaml", "r") as f:
        config = yaml.load(f, yaml.Loader)
    assert config["quantizers"] == {}

    # THEN The tensor-less quantizer state file is still valid SafeTensors, so
    #       consumers other than load_quantized_model can read it
    assert safetensors.torch.load_file(path / "quantizer_state.safetensors") == {}
    with safe_open(path / "quantizer_state.safetensors", framework="pt") as f:
        assert list(f.keys()) == []

    # WHEN/THEN Loading such an artifact still works and restores the weights
    loaded = _fresh_model()
    loaded.load_quantized_model(path, expected_name="test-model")
    for layer, expected_weight in zip(_layers(loaded), expected):
        torch.testing.assert_close(layer.weight, expected_weight)


def test_save_quantized_model_records_weight_encodings_when_active(tmp_path: Path) -> None:
    """Fusing without stubbing keeps weight scale/offset in the artifact."""
    # GIVEN An initialized quantized model with a known weight scale
    model = _quantized_model()
    expected_scales: list[torch.Tensor] = []
    for layer in _layers(model):
        quantizer = layer.weight_quantizer
        assert isinstance(quantizer, ff.nn.LinearQuantizer)
        expected_scales.append(torch.as_tensor(quantizer.scale).clone())
    path = tmp_path / "artifact"

    # WHEN Fusing (keeping the active quantizers) and saving
    _prepare_weights(model, "qdq_active")
    model.save_quantized_model(path, name_or_path="test-model")

    # THEN The weight quantizer encodings are present on disk
    state = safetensors.torch.load_file(path / "quantizer_state.safetensors")
    for index, expected_scale in enumerate(expected_scales):
        key = f"{index}.weight_quantizer.scale"
        assert key in state, f"missing weight encoding {key}"
        torch.testing.assert_close(state[key], expected_scale)


def test_load_quantized_model_missing_weight_key_raises(tmp_path: Path) -> None:
    """A weight key missing from the artifact is not silently ignored."""
    # GIVEN An artifact with one weight tensor removed
    model = _quantized_model()
    path = tmp_path / "artifact"
    model.save_quantized_model(path, name_or_path="test-model")
    weights = safetensors.torch.load_file(path / "weights.safetensors")
    del weights["0.weight"]
    safetensors.torch.save_file(weights, str(path / "weights.safetensors"))

    # WHEN Loading the tampered artifact
    # THEN The missing weight is reported rather than silently skipped
    loaded = _fresh_model()
    with pytest.raises(RuntimeError) as exc_info:
        loaded.load_quantized_model(path, expected_name="test-model")
    assert "0.weight" in str(exc_info.value)


def test_load_quantized_model_unexpected_weight_key_raises(tmp_path: Path) -> None:
    """An unexpected weight key in the artifact is reported."""
    # GIVEN An artifact with a spurious extra weight tensor
    model = _quantized_model()
    path = tmp_path / "artifact"
    model.save_quantized_model(path, name_or_path="test-model")
    weights = safetensors.torch.load_file(path / "weights.safetensors")
    weights["nonexistent.weight"] = torch.randn(4, 4)
    safetensors.torch.save_file(weights, str(path / "weights.safetensors"))

    # WHEN Loading the tampered artifact
    # THEN The unexpected key is reported
    loaded = _fresh_model()
    with pytest.raises(RuntimeError) as exc_info:
        loaded.load_quantized_model(path, expected_name="test-model")
    assert "nonexistent.weight" in str(exc_info.value)


def test_load_quantized_model_name_mismatch_raises(tmp_path: Path) -> None:
    """Loading an artifact saved for a different model is rejected."""
    # GIVEN An artifact saved under one model identifier
    model = _quantized_model()
    path = tmp_path / "artifact"
    model.save_quantized_model(path, name_or_path="model-A")

    # WHEN Loading it while expecting a different identifier
    # THEN The mismatch is reported
    loaded = _fresh_model()
    with pytest.raises(RuntimeError) as exc_info:
        loaded.load_quantized_model(path, expected_name="model-B")
    assert "mismatch" in str(exc_info.value)
