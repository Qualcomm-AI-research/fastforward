# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import json
import pathlib

from typing import Any, TypeAlias

import fastforward as ff
import pytest
import torch

from fastforward.export._export_schemas import (
    LegacySchemaHandler,
    V1SchemaHandler,
    V2SchemaHandler,
)
from fastforward.export.export import export
from fastforward.quantization.quant_init import QuantizerCollection
from fastforward.testing.initialization import initialize_quantizers_to_linear_quantizer

QuantizedModelFixture: TypeAlias = tuple[torch.nn.Module, QuantizerCollection, QuantizerCollection]

SchemaHandlerType = LegacySchemaHandler | V1SchemaHandler | V2SchemaHandler


@pytest.mark.parametrize(
    "handler_class", [(LegacySchemaHandler), (V1SchemaHandler), (V2SchemaHandler)]
)
@pytest.mark.slow
def test_encodings_propagation(
    tmp_path: pathlib.Path,
    simple_quant_model_with_non_quant_ops: QuantizedModelFixture,
    handler_class: type,
    _seed_prngs: int,
) -> None:
    # GIVEN a quantized model.
    granularity = ff.PerTensor()
    data = torch.randn(32, 10)
    quant_model, activation_quantizers, parameter_quantizers = simple_quant_model_with_non_quant_ops
    model_name = "test_export_function"
    output_directory = tmp_path / model_name
    output_model_directory = pathlib.Path(output_directory) / model_name
    encodings_file_path = output_model_directory.with_suffix(".encodings")
    schema_handler = handler_class()

    with ff.strict_quantization(False):
        estimate_model_ranges = initialize_quantizers_to_linear_quantizer(
            quant_model,
            activation_quantizers,
            parameter_quantizers,
            granularity_parameters=granularity,
        )
        estimate_model_ranges(data)

    # GIVEN the exported artifacts from that model and its original encodings file.
    export(
        quant_model,
        (data,),
        output_directory,
        model_name,
        enable_encodings_propagation=False,
        encoding_schema_handler=schema_handler,
    )

    with open(encodings_file_path, "r") as file:
        org_encodings_dictionary = json.load(file)

    # WHEN exporting the same model with encoding propagation.
    export(
        quant_model,
        (data,),
        output_directory,
        model_name,
        enable_encodings_propagation=True,
        encoding_schema_handler=schema_handler,
    )

    with open(encodings_file_path, "r") as file:
        new_encodings_dictionary = json.load(file)

    # THEN all original encodings should be preserved, and new ones added.
    org_names = get_names_from_encodings(org_encodings_dictionary, schema_handler)
    new_names = get_names_from_encodings(new_encodings_dictionary, schema_handler)
    assert org_names <= new_names

    # THEN the encodings for new activations should be associated as displayed in the below dictionary.
    # Case 1: these are derived from other activations
    activation_to_activation_association = {"ff_view_0": "ff_mm_0", "ff_view_1_0": "ff_mm_0"}

    for prop_encoding, org_encoding in activation_to_activation_association.items():
        org_encoding_value = get_encoding_values_from_name(
            org_encodings_dictionary, org_encoding, schema_handler
        )
        pror_encoding_value = get_encoding_values_from_name(
            new_encodings_dictionary, prop_encoding, schema_handler
        )
        assert_encodings_are_the_same(org_encoding_value, pror_encoding_value, schema_handler)

    # THEN the encodings for new activations should be associated as displayed in the below dictionary.
    # Case 2: these are derived from parameters
    activation_to_parameter_association = {
        "ff_permute_0": "ff_fc1.weight_0",
        "ff_permute_1_0": "ff_fc2.weight_0",
        "ff_permute_2_0": "ff_fc3.weight_0",
    }

    for prop_encoding, org_encoding in activation_to_parameter_association.items():
        org_encoding_value = get_encoding_values_from_name(
            org_encodings_dictionary, org_encoding, schema_handler
        )
        pror_encoding_value = get_encoding_values_from_name(
            new_encodings_dictionary, prop_encoding, schema_handler
        )
        assert_encodings_are_the_same(org_encoding_value, pror_encoding_value, schema_handler)

    # THEN encodings should not have propagated to the following parameters as these are not quantized.
    non_existing_names = ["extra_weight", "_softmax", "mm_3"]
    for name in non_existing_names:
        assert get_encoding_values_from_name(new_encodings_dictionary, name, schema_handler) is None


def get_names_from_encodings(
    encodings_dictionary: dict[str, Any], schema_handler: SchemaHandlerType
) -> set[str]:
    """Extract all encoding names from a dictionary based on schema type.

    Different schema handlers store encodings in different structures:
    - Legacy: separate activation_encodings and param_encodings dicts
    - V1: separate lists with 'name' fields in each encoding
    - V2: single list with 'name' fields in each encoding

    Args:
        encodings_dictionary: The encodings dictionary from export
        schema_handler: Handler instance to determine extraction method

    Returns:
        Set of all encoding names found in the dictionary
    """
    if isinstance(schema_handler, LegacySchemaHandler):
        activation_names = set(encodings_dictionary["activation_encodings"].keys())
        parameter_names = set(encodings_dictionary["param_encodings"].keys())

        return activation_names | parameter_names

    elif isinstance(schema_handler, V1SchemaHandler):
        activation_names = set([
            enc["name"] for enc in encodings_dictionary["activation_encodings"]
        ])
        parameter_names = set([enc["name"] for enc in encodings_dictionary["param_encodings"]])

        return activation_names | parameter_names
    else:
        names = set([enc["name"] for enc in encodings_dictionary["encodings"]])

        return names


def get_encoding_values_from_name(
    encodings_dictionary: dict[str, Any], name: str, schema_handler: SchemaHandlerType
) -> dict[str, Any] | None:
    """Retrieve encoding values for a specific name from the dictionary.

    Searches through the appropriate structure based on schema type and
    returns the requested encoding.

    Args:
        encodings_dictionary: The encodings dictionary from export
        name: The encoding name to search for
        schema_handler: Handler instance to determine search method

    Returns:
        The encoding dictionary if found, None otherwise
    """
    if isinstance(schema_handler, LegacySchemaHandler):
        legacy_encoding: tuple[dict[str, Any], ...] | None = encodings_dictionary[
            "activation_encodings"
        ].get(name) or encodings_dictionary["param_encodings"].get(name)
        if legacy_encoding is None:
            return legacy_encoding
        return legacy_encoding[0]

    elif isinstance(schema_handler, V1SchemaHandler):
        act_encs = encodings_dictionary["activation_encodings"]
        param_encs = encodings_dictionary["param_encodings"]

        v1_encoding: dict[str, Any] | None = next(
            (enc for enc in act_encs if enc["name"] == name), None
        ) or next((enc for enc in param_encs if enc["name"] == name), None)
        return v1_encoding

    else:
        v2_encoding: dict[str, Any] | None = next(
            (enc for enc in encodings_dictionary["encodings"] if enc["name"] == name), None
        )
        return v2_encoding


def assert_encodings_are_the_same(
    enc1: dict[str, Any] | None, enc2: dict[str, Any] | None, schema_handler: SchemaHandlerType
) -> None:
    """Assert that two encodings contain the same quantization parameters.

    Compares encodings while accounting for schema-specific differences:
    - Legacy: Direct dictionary comparison
    - V1/V2: Compare all fields except 'name' (which may differ)

    Args:
        enc1: First encoding to compare
        enc2: Second encoding to compare
        schema_handler: Handler instance to determine comparison method
    """
    assert enc1 is not None
    assert enc2 is not None

    if isinstance(schema_handler, LegacySchemaHandler):
        assert enc1 == enc2

    elif isinstance(schema_handler, (V1SchemaHandler, V2SchemaHandler)):
        for key in enc1:
            if key == "name":
                continue
            assert enc1[key] == enc2[key]
