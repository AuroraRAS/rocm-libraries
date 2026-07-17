# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from Tensile.Common.TypeValidationErrors import ConfigTypeError
from Tensile.LibraryLogicSchema import (
    normalizeLibraryLogicListSchema,
    normalizeLibraryLogicSchema,
)
from Tensile.SolutionStructs.Solution import validateParameterTypes


pytestmark = pytest.mark.unit


def _logic(solution):
    return {"Solutions": [solution]}


@pytest.mark.parametrize("legacy_length", [9, 10, 11])
def test_legacy_positional_logic_gets_explicit_matching_defaults(legacy_length):
    data = list(range(legacy_length))
    if legacy_length > 10:
        data[10] = None

    assert normalizeLibraryLogicListSchema(data) is data
    assert len(data) == 12
    assert data[10] == "DeviceEfficiency"
    assert data[11] == "Euclidean"


def test_current_positional_logic_is_not_changed():
    data = list(range(12))

    normalizeLibraryLogicListSchema(data)

    assert data == list(range(12))


@pytest.mark.parametrize(
    "key",
    [
        "DirectToLds",
        "PrefetchGlobalRead",
        "PrefetchLocalRead",
        "UseSgprForGRO",
        "VectorStore",
    ],
)
@pytest.mark.parametrize("legacy, canonical", [(False, 0), (True, 1)])
def test_known_legacy_boolean_switches_become_integers(key, legacy, canonical):
    data = _logic({key: legacy})

    assert normalizeLibraryLogicSchema(data) is data
    assert data["Solutions"][0][key] == canonical
    assert type(data["Solutions"][0][key]) is int


@pytest.mark.parametrize("legacy, canonical", [(0, False), (1, True)])
def test_known_legacy_integer_switch_becomes_boolean(legacy, canonical):
    data = _logic({"Use64bShadowLimit": legacy})

    normalizeLibraryLogicSchema(data)

    assert data["Solutions"][0]["Use64bShadowLimit"] is canonical


@pytest.mark.parametrize(
    "key, value",
    [
        ("DirectToLds", 2),
        ("Use64bShadowLimit", 2),
        ("Use64bShadowLimit", "1"),
        ("UnknownSwitch", True),
    ],
)
def test_values_outside_named_conversions_are_untouched(key, value):
    data = _logic({key: value})

    normalizeLibraryLogicSchema(data)

    assert data["Solutions"][0][key] == value
    assert type(data["Solutions"][0][key]) is type(value)


def test_only_named_non_runtime_metadata_is_removed():
    data = _logic({"AssertMinApproxSize": 3, "AssertFutureConstraint": 4})

    normalizeLibraryLogicSchema(data)

    assert "AssertMinApproxSize" not in data["Solutions"][0]
    assert data["Solutions"][0]["AssertFutureConstraint"] == 4


@pytest.mark.parametrize("value", [-1, 4, True, "3"])
def test_invalid_retired_metadata_is_not_silently_removed(value):
    data = _logic({"AssertMinApproxSize": value})

    normalizeLibraryLogicSchema(data)

    assert data["Solutions"][0]["AssertMinApproxSize"] == value
    assert type(data["Solutions"][0]["AssertMinApproxSize"]) is type(value)


def test_missing_custom_kernel_name_gets_current_default():
    solution = {}

    normalizeLibraryLogicSchema(_logic(solution))

    assert solution["CustomKernelName"] == ""


def test_legacy_vector_width_aliases_become_current_fields():
    solution = {
        "GlobalLoadVectorWidthA": 2,
        "GlobalLoadVectorWidthB": 4,
        "LocalReadVectorWidth": 8,
    }
    data = _logic(solution)

    normalizeLibraryLogicSchema(data)

    assert solution["GlobalReadVectorWidthA"] == 2
    assert solution["GlobalReadVectorWidthB"] == 4
    assert solution["LocalReadVectorWidthA"] == 8
    assert solution["LocalReadVectorWidthB"] == 8
    assert "GlobalLoadVectorWidthA" not in solution
    assert "GlobalLoadVectorWidthB" not in solution


def test_current_local_read_vector_width_overrides_are_preserved():
    solution = {
        "LocalReadVectorWidth": 8,
        "LocalReadVectorWidthA": 2,
        "LocalReadVectorWidthB": 4,
    }

    normalizeLibraryLogicSchema(_logic(solution))

    assert solution["LocalReadVectorWidthA"] == 2
    assert solution["LocalReadVectorWidthB"] == 4


def test_conflicting_vector_width_alias_is_rejected():
    data = _logic({"GlobalLoadVectorWidthA": 2, "GlobalReadVectorWidthA": 4})

    with pytest.raises(ConfigTypeError, match="conflicting library-logic fields"):
        normalizeLibraryLogicSchema(data, "legacy.yaml")


@pytest.mark.parametrize("batched", [False, True])
def test_canonical_tensor_contraction_becomes_gemm(batched):
    sum_index = 3 if batched else 2
    a_indices = [0, sum_index] + ([2] if batched else [])
    b_indices = [sum_index, 1] + ([2] if batched else [])
    data = {
        "ProblemType": {
            "OperationType": "TensorContraction",
            "Batched": batched,
            "TransposeA": False,
            "TransposeB": False,
            "IndexAssignmentsA": a_indices,
            "IndexAssignmentsB": b_indices,
            "NumIndicesC": 3 if batched else 2,
        },
        "Solutions": [],
    }

    normalizeLibraryLogicSchema(data)

    assert data["ProblemType"]["OperationType"] == "GEMM"


def test_activation_compute_type_defaults_to_compute_type():
    data = {"ProblemType": {"ComputeDataType": 0}, "Solutions": []}

    normalizeLibraryLogicSchema(data)

    assert data["ProblemType"]["ActivationComputeDataType"] == 0


def test_non_gemm_tensor_contraction_remains_explicitly_unsupported():
    data = {
        "ProblemType": {
            "OperationType": "TensorContraction",
            "Batched": False,
            "IndexAssignmentsA": [0, 2, 3],
            "IndexAssignmentsB": [2, 1, 3],
            "NumIndicesC": 2,
        },
        "Solutions": [],
    }

    normalizeLibraryLogicSchema(data)

    assert data["ProblemType"]["OperationType"] == "TensorContraction"


def test_normalized_fields_satisfy_current_strict_types():
    solution = {
        "DirectToLds": False,
        "PrefetchGlobalRead": True,
        "PrefetchLocalRead": True,
        "UseSgprForGRO": False,
        "VectorStore": True,
        "Use64bShadowLimit": 1,
    }

    normalizeLibraryLogicSchema(_logic(solution))

    assert validateParameterTypes(solution, "legacy.yaml") == []


def test_non_binary_boolean_encoding_still_fails_current_strict_types():
    solution = {"Use64bShadowLimit": 2}

    normalizeLibraryLogicSchema(_logic(solution))

    records = validateParameterTypes(solution, "legacy.yaml")
    assert records[0][0] == ("Use64bShadowLimit", "int", "bool")
