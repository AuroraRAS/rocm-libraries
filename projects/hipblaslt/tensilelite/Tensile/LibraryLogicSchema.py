# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Explicit compatibility conversion for legacy library-logic schemas.

Library logic is a persisted wire format.  Some classic Tensile logic files
encode integer-valued switches as booleans and one boolean-valued switch as
``0``/``1``.  Normalize only those known migrations before applying the current
strict schema.  Values outside these named conversions remain untouched so the
normal type validator reports them.
"""

from Tensile.Common.TypeValidationErrors import ConfigTypeError


LEGACY_BOOL_TO_INT_FIELDS = frozenset({
    "DirectToLds",
    "PrefetchGlobalRead",
    "PrefetchLocalRead",
    "UseSgprForGRO",
    "VectorStore",
})

LEGACY_INT_TO_BOOL_FIELDS = frozenset({"Use64bShadowLimit"})

# This was a library-generation heuristic, not a condition evaluated by the
# runtime.  Its meaning is not representable as a ContractionProblem predicate.
LEGACY_NON_RUNTIME_FIELD_VALUES = {
    "AssertMinApproxSize": frozenset({0, 1, 2, 3}),
}


def normalizeLibraryLogicListSchema(data):
    """Fill fields introduced after the original nine-item positional format."""
    if 9 <= len(data) < 12:
        data.extend([None] * (12 - len(data)))
        if data[10] is None:
            data[10] = "DeviceEfficiency"
        if data[11] is None:
            data[11] = "Euclidean"
    return data


def _copyAlias(solution, legacyKey, currentKeys, srcFile):
    if legacyKey not in solution:
        return

    value = solution[legacyKey]
    for currentKey in currentKeys:
        if currentKey in solution and solution[currentKey] != value:
            location = f"{srcFile}: " if srcFile else ""
            raise ConfigTypeError(
                f"{location}conflicting library-logic fields {legacyKey}="
                f"{value!r} and {currentKey}={solution[currentKey]!r}"
            )
        solution[currentKey] = value


def _isCanonicalGemmContraction(problemType):
    batched = problemType.get("Batched", False)
    sumIndex = 3 if batched else 2

    aIndices = [sumIndex, 0] if problemType.get("TransposeA", False) else [0, sumIndex]
    bIndices = [1, sumIndex] if problemType.get("TransposeB", False) else [sumIndex, 1]
    if batched:
        aIndices.append(2)
        bIndices.append(2)

    return (
        problemType.get("IndexAssignmentsA") == aIndices
        and problemType.get("IndexAssignmentsB") == bIndices
        and problemType.get("NumIndicesC") == (3 if batched else 2)
    )


def normalizeLibraryLogicSchema(data, srcFile=""):
    """Convert known legacy solution encodings to the current schema in place.

    The caller must pass the dictionary form produced by
    :func:`LibraryIO.parseLibraryLogicList`.  The same object is returned for
    convenient composition.
    """
    problemType = data.get("ProblemType", {})
    if "ComputeDataType" in problemType:
        problemType.setdefault("ActivationComputeDataType", problemType["ComputeDataType"])

    if (
        problemType.get("OperationType") == "TensorContraction"
        and _isCanonicalGemmContraction(problemType)
    ):
        problemType["OperationType"] = "GEMM"

    for solution in data.get("Solutions", []):
        solution.setdefault("CustomKernelName", "")

        _copyAlias(
            solution,
            "GlobalLoadVectorWidthA",
            ("GlobalReadVectorWidthA",),
            srcFile,
        )
        _copyAlias(
            solution,
            "GlobalLoadVectorWidthB",
            ("GlobalReadVectorWidthB",),
            srcFile,
        )
        if "LocalReadVectorWidth" in solution:
            solution.setdefault("LocalReadVectorWidthA", solution["LocalReadVectorWidth"])
            solution.setdefault("LocalReadVectorWidthB", solution["LocalReadVectorWidth"])

        solution.pop("GlobalLoadVectorWidthA", None)
        solution.pop("GlobalLoadVectorWidthB", None)

        for key in LEGACY_BOOL_TO_INT_FIELDS:
            value = solution.get(key)
            if type(value) is bool:
                solution[key] = int(value)

        for key in LEGACY_INT_TO_BOOL_FIELDS:
            value = solution.get(key)
            if type(value) is int and value in (0, 1):
                solution[key] = bool(value)

        for key, validValues in LEGACY_NON_RUNTIME_FIELD_VALUES.items():
            value = solution.get(key)
            if type(value) is int and value in validValues:
                solution.pop(key)

    return data
