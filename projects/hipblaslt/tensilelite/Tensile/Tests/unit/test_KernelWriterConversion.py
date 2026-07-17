################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

from Tensile.Common.DataType import DataType
from Tensile.KernelWriterConversion import KernelWriterConversion


def test_scalar_half_load_uses_scalar_type_and_one_load_lane():
    writer = object.__new__(KernelWriterConversion)
    writer.state = {"ProblemType": {"DataType": DataType("half")}}
    writer.datatype = DataType("half").toDevice("HIP")

    writer.num_dword_load, writer.is_sub_dword_load = writer._loadWidth(
        1, DataType("half"))

    assert writer.num_dword_load == 1
    assert writer.is_sub_dword_load
    assert writer._loadType("float") == writer.datatype


def _stride_writer(use_initial_strides, use_e=False):
    writer = object.__new__(KernelWriterConversion)
    writer.state = {"ProblemType": {
        "UseE": use_e,
        "UseInitialStridesCD": use_initial_strides,
    }}
    writer.indexChars = ["0I"]
    writer.endLine = "\n"
    return writer


def test_conversion_uses_supplied_initial_strides():
    defines = _stride_writer(True, use_e=True)._initialStrideDefines()

    for tensor in ("E", "D", "W", "C"):
        assert f"#define stride{tensor}0I arg.stride{tensor}0I\n" in defines
    assert "hard-coded initial strides" not in defines


def test_conversion_defaults_to_unit_initial_strides():
    defines = _stride_writer(False)._initialStrideDefines()

    for tensor in ("D", "W", "C"):
        assert f"#define stride{tensor}0I 1\n" in defines
    assert "strideE0I" not in defines
