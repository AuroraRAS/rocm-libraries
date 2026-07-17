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

import inspect
from types import SimpleNamespace

from Tensile import Components  # noqa: F401 - registers component implementations
from Tensile.Component import MAC
from Tensile.Components import MAC_F16


def _concrete_implementations(component):
    for implementation in component.implementations.values():
        if inspect.isabstract(implementation):
            yield from _concrete_implementations(implementation)
        else:
            yield implementation


def test_mac_components_share_call_interface():
    expected = ["self", "writer", "tPA", "tPB", "m", "innerUnroll"]

    for implementation in _concrete_implementations(MAC):
        assert list(inspect.signature(implementation.__call__).parameters) == expected, \
            implementation.__name__


def test_packed_f16_mac_generates_both_accumulator_halves(monkeypatch):
    class RecordingModule:
        def __init__(self, name):
            self.instructions = []

        def addComment(self, comment):
            pass

        def addInst(self, *args):
            self.instructions.append(args)

        def add(self, item):
            pass

    monkeypatch.setattr(MAC_F16, "Module", RecordingModule)

    kernel = {
        "ThreadTile0": 2,
        "ThreadTile1": 2,
    }
    writer = SimpleNamespace(states=SimpleNamespace(
        archCaps={},
        asmCaps={"v_pk_fma_f16": True},
        kernel=kernel,
    ))

    module = MAC_F16.FMA_F16_Packed()(writer, {}, {}, 0, 1)
    accumulators = [instruction[1] for instruction in module.instructions]

    assert accumulators == ["v[vgprValuC + 0]", "v[vgprValuC + 1]"]
