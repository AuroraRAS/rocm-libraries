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
from Tensile.Components import MAC_F16, MAC_F16_HPA


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


def test_packed_f16_mac_generates_both_accumulator_halves():
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
    generated = str(module)

    assert generated.count("v_pk_fma_f16") == 2
    assert "v[vgprValuC + 0]" in generated
    assert "v[vgprValuC + 1]" in generated


def test_f16_hpa_mad_mix_emits_typed_mac_instructions():
    kernel = {
        "ThreadTile0": 2,
        "ThreadTile1": 2,
    }
    writer = SimpleNamespace(states=SimpleNamespace(
        archCaps={},
        asmCaps={"v_fma_mix_f32": False},
        kernel=kernel,
    ))

    module = MAC_F16_HPA.FMA_F16_HPA_MAD_MIX()(
        writer,
        {"tileIdx": 0},
        {"tileIdx": 1},
        0,
        1,
    )
    generated = str(module)

    assert generated.count("v_mad_mix_f32") == 4
    assert "v[vgprValuA_X0_I0+0]" in generated
    assert "op_sel:[1,1,0] op_sel_hi:[1,1,0]" in generated


def test_gfx90c_f16_hpa_keeps_unpack_registers_reserved():
    class RegisterPool:
        def __init__(self):
            self.checkouts = []
            self.checkins = []

        def checkOutAligned(self, count, alignment, tag):
            self.checkouts.append((count, alignment, tag))
            return 100

        def checkIn(self, register):
            self.checkins.append(register)

    pool = RegisterPool()
    writer = SimpleNamespace(
        states=SimpleNamespace(
            archCaps={},
            asmCaps={"v_fma_mix_f32": False},
            gfx90cFp16HpaUnpackVgpr=None,
            kernel={"ThreadTile0": 2, "ThreadTile1": 2},
            version=(9, 0, 12),
        ),
        vgprPool=pool,
    )
    component = MAC_F16_HPA.FMA_F16_HPA_MAD_MIX()

    first = component(writer, {"tileIdx": 0}, {"tileIdx": 1}, 0, 1)
    second = component(writer, {"tileIdx": 0}, {"tileIdx": 1}, 1, 1)

    assert pool.checkouts == [(4, 1, "gfx90cFp16HpaUnpack")]
    assert pool.checkins == []
    assert writer.states.gfx90cFp16HpaUnpackVgpr == 100
    assert str(first).count("v_fma_f32") == 4
    assert str(second).count("v_fma_f32") == 4
