################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

import os
import shutil

import rocisa
from rocisa.container import vgpr
from rocisa.instruction import VMacF32


def test_vmac_f32_uses_fma_fallback_when_fmac_is_unavailable():
    isa = (9, 0, 12)
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    search_path = os.pathsep.join(
        [
            os.path.join(rocm_path, "bin"),
            os.path.join(rocm_path, "lib", "llvm", "bin"),
        ]
    )
    assembler = shutil.which("amdclang++", path=search_path) or "amdclang++"

    target = rocisa.rocIsa.getInstance()
    target.init(isa, assembler, False)
    target.setKernel(isa, 64)

    asm_caps = target.getAsmCaps()
    assert not asm_caps["v_fmac_f32"]
    assert asm_caps["v_fma_f32"]

    instruction = VMacF32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2))
    assert str(instruction).strip() == "v_fma_f32 v0, v1, v2, v0"
