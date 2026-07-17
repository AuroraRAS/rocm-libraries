# gfx90c classic Tensile logic

This directory is the rocBLAS-owned gfx90c logic package. It is consumed by
rocBLAS through the classic `TensileCreateLibraryFiles` CMake path. The
`vega10` schedule name is intentional: classic Tensile maps that schedule to
the Vega kernel family while the architecture field and every assembly
solution with an explicit ISA target `[9, 0, 12]` (`gfx90c`). Older HB/HHS
solutions omit the field and inherit gfx90c from the package architecture.

Generated `.dat`, `.co`, and `.hsaco` files do not belong here. They are built
and installed by rocBLAS and use the classic Tensile database format. Do not
copy this package's generated databases into hipBLASLt.
