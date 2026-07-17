# gfx90c TensileLite logic

This directory is the hipBLASLt-owned gfx90c logic package. Its `gfx90c`
schedule and architecture names ensure that TensileLite writes the lazy
mapping and solution shards for the gfx90c runtime architecture.

The package is canonical TensileLite YAML: it has no YAML aliases, uses
explicit non-MI ISA and scheduling fields, and contains only solutions the
current Lite backend can generate. Unsupported legacy LocalSplitU and flat
sub-dword kernels are omitted. Six HHS kernels whose non-prefetch variants
exceed the Lite half-store SGPR budget are replaced in exact-size selections
by supported kernels with the same tile and load geometry.

hipBLASLt's native CMake pipeline generates compressed `.dat.zlib` catalogs
and installs them directly. Generated catalogs and code objects do not belong
in this source tree and must not be copied from rocBLAS or manually
decompressed.
