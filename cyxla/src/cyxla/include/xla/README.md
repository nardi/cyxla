# Vendored PJRT C API header

`xla/pjrt/c/pjrt_c_api.h` is copied unchanged from
[openxla/xla](https://github.com/openxla/xla) at commit
`21d3caa7e06a33565c22927a7be5c5e68b4b94a5` (17 October 2026), which declares PJRT C API
version 0.116. The header has no dependencies beyond the C standard library. jaxlib does
not ship it, since `jax.ffi.include_dir()` only contains the XLA FFI headers.

Plugins accept argument structs that are larger than the version they were built for, so
this header works with every plugin from version 0.111 on, which is the oldest that
cyxla loads. `LICENSE` is XLA's Apache 2.0 license, which covers the header.

# Vendored XLA FFI C API header

`xla/ffi/api/c_api.h` is copied unchanged from the headers that jaxlib 0.10.2 ships
(`jaxlib/include/xla/ffi/api/c_api.h`), which declare XLA FFI API version 0.3. The
header has no dependencies beyond the C standard library. The handler for native
functions (`_native.py`) includes it to read the call frame that the runtime passes. The
structures of the header are the ones of the plugin, so the layouts need not be mirrored
by hand.
