Use match statements with unpacking where it makes sense (instead of if-else chains).
Try to avoid inheritance where possible, prefer defining Protocols and separate implementations (structural typing). You can reuse behaviors by defining functions.
Make sure all code is properly typed and typechecks with ty.
For Cython code, use the pure Python interface, and add accurate type stubs for exported functions.
Bump `ABI_VERSION` in `src/cyxla/_runtime.pxd` whenever a declared struct, enum or function signature changes.
The repository holds two projects, `cyxla/` and `cyxla-pjrt-cpu/`, and the same rules apply to both.
Keep the runtime portable to Linux, macOS and Windows. Operating system calls go through `src/cyxla/_platform.h`.
