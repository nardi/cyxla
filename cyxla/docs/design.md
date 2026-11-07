# Design

This page follows a JAX function from Python to a call in native code. The components
involved are JAX's export, the StableHLO representation, and the PJRT C API of OpenXLA.
[Bradbury et al. (2018)](references.md) describe JAX and [Frostig et al.
(2018)](references.md) its tracing design.

## From a function to StableHLO

`lower` jits the function and exports it with `jax.export` for the platform of the
loaded plugin. The export contains [StableHLO](https://openxla.org/stablehlo), a
portable set of operations that XLA compiles for every backend. An export for a platform
does not need its hardware, and its bytecode is stored as is in the disk cache. Custom
calls are the exception to this portability, because JAX only guarantees compatibility
for the custom call targets that are included in `jaxlib`, as the [custom
calls](guide/custom-calls.md) page describes.

## Compilation through PJRT

[PJRT](https://openxla.org/xla/pjrt/pjrt_integration) is the interface between
frameworks and XLA's runtimes. A plugin exports one function, `GetPjrtApi`, that returns
a table of C functions for clients, buffers and executables. cyxla loads the library,
checks that the API version is new enough, creates a client on the first device and
compiles the module with `PJRT_Client_Compile`.

All code in cyxla that runs a program uses this C interface, and none of it uses the
Python bindings of `jaxlib`. As a result, an extension module can call it without the
GIL, and why the same code works with the CPU, CUDA, ROCm and TPU plugins. The runtime
needs only a small part of the interface: clients, buffers from host memory,
executables, events and errors, plus the FFI extension for custom calls.

## Execution without copies

The CPU plugin can alias host memory, so input buffers that are created with the
`kImmutableZeroCopy` semantics point at the arrays that the extension module owns, and
the module fills them in place before each call. The outputs of the CPU plugin are also
in host memory, so the module reads them where they are and an execution copies nothing.
The module aligns the arrays to 64 bytes, because the plugin only aliases aligned
memory. When a plugin cannot alias, which includes all GPU plugins, `execute` copies
inputs to the device and outputs back.

`execute` waits on the plugin's events, and the CPU client dispatches synchronously, so
a call runs on the calling thread. A solver can call `execute` from several threads only
if it serializes the calls, because the runtime's state is shared.

## The separate plugin package

`jaxlib` links its CPU client statically, so it includes no library that implements the
PJRT interface for the CPU. [cyxla-pjrt-cpu](cpu-plugin.md) builds XLA's plugin from the
release that a JAX version pins and distributes it as a wheel. JAX's own CUDA, ROCm and
TPU plugins are already separate libraries, and cyxla loads them unmodified.

## Custom calls

XLA looks up a custom call's target in a registry inside the plugin when it compiles.
The registry of the plugin is not the one in `jaxlib`, even when both are XLA, so cyxla
registers handlers through the PJRT FFI extension. A registration contains the address
of an `XLA_FFI_Handler`, which the plugin calls once to read the handler's metadata.
`jaxlib`'s handlers and the handlers of third-party libraries are the same kind of
function, so cyxla registers both in the same way. JAX stores its registrations in two
places, a queue in `jaxlib.xla_client` until the backend of the platform starts and the
registry of `jaxlib` afterwards. cyxla reads both when the plugin loads. When cyxla is
imported, it wraps `register_custom_call_target` of `jaxlib`, so registrations made
after the import also reach the plugin.
