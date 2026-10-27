# Design

This page follows a JAX function from Python to a call in native code. The pieces are
JAX's export, the StableHLO representation, and the PJRT C API of OpenXLA. [Bradbury et
al. (2018)](references.md) describe JAX and [Frostig et al. (2018)](references.md) its
tracing design.

## From a function to StableHLO

`lower` jits the function and exports it with `jax.export` for the platform of the
loaded plugin. The export holds [StableHLO](https://openxla.org/stablehlo), a portable
set of operations that XLA compiles for every backend. Exporting lowers for a platform
without needing its hardware, and the bytecode is a plain serialization, which the disk
cache stores. Custom calls are the exception, since the export only knows compatibility
guarantees for the kernels that JAX ships, as the [custom calls](guide/custom-calls.md)
page describes.

## Compiling through PJRT

[PJRT](https://openxla.org/xla/pjrt/pjrt_integration) is the interface between
frameworks and XLA's runtimes. A plugin exports one function, `GetPjrtApi`, that returns
a table of C functions for clients, buffers and executables. cyxla loads the library,
checks that the API version is new enough, creates a client on the first device and
compiles the module with `PJRT_Client_Compile`.

Everything that runs a program uses this C interface and never the Python bindings of
jaxlib. That is why an extension module can call it without the GIL, and why the same
code works with the CPU, CUDA, ROCm and TPU plugins. The runtime needs only a small part
of the interface: clients, buffers from host memory, executables, events and errors,
plus the FFI extension for custom calls.

## Executing without copies

The CPU plugin can alias host memory. Input buffers made with `kImmutableZeroCopy` point
at the arrays that the extension module owns, and the module fills them in place before
each call. The outputs of the CPU plugin live in host memory too, so the module reads
them where they are. An execution therefore copies nothing. The arrays are 64-byte
aligned because the plugin only aliases aligned memory. When a plugin cannot alias,
which includes all GPU plugins, `execute` copies inputs to the device and outputs back.

Executions wait on the plugin's events inside `execute`, and the CPU client dispatches
synchronously, so a call runs on the calling thread. A solver may call `execute` from
several threads of its own only if it serializes them, since the runtime's state is
shared.

## Why a separate plugin package

jaxlib links its CPU client statically, so it ships no library that implements the PJRT
interface for the CPU. [cyxla-pjrt-cpu](cpu-plugin.md) builds XLA's plugin from the
release that a JAX version pins and packs it into a wheel. JAX's own CUDA, ROCm and TPU
plugins are already separate libraries, and cyxla loads them as they are.

## Custom calls

XLA looks up a custom call's target in a registry inside the plugin when it compiles.
The registry of the plugin is not the one in jaxlib, even when both are XLA, so cyxla
registers handlers through the PJRT FFI extension. A registration passes the address of
an `XLA_FFI_Handler`, which the plugin calls once to read the handler's metadata.
jaxlib's handlers and the handlers of third-party libraries are the same kind of
function, so both register the same way.
