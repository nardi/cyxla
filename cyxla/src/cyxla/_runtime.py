"""PJRT runtime: the plugin, the client and the executables compiled by it.

The functions declared in `_runtime.pxd` run without the GIL, so extension modules of other
packages can call them from callbacks of native code. On a CPU plugin, input buffers alias
host arrays that the caller fills in place, and outputs are read from the buffers' memory, so
an execution needs no copies. On other platforms, every execution copies its inputs to the
device and its outputs back.
"""

from collections.abc import Sequence

import cython
import numpy as np
from cython.cimports.cpython.bytes import (
    PyBytes_AsStringAndSize,
    PyBytes_FromStringAndSize,
)
from cython.cimports.cyxla._pjrt import (
    PJRT_ArgumentLists,
    PJRT_Buffer,
    PJRT_Buffer_Destroy_Args,
    PJRT_Buffer_Destroy_Args_STRUCT_SIZE,
    PJRT_Buffer_ToHostBuffer_Args,
    PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE,
    PJRT_Buffer_Type_F64,
    PJRT_Buffer_UnsafePointer_Args,
    PJRT_Buffer_UnsafePointer_Args_STRUCT_SIZE,
    PJRT_Client_AddressableDevices_Args,
    PJRT_Client_AddressableDevices_Args_STRUCT_SIZE,
    PJRT_Client_BufferFromHostBuffer_Args,
    PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE,
    PJRT_Client_Compile_Args,
    PJRT_Client_Compile_Args_STRUCT_SIZE,
    PJRT_Client_Create_Args,
    PJRT_Client_Create_Args_STRUCT_SIZE,
    PJRT_Client_PlatformName_Args,
    PJRT_Client_PlatformName_Args_STRUCT_SIZE,
    PJRT_Error,
    PJRT_Error_Destroy_Args,
    PJRT_Error_Destroy_Args_STRUCT_SIZE,
    PJRT_Error_Message_Args,
    PJRT_Error_Message_Args_STRUCT_SIZE,
    PJRT_Event,
    PJRT_Event_Await_Args,
    PJRT_Event_Await_Args_STRUCT_SIZE,
    PJRT_Event_Destroy_Args,
    PJRT_Event_Destroy_Args_STRUCT_SIZE,
    PJRT_Executable,
    PJRT_Executable_DeserializeAndLoad_Args,
    PJRT_Executable_DeserializeAndLoad_Args_STRUCT_SIZE,
    PJRT_Executable_Destroy_Args,
    PJRT_Executable_Destroy_Args_STRUCT_SIZE,
    PJRT_Executable_NumOutputs_Args,
    PJRT_Executable_NumOutputs_Args_STRUCT_SIZE,
    PJRT_Executable_Serialize_Args,
    PJRT_Executable_Serialize_Args_STRUCT_SIZE,
    PJRT_ExecuteOptions,
    PJRT_ExecuteOptions_STRUCT_SIZE,
    PJRT_HostBufferSemantics_kImmutableOnlyDuringCall,
    PJRT_HostBufferSemantics_kImmutableZeroCopy,
    PJRT_LoadedExecutable,
    PJRT_LoadedExecutable_Destroy_Args,
    PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE,
    PJRT_LoadedExecutable_Execute_Args,
    PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE,
    PJRT_LoadedExecutable_GetExecutable_Args,
    PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE,
    PJRT_NamedValue,
    PJRT_NamedValue_kBool,
    PJRT_NamedValue_kFloat,
    PJRT_NamedValue_kInt64,
    PJRT_NamedValue_kInt64List,
    PJRT_NamedValue_kString,
    PJRT_NamedValue_STRUCT_SIZE,
    PJRT_Plugin_Initialize_Args,
    PJRT_Plugin_Initialize_Args_STRUCT_SIZE,
    PJRT_Program,
    PJRT_Program_STRUCT_SIZE,
)
from cython.cimports.cyxla._platform import (
    cyxla_aligned_alloc,
    cyxla_aligned_free,
    cyxla_find_symbol,
    cyxla_library_error,
    cyxla_monotonic_seconds,
    cyxla_open_library,
)
from cython.cimports.libc.stdint import int64_t, uintptr_t
from cython.cimports.libc.stdlib import calloc, free
from cython.cimports.libc.string import memcpy, memset

from cyxla._types import ErrorInfo, OptionValue, RuntimeInfo

MINIMUM_API_MINOR_VERSION = 111
"""Oldest minor version of the PJRT C API that the runtime supports."""

api = cython.declare(cython.pointer[cython.const[PJRT_Api]], cython.NULL)
client = cython.declare(cython.pointer[PJRT_Client], cython.NULL)
device = cython.declare(cython.pointer[PJRT_Device], cython.NULL)
can_alias = cython.declare(cython.bint, False)
can_read_inplace = cython.declare(cython.bint, False)
error_count = cython.declare(cython.long, 0)


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def now() -> cython.double:
    """Returns a monotonic time in seconds."""
    return cyxla_monotonic_seconds()


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def check(error: cython.pointer[PJRT_Error]) -> cython.int:
    """Returns 0 when `error` is NULL, or records its message, frees it and returns -1."""
    global error_count
    if error == cython.NULL:
        return 0
    message: PJRT_Error_Message_Args
    memset(cython.address(message), 0, cython.sizeof(message))
    message.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE
    message.error = error
    api.PJRT_Error_Message(cython.address(message))
    size: cython.size_t = min(message.message_size, 1023)
    memcpy(cython.cast(cython.p_void, error_message), message.message, size)
    error_message[size] = 0
    error_count += 1
    destroy: PJRT_Error_Destroy_Args
    memset(cython.address(destroy), 0, cython.sizeof(destroy))
    destroy.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE
    destroy.error = error
    api.PJRT_Error_Destroy(cython.address(destroy))
    return -1


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def await_event(event: cython.pointer[PJRT_Event]) -> cython.int:
    """Waits for an event and frees it, returning -1 on error."""
    if event == cython.NULL:
        return 0  # pragma: no cover
    wait: PJRT_Event_Await_Args
    memset(cython.address(wait), 0, cython.sizeof(wait))
    wait.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE
    wait.event = event
    status: cython.int = check(api.PJRT_Event_Await(cython.address(wait)))
    destroy_event(event)
    return status


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def destroy_event(event: cython.pointer[PJRT_Event]) -> cython.void:
    """Frees an event without waiting for it."""
    destroy: PJRT_Event_Destroy_Args
    memset(cython.address(destroy), 0, cython.sizeof(destroy))
    destroy.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE
    destroy.event = event
    check(api.PJRT_Event_Destroy(cython.address(destroy)))


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def destroy_buffer(buffer: cython.pointer[PJRT_Buffer]) -> cython.void:
    """Frees a PJRT buffer, ignoring NULL."""
    if buffer == cython.NULL:
        return
    arguments: PJRT_Buffer_Destroy_Args
    memset(cython.address(arguments), 0, cython.sizeof(arguments))
    arguments.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE
    arguments.buffer = buffer
    check(api.PJRT_Buffer_Destroy(cython.address(arguments)))


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def buffer_from_host(
    data: cython.p_double, size: int64_t, alias: cython.bint
) -> cython.pointer[PJRT_Buffer]:
    """Returns a PJRT buffer of `size` doubles from host memory, or NULL on error.

    With `alias`, the buffer may share the host memory, which the caller then keeps alive and
    fills before each execution. Without it, the data is copied during the call.
    """
    arguments: PJRT_Client_BufferFromHostBuffer_Args
    memset(cython.address(arguments), 0, cython.sizeof(arguments))
    arguments.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE
    arguments.client = client
    arguments.data = data
    arguments.type = PJRT_Buffer_Type_F64
    arguments.dims = cython.address(size)
    arguments.num_dims = 1
    arguments.device = device
    if alias:
        arguments.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableZeroCopy
    else:
        arguments.host_buffer_semantics = (
            PJRT_HostBufferSemantics_kImmutableOnlyDuringCall
        )
    if check(api.PJRT_Client_BufferFromHostBuffer(cython.address(arguments))) != 0:
        return cython.NULL  # pragma: no cover

    # An aliased buffer reports its host memory as done only when the buffer is freed, so the
    # event is not awaited. A copied buffer's event is already complete when the call returns.
    if arguments.done_with_host_buffer != cython.NULL:
        if alias:
            destroy_event(arguments.done_with_host_buffer)
        else:
            await_event(arguments.done_with_host_buffer)
    return arguments.buffer


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def alias_buffer(data: cython.p_double, size: int64_t) -> cython.pointer[PJRT_Buffer]:
    """Returns a buffer that shares `size` doubles of host memory, or NULL without aliasing."""
    if not can_alias:
        return cython.NULL  # pragma: no cover
    return buffer_from_host(data, size, True)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def aligned_doubles(count: cython.int) -> cython.p_double:
    """Returns zeroed memory for `count` doubles at a 64-byte boundary, which XLA needs to alias it.

    The memory is freed with `free_doubles`, because Windows frees aligned memory with another
    function than `free`.
    """
    size: cython.size_t = max(count, 1) * cython.sizeof(cython.double)
    memory: cython.p_void = cyxla_aligned_alloc(ALIGNMENT, size)
    if memory == cython.NULL:
        return cython.NULL  # pragma: no cover
    memset(memory, 0, size)
    return cython.cast(cython.p_double, memory)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def free_doubles(memory: cython.p_double) -> cython.void:
    """Frees memory from `aligned_doubles`, ignoring NULL."""
    cyxla_aligned_free(memory)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def execute(
    program: cython.pointer[ProgramSpec],
    inputs: cython.pointer[Input],
    outputs: cython.pp_double,
    zero_copy: cython.bint,
) -> cython.int:
    """Executes a program and writes its outputs, returning -1 on error.

    `inputs` holds one host array per program input, which the caller fills before the call.
    With `zero_copy` and a plugin that aliases host memory, the inputs' aliased buffers are
    passed and outputs are read in place. Otherwise each input is copied into a new buffer and
    each output is copied back. Outputs with a NULL destination are dropped.
    """
    if program.handle == cython.NULL:
        return -1  # pragma: no cover
    use_aliases: cython.bint = zero_copy and can_alias
    arguments: cython.pointer[PJRT_Buffer][MAX_INPUTS]
    created: cython.pointer[PJRT_Buffer][MAX_INPUTS]
    index: cython.int
    for index in range(program.num_inputs):
        created[index] = cython.NULL
        if use_aliases:
            arguments[index] = inputs[index].buffer
        else:
            created[index] = buffer_from_host(
                inputs[index].host, inputs[index].size, False
            )
            arguments[index] = created[index]
            if created[index] == cython.NULL:
                release_buffers(created, index)  # pragma: no cover
                return -1  # pragma: no cover

    results: cython.pointer[PJRT_Buffer][MAX_OUTPUTS]
    for index in range(MAX_OUTPUTS):
        results[index] = cython.NULL
    result_list: cython.pointer[cython.pointer[PJRT_Buffer]] = results
    argument_list: cython.pointer[cython.pointer[PJRT_Buffer]] = arguments
    complete: cython.pointer[PJRT_Event] = cython.NULL
    options: PJRT_ExecuteOptions
    memset(cython.address(options), 0, cython.sizeof(options))
    options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE
    run: PJRT_LoadedExecutable_Execute_Args
    memset(cython.address(run), 0, cython.sizeof(run))
    run.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE
    run.executable = program.handle
    run.options = cython.address(options)
    run.argument_lists = cython.cast(PJRT_ArgumentLists, cython.address(argument_list))
    run.num_devices = 1
    run.num_args = program.num_inputs
    run.output_lists = cython.address(result_list)
    run.device_complete_events = cython.address(complete)
    status: cython.int = check(api.PJRT_LoadedExecutable_Execute(cython.address(run)))
    if status == 0:
        status = await_event(complete)

    pointer: PJRT_Buffer_UnsafePointer_Args
    transfer: PJRT_Buffer_ToHostBuffer_Args
    size: cython.size_t
    for index in range(program.num_outputs):
        size = program.output_sizes[index] * cython.sizeof(cython.double)
        if status == 0 and outputs[index] != cython.NULL:
            if use_aliases and can_read_inplace:
                # On the CPU device, buffer memory is host memory, so the result is read in place.
                memset(cython.address(pointer), 0, cython.sizeof(pointer))
                pointer.struct_size = PJRT_Buffer_UnsafePointer_Args_STRUCT_SIZE
                pointer.buffer = results[index]
                status = check(api.PJRT_Buffer_UnsafePointer(cython.address(pointer)))
                if status == 0:
                    memcpy(
                        outputs[index],
                        cython.cast(cython.p_void, pointer.buffer_pointer),
                        size,
                    )
            else:
                memset(cython.address(transfer), 0, cython.sizeof(transfer))
                transfer.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE
                transfer.src = results[index]
                transfer.dst = outputs[index]
                transfer.dst_size = size
                status = check(api.PJRT_Buffer_ToHostBuffer(cython.address(transfer)))
                if status == 0:
                    status = await_event(transfer.event)
        destroy_buffer(results[index])
    release_buffers(created, program.num_inputs)
    return status


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def release_buffers(
    buffers: cython.pointer[cython.pointer[PJRT_Buffer]], count: cython.int
) -> cython.void:
    """Frees the first `count` buffers of an array, ignoring NULL entries."""
    index: cython.int
    for index in range(count):
        destroy_buffer(buffers[index])


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def destroy_executable(executable: cython.pointer[PJRT_Executable]) -> cython.void:
    """Frees an unloaded executable handle."""
    arguments: PJRT_Executable_Destroy_Args
    memset(cython.address(arguments), 0, cython.sizeof(arguments))
    arguments.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE
    arguments.executable = executable
    check(api.PJRT_Executable_Destroy(cython.address(arguments)))


def require_runtime() -> None:
    """Raises an error unless `initialize` has run."""
    if api == cython.NULL:
        raise RuntimeError("The PJRT runtime is not initialized")


def raise_last_error(operation: str) -> None:
    """Raises the last recorded PJRT error as a `RuntimeError`."""
    message: bytes = error_message
    raise RuntimeError(f"{operation} failed: {message.decode('utf-8', 'replace')}")


def clock() -> float:
    """Returns the monotonic time that `now` reads, in seconds.

    Python code can compare it with timings that extension modules take with `now`.
    """
    return now()


def abi_version() -> int:
    """Returns the `ABI_VERSION` that this module was compiled with."""
    return ABI_VERSION


@cython.cfunc
def fill_option(
    option: cython.pointer[PJRT_NamedValue],
    name: bytes,
    value: OptionValue,
    keep_alive: list,
) -> cython.void:
    """Writes one client option into a named value.

    Strings and integer lists stay referenced by `keep_alive` until the client is created.
    The list's array is allocated with `calloc` and freed by the caller.
    """
    option.struct_size = PJRT_NamedValue_STRUCT_SIZE
    option.name = name
    option.name_size = len(name)
    option.value_size = 1
    keep_alive.append(name)
    index: cython.Py_ssize_t
    array: cython.pointer[int64_t]
    match value:
        case bool():
            option.type = PJRT_NamedValue_kBool
            option.bool_value = value
        case int():
            option.type = PJRT_NamedValue_kInt64
            option.int64_value = value
        case float():
            option.type = PJRT_NamedValue_kFloat
            option.float_value = value
        case str():
            encoded: bytes = value.encode()
            keep_alive.append(encoded)
            option.type = PJRT_NamedValue_kString
            option.string_value = encoded
            option.value_size = len(encoded)
        case tuple():
            array = cython.cast(
                cython.pointer[int64_t],
                calloc(max(len(value), 1), cython.sizeof(int64_t)),
            )
            for index in range(len(value)):
                array[index] = value[index]
            option.type = PJRT_NamedValue_kInt64List
            option.int64_array_value = array
            option.value_size = len(value)
        case _:
            raise TypeError(
                f"The client option {name.decode()} has the unsupported type "
                f"{type(value).__name__}."
            )


def initialize(
    plugin_path: str, create_options: Sequence[tuple[str, OptionValue]] = ()
) -> RuntimeInfo:
    """Loads a PJRT plugin, creates a client on its first device and returns its details.

    `create_options` are the options of `PJRT_Client_Create`, which differ between plugins.
    Calling it again returns the details of the runtime already loaded, whatever the path.
    """
    global api, client, device, can_alias, can_read_inplace
    if api != cython.NULL:
        return runtime_info()
    encoded: bytes = plugin_path.encode()
    library: cython.p_void = cyxla_open_library(encoded)
    if library == cython.NULL:
        reason: cython.char[512]
        cyxla_library_error(reason, 512)
        message: bytes = reason
        raise OSError(
            f"Cannot load the PJRT plugin {plugin_path}: "
            f"{message.decode('utf-8', 'replace')}"
        )
    symbol: cython.p_void = cyxla_find_symbol(library, b"GetPjrtApi")
    if symbol == cython.NULL:
        raise OSError(f"{plugin_path} does not export GetPjrtApi")
    loaded: cython.pointer[cython.const[PJRT_Api]] = cython.cast(GetPjrtApi, symbol)()
    major: cython.int = loaded.pjrt_api_version.major_version
    minor: cython.int = loaded.pjrt_api_version.minor_version
    if major != 0 or minor < MINIMUM_API_MINOR_VERSION:
        raise RuntimeError(
            f"The PJRT C API version {major}.{minor} is too old, "
            f"0.{MINIMUM_API_MINOR_VERSION} or newer is needed"
        )
    api = loaded

    plugin: PJRT_Plugin_Initialize_Args
    memset(cython.address(plugin), 0, cython.sizeof(plugin))
    plugin.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE
    if check(api.PJRT_Plugin_Initialize(cython.address(plugin))) != 0:
        api = cython.NULL  # pragma: no cover
        raise_last_error("PJRT_Plugin_Initialize")  # pragma: no cover

    count: cython.Py_ssize_t = len(create_options)
    options: cython.pointer[PJRT_NamedValue] = cython.cast(
        cython.pointer[PJRT_NamedValue],
        calloc(max(count, 1), cython.sizeof(PJRT_NamedValue)),
    )
    keep_alive: list = []
    index: cython.Py_ssize_t
    status: cython.int = 0
    try:
        for index in range(count):
            fill_option(
                cython.address(options[index]),
                create_options[index][0].encode(),
                create_options[index][1],
                keep_alive,
            )
        create: PJRT_Client_Create_Args
        memset(cython.address(create), 0, cython.sizeof(create))
        create.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE
        create.create_options = options
        create.num_options = count
        status = check(api.PJRT_Client_Create(cython.address(create)))
    except TypeError:
        api = cython.NULL
        raise
    finally:
        for index in range(count):
            if options[index].type == PJRT_NamedValue_kInt64List:
                free(cython.cast(cython.p_void, options[index].int64_array_value))
        free(options)
    if status != 0:
        api = cython.NULL
        raise_last_error("PJRT_Client_Create")
    client = create.client

    devices: PJRT_Client_AddressableDevices_Args
    memset(cython.address(devices), 0, cython.sizeof(devices))
    devices.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE
    devices.client = client
    check(api.PJRT_Client_AddressableDevices(cython.address(devices)))
    if devices.num_addressable_devices == 0:
        api = cython.NULL  # pragma: no cover
        raise RuntimeError("The PJRT client has no devices")  # pragma: no cover
    device = devices.addressable_devices[0]

    # Device memory is host memory only on the CPU, so outputs are read in place and inputs
    # may alias host arrays there. Aliasing is used only if the plugin really shares the host
    # memory of a zero-copy buffer.
    can_read_inplace = platform_name() == "cpu"
    if can_read_inplace:
        probe: cython.p_double = aligned_doubles(4)
        buffer: cython.pointer[PJRT_Buffer] = buffer_from_host(probe, 4, True)
        pointer: PJRT_Buffer_UnsafePointer_Args
        memset(cython.address(pointer), 0, cython.sizeof(pointer))
        pointer.struct_size = PJRT_Buffer_UnsafePointer_Args_STRUCT_SIZE
        pointer.buffer = buffer
        check(api.PJRT_Buffer_UnsafePointer(cython.address(pointer)))
        can_alias = pointer.buffer_pointer == cython.cast(uintptr_t, probe)
        destroy_buffer(buffer)
        free_doubles(probe)
    return runtime_info()


def platform_name() -> str:
    """Returns the platform name that the plugin reports."""
    platform: PJRT_Client_PlatformName_Args
    memset(cython.address(platform), 0, cython.sizeof(platform))
    platform.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE
    platform.client = client
    check(api.PJRT_Client_PlatformName(cython.address(platform)))
    name: bytes = platform.platform_name[: platform.platform_name_size]
    return name.decode()


def runtime_info() -> RuntimeInfo:
    """Returns the PJRT API version, the platform and whether executions avoid copies."""
    require_runtime()
    return RuntimeInfo(
        api_major_version=api.pjrt_api_version.major_version,
        api_minor_version=api.pjrt_api_version.minor_version,
        platform=platform_name(),
        aliased_inputs=bool(can_alias),
        inplace_outputs=bool(can_read_inplace),
    )


def is_initialized() -> bool:
    """Returns whether a PJRT plugin is loaded."""
    return api != cython.NULL


def error_info() -> ErrorInfo:
    """Returns the number of PJRT errors so far and the last message."""
    message: bytes = error_message
    return ErrorInfo(count=error_count, last_message=message.decode("utf-8", "replace"))


@cython.cclass
class Executable:
    """A program loaded into the PJRT client."""

    def __dealloc__(self) -> None:
        arguments: PJRT_LoadedExecutable_Destroy_Args
        if self.handle != cython.NULL:
            memset(cython.address(arguments), 0, cython.sizeof(arguments))
            arguments.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE
            arguments.executable = self.handle
            check(api.PJRT_LoadedExecutable_Destroy(cython.address(arguments)))

    @staticmethod
    def compile(code: bytes, compile_options: bytes) -> Executable:
        """Compiles StableHLO bytecode with a serialized `CompileOptionsProto`."""
        require_runtime()
        code_data: cython.p_char
        options_data: cython.p_char
        code_size: cython.Py_ssize_t
        options_size: cython.Py_ssize_t
        PyBytes_AsStringAndSize(
            code, cython.address(code_data), cython.address(code_size)
        )
        PyBytes_AsStringAndSize(
            compile_options, cython.address(options_data), cython.address(options_size)
        )
        program: PJRT_Program
        memset(cython.address(program), 0, cython.sizeof(program))
        program.struct_size = PJRT_Program_STRUCT_SIZE
        program.code = code_data
        program.code_size = code_size
        program.format = b"mlir"
        program.format_size = 4
        arguments: PJRT_Client_Compile_Args
        memset(cython.address(arguments), 0, cython.sizeof(arguments))
        arguments.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE
        arguments.client = client
        arguments.program = cython.address(program)
        arguments.compile_options = options_data
        arguments.compile_options_size = options_size
        error: cython.pointer[PJRT_Error]
        with cython.nogil:
            error = api.PJRT_Client_Compile(cython.address(arguments))
        if check(error) != 0:
            raise_last_error("PJRT_Client_Compile")
        return Executable.wrap(arguments.executable)

    @staticmethod
    def deserialize(serialized: bytes) -> Executable:
        """Loads an executable that `serialize` wrote with the same plugin."""
        require_runtime()
        data: cython.p_char
        size: cython.Py_ssize_t
        PyBytes_AsStringAndSize(serialized, cython.address(data), cython.address(size))
        arguments: PJRT_Executable_DeserializeAndLoad_Args
        memset(cython.address(arguments), 0, cython.sizeof(arguments))
        arguments.struct_size = PJRT_Executable_DeserializeAndLoad_Args_STRUCT_SIZE
        arguments.client = client
        arguments.serialized_executable = data
        arguments.serialized_executable_size = size
        if (
            check(api.PJRT_Executable_DeserializeAndLoad(cython.address(arguments)))
            != 0
        ):
            raise_last_error("PJRT_Executable_DeserializeAndLoad")
        return Executable.wrap(arguments.loaded_executable)

    @staticmethod
    @cython.cfunc
    def wrap(handle: cython.pointer[PJRT_LoadedExecutable]) -> Executable:
        """Wraps a loaded executable and reads its number of outputs."""
        executable: Executable = Executable.__new__(Executable)
        executable.handle = handle
        unloaded: cython.pointer[PJRT_Executable] = get_executable(handle)
        outputs: PJRT_Executable_NumOutputs_Args
        memset(cython.address(outputs), 0, cython.sizeof(outputs))
        outputs.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE
        outputs.executable = unloaded
        check(api.PJRT_Executable_NumOutputs(cython.address(outputs)))
        executable.num_outputs = outputs.num_outputs
        destroy_executable(unloaded)
        return executable

    def run(
        self,
        arguments: Sequence[np.ndarray],
        output_sizes: Sequence[int],
        zero_copy: bool = True,
    ) -> list[np.ndarray]:
        """Executes the program on float64 arguments and returns its flat outputs.

        It goes through `execute`, as extension modules do, which makes it useful for tests
        and interactive checks. Each output has the number of doubles in `output_sizes`.
        """
        require_runtime()
        num_inputs: cython.int = len(arguments)
        num_outputs: cython.int = len(output_sizes)
        if num_inputs > MAX_INPUTS or num_outputs != self.num_outputs:
            raise ValueError(
                f"The program takes at most {MAX_INPUTS} arguments and has "
                f"{self.num_outputs} outputs, not {num_inputs} and {num_outputs}"
            )
        spec: ProgramSpec
        spec.handle = self.handle
        spec.num_inputs = num_inputs
        spec.num_outputs = num_outputs
        inputs: Input[MAX_INPUTS]
        outputs: cython.p_double[MAX_OUTPUTS]
        results = [np.zeros(max(size, 1)) for size in output_sizes]
        index: cython.int
        for index in range(num_outputs):
            spec.output_sizes[index] = output_sizes[index]
            outputs[index] = cython.cast(
                cython.p_double, cython.cast(uintptr_t, results[index].ctypes.data)
            )
        for index in range(num_inputs):
            values = np.ascontiguousarray(arguments[index], dtype=np.float64).ravel()
            inputs[index].size = values.shape[0]
            inputs[index].host = aligned_doubles(values.shape[0])
            memcpy(
                inputs[index].host,
                cython.cast(cython.p_void, cython.cast(uintptr_t, values.ctypes.data)),
                values.shape[0] * cython.sizeof(cython.double),
            )
            inputs[index].buffer = cython.NULL
            if zero_copy:
                inputs[index].buffer = alias_buffer(inputs[index].host, values.shape[0])
        status: cython.int
        with cython.nogil:
            status = execute(cython.address(spec), inputs, outputs, zero_copy)
        for index in range(num_inputs):
            destroy_buffer(inputs[index].buffer)
            free_doubles(inputs[index].host)
        if status != 0:
            raise_last_error("PJRT_LoadedExecutable_Execute")
        return [
            result[:size] for result, size in zip(results, output_sizes, strict=True)
        ]

    def serialize(self) -> bytes:
        """Returns the executable in the plugin's serialized form."""
        unloaded: cython.pointer[PJRT_Executable] = get_executable(self.handle)
        arguments: PJRT_Executable_Serialize_Args
        memset(cython.address(arguments), 0, cython.sizeof(arguments))
        arguments.struct_size = PJRT_Executable_Serialize_Args_STRUCT_SIZE
        arguments.executable = unloaded
        status: cython.int = check(
            api.PJRT_Executable_Serialize(cython.address(arguments))
        )
        destroy_executable(unloaded)
        if status != 0:
            raise_last_error("PJRT_Executable_Serialize")  # pragma: no cover
        result: bytes = PyBytes_FromStringAndSize(
            arguments.serialized_bytes, arguments.serialized_bytes_size
        )
        arguments.serialized_executable_deleter(arguments.serialized_executable)
        return result


@cython.cfunc
def get_executable(
    handle: cython.pointer[PJRT_LoadedExecutable],
) -> cython.pointer[PJRT_Executable]:
    """Returns the unloaded executable behind a loaded one, which the caller frees."""
    arguments: PJRT_LoadedExecutable_GetExecutable_Args
    memset(cython.address(arguments), 0, cython.sizeof(arguments))
    arguments.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE
    arguments.loaded_executable = handle
    check(api.PJRT_LoadedExecutable_GetExecutable(cython.address(arguments)))
    return arguments.executable
