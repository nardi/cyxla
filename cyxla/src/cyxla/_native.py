"""Native functions: C functions that programs call by name, without the GIL.

`handle_call` is an XLA FFI handler. A program calls the target `cyxla_native_function` with
the attribute `name`, and the handler finds the function that was registered under that name
and passes it the buffers of the call. See `cyxla/native.h` for the interface of a function.
The module registers one function itself, `cyxla.python_callback`, which runs the Python
callbacks of programs that were lowered with `host_callbacks=True`.

The table of functions is replaced as a whole when a function is registered, and the old
table is never freed. The handler reads the table without the GIL and without a lock, so
functions are registered before programs run.
"""

import cython
from cython.cimports.cyxla._ffi_api import (
    XLA_FFI_API_MAJOR,
    XLA_FFI_API_MINOR,
    XLA_FFI_AttrType_SCALAR,
    XLA_FFI_AttrType_STRING,
    XLA_FFI_Buffer,
    XLA_FFI_ByteSpan,
    XLA_FFI_CallFrame,
    XLA_FFI_Error,
    XLA_FFI_Error_Code_INTERNAL,
    XLA_FFI_Error_Create_Args,
    XLA_FFI_Error_Create_Args_STRUCT_SIZE,
    XLA_FFI_Extension_Metadata,
    XLA_FFI_Metadata_Extension,
    XLA_FFI_Scalar,
)
from cython.cimports.cyxla._native import (
    ATTRIBUTE_NAME_SIZE,
    ATTRIBUTE_NAMES_SIZE,
    MAX_ARRAYS,
    MAX_ATTRIBUTES,
    MESSAGE_SIZE,
    NATIVE_ABI_VERSION,
    NATIVE_ATTRIBUTE_SCALAR,
    NATIVE_ATTRIBUTE_STRING,
    NATIVE_U64,
    NativeArray,
    NativeAttribute,
    NativeEntry,
    NativeFunction,
    entries,
    entry_count,
    native_find_attribute,
    native_size,
    registration_hook,
)
from cython.cimports.cyxla._platform import (
    THREAD_MESSAGE_SIZE,
    cyxla_thread_message,
)
from cython.cimports.libc.stdint import int64_t
from cython.cimports.libc.stdio import snprintf
from cython.cimports.libc.string import memcmp, memcpy, memset, strlen


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def make_error(
    frame: cython.pointer[XLA_FFI_CallFrame], message: cython.p_char
) -> cython.pointer[XLA_FFI_Error]:
    """Returns the FFI error that carries a message, which the runtime copies."""
    arguments: XLA_FFI_Error_Create_Args
    memset(cython.address(arguments), 0, cython.sizeof(arguments))
    arguments.struct_size = XLA_FFI_Error_Create_Args_STRUCT_SIZE
    arguments.message = message
    arguments.errc = XLA_FFI_Error_Code_INTERNAL
    return frame.api.XLA_FFI_Error_Create(cython.address(arguments))


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def find_entry(
    name: cython.p_char, length: cython.size_t
) -> cython.pointer[NativeEntry]:
    """Returns the function registered under a name, or `NULL` if there is none."""
    position: cython.size_t
    for position in range(entry_count):
        if entries[position].name_length == length and (
            memcmp(entries[position].name, name, length) == 0
        ):
            return cython.address(entries[position])
    return cython.NULL


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def handle_call(
    frame: cython.pointer[XLA_FFI_CallFrame],
) -> cython.pointer[XLA_FFI_Error]:
    """Runs a call of `cyxla_native_function`, or answers the question for the metadata."""
    if (
        frame.extension_start != cython.NULL
        and frame.extension_start.type == XLA_FFI_Extension_Metadata
    ):
        extension: cython.pointer[XLA_FFI_Metadata_Extension] = cython.cast(
            cython.pointer[XLA_FFI_Metadata_Extension], frame.extension_start
        )
        extension.metadata.api_version.major_version = XLA_FFI_API_MAJOR
        extension.metadata.api_version.minor_version = XLA_FFI_API_MINOR
        extension.metadata.traits = 0
        return cython.NULL

    message: cython.char[MESSAGE_SIZE]
    attributes: NativeAttribute[MAX_ATTRIBUTES]
    attribute_names: cython.char[ATTRIBUTE_NAMES_SIZE]
    arguments: NativeArray[MAX_ARRAYS]
    results: NativeArray[MAX_ARRAYS]
    function_name: cython.p_char = cython.NULL
    function_length: cython.size_t = 0
    attribute_count: cython.int = 0
    position: cython.int
    span: cython.pointer[XLA_FFI_ByteSpan]
    text: cython.pointer[XLA_FFI_ByteSpan]
    scalar: cython.pointer[XLA_FFI_Scalar]
    buffer: cython.pointer[XLA_FFI_Buffer]

    for position in range(frame.attrs.size):
        span = frame.attrs.names[position]
        if span.len == 4 and memcmp(span.ptr, b"name", 4) == 0:
            if frame.attrs.types[position] != XLA_FFI_AttrType_STRING:
                return make_error(frame, b"The attribute name must be a string.")
            text = cython.cast(
                cython.pointer[XLA_FFI_ByteSpan], frame.attrs.attrs[position]
            )
            function_name = cython.cast(cython.p_char, text.ptr)
            function_length = text.len
            continue
        if attribute_count == MAX_ATTRIBUTES or span.len >= ATTRIBUTE_NAME_SIZE:
            return make_error(
                frame, b"A native function call has too many or too long attributes."
            )
        memcpy(
            cython.address(attribute_names[attribute_count * ATTRIBUTE_NAME_SIZE]),
            span.ptr,
            span.len,
        )
        attribute_names[attribute_count * ATTRIBUTE_NAME_SIZE + span.len] = 0
        attributes[attribute_count].name = cython.address(
            attribute_names[attribute_count * ATTRIBUTE_NAME_SIZE]
        )
        if frame.attrs.types[position] == XLA_FFI_AttrType_SCALAR:
            scalar = cython.cast(
                cython.pointer[XLA_FFI_Scalar], frame.attrs.attrs[position]
            )
            attributes[attribute_count].kind = NATIVE_ATTRIBUTE_SCALAR
            attributes[attribute_count].dtype = scalar.dtype
            attributes[attribute_count].value = scalar.value
            attributes[attribute_count].length = 0
        elif frame.attrs.types[position] == XLA_FFI_AttrType_STRING:
            text = cython.cast(
                cython.pointer[XLA_FFI_ByteSpan], frame.attrs.attrs[position]
            )
            attributes[attribute_count].kind = NATIVE_ATTRIBUTE_STRING
            attributes[attribute_count].dtype = 0
            attributes[attribute_count].value = text.ptr
            attributes[attribute_count].length = text.len
        else:
            return make_error(
                frame, b"A native function takes scalar and string attributes only."
            )
        attribute_count += 1

    if function_name == cython.NULL:
        return make_error(frame, b"A native function call needs the attribute name.")
    entry: cython.pointer[NativeEntry] = find_entry(function_name, function_length)
    if entry == cython.NULL:
        snprintf(
            message,
            MESSAGE_SIZE,
            b"No native function is registered under the name %.*s. Import the library that registers it.",
            cython.cast(cython.int, function_length),
            function_name,
        )
        return make_error(frame, message)

    if frame.args.size > MAX_ARRAYS or frame.rets.size > MAX_ARRAYS:
        return make_error(
            frame, b"A native function call has too many arguments or results."
        )
    for position in range(frame.args.size):
        buffer = cython.cast(cython.pointer[XLA_FFI_Buffer], frame.args.args[position])
        arguments[position].dtype = buffer.dtype
        arguments[position].rank = buffer.rank
        arguments[position].dimensions = buffer.dims
        arguments[position].data = buffer.data
    for position in range(frame.rets.size):
        buffer = cython.cast(cython.pointer[XLA_FFI_Buffer], frame.rets.rets[position])
        results[position].dtype = buffer.dtype
        results[position].rank = buffer.rank
        results[position].dimensions = buffer.dims
        results[position].data = buffer.data

    failure: cython.p_char = cython.cast(
        cython.p_char,
        entry.function(
            entry.user_data,
            frame.args.size,
            arguments,
            frame.rets.size,
            results,
            attribute_count,
            attributes,
        ),
    )
    if failure != cython.NULL:
        return make_error(frame, failure)
    return cython.NULL


tables: list = []
"""The tables and names that were ever registered, kept alive for the handler."""


@cython.cfunc
@cython.exceptval(-1, check=True)
def register_native(
    name: cython.p_char, function: NativeFunction, user_data: cython.p_void
) -> cython.int:
    """Registers a function under a name for the programs of this process.

    Registering the same function again does nothing. Another function under the same name
    is an error. The table is replaced as a whole, and the old one stays in memory because
    the handler may be reading it.
    """
    global entries, entry_count
    length: cython.size_t = strlen(name)
    position: cython.size_t
    for position in range(entry_count):
        if entries[position].name_length == length and (
            memcmp(entries[position].name, name, length) == 0
        ):
            if entries[position].function == function and (
                entries[position].user_data == user_data
            ):
                return 0
            raise ValueError(
                f"The native function {name.decode()} is registered with another function."
            )
    # The new table and the copy of the name are Python objects, so a failed allocation
    # raises a `MemoryError` by itself. The tables stay in `tables` for good, because the
    # handler may still be reading an old one.
    owned_name: bytes = name
    table: bytearray = bytearray((entry_count + 1) * cython.sizeof(NativeEntry))
    tables.append((owned_name, table))
    fresh: cython.pointer[NativeEntry] = cython.cast(
        cython.pointer[NativeEntry], cython.cast(cython.p_char, table)
    )
    if entry_count > 0:
        memcpy(fresh, entries, entry_count * cython.sizeof(NativeEntry))
    fresh[entry_count].name = cython.cast(cython.p_char, owned_name)
    fresh[entry_count].name_length = length
    fresh[entry_count].function = function
    fresh[entry_count].user_data = user_data
    entries = fresh
    entry_count += 1
    if registration_hook is not None:
        registration_hook(owned_name.decode())
    return 0


def set_registration_hook(hook: object) -> None:
    """Sets the function that is called with the name of every registered function."""
    global registration_hook
    registration_hook = hook


def register_native_address(
    name: str,
    function: cython.size_t,
    user_data: cython.size_t,
    abi_version: cython.int,
) -> None:
    """Registers a function that is given by its address, as other languages pass it."""
    if abi_version != NATIVE_ABI_VERSION:
        raise ValueError(
            f"The native function {name} was built for version {abi_version} of the native "
            f"interface, but cyxla has version {NATIVE_ABI_VERSION}."
        )
    encoded: bytes = name.encode()
    register_native(
        encoded,
        cython.cast(NativeFunction, function),
        cython.cast(cython.p_void, user_data),
    )


python_dispatch: object = None
"""The function that runs a Python callback, set by `cyxla._callbacks`."""


def set_python_dispatch(dispatch: object) -> None:
    """Sets the function that `python_callback` calls with the GIL held."""
    global python_dispatch
    python_dispatch = dispatch


@cython.cfunc
def describe_arrays(
    count: int64_t, arrays: cython.pointer[cython.const[NativeArray]]
) -> list:
    """Returns the element type, shape, element count and address of each array.

    The Python dispatch builds its views from these.
    """
    described: list = []
    position: int64_t
    axis: int64_t
    for position in range(count):
        dimensions: list = []
        for axis in range(arrays[position].rank):
            dimensions.append(arrays[position].dimensions[axis])
        described.append(
            (
                arrays[position].dtype,
                tuple(dimensions),
                native_size(cython.address(arrays[position])),
                cython.cast(cython.size_t, arrays[position].data),
            )
        )
    return described


@cython.cfunc
@cython.exceptval(check=False)
def run_dispatch(
    index: cython.ulonglong,
    argument_count: int64_t,
    arguments: cython.pointer[cython.const[NativeArray]],
    result_count: int64_t,
    results: cython.pointer[NativeArray],
    message: cython.p_char,
) -> cython.int:
    """Calls the Python dispatch and returns 1 after writing the message of an exception.

    The message has the type of the exception followed by its text. It is cut to the size of
    the buffer. This runs with the GIL held, and it catches everything, because the runtime
    that called the native function cannot handle a Python exception.
    """
    try:
        python_dispatch(
            index,
            describe_arrays(argument_count, arguments),
            describe_arrays(result_count, results),
        )
    except BaseException as error:
        text: bytes = f"{type(error).__name__}: {error}".encode("utf-8", "replace")
        snprintf(message, THREAD_MESSAGE_SIZE, b"%s", cython.cast(cython.p_char, text))
        return 1
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def python_callback(
    user_data: cython.p_void,
    argument_count: int64_t,
    arguments: cython.pointer[cython.const[NativeArray]],
    result_count: int64_t,
    results: cython.pointer[NativeArray],
    attribute_count: int64_t,
    attributes: cython.pointer[cython.const[NativeAttribute]],
) -> cython.pointer[cython.const[cython.char]]:
    """Calls the Python callback that the attribute `index` names, with the GIL taken.

    Programs with callbacks call this native function. The message of a failure is in a
    buffer of the calling thread, which stays valid until the thread calls a native function
    again.
    """
    attribute: cython.pointer[cython.const[NativeAttribute]] = native_find_attribute(
        attributes, attribute_count, b"index"
    )
    if (
        attribute == cython.NULL
        or attribute.kind != NATIVE_ATTRIBUTE_SCALAR
        or attribute.dtype != NATIVE_U64
    ):
        return b"ValueError: The call of the callback has no attribute index."
    index: cython.ulonglong = cython.cast(
        cython.pointer[cython.const[cython.ulonglong]], attribute.value
    )[0]
    message: cython.p_char = cyxla_thread_message()
    failed: cython.int
    with cython.gil:
        failed = run_dispatch(
            index, argument_count, arguments, result_count, results, message
        )
    if failed:
        return message
    return cython.NULL


register_native(b"cyxla.python_callback", python_callback, cython.NULL)


def handler_address() -> cython.size_t:
    """Returns the address of the FFI handler, which the runtimes register."""
    return cython.cast(cython.size_t, handle_call)


def native_abi_version() -> cython.int:
    """Returns the version of the native interface that this build has."""
    return NATIVE_ABI_VERSION
