"""Declarations of native functions, which programs call through the handler of cyxla.

The structures come from `cyxla/native.h`, the header that C, Rust and C++ code includes.
A Cython module registers a function of the type `NativeFunction` with `register_native`,
which checks the type at compile time, and programs call it by name with
`cyxla.native_call`. See the header for the rules a function has to follow.
"""

from libc.math cimport NAN
from libc.stdint cimport int32_t, int64_t
from libc.string cimport strcmp


cdef extern from "cyxla/native.h" nogil:
    enum:
        NATIVE_ABI_VERSION "CYXLA_NATIVE_ABI_VERSION"
        NATIVE_PRED "CYXLA_NATIVE_PRED"
        NATIVE_S8 "CYXLA_NATIVE_S8"
        NATIVE_S16 "CYXLA_NATIVE_S16"
        NATIVE_S32 "CYXLA_NATIVE_S32"
        NATIVE_S64 "CYXLA_NATIVE_S64"
        NATIVE_U8 "CYXLA_NATIVE_U8"
        NATIVE_U16 "CYXLA_NATIVE_U16"
        NATIVE_U32 "CYXLA_NATIVE_U32"
        NATIVE_U64 "CYXLA_NATIVE_U64"
        NATIVE_F16 "CYXLA_NATIVE_F16"
        NATIVE_F32 "CYXLA_NATIVE_F32"
        NATIVE_F64 "CYXLA_NATIVE_F64"
        NATIVE_C64 "CYXLA_NATIVE_C64"
        NATIVE_BF16 "CYXLA_NATIVE_BF16"
        NATIVE_C128 "CYXLA_NATIVE_C128"
        NATIVE_ATTRIBUTE_SCALAR "CYXLA_NATIVE_ATTRIBUTE_SCALAR"
        NATIVE_ATTRIBUTE_STRING "CYXLA_NATIVE_ATTRIBUTE_STRING"

    ctypedef struct NativeArray "CyxlaNativeArray":
        int32_t dtype
        int64_t rank
        const int64_t* dimensions
        void* data

    ctypedef struct NativeAttribute "CyxlaNativeAttribute":
        const char* name
        int32_t kind
        int32_t dtype
        const void* value
        int64_t length

    ctypedef const char* (*NativeFunction "CyxlaNativeFunction")(
        void* user_data,
        int64_t argument_count,
        const NativeArray* arguments,
        int64_t result_count,
        NativeArray* results,
        int64_t attribute_count,
        const NativeAttribute* attributes,
    ) noexcept nogil


cdef enum:
    # The most arguments, and the most results, that a call may have.
    MAX_ARRAYS = 32
    # The most attributes besides the name that a call may have.
    MAX_ATTRIBUTES = 16
    # The most bytes of the name of an attribute, with the terminator.
    ATTRIBUTE_NAME_SIZE = 64
    # The size of the buffer that holds the names of all attributes.
    ATTRIBUTE_NAMES_SIZE = 1024
    # The size of the buffer for the messages that the handler writes itself.
    MESSAGE_SIZE = 256


cdef struct NativeEntry:
    # A registered function. The name is a copy that the entry owns.
    char* name
    size_t name_length
    NativeFunction function
    void* user_data


# The functions that programs may call, as an array that is replaced as a whole and never
# freed, because the handler reads it from other threads without the GIL.
cdef NativeEntry* entries
cdef size_t entry_count


# Called with the name of every function that is registered, so the Python layer can record
# which package registered it. It is also called for functions registered from Cython.
cdef object registration_hook


cdef int register_native(char* name, NativeFunction function, void* user_data) except? -1


cdef inline int64_t native_size(const NativeArray* array) noexcept nogil:
    # The number of elements of an argument or a result.
    cdef int64_t size = 1
    cdef int64_t axis
    for axis in range(array.rank):
        size *= array.dimensions[axis]
    return size


cdef inline const NativeAttribute* native_find_attribute(
    const NativeAttribute* attributes, int64_t count, const char* name
) noexcept nogil:
    # The attribute with a name, or `NULL` if the call has none.
    cdef int64_t position
    for position in range(count):
        if strcmp(attributes[position].name, name) == 0:
            return &attributes[position]
    return NULL


cdef inline double native_attribute_double(const NativeAttribute* attribute) noexcept nogil:
    # The value of a scalar attribute as a double, or `NaN` for another kind or type.
    if attribute == NULL or attribute.kind != NATIVE_ATTRIBUTE_SCALAR:
        return NAN
    if attribute.dtype == NATIVE_F64:
        return (<const double*> attribute.value)[0]
    if attribute.dtype == NATIVE_F32:
        return (<const float*> attribute.value)[0]
    if attribute.dtype == NATIVE_S64:
        return <double> (<const int64_t*> attribute.value)[0]
    if attribute.dtype == NATIVE_S32:
        return <double> (<const int32_t*> attribute.value)[0]
    return NAN
