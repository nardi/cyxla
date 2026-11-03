"""Declarations of the part of the XLA FFI C API that the handler for native functions uses.

The declarations follow `xla/ffi/api/c_api.h` (vendored under `include`). Cython only needs
the fields that the module reads or writes, so each struct lists a subset of its fields, and
the C compiler uses the real layout from the header.
"""

from libc.stdint cimport int64_t, uint32_t


cdef extern from "xla/ffi/api/c_api.h" nogil:
    enum:
        XLA_FFI_API_MAJOR
        XLA_FFI_API_MINOR
        XLA_FFI_Extension_Metadata
        XLA_FFI_AttrType_SCALAR
        XLA_FFI_AttrType_STRING
        XLA_FFI_Error_Code_INTERNAL
        XLA_FFI_Error_Create_Args_STRUCT_SIZE

    ctypedef struct XLA_FFI_Extension_Base:
        size_t struct_size
        int type
        XLA_FFI_Extension_Base* next

    ctypedef struct XLA_FFI_Api_Version:
        int major_version
        int minor_version

    ctypedef struct XLA_FFI_Error

    ctypedef struct XLA_FFI_Error_Create_Args:
        size_t struct_size
        XLA_FFI_Extension_Base* extension_start
        const char* message
        int errc

    ctypedef struct XLA_FFI_Api:
        XLA_FFI_Error* (*XLA_FFI_Error_Create)(XLA_FFI_Error_Create_Args* args) noexcept nogil

    ctypedef struct XLA_FFI_Metadata:
        XLA_FFI_Api_Version api_version
        uint32_t traits

    ctypedef struct XLA_FFI_Metadata_Extension:
        XLA_FFI_Extension_Base extension_base
        XLA_FFI_Metadata* metadata

    ctypedef struct XLA_FFI_ByteSpan:
        const char* ptr
        size_t len

    ctypedef struct XLA_FFI_Scalar:
        int dtype
        void* value

    ctypedef struct XLA_FFI_Buffer:
        int dtype
        void* data
        int64_t rank
        int64_t* dims

    ctypedef struct XLA_FFI_Args:
        int64_t size
        void** args

    ctypedef struct XLA_FFI_Rets:
        int64_t size
        void** rets

    ctypedef struct XLA_FFI_Attrs:
        int64_t size
        int* types
        XLA_FFI_ByteSpan** names
        void** attrs

    ctypedef struct XLA_FFI_CallFrame:
        XLA_FFI_Extension_Base* extension_start
        const XLA_FFI_Api* api
        XLA_FFI_Args args
        XLA_FFI_Rets rets
        XLA_FFI_Attrs attrs
