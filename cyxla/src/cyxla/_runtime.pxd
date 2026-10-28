from libc.stdint cimport int64_t

from cyxla._pjrt cimport (
    PJRT_Api,
    PJRT_Buffer,
    PJRT_Buffer_Type,
    PJRT_Client,
    PJRT_Device,
    PJRT_Error,
    PJRT_Event,
    PJRT_Executable,
    PJRT_LoadedExecutable,
)


cdef enum:
    # The version of the declarations below that other extension modules compile against.
    # It changes whenever a struct or a function signature changes, so `cyxla.check_abi`
    # can reject a module compiled against other declarations.
    ABI_VERSION = 3
    ALIGNMENT = 64
    MAX_INPUTS = 8
    MAX_OUTPUTS = 4


cdef struct ProgramSpec:
    # An executable with its number of inputs and the size in bytes of each output.
    PJRT_LoadedExecutable* handle
    int num_inputs
    int num_outputs
    int64_t output_bytes[MAX_OUTPUTS]


cdef struct Input:
    # A host array of `size` elements of `type` and its aliased buffer, which is NULL when
    # the plugin cannot alias host memory.
    void* host
    PJRT_Buffer* buffer
    PJRT_Buffer_Type type
    int64_t size


ctypedef const PJRT_Api* (*GetPjrtApi)() noexcept nogil


# Runtime state shared by all functions. Only one thread may use the runtime at a time.
cdef const PJRT_Api* api
cdef PJRT_Client* client
cdef PJRT_Device* device
cdef bint can_alias
cdef bint can_read_inplace
cdef char error_message[1024]
cdef long error_count


cdef double now() noexcept nogil
cdef int check(PJRT_Error* error) noexcept nogil
cpdef int itemsize(PJRT_Buffer_Type type) noexcept nogil
cdef PJRT_Buffer* alias_buffer(void* data, PJRT_Buffer_Type type, int64_t size) noexcept nogil
cdef void destroy_buffer(PJRT_Buffer* buffer) noexcept nogil
cdef void* aligned_memory(size_t size) noexcept nogil
cdef void free_aligned(void* memory) noexcept nogil
cdef int await_event(PJRT_Event* event) noexcept nogil
cdef void destroy_event(PJRT_Event* event) noexcept nogil
cdef PJRT_Buffer* buffer_from_host(void* data, PJRT_Buffer_Type type, int64_t size, bint alias) noexcept nogil
cdef void destroy_executable(PJRT_Executable* executable) noexcept nogil
cdef int execute(ProgramSpec* program, Input* inputs, void** outputs, bint zero_copy) noexcept nogil


cdef class Executable:
    cdef PJRT_LoadedExecutable* handle
    cdef readonly int num_outputs
    cdef readonly tuple output_types
    cdef public object input_types

    @staticmethod
    cdef Executable wrap(PJRT_LoadedExecutable* handle)
