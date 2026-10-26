# The operating system calls of `_platform.h`, which only `_runtime` uses.


cdef extern from "_platform.h" nogil:
    void* cyxla_open_library(const char* path)
    void* cyxla_find_symbol(void* library, const char* name)
    void cyxla_library_error(char* buffer, size_t size)
    void* cyxla_aligned_alloc(size_t alignment, size_t size)
    void cyxla_aligned_free(void* memory)
    double cyxla_monotonic_seconds()
