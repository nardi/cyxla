/*
 * The operating system calls of the runtime, for POSIX systems and for Windows.
 *
 * Only `_runtime` includes this header. The declarations that other extension modules
 * cimport from `_runtime.pxd` do not depend on it.
 */

#ifndef CYXLA_PLATFORM_H_
#define CYXLA_PLATFORM_H_

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <malloc.h>
#include <windows.h>
#else
#include <dlfcn.h>
#include <time.h>
#endif

/* Opens a shared library by its UTF-8 path and returns its handle, or NULL on failure. */
static void* cyxla_open_library(const char* path) {
#ifdef _WIN32
  int size = MultiByteToWideChar(CP_UTF8, 0, path, -1, NULL, 0);
  if (size <= 0) {
    return NULL;
  }
  wchar_t* wide = (wchar_t*)malloc((size_t)size * sizeof(wchar_t));
  if (wide == NULL) {
    return NULL;
  }
  MultiByteToWideChar(CP_UTF8, 0, path, -1, wide, size);
  /* The directory of the library comes first when Windows resolves its dependencies. */
  HMODULE module = LoadLibraryExW(
      wide, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
  free(wide);
  return (void*)module;
#else
  return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

/* Returns the address of a symbol of an open library, or NULL if it has none. */
static void* cyxla_find_symbol(void* library, const char* name) {
#ifdef _WIN32
  return (void*)(uintptr_t)GetProcAddress((HMODULE)library, name);
#else
  return dlsym(library, name);
#endif
}

/* Writes the reason why the last library failed to open into `buffer`. */
static void cyxla_library_error(char* buffer, size_t size) {
  if (size == 0) {
    return;
  }
  buffer[0] = '\0';
#ifdef _WIN32
  DWORD length = FormatMessageA(
      FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, GetLastError(), 0,
      buffer, (DWORD)size, NULL);
  /* Windows ends its messages with a line break, which the error message does not need. */
  while (length > 0 && (buffer[length - 1] == '\n' || buffer[length - 1] == '\r')) {
    buffer[--length] = '\0';
  }
#else
  const char* reason = dlerror();
  if (reason != NULL) {
    strncpy(buffer, reason, size - 1);
    buffer[size - 1] = '\0';
  }
#endif
}

/* Returns `size` bytes at a multiple of `alignment`, or NULL, freed with `cyxla_aligned_free`. */
static void* cyxla_aligned_alloc(size_t alignment, size_t size) {
#ifdef _WIN32
  return _aligned_malloc(size, alignment);
#else
  void* memory = NULL;
  if (posix_memalign(&memory, alignment, size) != 0) {
    return NULL;
  }
  return memory;
#endif
}

/* Frees memory from `cyxla_aligned_alloc`, ignoring NULL. */
static void cyxla_aligned_free(void* memory) {
#ifdef _WIN32
  _aligned_free(memory);
#else
  free(memory);
#endif
}

/* Returns a monotonic time in seconds. */
static double cyxla_monotonic_seconds(void) {
#ifdef _WIN32
  LARGE_INTEGER frequency;
  LARGE_INTEGER counter;
  QueryPerformanceFrequency(&frequency);
  QueryPerformanceCounter(&counter);
  return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
  struct timespec moment;
  clock_gettime(CLOCK_MONOTONIC, &moment);
  return (double)moment.tv_sec + 1e-9 * (double)moment.tv_nsec;
#endif
}

#endif  /* CYXLA_PLATFORM_H_ */
