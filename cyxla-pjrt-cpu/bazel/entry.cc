// Entry point of the plugin library, defined here so that it has default visibility.
//
// XLA's own definition of `GetPjrtApi` is compiled with `-fvisibility=hidden` under the
// Bazel options of jaxlib, so no linker option can export it afterwards.

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_cpu_internal.h"

#if defined(_WIN32)
#define CYXLA_EXPORT __declspec(dllexport)
#else
#define CYXLA_EXPORT __attribute__((visibility("default")))
#endif

extern "C" CYXLA_EXPORT const PJRT_Api* GetPjrtApi() {
  return pjrt::cpu_plugin::GetCpuPjrtApi();
}
