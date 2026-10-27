// A typed FFI handler of the kind that third-party libraries register: it multiplies a
// float64 vector by an attribute.

#include <cstddef>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

static ffi::Error ScaleImpl(double factor, ffi::Buffer<ffi::F64> input,
                            ffi::ResultBuffer<ffi::F64> output) {
  const double* values = input.typed_data();
  double* scaled = output->typed_data();
  for (std::size_t index = 0; index < input.element_count(); ++index) {
    scaled[index] = factor * values[index];
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(Scale, ScaleImpl,
                       ffi::Ffi::Bind()
                           .Attr<double>("factor")
                           .Arg<ffi::Buffer<ffi::F64>>()
                           .Ret<ffi::Buffer<ffi::F64>>());

#if defined(_WIN32)
#define SCALE_EXPORT extern "C" __declspec(dllexport)
#else
#define SCALE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// Returns the address of the handler, which `cyxla.register_ffi_target` takes.
SCALE_EXPORT void* scale_handler() { return reinterpret_cast<void*>(Scale); }
