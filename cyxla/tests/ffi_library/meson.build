# A shared library with a typed FFI handler, built with the C++ compiler of each system
# against the headers that jaxlib ships.
project('test-ffi-library', 'cpp', default_options: ['cpp_std=c++17'])

shared_library(
  'scale',
  'scale.cc',
  include_directories: include_directories(get_option('ffi_include'), is_system: true),
)
