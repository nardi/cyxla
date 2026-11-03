# A shared library with native functions, built with the C compiler of each system against
# the header that cyxla installs.
project('test-native-library', 'c', default_options: ['c_std=c11'])

shared_library(
  'native_functions',
  'native_functions.c',
  include_directories: include_directories(get_option('cyxla_include')),
  dependencies: meson.get_compiler('c').find_library('m', required: false),
)
