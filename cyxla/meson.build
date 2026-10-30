project(
  'cyxla',
  'c',
  version: '0.1.1',
  meson_version: '>=1.8',
  default_options: ['c_std=c11', 'buildtype=release', 'b_ndebug=if-release'],
)

python = import('python').find_installation(pure: false)
cython = find_program('cython')

# The PJRT C API is a single header, installed with the package so that extension modules
# built against cyxla can include it. The plugin itself is opened with `dlopen` at run time.
pjrt = declare_dependency(include_directories: include_directories('src/cyxla/include'))

# `dlopen` lives in libdl on older glibc and in libc or libSystem elsewhere.
dl = meson.get_compiler('c').find_library('dl', required: false)

subdir('src/cyxla')
