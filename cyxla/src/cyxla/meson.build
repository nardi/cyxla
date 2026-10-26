python.install_sources(
  '__init__.py',
  '__main__.py',
  '_cache.py',
  '_lowering.py',
  '_plugins.py',
  '_types.py',
  '_pjrt.pxd',
  '_runtime.pxd',
  '_runtime.pyi',
  'py.typed',
  subdir: 'cyxla',
)
install_subdir('include', install_dir: python.get_install_dir() / 'cyxla')

cython_args = ['-3', '--fast-fail']
# MSVC knows none of these options, so only the supported ones are passed.
c_args = meson.get_compiler('c').get_supported_arguments(
  '-Wno-unused-function', '-Wno-unused-variable', '-Wno-unused-but-set-variable',
)
if get_option('coverage')
  cython_args += ['-X', 'linetrace=True']
  # Coverage measures branches through `sys.settrace` before Python 3.14, so Cython has to
  # report lines through the same hooks rather than through `sys.monitoring`.
  c_args += ['-DCYTHON_TRACE_NOGIL=1', '-DCYTHON_USE_SYS_MONITORING=0']
endif

# Meson compiles Cython from `.pyx` sources only, so the pure Python mode module is
# translated to C by an explicit `cython` step. The declarations are listed as dependencies,
# because the depfile would record paths relative to the project root rather than the build
# directory.
runtime_c = custom_target(
  '_runtime.c',
  input: '_runtime.py',
  output: '_runtime.c',
  depend_files: files('_pjrt.pxd', '_platform.h', '_platform.pxd', '_runtime.pxd'),
  command: [
    cython, cython_args,
    # Running from the project root makes Cython record the source as
    # `src/cyxla/_runtime.py`, the path coverage resolves line events against.
    '-w', meson.project_source_root(),
    '-I', 'src',
    '--module-name', 'cyxla._runtime',
    meson.current_source_dir() / '_runtime.py',
    '-o', meson.current_build_dir() / '_runtime.c',
  ],
)
if get_option('coverage')
  # The `Cython.Coverage` plugin looks for the generated C file next to the source, so
  # coverage builds place a copy there.
  custom_target(
    '_runtime.c.copy',
    input: runtime_c,
    output: '_runtime.c.stamp',
    command: [
      python, '-c', 'import shutil, sys; shutil.copyfile(sys.argv[1], sys.argv[2]); open(sys.argv[3], "w").close()',
      '@INPUT@', meson.current_source_dir() / '_runtime.c', '@OUTPUT@',
    ],
    build_by_default: true,
  )
endif
python.extension_module(
  '_runtime',
  runtime_c,
  dependencies: [pjrt, dl, python.dependency()],
  # `_platform.h` sits next to the sources and is not installed, as only this module needs it.
  include_directories: include_directories('.'),
  c_args: c_args,
  install: true,
  subdir: 'cyxla',
)
