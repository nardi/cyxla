# An extension module of another package, built against an installed cyxla the way a
# downstream project would build it.
project('downstream', 'c', default_options: ['c_std=c11'])

add_project_arguments(
  meson.get_compiler('c').get_supported_arguments(
    '-Werror=incompatible-pointer-types', '-Werror=incompatible-function-pointer-types',
  ),
  language: 'c',
)

python = import('python').find_installation(pure: false)
cython = find_program('cython')

# The first directory holds the PJRT header, the second the `.pxd` files.
cyxla_include = run_command(
  python, '-c', 'import cyxla; print("\\n".join(cyxla.get_include()))', check: true,
).stdout().strip().split('\n')

# Each module is a pure Python mode source (`py`) or a classic Cython source (`pyx`).
foreach module : [
  ['downstream_square', 'py'],
  ['downstream_native', 'py'],
  ['downstream_native_pyx', 'pyx'],
]
name = module[0]
module_c = custom_target(
  name + '.c',
  input: 'src/' + name + '.' + module[1],
  output: name + '.c',
  command: [
    cython, '-3', '--fast-fail', '-I', cyxla_include[1],
    '--module-name', name, '@INPUT@', '-o', '@OUTPUT@',
  ],
)
python.extension_module(
  name,
  module_c,
  include_directories: include_directories(cyxla_include[0]),
  dependencies: python.dependency(),
)
endforeach
