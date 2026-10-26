# An extension module of another package, built against an installed cyxla the way a
# downstream project would build it.
project('downstream', 'c', default_options: ['c_std=c11'])

python = import('python').find_installation(pure: false)
cython = find_program('cython')

# The first directory holds the PJRT header, the second the `.pxd` files.
cyxla_include = run_command(
  python, '-c', 'import cyxla; print("\\n".join(cyxla.get_include()))', check: true,
).stdout().strip().split('\n')

module_c = custom_target(
  'downstream_square.c',
  input: 'src/downstream_square.py',
  output: 'downstream_square.c',
  command: [
    cython, '-3', '--fast-fail', '-I', cyxla_include[1],
    '--module-name', 'downstream_square', '@INPUT@', '-o', '@OUTPUT@',
  ],
)
python.extension_module(
  'downstream_square',
  module_c,
  include_directories: include_directories(cyxla_include[0]),
  dependencies: python.dependency(),
)
