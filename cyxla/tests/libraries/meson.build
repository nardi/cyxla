# Shared libraries that the tests load as broken or extra plugins, built with the compiler
# of each system.
project('test-libraries', 'c', default_options: ['c_std=c11'])

pjrt = include_directories(get_option('pjrt_include'))

shared_library('not_plugin', 'not_plugin.c')
shared_library('old_plugin', 'old_plugin.c', include_directories: pjrt)
shared_library('preloaded', 'preloaded.c')
