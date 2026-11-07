# Program serialization

The [disk cache](caching.md) stores lowered programs and executables on one machine,
under keys that a library computes from the functions that they were lowered from.
Saving an object in one process and loading it in another has further requirements. The
saved file has to be self-contained, it has to contain the StableHLO in case the
executables cannot be used on the loading machine, and it has to load fast. cyxla
provides the building blocks for such a file, and its contents are defined by each
library.

## Bundles

A [`BundleWriter`][cyxla.BundleWriter] collects named blobs and a JSON header into one
file. The header contains the kind of object in the file and the version of the layout
of the library. A reader passes the expected kind and version, and a mismatch raises an
error.

<!-- name: test_docs_serialization -->
```python
import numpy as np

import cyxla as cx

writer = cx.BundleWriter("example.object", 1)
writer.add("indices", np.arange(4, dtype=np.int32))
data = writer.to_bytes({"size": 4})

bundle = cx.Bundle(data, "example.object", 1)
print(bundle.meta, bundle.array("indices", np.int32))
```

<!-- name: test_docs_serialization -->
```python
assert bundle.meta == {"size": 4}
assert bundle.array("indices", np.int32).tolist() == [0, 1, 2, 3]
```

Each blob starts at a multiple of 64 bytes and has a CRC, which the reader checks the
first time it reads the blob. A reader takes a blob as a `memoryview`, so blobs that are
never used are not read or checked. `BundleWriter.write` writes the file under a
temporary name and renames it once it is complete, and `Bundle.read` reads one.

## Programs

[`pack_program`][cyxla.pack_program] stores an executable in a bundle. It writes the
StableHLO of the program and the serialized executable, and it returns a record for the
header with the argument types, the platform, the custom call targets and the name of
the executable's blob. The executable has to come from `compile_program` or
`load_program`, which set its `program` attribute.

[`load_program`][cyxla.load_program] loads the executable again. The bundle contains an
executable for each target that stored one, under the key of a
[`TargetFingerprint`][cyxla.TargetFingerprint]. If the current target has an executable,
`load_program` deserializes it. Otherwise, or if the plugin rejects the executable, it
compiles the StableHLO again, and `on_mismatch="raise"` raises a
[`TargetMismatchError`][cyxla.TargetMismatchError] instead. The result is the executable
and the [`CompileStatistics`][cyxla.CompileStatistics] of loading or compiling it.

<!-- name: test_docs_serialization -->
```python
import jax
import jax.numpy as jnp

shapes = [jax.ShapeDtypeStruct((3,), jnp.float64)]
program = cx.lower(lambda values: jnp.sin(values) * 2, shapes)
executable, _ = cx.compile_program(program, disk_cache=False)

writer = cx.BundleWriter("example.program", 1)
record = cx.pack_program(writer, "sine", executable)
bundle = cx.Bundle(writer.to_bytes({"record": record}), "example.program", 1)

loaded, statistics = cx.load_program(bundle, bundle.meta["record"])
(result,) = loaded.run([np.array([0.0, 1.0, 2.0])])
```

<!-- name: test_docs_serialization -->
```python
assert statistics.loaded == 1
assert np.allclose(result, 2 * np.sin([0.0, 1.0, 2.0]))
```

A file that was written on another operating system, architecture or plugin version has
no executable for the loading machine, so loading compiles the StableHLO. This adds the
compile time to every load. The compile options of the plugin and its code generation
for the host CPU make executables machine-specific, which is why the bytecode is always
stored.

## Processes without JAX

Importing cyxla does not import JAX, so a process that loads a bundle and runs its
programs does not need to import JAX, which saves about half a second. Call
`initialize(jax_targets=False)` unless a program uses custom calls.
[`record_custom_calls`][cyxla.record_custom_calls] returns the targets that the records
of a bundle require, and with its default arguments `initialize` registers the handlers
of `jaxlib` for them. `Executable.warm_up` executes a program once on zeros, so that the
code of the program is already loaded at the first real call. A program that checks its
inputs can fail on zeros, so `warm_up` also accepts the arguments to run on.

## Targets of libraries

A program that calls the handlers of a library can only be loaded when that library is
imported, because the handlers are defined in the library. A record stores the package
that registered each target and the version of its distribution.
[`check_targets`][cyxla.check_targets] raises a
[`MissingTargetError`][cyxla.MissingTargetError] after `initialize` and before the first
run, and the message names each missing target with its provider.

```python
cx.check_targets(records)
```

The provider is only known for handlers that were registered after cyxla was imported.
Handlers that were already in the registry of JAX when the plugin loaded have no
recorded provider. The caller can import the missing providers, or pass
`import_providers=True` to `check_targets` to import them. Importing a package whose
name is read from a file runs the code of that package, so the option is off by default.

## Tree fingerprints

A saved object operates on flat vectors, so the trees that these vectors are flattened
from have to be passed again when the object is loaded.
[`tree_fingerprint`][cyxla.tree_fingerprint] records the path and shape of each leaf of
a tree, and [`check_fingerprint`][cyxla.check_fingerprint] compares it with the
fingerprint of the tree that is passed when loading. The error names the first leaf that
differs. [`encode_fingerprint`][cyxla.encode_fingerprint] and
[`decode_fingerprint`][cyxla.decode_fingerprint] convert fingerprints to and from JSON
for the header, and [`encode_record`][cyxla.encode_record] and
[`decode_record`][cyxla.decode_record] do the same for option records made of named
tuples, enums and numbers.

## Trust

A serialized executable is machine code, and the plugin loads it without verification.
Load only files that you trust. The CRC of a blob detects corruption, but it does not
protect against a file that was modified deliberately.
