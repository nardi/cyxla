# Serializing programs

The [disk cache](caching.md) keeps lowered programs and executables on one machine,
under keys that a library computes from the functions it builds them from. A library
that wants to save a finished object in one process and load it in another needs more.
The saved file has to stand alone, it has to carry the StableHLO in case the loading
machine cannot use the executables, and it has to load fast. cyxla provides the parts
for this, and each library decides what its own file holds.

## Bundles

A [`BundleWriter`][cyxla.BundleWriter] collects named blobs and a JSON header into one
file. The header says what kind of object the file holds and which version of that
library's layout wrote it. A reader asks for the same kind and version and gets an error
otherwise.

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

Each blob starts at a multiple of 64 bytes and has a CRC, which is checked the first
time the blob is read. A reader takes a blob as a `memoryview`, so blobs that it never
uses cost nothing. `BundleWriter.write` writes a file that appears under its name only
when it is complete, and `Bundle.read` reads one.

## Programs

[`pack_program`][cyxla.pack_program] stores an executable in a bundle. It writes the
StableHLO of the program and the serialized executable, and it returns a record for the
header with the argument types, the platform, the custom call targets and the name of
the executable's blob. The executable has to come from `compile_program` or
`load_program`, which set its `program` attribute.

[`load_program`][cyxla.load_program] reverses it. The bundle holds an executable for
each target that stored one, under the key of a
[`TargetFingerprint`][cyxla.TargetFingerprint]. If the current target has an executable,
it is deserialized. Otherwise, or if the plugin rejects the executable, the StableHLO is
compiled again, and `on_mismatch="raise"` raises a
[`TargetMismatchError`][cyxla.TargetMismatchError] instead. The result is the executable
and the [`CompileStatistics`][cyxla.CompileStatistics] of getting it.

<!-- name: test_docs_serialization -->
```python
import jax
import jax.numpy as jnp

program = cx.lower(lambda x: jnp.sin(x) * 2, [jax.ShapeDtypeStruct((3,), jnp.float64)])
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
no executable for the loading machine, so loading compiles the StableHLO. That costs the
compile time once per load. The compile options of the plugin and its code generation
for the host CPU make executables machine-specific, which is why the bytecode is always
kept.

## Loading in a process without JAX

Importing cyxla does not import JAX. A process that loads a bundle and runs its programs
can stay without it, which saves about half a second. Call
`initialize(jax_targets=False)` unless a program uses custom calls.
[`record_custom_calls`][cyxla.record_custom_calls] returns the targets that the records
of a bundle need, and `initialize` registers the handlers of jaxlib for them when it is
left at its default. `Executable.warm_up` executes a program once on zeros, so that the
first real call does not pay for loading its code.

## Checking trees

A saved object works on flat vectors, so the caller supplies the trees that those
vectors are cut from when loading. [`tree_fingerprint`][cyxla.tree_fingerprint] records
the path and shape of each leaf of a tree, and
[`check_fingerprint`][cyxla.check_fingerprint] compares it with the fingerprint of the
tree that the caller supplies. The error names the first leaf that differs.
[`encode_fingerprint`][cyxla.encode_fingerprint] and
[`decode_fingerprint`][cyxla.decode_fingerprint] move fingerprints through JSON headers,
and [`encode_record`][cyxla.encode_record] and [`decode_record`][cyxla.decode_record] do
the same for option records made of named tuples, enums and numbers.

## Trust

A serialized executable is machine code, and the plugin loads it as it is. Load only
files that you trust. The CRC of a blob detects damage, and it does not protect against
someone who writes a file on purpose.
