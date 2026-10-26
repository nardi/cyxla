"""A PEP 517 build backend that packages a PJRT CPU plugin library as a platform wheel.

`build_plugin.py` builds the library from XLA. The backend then writes a wheel with the
library next to the `cyxla_pjrt_cpu` module, and reads the platform and the wheel tag from
the library itself. It takes the library from the config setting `library`, from the
environment variable `CYXLA_PJRT_CPU_LIBRARY`, or else downloads the library of this machine
from the release in `plugin.toml` and checks its hash.

    uv build --wheel -C library=dist/library/libpjrt_cpu.so
"""

import base64
import gzip
import hashlib
import io
import mmap
import os
import platform
import re
import shutil
import struct
import sys
import tarfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

PROJECT_DIRECTORY = Path(__file__).parent
"""Directory of `pyproject.toml`, which the build frontend runs the backend from."""

LIBRARY_ENVIRONMENT_VARIABLE = "CYXLA_PJRT_CPU_LIBRARY"
"""Environment variable with the path of a library to package."""

PACKAGE = "cyxla_pjrt_cpu"
"""Name of the importable package."""

SDIST_FILES = (
    "pyproject.toml",
    "backend.py",
    "build_plugin.py",
    "plugin.toml",
    "README.md",
    "LICENSE",
    "bazel/BUILD.bazel",
    "bazel/exports.def",
    "bazel/version_script.lds",
)
"""Files of the project directory that the sdist holds, besides the package sources."""

ConfigSettings = dict[str, str | list[str]] | None
"""The config settings that a build frontend passes, with string or repeated values."""

ELF_MACHINES = {62: "x86_64", 183: "aarch64"}
"""Architectures of ELF `e_machine` values."""

MACHO_CPU_TYPES = {0x01000007: "x86_64", 0x0100000C: "arm64"}
"""Architectures of Mach-O `cputype` values."""

PE_MACHINES = {0x8664: "amd64", 0xAA64: "arm64"}
"""Architectures of PE `Machine` values."""

GLIBC_VERSION = re.compile(rb"GLIBC_(\d+)\.(\d+)")
"""A glibc symbol version that a library needs."""


def read_toml(name: str) -> dict[str, Any]:
    """Returns a TOML file of the project directory.

    The values are typed `Any` because they are whatever the file holds.
    """
    with open(PROJECT_DIRECTORY / name, "rb") as file:
        return tomllib.load(file)


def host_platform() -> str:
    """Returns the name in `plugin.toml` of the platform of this machine."""
    match sys.platform, platform.machine().lower():
        case "linux", "x86_64" | "amd64":
            return "linux-x86_64"
        case "linux", "aarch64" | "arm64":
            return "linux-aarch64"
        case "darwin", "arm64" | "aarch64":
            return "macos-arm64"
        case "win32", "amd64" | "x86_64":
            return "windows-amd64"
        case system, machine:
            raise RuntimeError(
                f"There is no PJRT CPU plugin for {system} on {machine}."
            )


def elf_glibc_version(data: bytes | mmap.mmap) -> tuple[int, int]:
    """Returns the newest glibc symbol version that a 64-bit ELF library needs.

    The versions are the names of the `Vernaux` entries in the `SHT_GNU_verneed` section,
    which the section's linked string table holds. A library that needs no versioned glibc
    symbol gets glibc 2.17, the oldest that manylinux wheels can declare.
    """
    (section_offset,) = struct.unpack_from("<Q", data, 0x28)
    section_size, section_count = struct.unpack_from("<HH", data, 0x3A)
    sections = [
        struct.unpack_from("<IIQQQQII", data, section_offset + index * section_size)
        for index in range(section_count)
    ]
    newest = (2, 17)
    for _, kind, _, _, offset, _, link, count in sections:
        if kind != 0x6FFFFFFE:
            continue
        strings = sections[link][4]
        entry = offset
        for _ in range(count):
            _, auxiliary_count, _, auxiliary, following = struct.unpack_from(
                "<HHIII", data, entry
            )
            position = entry + auxiliary
            for _ in range(auxiliary_count):
                _, _, _, name, next_auxiliary = struct.unpack_from(
                    "<IHHII", data, position
                )
                end = data.find(b"\0", strings + name)
                found = GLIBC_VERSION.fullmatch(bytes(data[strings + name : end]))
                if found:
                    newest = max(newest, (int(found[1]), int(found[2])))
                position += next_auxiliary
            entry += following
    return newest


def macos_minimum_version(data: bytes | mmap.mmap) -> tuple[int, int]:
    """Returns the minimum macOS version in the load commands of a 64-bit Mach-O library."""
    (command_count,) = struct.unpack_from("<I", data, 16)
    offset = 32
    for _ in range(command_count):
        command, size = struct.unpack_from("<II", data, offset)
        match command:
            case 0x32:
                (version,) = struct.unpack_from("<I", data, offset + 12)
                return version >> 16, (version >> 8) & 0xFF
            case 0x24:
                (version,) = struct.unpack_from("<I", data, offset + 8)
                return version >> 16, (version >> 8) & 0xFF
        offset += size
    raise ValueError("The Mach-O library states no minimum macOS version.")


def library_platform(path: Path) -> tuple[str, str]:
    """Returns the platform name in `plugin.toml` and the wheel tag of a plugin library."""
    with (
        open(path, "rb") as file,
        mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data,
    ):
        match bytes(data[:4]):
            case b"\x7fELF":
                (machine,) = struct.unpack_from("<H", data, 18)
                architecture = ELF_MACHINES[machine]
                major, minor = elf_glibc_version(data)
                return (
                    f"linux-{architecture}",
                    f"manylinux_{major}_{minor}_{architecture}",
                )
            case b"\xcf\xfa\xed\xfe":
                (cpu,) = struct.unpack_from("<I", data, 4)
                architecture = MACHO_CPU_TYPES[cpu]
                major, minor = macos_minimum_version(data)
                if major >= 11:
                    minor = 0
                return f"macos-{architecture}", f"macosx_{major}_{minor}_{architecture}"
            case header if header[:2] == b"MZ":
                (pe_offset,) = struct.unpack_from("<I", data, 0x3C)
                if bytes(data[pe_offset : pe_offset + 4]) != b"PE\0\0":
                    raise ValueError(f"{path} is not a PE library.")
                (machine,) = struct.unpack_from("<H", data, pe_offset + 4)
                architecture = PE_MACHINES[machine]
                return f"windows-{architecture}", f"win_{architecture}"
            case _:
                raise ValueError(f"{path} is not an ELF, Mach-O or PE library.")


def wheel_tag(path: Path) -> str:
    """Returns the platform tag of the wheel that holds a plugin library."""
    return library_platform(path)[1]


def file_hash(path: Path) -> str:
    """Returns the SHA-256 hash of a file as hexadecimal."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while block := file.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def download_directory(version: str) -> Path:
    """Returns the directory that downloaded libraries of a version are kept in."""
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    directory = Path(base) / "cyxla-pjrt-cpu" / version
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def released_library(settings: dict[str, Any], version: str) -> Path:
    """Returns the released library of this machine, downloading and checking it if needed.

    A library whose hash does not match `plugin.toml` is removed and raises a
    `RuntimeError`, so the next build downloads it again.
    """
    platform_name = host_platform()
    entry = settings["platforms"][platform_name]
    expected = entry["sha256"]
    if not expected:
        raise RuntimeError(
            f"No library of {platform_name} is released yet. Build one with "
            f"`python build_plugin.py --output DIRECTORY` and pass it with "
            f"`-C library=PATH` or {LIBRARY_ENVIRONMENT_VARIABLE}."
        )
    path = download_directory(version) / f"{platform_name}-{entry['library']}"
    if not path.exists():
        url = settings["release_url"].format(
            version=version, platform=platform_name, library=entry["library"]
        )
        temporary = path.with_suffix(f".{os.getpid()}.part")
        with urllib.request.urlopen(url) as response, open(temporary, "wb") as file:
            shutil.copyfileobj(response, file)
        temporary.replace(path)
    actual = file_hash(path)
    if actual != expected:
        path.unlink()
        raise RuntimeError(
            f"The library {path} has the SHA-256 hash {actual}, but {expected} is expected."
        )
    return path


def library_source(config_settings: ConfigSettings, version: str) -> Path:
    """Returns the library to package, from the config setting, the environment or a release."""
    match (config_settings or {}).get("library") or os.environ.get(
        LIBRARY_ENVIRONMENT_VARIABLE
    ):
        case None | "":
            return released_library(read_toml("plugin.toml"), version)
        case str() as given:
            return Path(given)
        case values:
            raise ValueError(f"Pass one library, not {values}.")


def metadata_text(project: dict[str, Any]) -> str:
    """Returns the core metadata of the distribution, in the format of `METADATA`."""
    lines = [
        "Metadata-Version: 2.4",
        f"Name: {project['name']}",
        f"Version: {project['version']}",
        f"Summary: {project['description']}",
        f"License-Expression: {project['license']}",
        *(f"License-File: {name}" for name in project["license-files"]),
        f"Requires-Python: {project['requires-python']}",
        *(
            f"Author-email: {author['name']} <{author['email']}>"
            for author in project["authors"]
        ),
        *(f"Classifier: {classifier}" for classifier in project["classifiers"]),
        *(f"Project-URL: {label}, {url}" for label, url in project["urls"].items()),
        "Description-Content-Type: text/markdown",
    ]
    readme = (PROJECT_DIRECTORY / project["readme"]).read_text()
    return "\n".join(lines) + "\n\n" + readme


def entry_points_text(project: dict[str, Any]) -> str:
    """Returns the entry points of the distribution, in the format of `entry_points.txt`."""
    sections = []
    for group, entries in project["entry-points"].items():
        lines = [
            f"[{group}]",
            *(f"{name} = {value}" for name, value in entries.items()),
        ]
        sections.append("\n".join(lines) + "\n")
    return "\n".join(sections)


def distribution_name(project: dict[str, Any]) -> str:
    """Returns the normalized distribution name that file names use."""
    return project["name"].replace("-", "_").lower()


def package_files() -> dict[str, bytes]:
    """Returns the sources of the package, keyed by their path in the wheel."""
    source = PROJECT_DIRECTORY / "src"
    return {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in sorted((source / PACKAGE).rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def record_line(path: str, data: bytes) -> str:
    """Returns the `RECORD` line of a file, with its hash in URL-safe base64."""
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"{path},sha256={digest.decode()},{len(data)}"


def write_wheel(
    directory: Path, project: dict[str, Any], tag: str, files: dict[str, bytes]
) -> str:
    """Writes a wheel with the given files and its metadata, and returns its file name."""
    name = distribution_name(project)
    info = f"{name}-{project['version']}.dist-info"
    wheel_tag_text = f"py3-none-{tag}"
    contents = dict(files)
    contents[f"{info}/METADATA"] = metadata_text(project).encode()
    contents[f"{info}/WHEEL"] = (
        "Wheel-Version: 1.0\n"
        "Generator: cyxla-pjrt-cpu backend\n"
        "Root-Is-Purelib: false\n"
        f"Tag: {wheel_tag_text}\n"
    ).encode()
    contents[f"{info}/entry_points.txt"] = entry_points_text(project).encode()
    for license_file in project["license-files"]:
        contents[f"{info}/licenses/{license_file}"] = (
            PROJECT_DIRECTORY / license_file
        ).read_bytes()
    record_path = f"{info}/RECORD"
    record = [record_line(path, data) for path, data in contents.items()]
    contents[record_path] = ("\n".join([*record, f"{record_path},,"]) + "\n").encode()
    file_name = f"{name}-{project['version']}-{wheel_tag_text}.whl"
    with zipfile.ZipFile(directory / file_name, "w", zipfile.ZIP_DEFLATED) as wheel:
        for path, data in contents.items():
            entry = zipfile.ZipInfo(path, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            # Libraries keep their executable bit, which loading them does not need but
            # tools that inspect wheels expect.
            mode = 0o755 if path.endswith((".so", ".dylib", ".dll")) else 0o644
            entry.external_attr = (0o100000 | mode) << 16
            wheel.writestr(entry, data)
    return file_name


def get_requires_for_build_wheel(config_settings: ConfigSettings = None) -> list[str]:
    """Returns the build requirements of a wheel, which are none."""
    return []


def get_requires_for_build_sdist(config_settings: ConfigSettings = None) -> list[str]:
    """Returns the build requirements of an sdist, which are none."""
    return []


def build_wheel(
    wheel_directory: str,
    config_settings: ConfigSettings = None,
    metadata_directory: str | None = None,
) -> str:
    """Builds the wheel of a plugin library and returns its file name.

    The library goes into the package under the name that `plugin.toml` gives for its
    platform, which is the name that `cyxla_pjrt_cpu.library_path` returns.
    """
    project = read_toml("pyproject.toml")["project"]
    settings = read_toml("plugin.toml")
    library = library_source(config_settings, project["version"])
    platform_name, tag = library_platform(library)
    if platform_name not in settings["platforms"]:
        raise ValueError(
            f"{library} is a library of {platform_name}, which has no plugin."
        )
    files = package_files()
    name = settings["platforms"][platform_name]["library"]
    files[f"{PACKAGE}/{name}"] = library.read_bytes()
    return write_wheel(Path(wheel_directory), project, tag, files)


def build_sdist(sdist_directory: str, config_settings: ConfigSettings = None) -> str:
    """Builds an sdist of the project and returns its file name.

    The sdist holds no library, so building a wheel from it downloads the released one.
    """
    project = read_toml("pyproject.toml")["project"]
    base = f"{distribution_name(project)}-{project['version']}"
    files = {name: (PROJECT_DIRECTORY / name).read_bytes() for name in SDIST_FILES}
    files |= {f"src/{path}": data for path, data in package_files().items()}
    files["PKG-INFO"] = metadata_text(project).encode()
    file_name = f"{base}.tar.gz"
    with open(Path(sdist_directory) / file_name, "wb") as raw:
        # The timestamp is fixed so that the same sources give the same archive.
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(
                fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
            ) as tar:
                for path, data in sorted(files.items()):
                    member = tarfile.TarInfo(f"{base}/{path}")
                    member.size = len(data)
                    member.mode = 0o644
                    tar.addfile(member, io.BytesIO(data))
    return file_name
