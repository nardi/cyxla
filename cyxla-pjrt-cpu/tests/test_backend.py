"""Tests of the build backend with synthetic libraries of each system."""

import hashlib
import io
import struct
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import backend  # noqa: E402


def elf_library(machine: int, versions: list[bytes]) -> bytes:
    """Returns a 64-bit ELF file that needs the given symbol versions from one library.

    It has a null section, a string table and an `SHT_GNU_verneed` section, which is all
    that the backend reads.
    """
    strings = b"\0libc.so.6\0" + b"".join(version + b"\0" for version in versions)
    names = [strings.index(version + b"\0") for version in versions]
    auxiliaries = b"".join(
        struct.pack("<IHHII", 0, 0, 0, name, 16 if index < len(names) - 1 else 0)
        for index, name in enumerate(names)
    )
    needed = struct.pack("<HHIII", 1, len(names), 1, 16, 0) + auxiliaries
    strings_offset = 64
    needed_offset = strings_offset + len(strings)
    sections_offset = needed_offset + len(needed)
    header = bytearray(64)
    header[:4] = b"\x7fELF"
    struct.pack_into("<H", header, 18, machine)
    struct.pack_into("<Q", header, 0x28, sections_offset)
    struct.pack_into("<HH", header, 0x3A, 64, 3)
    sections = (
        bytes(64)
        + struct.pack(
            "<IIQQQQIIQQ", 0, 3, 0, 0, strings_offset, len(strings), 0, 0, 1, 0
        )
        + struct.pack(
            "<IIQQQQIIQQ", 0, 0x6FFFFFFE, 0, 0, needed_offset, len(needed), 1, 1, 8, 0
        )
    )
    return bytes(header) + strings + needed + sections


def macho_library(cpu: int, command: int, version: int) -> bytes:
    """Returns a 64-bit Mach-O header with one version load command."""
    header = struct.pack("<IiiIIIII", 0xFEEDFACF, cpu, 0, 6, 2, 0, 0, 0)
    other = struct.pack("<II", 0x19, 16) + bytes(8)
    match command:
        case 0x32:
            versioned = struct.pack("<IIIIII", 0x32, 24, 1, version, version, 0)
        case _:
            versioned = struct.pack("<IIII", 0x24, 16, version, version)
    return header + other + versioned


def pe_library(machine: int) -> bytes:
    """Returns the start of a PE file for a machine."""
    data = bytearray(256)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 128)
    data[128:132] = b"PE\0\0"
    struct.pack_into("<H", data, 132, machine)
    return bytes(data)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (
            elf_library(62, [b"GLIBC_2.17", b"GLIBC_2.27", b"GLIBC_2.3.4"]),
            ("linux-x86_64", "manylinux_2_27_x86_64"),
        ),
        (
            elf_library(183, [b"CXXABI_1.3"]),
            ("linux-aarch64", "manylinux_2_17_aarch64"),
        ),
        (
            macho_library(0x0100000C, 0x32, (12 << 16) | (3 << 8)),
            ("macos-arm64", "macosx_12_0_arm64"),
        ),
        (
            macho_library(0x01000007, 0x24, (10 << 16) | (15 << 8)),
            ("macos-x86_64", "macosx_10_15_x86_64"),
        ),
        (pe_library(0x8664), ("windows-amd64", "win_amd64")),
    ],
)
def test_library_platform(
    data: bytes, expected: tuple[str, str], tmp_path: Path
) -> None:
    """Checks the platform and wheel tag of libraries of each system."""
    path = tmp_path / "library"
    path.write_bytes(data)
    assert backend.library_platform(path) == expected
    assert backend.wheel_tag(path) == expected[1]


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"not a library", "not an ELF, Mach-O or PE"),
        (pe_library(0x8664).replace(b"PE\0\0", b"XX\0\0"), "not a PE"),
        (macho_library(0x0100000C, 0x32, 0)[:48] + bytes(16), "no minimum macOS"),
    ],
)
def test_bad_libraries(data: bytes, message: str, tmp_path: Path) -> None:
    """Checks the errors of files that are not plugin libraries."""
    path = tmp_path / "library"
    path.write_bytes(data)
    with pytest.raises(ValueError, match=message):
        backend.library_platform(path)


def test_wheel_of_a_windows_library(tmp_path: Path) -> None:
    """Checks that a Windows library becomes `pjrt_cpu.dll` in a `win_amd64` wheel."""
    library = tmp_path / "built.dll"
    library.write_bytes(pe_library(0x8664))
    name = backend.build_wheel(str(tmp_path), {"library": str(library)})
    assert name == "cyxla_pjrt_cpu-0.12.0.1-py3-none-win_amd64.whl"
    with zipfile.ZipFile(tmp_path / name) as wheel:
        assert wheel.read("cyxla_pjrt_cpu/pjrt_cpu.dll") == library.read_bytes()
        info = "cyxla_pjrt_cpu-0.12.0.1.dist-info"
        assert "Tag: py3-none-win_amd64" in wheel.read(f"{info}/WHEEL").decode()
        assert (
            wheel.read(f"{info}/entry_points.txt")
            == b"[cyxla.plugins]\ncpu = cyxla_pjrt_cpu\n"
        )
        record = wheel.read(f"{info}/RECORD").decode()
        assert "cyxla_pjrt_cpu/pjrt_cpu.dll,sha256=" in record


def test_wheel_of_an_unknown_platform(tmp_path: Path) -> None:
    """Checks that a library of a platform without a plugin is rejected."""
    library = tmp_path / "built.dylib"
    library.write_bytes(macho_library(0x01000007, 0x24, 11 << 16))
    with pytest.raises(ValueError, match="macos-x86_64, which has no plugin"):
        backend.build_wheel(str(tmp_path), {"library": str(library)})


def test_library_sources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Checks the config setting, the environment variable and the released library."""
    monkeypatch.delenv(backend.LIBRARY_ENVIRONMENT_VARIABLE, raising=False)
    assert backend.library_source({"library": "a.so"}, "1") == Path("a.so")
    monkeypatch.setenv(backend.LIBRARY_ENVIRONMENT_VARIABLE, "b.so")
    assert backend.library_source(None, "1") == Path("b.so")
    with pytest.raises(ValueError, match="Pass one library"):
        backend.library_source({"library": ["a", "b"]}, "1")
    monkeypatch.delenv(backend.LIBRARY_ENVIRONMENT_VARIABLE)
    monkeypatch.setattr(
        backend, "released_library", lambda settings, version: Path(version)
    )
    assert backend.library_source({}, "2") == Path("2")


def test_released_library(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Checks the download, the hash check and the missing release."""
    content = b"library"
    monkeypatch.setattr(backend, "host_platform", lambda: "linux-x86_64")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    settings = {
        "release_url": "https://example.com/{version}/{platform}-{library}",
        "platforms": {"linux-x86_64": {"library": "libpjrt_cpu.so", "sha256": ""}},
    }
    with pytest.raises(RuntimeError, match="No library of linux-x86_64"):
        backend.released_library(settings, "1.0")
    urls = []

    def urlopen(url: str) -> io.BytesIO:
        urls.append(url)
        return io.BytesIO(content)

    monkeypatch.setattr(backend.urllib.request, "urlopen", urlopen)
    entry = settings["platforms"]["linux-x86_64"]
    entry["sha256"] = hashlib.sha256(content).hexdigest()
    path = backend.released_library(settings, "1.0")
    assert path.read_bytes() == content
    assert urls == ["https://example.com/1.0/linux-x86_64-libpjrt_cpu.so"]
    assert backend.released_library(settings, "1.0") == path and len(urls) == 1
    entry["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="is expected"):
        backend.released_library(settings, "1.0")
    assert not path.exists()


def test_host_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks the platform names of each system and the error of an unknown one."""
    for system, machine, expected in (
        ("linux", "x86_64", "linux-x86_64"),
        ("linux", "aarch64", "linux-aarch64"),
        ("darwin", "arm64", "macos-arm64"),
        ("win32", "AMD64", "windows-amd64"),
    ):
        monkeypatch.setattr(backend.sys, "platform", system)
        monkeypatch.setattr(backend.platform, "machine", lambda: machine)
        assert backend.host_platform() == expected
    monkeypatch.setattr(backend.sys, "platform", "sunos5")
    with pytest.raises(RuntimeError, match="no PJRT CPU plugin for sunos5"):
        backend.host_platform()


def test_sdist(tmp_path: Path) -> None:
    """Checks that the sdist holds the build files and no library."""
    name = backend.build_sdist(str(tmp_path))
    with tarfile.open(tmp_path / name) as tar:
        members = {member.name.split("/", 1)[1] for member in tar.getmembers()}
    assert {
        "plugin.toml",
        "build_plugin.py",
        "bazel/BUILD.bazel",
        "PKG-INFO",
    } <= members
    assert not any(member.endswith((".so", ".dylib", ".dll")) for member in members)
