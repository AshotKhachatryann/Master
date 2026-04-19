#!/usr/bin/env python3
"""
Bridge to use a C++ Zstd codec binary from Python.

Provides a Python-friendly interface (compress/decompress on raw bytes)
that delegates to the zstd_codec CLI binary.  The binary is
auto-compiled from zstd_codec.cpp on first use if not already present.
Unlike the Huffman and arithmetic bridges, this one also links against
libzstd at build time and accepts a compression level parameter.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


class ZstdCompressor:
    """Use the C++ Zstd codec CLI for compression/decompression.

    On construction the class locates (or builds) the zstd_codec
    binary in *base_dir*.  compress() and decompress() shuttle data
    through temp files so the caller only deals with bytes objects.
    The compression *level* (1-22) is forwarded to the CLI on each
    compress call.
    """

    def __init__(self, level: int = 3, base_dir: Path | None = None):
        # Zstd compression level (1 = fastest, 22 = best ratio).
        self.level = level
        # Default to the directory containing this script.
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.source_path = self.base_dir / "zstd_codec.cpp"
        self.binary_path = self.base_dir / "zstd_codec"
        self._ensure_binary()

    def _ensure_binary(self) -> None:
        """Compile the C++ codec if the binary is missing or not executable.

        Tries several link targets for libzstd because the development
        symlink (libzstd.so) may not exist; the runtime .so.1 usually does.
        Falls back to the standard -lzstd flag as a last resort.
        """
        if self.binary_path.exists() and os.access(self.binary_path, os.X_OK):
            return

        compiler = shutil.which("g++")
        if compiler is None:
            raise RuntimeError(
                "g++ compiler not found. Install g++ to build zstd_codec.cpp"
            )

        if not self.source_path.exists():
            raise RuntimeError(f"Missing C++ source file: {self.source_path}")

        # Ordered list of link targets to try (most specific first).
        link_targets = ["/usr/lib64/libzstd.so.1", "/lib64/libzstd.so.1", "-lzstd"]
        last_err = ""
        for link_target in link_targets:
            # Skip absolute paths that don't exist on this system.
            if link_target.endswith(".so.1") and not Path(link_target).exists():
                continue
            # Build with -O2 optimisation, C++17 standard.
            cmd = [
                compiler,
                "-O2",
                "-std=c++17",
                str(self.source_path),
                "-o",
                str(self.binary_path),
                link_target,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                return
            last_err = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

        raise RuntimeError("Failed to build zstd codec binary.\n" + last_err)

    def _run_codec(self, mode: str, data: bytes, level: int | None = None) -> bytes:
        """Run the codec binary in the given mode ('c' or 'd').

        Writes *data* to a temp file, invokes the binary, reads the
        output file, then cleans up both temp files.  In compress mode
        the compression level is appended as an extra CLI argument.
        If *level* is None the instance default (self.level) is used.
        """
        effective_level = level if level is not None else self.level
        # Write input bytes to a temporary file.
        with tempfile.NamedTemporaryFile(delete=False, dir=self.base_dir) as in_file:
            in_file.write(data)
            input_path = Path(in_file.name)

        # Create a separate temp file for the codec output.
        out_file = tempfile.NamedTemporaryFile(delete=False, dir=self.base_dir)
        output_path = Path(out_file.name)
        out_file.close()

        try:
            cmd = [str(self.binary_path), mode, str(input_path), str(output_path)]
            # Pass compression level only when compressing.
            if mode == "c":
                cmd.append(str(effective_level))
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Zstd codec failed (mode={mode}): {result.stderr.strip()}"
                )
            return output_path.read_bytes()
        finally:
            # Always clean up temp files regardless of success/failure.
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def compress(self, data: bytes, level: int | None = None) -> bytes:
        """Compress raw bytes using Zstd.

        If *level* is given it overrides the instance default for this
        call only (valid range: 1-22).
        """
        return self._run_codec("c", data, level)

    def decompress(self, data: bytes) -> bytes:
        """Decompress a Zstd-compressed byte buffer back to the original data."""
        return self._run_codec("d", data)
