#!/usr/bin/env python3
"""
Bridge to use a C++ arithmetic codec binary from Python.

Provides a Python-friendly interface (compress/decompress on raw bytes)
that delegates to the arithmetic_codec CLI binary.  The binary is
auto-compiled from arithmetic_codec.cpp on first use if not already
present.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


class ArithmeticCompressor:
    """Use the C++ arithmetic codec CLI for compression/decompression.

    On construction the class locates (or builds) the arithmetic_codec
    binary in *base_dir*.  compress() and decompress() shuttle data
    through temp files so the caller only deals with bytes objects.
    """

    def __init__(self, base_dir: Path | None = None):
        # Default to the directory containing this script.
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.source_path = self.base_dir / "arithmetic_codec.cpp"
        self.binary_path = self.base_dir / "arithmetic_codec"
        self._ensure_binary()

    def _ensure_binary(self) -> None:
        """Compile the C++ codec if the binary is missing or not executable."""
        if self.binary_path.exists() and os.access(self.binary_path, os.X_OK):
            return

        compiler = shutil.which("g++")
        if compiler is None:
            raise RuntimeError(
                "g++ compiler not found. Install g++ to build arithmetic_codec.cpp"
            )

        if not self.source_path.exists():
            raise RuntimeError(f"Missing C++ source file: {self.source_path}")

        # Build with -O2 optimisation, C++17 standard.
        cmd = [
            compiler,
            "-O2",
            "-std=c++17",
            str(self.source_path),
            "-o",
            str(self.binary_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                "Failed to build arithmetic codec:\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

    def _run_codec(self, mode: str, data: bytes) -> bytes:
        """Run the codec binary in the given mode ('c' or 'd').

        Writes *data* to a temp file, invokes the binary, reads the
        output file, then cleans up both temp files.
        """
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
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Arithmetic codec failed (mode={mode}): {result.stderr.strip()}"
                )
            return output_path.read_bytes()
        finally:
            # Always clean up temp files regardless of success/failure.
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def compress(self, data: bytes) -> bytes:
        """Compress raw bytes using arithmetic coding."""
        return self._run_codec("c", data)

    def decompress(self, data: bytes) -> bytes:
        """Decompress an arithmetic-coded byte buffer back to the original data."""
        return self._run_codec("d", data)

        """Decompress an arithmetic-coded byte buffer back to the original data."""
        return self._run_codec("d", data)
