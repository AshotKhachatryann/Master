#!/usr/bin/env python3
"""
Pipeline Comparison Benchmark — Zstd+Arithmetic vs Zstd+Huffman.

Compares two independent two-stage compression pipelines on the
same input data and produces a side-by-side report plus JSON:

  1. Zstd → Arithmetic  — LZ first, then entropy (arithmetic) coder
  2. Zstd → Huffman     — LZ first, then entropy (Huffman) coder

Both pipelines are round-trip verified.  The delta section shows
Which second-stage entropy coder squeezes Zstd output better?

Compression engines are in C++; this script is a Python benchmark
orchestration using the project-local bridge modules.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Project-local bridges — each auto-compiles its C++ binary on first use.
from cpp_arithmetic_bridge import ArithmeticCompressor
from cpp_huffman_bridge import HuffmanCompressor
from cpp_zstd_bridge import ZstdCompressor


def format_size(num_bytes: int, signed: bool = False) -> str:
    """Render *num_bytes* using the largest unit that yields a value >= 1.

    Picks GB → MB → KB → B (binary, 1024-based).  When *signed* is True
    a leading '+' is shown for positive values (negatives always carry '-').
    """
    n = float(num_bytes)
    fmt = "+.2f" if signed else ".2f"
    for unit, threshold in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if abs(n) >= threshold:
            return f"{n / threshold:{fmt}} {unit}"
    int_fmt = "+d" if signed else "d"
    return f"{int(num_bytes):{int_fmt}} B"


class PipelineComparisonBenchmark:
    """Compare Zstd+Arithmetic vs Zstd+Huffman two-stage pipelines.

    Instantiate once, then call run() for each test payload.
    Pass the returned dict to print_table() for formatted output.
    """

    def __init__(self, zstd_level: int = 3):
        # Reusable compressor instances (binary built once, reused).
        self.zstd = ZstdCompressor(level=zstd_level)
        self.arith = ArithmeticCompressor()
        self.huff = HuffmanCompressor()

    # ── Zstd → Arithmetic pipeline ────────────────────────────────────────

    def compress_zstd_arith(self, data: bytes) -> tuple[bytes, float]:
        """Two-stage: Zstd then Arithmetic.  Returns (compressed, elapsed_s)."""
        start = time.time()
        stage1 = self.zstd.compress(data)
        final = self.arith.compress(stage1)
        return final, (time.time() - start)

    def decompress_zstd_arith(self, data: bytes) -> tuple[bytes, float]:
        """Reverse: undo Arithmetic then Zstd.  Returns (decompressed, elapsed_s)."""
        start = time.time()
        stage1 = self.arith.decompress(data)
        final = self.zstd.decompress(stage1)
        return final, (time.time() - start)

    # ── Zstd → Huffman pipeline ───────────────────────────────────────────

    def compress_zstd_huff(self, data: bytes) -> tuple[bytes, float]:
        """Two-stage: Zstd then Huffman.  Returns (compressed, elapsed_s)."""
        start = time.time()
        stage1 = self.zstd.compress(data)
        final = self.huff.compress(stage1)
        return final, (time.time() - start)

    def decompress_zstd_huff(self, data: bytes) -> tuple[bytes, float]:
        """Reverse: undo Huffman then Zstd.  Returns (decompressed, elapsed_s)."""
        start = time.time()
        stage1 = self.huff.decompress(data)
        final = self.zstd.decompress(stage1)
        return final, (time.time() - start)

    # ── Core benchmark driver ─────────────────────────────────────────────

    def run(self, data: bytes, label: str) -> dict:
        """Run both pipelines on *data*, verify round-trips, return stats dict."""
        original_size = len(data)

        # ── Compress with both pipelines ──────────────────────────────────
        arith_comp, arith_ct = self.compress_zstd_arith(data)
        huff_comp, huff_ct = self.compress_zstd_huff(data)

        # ── Round-trip verification ───────────────────────────────────────
        arith_decomp, arith_dt = self.decompress_zstd_arith(arith_comp)
        huff_decomp, huff_dt = self.decompress_zstd_huff(huff_comp)

        if arith_decomp != data:
            raise RuntimeError(
                f"Round-trip FAILED for Zstd+Arithmetic: "
                f"original {len(data)} bytes, recovered {len(arith_decomp)} bytes"
            )
        if huff_decomp != data:
            raise RuntimeError(
                f"Round-trip FAILED for Zstd+Huffman: "
                f"original {len(data)} bytes, recovered {len(huff_decomp)} bytes"
            )

        arith_size = len(arith_comp)
        huff_size = len(huff_comp)

        arith_ratio = original_size / arith_size if arith_size else 0.0
        huff_ratio = original_size / huff_size if huff_size else 0.0

        best = "zstd_arithmetic" if arith_ratio >= huff_ratio else "zstd_huffman"

        return {
            "label": label,
            "original_size_bytes": original_size,
            "methods": {
                "zstd_arithmetic": {
                    "compressed_size_bytes": arith_size,
                    "compression_ratio": round(arith_ratio, 4),
                    "compression_time_s": round(arith_ct, 4),
                    "decompression_time_s": round(arith_dt, 4),
                },
                "zstd_huffman": {
                    "compressed_size_bytes": huff_size,
                    "compression_ratio": round(huff_ratio, 4),
                    "compression_time_s": round(huff_ct, 4),
                    "decompression_time_s": round(huff_dt, 4),
                },
            },
            "delta": {
                "huffman_minus_arithmetic_bytes": huff_size - arith_size,
                "huffman_minus_arithmetic_percent": round(((huff_size - arith_size) / arith_size * 100) if arith_size else 0.0, 2),
                "ratio_huffman_minus_arithmetic": round(huff_ratio - arith_ratio, 4),
            },
            "best_method": best,
        }


def print_table(result: dict) -> None:
    print("=" * 116)
    print("CUSTOM PIPELINE COMPARISON: ZSTD+ARITHMETIC vs ZSTD+HUFFMAN")
    print("=" * 116)
    print(f"Test: {result['label']}")
    print(f"Original Size: {format_size(result['original_size_bytes'])} ({result['original_size_bytes']:,} bytes)")
    print()

    # Column inner widths (must match header labels and row data formats).
    # Each cell is rendered as: " " + value(width=W) + " ", so the border
    # segment is W+2 box-drawing characters wide.
    W_NAME, W_SIZE, W_RATIO, W_CT, W_DT = 24, 32, 18, 20, 22
    h = "─"
    top    = f"┌{h*(W_NAME+2)}┬{h*(W_SIZE+2)}┬{h*(W_RATIO+2)}┬{h*(W_CT+2)}┬{h*(W_DT+2)}┐"
    mid    = f"├{h*(W_NAME+2)}┼{h*(W_SIZE+2)}┼{h*(W_RATIO+2)}┼{h*(W_CT+2)}┼{h*(W_DT+2)}┤"
    bottom = f"└{h*(W_NAME+2)}┴{h*(W_SIZE+2)}┴{h*(W_RATIO+2)}┴{h*(W_CT+2)}┴{h*(W_DT+2)}┘"

    print(top)
    print(f"│ {'Pipeline':<{W_NAME}} │ {'Compressed Size':<{W_SIZE}} │ {'Compression Ratio':<{W_RATIO}} │ {'Compress Time (s)':<{W_CT}} │ {'Decompress Time (s)':<{W_DT}} │")
    print(mid)

    m1 = result["methods"]["zstd_arithmetic"]
    m2 = result["methods"]["zstd_huffman"]

    for label, m in (("Zstd+Arithmetic", m1), ("Zstd+Huffman", m2)):
        # Format size with the largest unit that fits (GB/MB/KB/B),
        # plus the exact byte count for unambiguous comparison.
        size_str = f"{format_size(m['compressed_size_bytes'])} ({m['compressed_size_bytes']:,} B)"
        print(
            f"│ {label:<{W_NAME}} "
            f"│ {size_str:>{W_SIZE}} "
            f"│ {m['compression_ratio']:>{W_RATIO}.4f} "
            f"│ {m['compression_time_s']:>{W_CT}.4f} "
            f"│ {m['decompression_time_s']:>{W_DT}.4f} │"
        )
    print(bottom)

    d = result["delta"]
    print()
    print("Delta (Zstd+Huffman - Zstd+Arithmetic):")
    print(f"  Size:  {format_size(d['huffman_minus_arithmetic_bytes'], signed=True)} ({d['huffman_minus_arithmetic_percent']:+.2f}%)")
    print(f"  Ratio: {d['ratio_huffman_minus_arithmetic']:+.4f}x")
    print(f"Best Method: {result['best_method']}")
    print("=" * 116)


def main() -> None:
    """Run pipeline comparison with WAV data, falling back to synthetic data."""
    parser = argparse.ArgumentParser(
        description="Pipeline comparison: Zstd+Arithmetic vs Zstd+Huffman."
    )
    parser.add_argument(
        "wav", nargs="?", default=None,
        help="Path to a .wav file to benchmark.  Defaults to test_audio.wav in the script directory."
    )
    parser.add_argument(
        "-l", "--level", type=int, default=3, metavar="N",
        help="Zstd compression level (default: 3).  Valid range: 1-22."
    )
    args = parser.parse_args()

    base = Path(__file__).parent

    # Use user-supplied path, or fall back to the bundled test file.
    if args.wav is not None:
        wav = Path(args.wav)
    else:
        wav = base / "test_audio.wav"

    bench = PipelineComparisonBenchmark(zstd_level=args.level)

    if wav.exists():
        print(f"Reading WAV file: {wav}")
        data = wav.read_bytes()
        print(f"WAV file size: {format_size(len(data))} ({len(data):,} bytes)")
        print("Running pipeline comparison benchmark...")
        result = bench.run(data, f"WAV Audio File ({format_size(len(data))})")
    else:
        print(f"WAV file not found: {wav}")
        print("Using synthetic test data instead...")
        # Fallback: incompressible random data.
        data = os.urandom(10000)
        result = bench.run(data, "Random binary (10KB)")

    print_table(result)

    # Persist raw results as JSON for later analysis or plotting.
    out_json = base / "compression_benchmark_zstd_arith_vs_huff_results.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"\nDetailed results saved to: {out_json}")


if __name__ == "__main__":
    main()
