#!/usr/bin/env python3
"""
Compression Ratio Comparison Benchmark — Zstd + Arithmetic Focus (WAV edition).

Compares four compression strategies on the same input data and
produces a human-readable report plus a JSON results file:

  1. Zstd-only          — single-pass general-purpose compression
  2. Arithmetic-only    — single-pass statistical/entropy coder
  3. Zstd → Arithmetic  — two-stage pipeline (LZ first, then entropy)
  4. Arithmetic → Zstd  — two-stage pipeline (entropy first, then LZ)

This version reads an actual WAV file (test_audio.wav) for testing,
falling back to synthetic data if the file is missing.
"""

import argparse
import os
import time
import json
from pathlib import Path

# Project-local bridges — each auto-compiles its C++ binary on first use.
from cpp_arithmetic_bridge import ArithmeticCompressor
from cpp_zstd_bridge import ZstdCompressor


class CompressionBenchmarkZstdArith:
    """Benchmark comparing Zstd-first pipeline approach.

    Instantiate once, then call benchmark_on_data() for each test
    payload.  Collect the returned dicts and pass them to
    generate_report() or generate_table_report() for a formatted summary.
    """

    def __init__(self, zstd_level: int = 3):
        # Reusable compressor instances (binary built once, reused).
        self.arithmetic_coder = ArithmeticCompressor()
        self.zstd_coder = ZstdCompressor(level=zstd_level)

    # ── Single-pass compression ───────────────────────────────────────────

    def compress_zstd_only(self, data: bytes) -> tuple:
        """Single-pass Zstd compression.  Returns (compressed_bytes, elapsed_s)."""
        start = time.time()
        compressed = self.zstd_coder.compress(data)
        elapsed = time.time() - start
        return compressed, elapsed

    def compress_arithmetic_only(self, data: bytes) -> tuple:
        """Single-pass arithmetic compression.  Returns (compressed_bytes, elapsed_s)."""
        start = time.time()
        compressed = self.arithmetic_coder.compress(data)
        elapsed = time.time() - start
        return compressed, elapsed

    # ── Two-stage pipeline ────────────────────────────────────────────────

    def compress_reverse_pipeline_zstd_arith(self, data: bytes) -> tuple:
        """Two-stage reverse pipeline: Zstd → Arithmetic.

        Stage 1 removes general-purpose (LZ77) redundancies via Zstd.
        Stage 2 applies arithmetic coding to exploit any remaining
        statistical patterns in the Zstd output.

        Returns (compressed_bytes, elapsed_s) for both stages combined.
        """
        start = time.time()

        # Stage 1: Zstd compression — general-purpose LZ pass.
        zstd_compressed = self.zstd_coder.compress(data)

        # Stage 2: Arithmetic coding on Zstd output — entropy pass.
        reverse_compressed = self.arithmetic_coder.compress(zstd_compressed)

        elapsed = time.time() - start
        return reverse_compressed, elapsed

    def compress_forward_pipeline_arith_zstd(self, data: bytes) -> tuple:
        """Two-stage forward pipeline: Arithmetic → Zstd.

        Stage 1 applies arithmetic coding to exploit statistical patterns.
        Stage 2 applies Zstd to remove any remaining LZ77 redundancies.

        Returns (compressed_bytes, elapsed_s) for both stages combined.
        """
        start = time.time()

        # Stage 1: Arithmetic coding — entropy pass.
        arith_compressed = self.arithmetic_coder.compress(data)

        # Stage 2: Zstd compression on arithmetic output — LZ pass.
        forward_compressed = self.zstd_coder.compress(arith_compressed)

        elapsed = time.time() - start
        return forward_compressed, elapsed

    # ── Decompression (mirrors the above, reverse stage order) ────────────

    def decompress_zstd_only(self, data: bytes) -> tuple:
        """Decompress Zstd-only data.  Returns (decompressed_bytes, elapsed_s)."""
        start = time.time()
        decompressed = self.zstd_coder.decompress(data)
        elapsed = time.time() - start
        return decompressed, elapsed

    def decompress_arithmetic_only(self, data: bytes) -> tuple:
        """Decompress arithmetic-only data.  Returns (decompressed_bytes, elapsed_s)."""
        start = time.time()
        decompressed = self.arithmetic_coder.decompress(data)
        elapsed = time.time() - start
        return decompressed, elapsed

    def decompress_reverse_pipeline_zstd_arith(self, data: bytes) -> tuple:
        """Decompress reverse pipeline data (undo Arithmetic then Zstd).

        Stages are reversed compared to compression:
          1. Arithmetic decode → recovers Zstd-compressed stream
          2. Zstd decompress   → recovers original data
        """
        start = time.time()

        # Stage 1: Strip the arithmetic layer.
        zstd_compressed = self.arithmetic_coder.decompress(data)

        # Stage 2: Strip the Zstd layer.
        decompressed = self.zstd_coder.decompress(zstd_compressed)

        elapsed = time.time() - start
        return decompressed, elapsed

    def decompress_forward_pipeline_arith_zstd(self, data: bytes) -> tuple:
        """Decompress forward pipeline data (undo Zstd then Arithmetic).

        Stages are reversed compared to compression:
          1. Zstd decompress   → recovers arithmetic-compressed stream
          2. Arithmetic decode  → recovers original data
        """
        start = time.time()

        # Stage 1: Strip the Zstd layer.
        arith_compressed = self.zstd_coder.decompress(data)

        # Stage 2: Strip the arithmetic layer.
        decompressed = self.arithmetic_coder.decompress(arith_compressed)

        elapsed = time.time() - start
        return decompressed, elapsed

    # ── Core benchmark driver ─────────────────────────────────────────────

    def _verify_round_trip(self, original: bytes, compressed: bytes,
                            decompress_fn, method_name: str) -> None:
        """Decompress *compressed* and assert it matches *original*.

        Raises RuntimeError if the round-trip fails, so corrupt codecs
        are caught immediately rather than producing silent bad data.
        """
        recovered, _ = decompress_fn(compressed)
        if recovered != original:
            raise RuntimeError(
                f"Round-trip verification FAILED for {method_name}: "
                f"original {len(original)} bytes, recovered {len(recovered)} bytes"
            )

    def benchmark_on_data(self, data: bytes, label: str = "Unknown") -> dict:
        """Run all four compression methods on *data* and return a stats dict.

        The returned dict contains per-method sizes/ratios/times, the
        best method, delta analysis, and pipeline viability assessments.
        Each method is round-trip verified after compression.
        """
        original_size = len(data)
        stats = {
            "label": label,
            "original_size_bytes": original_size,
            "methods": {}
        }

        # ── Method 1: Zstd only ──────────────────────────────────────────
        zstd_comp, zstd_time = self.compress_zstd_only(data)
        self._verify_round_trip(data, zstd_comp,
                                self.decompress_zstd_only, "zstd_only")
        zstd_comp_size = len(zstd_comp)
        # Ratio >1 means data got smaller; 0 guards against empty output.
        zstd_ratio = original_size / zstd_comp_size if zstd_comp_size > 0 else 0
        stats["methods"]["zstd_only"] = {
            "compressed_size_bytes": zstd_comp_size,
            "compression_ratio": round(zstd_ratio, 4),
            "compression_time_ms": round(zstd_time * 1000, 2)
        }

        # ── Method 2: Arithmetic only ────────────────────────────────────
        arith_comp, arith_time = self.compress_arithmetic_only(data)
        self._verify_round_trip(data, arith_comp,
                                self.decompress_arithmetic_only, "arithmetic_only")
        arith_comp_size = len(arith_comp)
        arith_ratio = original_size / arith_comp_size if arith_comp_size > 0 else 0
        stats["methods"]["arithmetic_only"] = {
            "compressed_size_bytes": arith_comp_size,
            "compression_ratio": round(arith_ratio, 4),
            "compression_time_ms": round(arith_time * 1000, 2)
        }

        # ── Method 3: Reverse pipeline (Zstd → Arithmetic) ──────────────
        rev_comp, rev_time = self.compress_reverse_pipeline_zstd_arith(data)
        self._verify_round_trip(data, rev_comp,
                                self.decompress_reverse_pipeline_zstd_arith,
                                "reverse_pipeline_zstd_arith")
        rev_comp_size = len(rev_comp)
        rev_ratio = original_size / rev_comp_size if rev_comp_size > 0 else 0
        stats["methods"]["reverse_pipeline_zstd_arith"] = {
            "compressed_size_bytes": rev_comp_size,
            "compression_ratio": round(rev_ratio, 4),
            "compression_time_ms": round(rev_time * 1000, 2)
        }

        # ── Method 4: Forward pipeline (Arithmetic → Zstd) ──────────────
        fwd_comp, fwd_time = self.compress_forward_pipeline_arith_zstd(data)
        self._verify_round_trip(data, fwd_comp,
                                self.decompress_forward_pipeline_arith_zstd,
                                "forward_pipeline_arith_zstd")
        fwd_comp_size = len(fwd_comp)
        fwd_ratio = original_size / fwd_comp_size if fwd_comp_size > 0 else 0
        stats["methods"]["forward_pipeline_arith_zstd"] = {
            "compressed_size_bytes": fwd_comp_size,
            "compression_ratio": round(fwd_ratio, 4),
            "compression_time_ms": round(fwd_time * 1000, 2)
        }

        # ── Winner selection ─────────────────────────────────────────────
        ratios = {
            "zstd_only": zstd_ratio,
            "arithmetic_only": arith_ratio,
            "reverse_pipeline_zstd_arith": rev_ratio,
            "forward_pipeline_arith_zstd": fwd_ratio
        }
        best_method = max(ratios, key=ratios.get)
        stats["best_method"] = best_method
        stats["best_ratio"] = round(ratios[best_method], 4)

        # ── Delta analysis ───────────────────────────────────────────────
        # Negative values mean the reverse pipeline produced a *smaller*
        # (better) output than the single-pass method.
        delta_reverse_vs_zstd = rev_comp_size - zstd_comp_size
        delta_reverse_vs_arith = rev_comp_size - arith_comp_size
        # NOTE: this is a signed percentage — negative = pipeline is better.
        reverse_overhead_percent = round(
            (delta_reverse_vs_zstd / zstd_comp_size * 100) if zstd_comp_size > 0 else 0, 2
        )

        stats["delta_analysis"] = {
            "reverse_vs_zstd_bytes": delta_reverse_vs_zstd,
            "reverse_vs_zstd_percent": reverse_overhead_percent,
            "reverse_vs_arithmetic_bytes": delta_reverse_vs_arith,
            "reverse_vs_arithmetic_percent": round(
                (delta_reverse_vs_arith / arith_comp_size * 100) if arith_comp_size > 0 else 0, 2
            ),
            "zstd_vs_arithmetic_bytes": zstd_comp_size - arith_comp_size,
            "zstd_vs_arithmetic_percent": round(
                ((zstd_comp_size - arith_comp_size) / arith_comp_size * 100) if arith_comp_size > 0 else 0, 2
            ),
            "forward_vs_zstd_bytes": fwd_comp_size - zstd_comp_size,
            "forward_vs_zstd_percent": round(
                ((fwd_comp_size - zstd_comp_size) / zstd_comp_size * 100) if zstd_comp_size > 0 else 0, 2
            ),
            "forward_vs_arithmetic_bytes": fwd_comp_size - arith_comp_size,
            "forward_vs_arithmetic_percent": round(
                ((fwd_comp_size - arith_comp_size) / arith_comp_size * 100) if arith_comp_size > 0 else 0, 2
            ),
            "forward_vs_reverse_bytes": fwd_comp_size - rev_comp_size,
            "forward_vs_reverse_percent": round(
                ((fwd_comp_size - rev_comp_size) / rev_comp_size * 100) if rev_comp_size > 0 else 0, 2
            )
        }

        # ── Reverse pipeline viability ───────────────────────────────────
        # Use actual percentage size reduction for the >1 % threshold:
        #   pct = (baseline_size - pipeline_size) / baseline_size * 100
        # Positive pct means the pipeline is smaller (better).
        pct_smaller_vs_zstd = (
            (zstd_comp_size - rev_comp_size) / zstd_comp_size * 100
            if zstd_comp_size > 0 else 0
        )
        pct_smaller_vs_arith = (
            (arith_comp_size - rev_comp_size) / arith_comp_size * 100
            if arith_comp_size > 0 else 0
        )
        stats["reverse_pipeline_analysis"] = {
            "pct_smaller_vs_zstd": round(pct_smaller_vs_zstd, 2),
            "pct_smaller_vs_arithmetic": round(pct_smaller_vs_arith, 2),
            "worthwhile": pct_smaller_vs_zstd > 1.0 or pct_smaller_vs_arith > 1.0
        }

        # ── Forward pipeline viability ───────────────────────────────────
        fwd_pct_vs_zstd = (
            (zstd_comp_size - fwd_comp_size) / zstd_comp_size * 100
            if zstd_comp_size > 0 else 0
        )
        fwd_pct_vs_arith = (
            (arith_comp_size - fwd_comp_size) / arith_comp_size * 100
            if arith_comp_size > 0 else 0
        )
        stats["forward_pipeline_analysis"] = {
            "pct_smaller_vs_zstd": round(fwd_pct_vs_zstd, 2),
            "pct_smaller_vs_arithmetic": round(fwd_pct_vs_arith, 2),
            "worthwhile": fwd_pct_vs_zstd > 1.0 or fwd_pct_vs_arith > 1.0
        }

        return stats

    # ── Report generators ─────────────────────────────────────────────────

    def generate_table_report(self, results_list: list) -> str:
        """Generate a clean table-based report for easy comparison."""
        report = []
        report.append("")
        report.append("=" * 120)
        report.append("COMPRESSION COMPARISON TABLE - ALL METHODS")
        report.append("=" * 120)
        report.append("")

        for result in results_list:
            report.append(f"Test: {result['label']}")
            report.append(f"Original Size: {result['original_size_bytes']:,} bytes")
            report.append("")

            # Create comparison table
            report.append("┌─────────────────────────────┬─────────────────┬──────────────────┬────────────────────┬─────────────────────┐")
            report.append("│ Compression Method          │ Compressed Size │ Compression Ratio│ Time (ms)          │ Size vs Zstd-only   │")
            report.append("├─────────────────────────────┼─────────────────┼──────────────────┼────────────────────┼─────────────────────┤")

            methods = ["zstd_only", "arithmetic_only", "reverse_pipeline_zstd_arith", "forward_pipeline_arith_zstd"]
            method_labels = ["Zstd Only", "Arithmetic Only", "Zstd->Arithmetic Pipeline", "Arithmetic->Zstd Pipeline"]

            zstd_size = result['methods']['zstd_only']['compressed_size_bytes']

            for method, label in zip(methods, method_labels):
                data = result['methods'][method]
                size = data['compressed_size_bytes']
                ratio = data['compression_ratio']
                time_ms = data['compression_time_ms']

                # Calculate delta vs zstd
                if method == "zstd_only":
                    delta_str = "-"
                else:
                    delta = size - zstd_size
                    delta_pct = (delta / zstd_size * 100) if zstd_size > 0 else 0
                    delta_str = f"{delta:+,} bytes ({delta_pct:+.2f}%)"

                report.append(f"│ {label:<27} │ {size:>14,} │ {ratio:>16.4f} │ {time_ms:>18.2f} │ {delta_str:>19} │")

            report.append("└─────────────────────────────┴─────────────────┴──────────────────┴────────────────────┴─────────────────────┘")
            report.append("")

            # Summary
            report.append("SUMMARY:")
            report.append(f"  Best Method: {result['best_method'].upper().replace('_', ' ')} (ratio: {result['best_ratio']:.4f}x)")

            rev_analysis = result['reverse_pipeline_analysis']
            rev_worth = "✓ WORTHWHILE (>1% smaller)" if rev_analysis['worthwhile'] else "✗ NOT WORTHWHILE (<1% smaller)"
            report.append(f"  Zstd->Arithmetic Pipeline: {rev_worth}")
            report.append(f"    - Size reduction vs Zstd:       {rev_analysis['pct_smaller_vs_zstd']:+.2f}%")
            report.append(f"    - Size reduction vs Arithmetic: {rev_analysis['pct_smaller_vs_arithmetic']:+.2f}%")

            fwd_analysis = result['forward_pipeline_analysis']
            fwd_worth = "✓ WORTHWHILE (>1% smaller)" if fwd_analysis['worthwhile'] else "✗ NOT WORTHWHILE (<1% smaller)"
            report.append(f"  Arithmetic->Zstd Pipeline: {fwd_worth}")
            report.append(f"    - Size reduction vs Zstd:       {fwd_analysis['pct_smaller_vs_zstd']:+.2f}%")
            report.append(f"    - Size reduction vs Arithmetic: {fwd_analysis['pct_smaller_vs_arithmetic']:+.2f}%")
            report.append("")

        report.append("=" * 120)
        return "\n".join(report)

    def generate_report(self, results_list: list) -> str:
        """Generate a human-readable benchmark report from a list of stats dicts."""
        report = []
        report.append("=" * 90)
        report.append("COMPRESSION RATIO COMPARISON REPORT - ZSTD + ARITHMETIC FOCUS")
        report.append("Focus: Reverse pipeline (Zstd -> Arithmetic) vs single-pass methods")
        report.append("=" * 90)
        report.append("")

        for result in results_list:
            report.append(f"Test: {result['label']}")
            report.append(f"Original size: {result['original_size_bytes']:,} bytes")
            report.append("")

            # ── Single-pass results ──────────────────────────────────────
            report.append("  SINGLE-PASS METHODS:")
            for method in ["zstd_only", "arithmetic_only"]:
                if method in result['methods']:
                    data = result['methods'][method]
                    report.append(f"    {method.upper().replace('_', ' ')}")
                    report.append(f"      Compressed size: {data['compressed_size_bytes']:,} bytes")
                    report.append(f"      Compression ratio: {data['compression_ratio']:.4f}x")
                    report.append(f"      Time: {data['compression_time_ms']} ms")

            # ── Pipeline results ─────────────────────────────────────────
            report.append("")
            report.append("  TWO-STAGE PIPELINES:")
            for pipe_method, pipe_label in [
                ("reverse_pipeline_zstd_arith", "ZSTD -> ARITHMETIC"),
                ("forward_pipeline_arith_zstd", "ARITHMETIC -> ZSTD"),
            ]:
                if pipe_method in result['methods']:
                    data = result['methods'][pipe_method]
                    report.append(f"    {pipe_label}:")
                    report.append(f"      Compressed size: {data['compressed_size_bytes']:,} bytes")
                    report.append(f"      Compression ratio: {data['compression_ratio']:.4f}x")
                    report.append(f"      Time: {data['compression_time_ms']} ms")

            report.append("")
            report.append(f"  BEST OVERALL: {result['best_method'].upper().replace('_', ' ')} ({result['best_ratio']:.4f}x)")
            report.append("")

            # ── Delta analysis ───────────────────────────────────────────
            report.append("  DELTA ANALYSIS (Reverse Pipeline Impact):")
            delta = result['delta_analysis']
            report.append(f"    vs Zstd-only:      {delta['reverse_vs_zstd_bytes']:+,} bytes ({delta['reverse_vs_zstd_percent']:+.2f}%)")
            report.append(f"    vs Arithmetic-only: {delta['reverse_vs_arithmetic_bytes']:+,} bytes ({delta['reverse_vs_arithmetic_percent']:+.2f}%)")
            report.append(f"    Zstd vs Arithmetic: {delta['zstd_vs_arithmetic_bytes']:+,} bytes ({delta['zstd_vs_arithmetic_percent']:+.2f}%)")
            report.append(f"    Forward vs Zstd:    {delta['forward_vs_zstd_bytes']:+,} bytes ({delta['forward_vs_zstd_percent']:+.2f}%)")
            report.append(f"    Forward vs Reverse: {delta['forward_vs_reverse_bytes']:+,} bytes ({delta['forward_vs_reverse_percent']:+.2f}%)")
            report.append("")

            # ── Viability verdicts ────────────────────────────────────────
            report.append("  PIPELINE VIABILITY:")
            rev_a = result['reverse_pipeline_analysis']
            report.append(f"    Zstd->Arithmetic:")
            report.append(f"      Size reduction vs Zstd:       {rev_a['pct_smaller_vs_zstd']:+.2f}%")
            report.append(f"      Size reduction vs Arithmetic: {rev_a['pct_smaller_vs_arithmetic']:+.2f}%")
            rev_status = "✓ WORTHWHILE (>1% smaller)" if rev_a['worthwhile'] else "✗ NOT WORTHWHILE (<1% smaller)"
            report.append(f"      Assessment: {rev_status}")
            fwd_a = result['forward_pipeline_analysis']
            report.append(f"    Arithmetic->Zstd:")
            report.append(f"      Size reduction vs Zstd:       {fwd_a['pct_smaller_vs_zstd']:+.2f}%")
            report.append(f"      Size reduction vs Arithmetic: {fwd_a['pct_smaller_vs_arithmetic']:+.2f}%")
            fwd_status = "✓ WORTHWHILE (>1% smaller)" if fwd_a['worthwhile'] else "✗ NOT WORTHWHILE (<1% smaller)"
            report.append(f"      Assessment: {fwd_status}")
            report.append("")

        # ── Footer / recommendations ─────────────────────────────────────
        report.append("=" * 90)
        report.append("SUMMARY & RECOMMENDATION:")
        report.append("- Pipelines worthwhile only if size reduction >1% vs single-pass")
        report.append("- Compare both pipeline orders to find the better stage sequence")
        report.append("- Consider CPU overhead of two-stage compression vs size reduction gain")
        report.append("- Test on representative WAV data (speech, music, noise, silence)")
        report.append("=" * 90)

        return "\n".join(report)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    """Run benchmark with WAV file data, falling back to synthetic data."""
    parser = argparse.ArgumentParser(
        description="Compression benchmark: Zstd vs Arithmetic vs Zstd+Arithmetic pipeline (WAV edition)."
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

    benchmark = CompressionBenchmarkZstdArith(zstd_level=args.level)
    results = []

    # Use the user-supplied path, or fall back to the bundled test file.
    if args.wav is not None:
        wav_file = Path(args.wav)
    else:
        wav_file = Path(__file__).parent / "test_audio.wav"

    if wav_file.exists():
        print(f"Reading WAV file: {wav_file}")
        with open(wav_file, 'rb') as f:
            wav_data = f.read()

        print(f"WAV file size: {len(wav_data):,} bytes")
        print("Running compression benchmark...")

        result = benchmark.benchmark_on_data(wav_data, f"WAV Audio File ({len(wav_data):,} bytes)")
        results.append(result)
    else:
        print(f"WAV file not found: {wav_file}")
        print("Using synthetic test data instead...")

        # Fallback: incompressible random data.
        random_data = os.urandom(10000)
        result1 = benchmark.benchmark_on_data(random_data, "Random binary (10KB)")
        results.append(result1)

    # ── Output ────────────────────────────────────────────────────────────
    if results:
        print("\n" + benchmark.generate_table_report(results))
        print("\n" + benchmark.generate_report(results))

        # Persist raw results as JSON for later analysis or plotting.
        output_file = Path(__file__).parent / "compression_benchmark_zstd_arith_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
