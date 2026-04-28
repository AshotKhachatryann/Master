/**
 * zstd_arithmetic_codec.cpp
 *
 * Two-stage compression pipeline: Zstd → Arithmetic
 *
 * Compress:   raw bytes → zstd compress → arithmetic encode → output
 * Decompress: input → arithmetic decode → zstd decompress → raw bytes
 *
 * Usage:
 *   zstd_arithmetic_codec  c|−c|−compress  <inputFile> <outputFile> [zstdLevel]
 *   zstd_arithmetic_codec  d|−d|−decompress  <inputFile> <outputFile>
 *
 * Build (pick whichever works on your system):
 *   g++ -O2 -std=c++17 -o zstd_arithmetic_codec zstd_arithmetic_codec.cpp -lzstd
 *   g++ -O2 -std=c++17 -o zstd_arithmetic_codec zstd_arithmetic_codec.cpp /lib64/libzstd.so.1
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Zstd C API — declared directly so no zstd development headers are needed.
// Only requires libzstd runtime (.so/.dylib/.dll) at link and run time.
extern "C" {
size_t ZSTD_compress(void* dst, size_t dstCapacity,
           const void* src, size_t srcSize, int compressionLevel);
size_t ZSTD_decompress(void* dst, size_t dstCapacity,
             const void* src, size_t compressedSize);
size_t ZSTD_compressBound(size_t srcSize);
unsigned long long ZSTD_getFrameContentSize(const void* src, size_t srcSize);
unsigned ZSTD_isError(size_t code);
const char* ZSTD_getErrorName(size_t code);
}

namespace {

// ── Constants ────────────────────────────────────────────────────────────────

// Zstd sentinel values returned by ZSTD_getFrameContentSize() to signal
// that the decompressed size is either unknown or the frame is invalid.
constexpr unsigned long long ZSTD_CONTENTSIZE_UNKNOWN = ~0ULL;
constexpr unsigned long long ZSTD_CONTENTSIZE_ERROR   = ~1ULL;

// Arithmetic coder alphabet: 256 byte values (0x00–0xFF) plus one
// end-of-file marker so the decoder knows when to stop.
constexpr int    SYMBOL_COUNT  = 257;
constexpr int    EOF_SYMBOL    = 256;

// 32-bit range boundaries used by the arithmetic coder.
// The interval [low, high] is maintained within 32-bit precision;
// HALF/QUARTER/THREE_QUARTERS define the renormalization thresholds.
constexpr uint32_t TOP_VALUE     = 0xFFFFFFFFu;   // Full 32-bit range
constexpr uint32_t HALF      = 0x80000000u;   // Mid-point
constexpr uint32_t QUARTER     = 0x40000000u;   // 25%
constexpr uint32_t THREE_QUARTERS  = 0xC0000000u;   // 75%


// ── Bit I/O helpers ──────────────────────────────────────────────────────────

/**
 * @brief Writes individual bits to an output stream, packing 8 bits into
 *    each byte. Bits are written MSB-first (most-significant bit first).
 *    Call flush() after all bits are written to emit any partial byte.
 */
class BitWriter {
public:
  explicit BitWriter(std::ostream& out) : m_out(out), m_buffer(0), m_bitCount(0) {}

  /// Appends a single bit (0 or 1) to the buffer; emits a byte when full.
  void writeBit(int bit) {
    m_buffer = static_cast<uint8_t>((m_buffer << 1) | (bit & 1));
    if (++m_bitCount == 8) {
      m_out.put(static_cast<char>(m_buffer));
      m_buffer = 0;
      m_bitCount = 0;
    }
  }

  /// Pads the remaining bits with zeros and emits the final partial byte.
  void flush() {
    if (m_bitCount > 0) {
      m_buffer <<= (8 - m_bitCount);
      m_out.put(static_cast<char>(m_buffer));
      m_buffer = 0;
      m_bitCount = 0;
    }
  }

private:
  std::ostream& m_out;
  uint8_t m_buffer;
  int m_bitCount;
};

/**
 * @brief Reads individual bits from an input stream, unpacking each byte
 *    MSB-first. Returns 0 bits on EOF to allow the decoder to finish.
 */
class BitReader {
public:
  explicit BitReader(std::istream& in) : m_in(in), m_buffer(0), m_bitsLeft(0) {}

  /// Returns the next bit (0 or 1). Reads a fresh byte when the buffer
  /// is exhausted; returns 0 past end-of-stream.
  int readBit() {
    if (m_bitsLeft == 0) {
      int byte = m_in.get();
      if (byte == EOF) return 0;
      m_buffer = static_cast<uint8_t>(byte);
      m_bitsLeft = 8;
    }
    int bit = (m_buffer >> 7) & 1;
    m_buffer <<= 1;
    --m_bitsLeft;
    return bit;
  }

private:
  std::istream& m_in;
  uint8_t m_buffer;
  int m_bitsLeft;
};

// ── Frequency / cumulative helpers ───────────────────────────────────────────

/**
 * @brief Counts byte frequencies for the arithmetic model.
 *
 * Every symbol (including EOF) starts at 1 so that no symbol ever has
 * a zero probability, which would make the arithmetic interval degenerate.
 *
 * @param data  Raw input bytes to analyze.
 * @return    Frequency table of size SYMBOL_COUNT (257 entries).
 */
std::vector<uint32_t> buildFrequencies(const std::vector<uint8_t>& data) {
  std::vector<uint32_t> freq(SYMBOL_COUNT, 1);
  for (uint8_t b : data) {
    if (freq[b] < UINT32_MAX - 1) freq[b]++;
  }
  return freq;
}

/**
 * @brief Converts a frequency table to a cumulative distribution.
 *
 * cum[i] = sum of freq[0..i-1].  The arithmetic coder maps each symbol
 * to the sub-interval [cum[symbol]/total, cum[symbol+1]/total).
 */
std::vector<uint64_t> buildCumulative(const std::vector<uint32_t>& freq) {
  std::vector<uint64_t> cum(SYMBOL_COUNT + 1, 0);
  for (int i = 0; i < SYMBOL_COUNT; ++i) {
    cum[i + 1] = cum[i] + freq[i];
  }
  return cum;
}

/// Writes a 32-bit unsigned integer in little-endian byte order.
void writeU32Le(std::ostream& out, uint32_t v) {
  out.put(static_cast<char>( v    & 0xFF));
  out.put(static_cast<char>((v >>  8) & 0xFF));
  out.put(static_cast<char>((v >> 16) & 0xFF));
  out.put(static_cast<char>((v >> 24) & 0xFF));
}

/// Reads a 32-bit unsigned integer in little-endian byte order.
uint32_t readU32Le(std::istream& in) {
  uint32_t b0 = static_cast<uint32_t>(in.get());
  uint32_t b1 = static_cast<uint32_t>(in.get());
  uint32_t b2 = static_cast<uint32_t>(in.get());
  uint32_t b3 = static_cast<uint32_t>(in.get());
  if (!in.good()) throw std::runtime_error("Unexpected EOF reading header");
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

// ── Arithmetic encoder (raw bytes → arithmetic stream in memory) ─────────────

/**
 * @brief Encodes raw bytes into an arithmetic-coded bitstream.
 *
 * Output format (all values little-endian):
 *   [4 bytes] "ACOD" magic marker
 *   [257 × 4 bytes] frequency table (one uint32 per symbol)
 *   [variable] arithmetic-coded bitstream (MSB-first packed bytes)
 *
 * The encoder maintains a 32-bit [low, high] interval that is progressively
 * narrowed for each input symbol according to its cumulative probability.
 * Renormalization emits bits whenever the interval falls entirely within
 * the upper or lower half (or straddles the midpoint, tracked via
 * pendingBits for the "middle-follows" case).
 *
 * @param input  Raw bytes to compress.
 * @return     Arithmetic-coded byte buffer (self-contained, decodable).
 */
std::vector<uint8_t> arithmeticEncode(const std::vector<uint8_t>& input) {
  auto freq  = buildFrequencies(input);
  auto cum   = buildCumulative(freq);
  uint64_t total = cum.back();  // Sum of all frequencies = denominator

  std::ostringstream oss(std::ios::binary);

  // Write magic header + full frequency table so the decoder can
  // rebuild the same probability model without a separate side-channel.
  oss.write("ACOD", 4);
  for (int i = 0; i < SYMBOL_COUNT; ++i) {
    writeU32Le(oss, freq[i]);
  }

  BitWriter bw(oss);
  uint32_t low  = 0;
  uint32_t high = TOP_VALUE;
  uint32_t pendingBits = 0;

  // Encode a single symbol by narrowing [low, high] to the sub-interval
  // corresponding to the symbol's cumulative probability range, then
  // renormalize by shifting out resolved bits.
  auto encodeSymbol = [&](int symbol) {
    uint64_t range = static_cast<uint64_t>(high) - low + 1;
    // Map symbol's probability interval onto the current [low, high] range
    uint32_t newLow  = low + static_cast<uint32_t>((range * cum[symbol])   / total);
    uint32_t newHigh = low + static_cast<uint32_t>((range * cum[symbol + 1]) / total) - 1;
    low  = newLow;
    high = newHigh;

    // Renormalization loop: emit bits and widen the interval whenever
    // the MSBs of low and high agree (or nearly agree).
    for (;;) {
      if (high < HALF) {
        // Interval is entirely in the lower half → emit 0
        bw.writeBit(0);
        while (pendingBits > 0) { bw.writeBit(1); --pendingBits; }
      } else if (low >= HALF) {
        // Interval is entirely in the upper half → emit 1
        bw.writeBit(1);
        while (pendingBits > 0) { bw.writeBit(0); --pendingBits; }
        low  -= HALF;
        high -= HALF;
      } else if (low >= QUARTER && high < THREE_QUARTERS) {
        // Interval straddles the midpoint → defer the bit decision
        ++pendingBits;
        low  -= QUARTER;
        high -= QUARTER;
      } else {
        break;  // Interval spans a wide enough range; stop
      }
      low  <<= 1;       // Widen interval: shift left
      high = (high << 1) | 1;
    }
  };

  // Encode every input byte, then send the EOF marker so the decoder
  // knows exactly when to stop.
  for (uint8_t b : input) encodeSymbol(static_cast<int>(b));
  encodeSymbol(EOF_SYMBOL);

  // Flush the final interval: emit enough bits to uniquely identify
  // a value inside the remaining [low, high] range.
  ++pendingBits;
  if (low < QUARTER) {
    bw.writeBit(0);
    while (pendingBits > 0) { bw.writeBit(1); --pendingBits; }
  } else {
    bw.writeBit(1);
    while (pendingBits > 0) { bw.writeBit(0); --pendingBits; }
  }
  bw.flush();

  // Collect bytes from stringstream
  std::string s = oss.str();
  return std::vector<uint8_t>(s.begin(), s.end());
}

// ── Arithmetic decoder (arithmetic stream → raw bytes) ───────────────────────

/**
 * @brief Decodes an arithmetic-coded byte buffer back to the original data.
 *
 * Reverses arithmeticEncode(): reads the "ACOD" header + frequency table,
 * rebuilds the cumulative distribution, and then iteratively determines
 * which symbol each successive interval subdivision corresponds to.
 * Stops when the EOF_SYMBOL is decoded.
 *
 * @param compressed  Arithmetic-coded byte buffer produced by arithmeticEncode().
 * @return      Reconstructed original bytes (lossless).
 */
std::vector<uint8_t> arithmeticDecode(const std::vector<uint8_t>& compressed) {
  std::string s(compressed.begin(), compressed.end());
  std::istringstream iss(s, std::ios::binary);

  // Validate magic header
  char magic[4];
  iss.read(magic, 4);
  if (!iss.good() || std::string(magic, 4) != "ACOD") {
    throw std::runtime_error("Invalid arithmetic stream magic header");
  }

  // Rebuild the same frequency table and cumulative distribution
  // that the encoder used.
  std::vector<uint32_t> freq(SYMBOL_COUNT, 0);
  for (int i = 0; i < SYMBOL_COUNT; ++i) {
    freq[i] = readU32Le(iss);
    if (freq[i] == 0) throw std::runtime_error("Invalid frequency table");
  }

  auto cum = buildCumulative(freq);
  uint64_t total = cum.back();

  BitReader br(iss);
  uint32_t low  = 0;
  uint32_t high = TOP_VALUE;
  // Prime the decoder with the first 32 bits from the bitstream
  // to establish the initial "code" value inside [low, high].
  uint32_t code = 0;
  for (int i = 0; i < 32; ++i) code = (code << 1) | br.readBit();

  std::vector<uint8_t> output;

  // Decode loop: for each iteration, determine which symbol the current
  // "code" value falls into, emit it, then narrow the interval and
  // renormalize — mirroring the encoder's logic.
  while (true) {
    uint64_t range = static_cast<uint64_t>(high) - low + 1;
    // Scale the code value into [0, total) to find the target symbol
    uint64_t scaled = ((static_cast<uint64_t>(code - low) + 1) * total - 1) / range;

    // Binary search: find symbol whose cumulative interval contains 'scaled'
    int loS = 0, hiS = SYMBOL_COUNT;
    while (loS + 1 < hiS) {
      int mid = (loS + hiS) / 2;
      if (cum[mid] <= scaled) loS = mid; else hiS = mid;
    }
    int symbol = loS;

    if (symbol == EOF_SYMBOL) break;
    output.push_back(static_cast<uint8_t>(symbol));

    uint32_t newLow  = low + static_cast<uint32_t>((range * cum[symbol])   / total);
    uint32_t newHigh = low + static_cast<uint32_t>((range * cum[symbol + 1]) / total) - 1;
    low  = newLow;
    high = newHigh;

    // Renormalization: shift out resolved bits (same logic as encoder)
    // and read fresh bits from the stream into 'code'.
    for (;;) {
      if (high < HALF) {
        // Both in lower half — no adjustment needed
      } else if (low >= HALF) {
        code -= HALF; low -= HALF; high -= HALF;
      } else if (low >= QUARTER && high < THREE_QUARTERS) {
        code -= QUARTER; low -= QUARTER; high -= QUARTER;
      } else {
        break;
      }
      low  <<= 1;
      high = (high << 1) | 1;
      code = (code << 1) | br.readBit();  // Shift in next bit
    }
  }

  return output;
}

// ── Zstd wrappers ────────────────────────────────────────────────────────────

/**
 * @brief Compresses a byte buffer using Zstandard.
 *
 * @param input  Raw bytes to compress.
 * @param level  Zstd compression level (1 = fast, 22 = best ratio; default 3).
 * @return     Zstd-compressed frame (includes embedded frame metadata).
 */
std::vector<uint8_t> compressZstd(const std::vector<uint8_t>& input, int level) {
  // ZSTD_compressBound gives a safe upper-bound for the output buffer size
  size_t bound = ZSTD_compressBound(input.size());
  std::vector<uint8_t> out(bound);
  size_t written = ZSTD_compress(out.data(), out.size(),
                   input.data(), input.size(), level);
  if (ZSTD_isError(written)) {
    throw std::runtime_error(std::string("ZSTD_compress failed: ") +
                 ZSTD_getErrorName(written));
  }
  out.resize(written);
  return out;
}

/**
 * @brief Decompresses a Zstd frame back to the original byte buffer.
 *
 * Reads the expected decompressed size from the frame header and allocates
 * the output buffer accordingly. Throws on corrupt or incomplete frames.
 *
 * @param input  Zstd-compressed frame.
 * @return     Decompressed original bytes.
 */
std::vector<uint8_t> decompressZstd(const std::vector<uint8_t>& input) {
  // Read the original size from the zstd frame header
  unsigned long long expected = ZSTD_getFrameContentSize(input.data(), input.size());
  if (expected == ZSTD_CONTENTSIZE_ERROR)
    throw std::runtime_error("Invalid zstd frame");
  if (expected == ZSTD_CONTENTSIZE_UNKNOWN)
    throw std::runtime_error("Unknown decompressed size in zstd frame");

  std::vector<uint8_t> out(static_cast<size_t>(expected));
  size_t written = ZSTD_decompress(out.data(), out.size(),
                   input.data(), input.size());
  if (ZSTD_isError(written)) {
    throw std::runtime_error(std::string("ZSTD_decompress failed: ") +
                 ZSTD_getErrorName(written));
  }
  out.resize(written);
  return out;
}

// ── File I/O ─────────────────────────────────────────────────────────────────

/// Reads an entire file into a byte vector.
std::vector<uint8_t> readAllBytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open input file: " + path);
  in.seekg(0, std::ios::end);
  auto size = in.tellg();
  in.seekg(0, std::ios::beg);
  std::vector<uint8_t> data(static_cast<size_t>(size));
  if (size > 0) in.read(reinterpret_cast<char*>(data.data()), size);
  return data;
}

/// Writes a byte vector to a file, creating or overwriting it.
void writeAllBytes(const std::string& path, const std::vector<uint8_t>& data) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("Cannot open output file: " + path);
  if (!data.empty())
    out.write(reinterpret_cast<const char*>(data.data()),
          static_cast<std::streamsize>(data.size()));
}

}  // namespace

// ── Main: two-stage pipeline CLI ─────────────────────────────────────────────

/**
 * @brief Entry point — dispatches compress or decompress based on mode flag.
 *
 * Compress path (mode "c"):
 *   1. Read raw file into memory
 *   2. Stage 1: Zstd compress    →  compact byte buffer
 *   3. Stage 2: Arithmetic encode  →  final compressed output
 *   4. Write to the output file
 *
 * Decompress path (mode "d") — reverses the pipeline:
 *   1. Read compressed file
 *   2. Stage 1: Arithmetic decode  →  zstd-compressed buffer
 *   3. Stage 2: Zstd decompress  →  original raw bytes
 *   4. Write to the output file
 */
namespace {

/// Default file extension used for Zstd→Arithmetic compressed output.
constexpr const char* ZAC_EXTENSION = ".zac";

/// Returns true if *path* ends with the given suffix (case-sensitive).
bool endsWith(const std::string& path, const std::string& suffix) {
  return path.size() >= suffix.size() &&
       path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/// Derives the default compressed-output path by appending ".zac" to *input*.
std::string defaultCompressedPath(const std::string& input) {
  return input + ZAC_EXTENSION;
}

/// Derives the default decompressed-output path by stripping ".zac" from
/// *input*.  If the input does not end in ".zac", appends ".out" instead so
/// the result is never the same as the input file.
std::string defaultDecompressedPath(const std::string& input) {
  if (endsWith(input, ZAC_EXTENSION)) {
    return input.substr(0, input.size() - std::string(ZAC_EXTENSION).size());
  }
  return input + ".out";
}

/// Renders *bytes* using the largest 1024-based unit that yields a value
/// >= 1 (GB → MB → KB → B).  Two decimals for non-byte units.
std::string formatSize(uint64_t bytes) {
  constexpr uint64_t KB = 1024ULL;
  constexpr uint64_t MB = KB * 1024ULL;
  constexpr uint64_t GB = MB * 1024ULL;
  char buf[32];
  if (bytes >= GB) {
    std::snprintf(buf, sizeof(buf), "%.2f GB", static_cast<double>(bytes) / GB);
  } else if (bytes >= MB) {
    std::snprintf(buf, sizeof(buf), "%.2f MB", static_cast<double>(bytes) / MB);
  } else if (bytes >= KB) {
    std::snprintf(buf, sizeof(buf), "%.2f KB", static_cast<double>(bytes) / KB);
  } else {
    std::snprintf(buf, sizeof(buf), "%llu B", static_cast<unsigned long long>(bytes));
  }
  return std::string(buf);
}

}  // namespace

int main(int argc, char* argv[])
{
  // Pre-pass: extract -l/--level flag (with optional '=' form) from argv,
  // leaving only positional args for the legacy parser below.  The flag
  // may appear anywhere on the command line.
  int zstdLevel = 3;
  std::vector<std::string> args;
  args.reserve(argc);
  args.emplace_back(argv[0]);

  auto setLevel = [&](const std::string& v) -> bool {
    try {
      zstdLevel = std::stoi(v);
    } catch (...) {
      std::cerr << "Invalid zstdLevel value: '" << v << "'\n";
      return false;
    }
    if (zstdLevel < 1 || zstdLevel > 22) {
      std::cerr << "Invalid zstdLevel: " << zstdLevel << " (must be 1–22)\n";
      return false;
    }
    return true;
  }
 for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-l" || a == "--level") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << a << "\n";
        return 1;
      }
      if (!setLevel(argv[++i])) return 1;
    } else if (a.rfind("-l=", 0) == 0) {
      if (!setLevel(a.substr(3))) return 1;
    } else if (a.rfind("--level=", 0) == 0) {
      if (!setLevel(a.substr(8))) return 1;
    } else {
      args.push_back(std::move(a));
    }
  }

  // Positional contract after flag stripping:
  //   args = [argv0, mode, input]            — output defaulted
  //   args = [argv0, mode, input, output]    — explicit output
  if (args.size() < 3 || args.size() > 4) {
    std::cerr << "Zstd+Arith two-stage pipeline codec\n\n"
          << "Usage:\n"
          << "  " << argv[0] << " <mode> <input> [output] [-l N | --level N]\n\n"
          << "Modes:\n"
          << "  c, -c, -compress    Compress (Zstd → Arithmetic)\n"
          << "  d, -d, -decompress  Decompress (Arithmetic decode → Zstd)\n\n"
          << "If [output] is omitted:\n"
          << "  - compress writes to <input>.zac\n"
          << "  - decompress strips the trailing .zac (or appends .out)\n\n"
          << "-l, --level: Zstd level 1 (fastest) to 22 (best ratio), default 3.\n"
          << "             Accepts '-l 9', '-l=9', '--level 9', or '--level=9'.\n\n"
          << "Compress:   input → Zstd → Arithmetic → output\n"
          << "Decompress: input → Arithmetic decode → Zstd decompress → output\n";
    return 1;
  }

  const std::string mode    = args[1];
  const std::string inputFile  = args[2];

  // Normalize mode flag: accept c, -c, -compress for compression
  //             and  d, -d, -decompress for decompression
  const bool doCompress   = (mode == "c" || mode == "-c" || mode == "-compress");
  const bool doDecompress = (mode == "d" || mode == "-d" || mode == "-decompress");

  // Resolve output path: explicit arg wins, otherwise use the mode default.
  std::string outputFile;
  if (args.size() == 4) {
    outputFile = args[3];
  } else if (doCompress) {
    outputFile = defaultCompressedPath(inputFile);
  } else if (doDecompress) {
    outputFile = defaultDecompressedPath(inputFile);
  }

  try {
    std::vector<uint8_t> input = readAllBytes(inputFile);

    if (doCompress) {
      // Stage 1: Zstd compress
      std::vector<uint8_t> zstdBytes = compressZstd(input, zstdLevel);
      std::cerr << "  Stage 1 (Zstd L" << zstdLevel << "): "
            << formatSize(input.size()) << " -> " << formatSize(zstdBytes.size()) << "\n";

      // Stage 2: Arithmetic encode
      std::vector<uint8_t> finalBytes = arithmeticEncode(zstdBytes);
      std::cerr << "  Stage 2 (Arithmetic): "
            << formatSize(zstdBytes.size()) << " -> " << formatSize(finalBytes.size()) << "\n";

      std::cerr << "  Total: " << formatSize(input.size()) << " -> " << formatSize(finalBytes.size())
            << " (ratio: "
            << static_cast<double>(input.size()) / finalBytes.size() << "x)"
            << "  (" << outputFile << ")\n";

      writeAllBytes(outputFile, finalBytes);
      return 0;
    }

    if (doDecompress) {
      // Stage 1: Arithmetic decode
      std::vector<uint8_t> zstdBytes = arithmeticDecode(input);
      std::cerr << "  Stage 1 (Arithmetic decode): "
            << formatSize(input.size()) << " -> " << formatSize(zstdBytes.size()) << "\n";

      // Stage 2: Zstd decompress
      std::vector<uint8_t> original = decompressZstd(zstdBytes);
      std::cerr << "  Stage 2 (Zstd decompress): "
            << formatSize(zstdBytes.size()) << " -> " << formatSize(original.size())
            << "  (" << outputFile << ")\n";

      writeAllBytes(outputFile, original);
      return 0;
    }

    std::cerr << "Unknown mode: " << mode << " (use c or d)\n";
    return 1;

  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 2;
  }
}


