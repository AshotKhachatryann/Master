/**
 * arithmetic_codec.cpp
 *
 * Standalone arithmetic compression codec.
 *
 * Compress:   raw bytes → arithmetic encode → output
 * Decompress: input → arithmetic decode → raw bytes
 *
 * Usage:
 *   arithmetic_codec  c|−c|−compress  <inputFile> <outputFile>
 *   arithmetic_codec  d|−d|−decompress  <inputFile> <outputFile>
 *
 * Build:
 *   g++ -O2 -std=c++17 -o arithmetic_codec arithmetic_codec.cpp
 */

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ── Constants ────────────────────────────────────────────────────────────────

// Arithmetic coder alphabet: 256 byte values (0x00–0xFF) plus one
// end-of-file marker so the decoder knows when to stop.
constexpr int    SYMBOL_COUNT    = 257;
constexpr int    EOF_SYMBOL      = 256;

// 32-bit range boundaries used by the arithmetic coder.
// The interval [low, high] is maintained within 32-bit precision;
// HALF/QUARTER/THREE_QUARTERS define the renormalization thresholds.
constexpr uint32_t TOP_VALUE       = 0xFFFFFFFFu;   // Full 32-bit range
constexpr uint32_t HALF            = 0x80000000u;   // Mid-point
constexpr uint32_t QUARTER         = 0x40000000u;   // 25%
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
  out.put(static_cast<char>( v        & 0xFF));
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

// ── Arithmetic encoder ───────────────────────────────────────────────────────

/**
 * @brief Encodes raw bytes into an arithmetic-coded bitstream.
 *
 * Output format (all values little-endian):
 *   [4 bytes] "ACOD" magic marker
 *   [257 × 4 bytes] frequency table (one uint32 per symbol)
 *   [variable] arithmetic-coded bitstream (MSB-first packed bytes)
 */
void encode(const std::vector<uint8_t>& input, std::ostream& out) {
  auto freq  = buildFrequencies(input);
  auto cum   = buildCumulative(freq);
  uint64_t total = cum.back();

  // Write magic header + full frequency table
  out.write("ACOD", 4);
  for (int i = 0; i < SYMBOL_COUNT; ++i) writeU32Le(out, freq[i]);

  BitWriter bw(out);
  uint32_t low  = 0;
  uint32_t high = TOP_VALUE;
  uint32_t pendingBits = 0;

  // Encode a single symbol by narrowing [low, high] and renormalizing.
  auto encodeSymbol = [&](int symbol) {
    uint64_t range = static_cast<uint64_t>(high) - low + 1;
    uint32_t newLow  = low + static_cast<uint32_t>((range * cum[symbol])     / total);
    uint32_t newHigh = low + static_cast<uint32_t>((range * cum[symbol + 1]) / total) - 1;
    low  = newLow;
    high = newHigh;

    // Renormalization loop
    for (;;) {
      if (high < HALF) {
        bw.writeBit(0);
        while (pendingBits > 0) { bw.writeBit(1); --pendingBits; }
      } else if (low >= HALF) {
        bw.writeBit(1);
        while (pendingBits > 0) { bw.writeBit(0); --pendingBits; }
        low  -= HALF;
        high -= HALF;
      } else if (low >= QUARTER && high < THREE_QUARTERS) {
        ++pendingBits;
        low  -= QUARTER;
        high -= QUARTER;
      } else {
        break;
      }
      low  <<= 1;
      high = (high << 1) | 1;
    }
  };

  // Encode every input byte, then send EOF marker.
  for (uint8_t b : input) encodeSymbol(static_cast<int>(b));
  encodeSymbol(EOF_SYMBOL);

  // Flush the final interval
  ++pendingBits;
  if (low < QUARTER) {
    bw.writeBit(0);
    while (pendingBits > 0) { bw.writeBit(1); --pendingBits; }
  } else {
    bw.writeBit(1);
    while (pendingBits > 0) { bw.writeBit(0); --pendingBits; }
  }
  bw.flush();
}

// ── Arithmetic decoder ───────────────────────────────────────────────────────

/**
 * @brief Decodes an arithmetic-coded stream back to the original data.
 *
 * Reads the "ACOD" header + frequency table, rebuilds the cumulative
 * distribution, and iteratively determines which symbol each successive
 * interval subdivision corresponds to. Stops when EOF_SYMBOL is decoded.
 */
std::vector<uint8_t> decode(std::istream& in) {
  char magic[4];
  in.read(magic, 4);
  if (!in.good() || std::string(magic, 4) != "ACOD") {
    throw std::runtime_error("Invalid arithmetic stream magic header");
  }

  // Rebuild frequency table and cumulative distribution
  std::vector<uint32_t> freq(SYMBOL_COUNT, 0);
  for (int i = 0; i < SYMBOL_COUNT; ++i) {
    freq[i] = readU32Le(in);
    if (freq[i] == 0) throw std::runtime_error("Invalid frequency table");
  }

  auto cum = buildCumulative(freq);
  uint64_t total = cum.back();

  BitReader br(in);
  uint32_t low  = 0;
  uint32_t high = TOP_VALUE;
  // Prime the decoder with the first 32 bits from the bitstream
  uint32_t code = 0;
  for (int i = 0; i < 32; ++i) code = (code << 1) | br.readBit();

  std::vector<uint8_t> output;

  // Decode loop
  while (true) {
    uint64_t range = static_cast<uint64_t>(high) - low + 1;
    uint64_t scaled = ((static_cast<uint64_t>(code - low) + 1) * total - 1) / range;

    // Binary search for the symbol
    int loS = 0, hiS = SYMBOL_COUNT;
    while (loS + 1 < hiS) {
      int mid = (loS + hiS) / 2;
      if (cum[mid] <= scaled) loS = mid; else hiS = mid;
    }
    int symbol = loS;

    if (symbol == EOF_SYMBOL) break;
    output.push_back(static_cast<uint8_t>(symbol));

    uint32_t newLow  = low + static_cast<uint32_t>((range * cum[symbol])     / total);
    uint32_t newHigh = low + static_cast<uint32_t>((range * cum[symbol + 1]) / total) - 1;
    low  = newLow;
    high = newHigh;

    // Renormalization
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
      code = (code << 1) | br.readBit();
    }
  }

  return output;
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

// ── Main: Arithmetic codec CLI ───────────────────────────────────────────────

namespace {

/// Default file extension used for arithmetic-compressed output.
constexpr const char* ART_EXTENSION = ".art";

/// Returns true if *path* ends with the given suffix (case-sensitive).
bool endsWith(const std::string& path, const std::string& suffix) {
  return path.size() >= suffix.size() &&
       path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/// Derives the default compressed-output path by appending ".art" to *input*.
std::string defaultCompressedPath(const std::string& input) {
  return input + ART_EXTENSION;
}

/// Derives the default decompressed-output path by stripping ".art" from
/// *input*.  If the input does not end in ".art", appends ".out" instead so
/// the result is never the same as the input file.
std::string defaultDecompressedPath(const std::string& input) {
  if (endsWith(input, ART_EXTENSION)) {
    return input.substr(0, input.size() - std::string(ART_EXTENSION).size());
  }
  return input + ".out";
}

}  // namespace

int main(int argc, char* argv[]) {
  // The output argument is optional:
  //   - On compress:   default is <input>.art
  //   - On decompress: default is <input> with ".art" stripped (or +".out")
  if (argc < 3 || argc > 4) {
    std::cerr << "Arithmetic codec\n\n"
          << "Usage:\n"
          << "  " << argv[0] << " <mode> <input> [output]\n\n"
          << "Modes:\n"
          << "  c, -c, -compress    Compress (Arithmetic encode)\n"
          << "  d, -d, -decompress  Decompress (Arithmetic decode)\n\n"
          << "If [output] is omitted:\n"
          << "  - compress writes to <input>.art\n"
          << "  - decompress strips the trailing .art (or appends .out)\n";
    return 1;
  }

  const std::string mode       = argv[1];
  const std::string inputFile  = argv[2];

  const bool doCompress   = (mode == "c" || mode == "-c" || mode == "-compress");
  const bool doDecompress = (mode == "d" || mode == "-d" || mode == "-decompress");

  // Resolve output path: explicit arg wins, otherwise use the mode default.
  std::string outputFile;
  if (argc == 4) {
    outputFile = argv[3];
  } else if (doCompress) {
    outputFile = defaultCompressedPath(inputFile);
  } else if (doDecompress) {
    outputFile = defaultDecompressedPath(inputFile);
  }

  try {
    if (doCompress) {
      auto inData = readAllBytes(inputFile);
      std::ofstream out(outputFile, std::ios::binary);
      if (!out) throw std::runtime_error("Cannot open output file: " + outputFile);
      encode(inData, out);
      std::cerr << "  Arithmetic: " << inData.size() << " -> " << out.tellp()
            << " bytes  (" << outputFile << ")\n";
      return 0;
    }

    if (doDecompress) {
      std::ifstream in(inputFile, std::ios::binary);
      if (!in) throw std::runtime_error("Cannot open input file: " + inputFile);
      auto outData = decode(in);
      writeAllBytes(outputFile, outData);
      std::cerr << "  Arithmetic decode: " << outData.size()
            << " bytes  (" << outputFile << ")\n";
      return 0;
    }

    std::cerr << "Unknown mode: " << mode << " (use c or d)\n";
    return 1;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 2;
  }
}

