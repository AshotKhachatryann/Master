/**
 * zstd_codec.cpp
 *
 * Standalone Zstandard compression codec.
 *
 * Compress:   raw bytes → Zstd compress → output
 * Decompress: input → Zstd decompress → raw bytes
 *
 * Usage:
 *   zstd_codec  c|−c|−compress  <inputFile> <outputFile> [zstdLevel]
 *   zstd_codec  d|−d|−decompress  <inputFile> <outputFile>
 *
 * Build (pick whichever works on your system):
 *   g++ -O2 -std=c++17 -o zstd_codec zstd_codec.cpp -lzstd
 *   g++ -O2 -std=c++17 -o zstd_codec zstd_codec.cpp /lib64/libzstd.so.1
 */

#include <cstdint>
#include <fstream>
#include <iostream>
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

// ── Zstd wrappers ────────────────────────────────────────────────────────────

/**
 * @brief Compresses a byte buffer using Zstandard.
 *
 * @param input  Raw bytes to compress.
 * @param level  Zstd compression level (1 = fast, 22 = best ratio; default 3).
 * @return     Zstd-compressed frame (includes embedded frame metadata).
 */
std::vector<uint8_t> compressZstd(const std::vector<uint8_t>& input, int level) {
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

}  // namespace

// ── Main: Zstd codec CLI ────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  if (argc < 4 || argc > 5) {
    std::cerr << "Zstd codec\n\n"
          << "Usage:\n"
          << "  " << argv[0] << " <mode> <input> <output> [zstdLevel]\n\n"
          << "Modes:\n"
          << "  c, -c, -compress    Compress (Zstd)\n"
          << "  d, -d, -decompress  Decompress (Zstd)\n\n"
          << "zstdLevel: 1 (fastest) to 22 (best ratio), default 3\n";
    return 1;
  }

  const std::string mode       = argv[1];
  const std::string inputFile  = argv[2];
  const std::string outputFile = argv[3];
  int zstdLevel = 3;
  if (argc == 5) {
    zstdLevel = std::stoi(argv[4]);
    if (zstdLevel < 1 || zstdLevel > 22) {
      std::cerr << "Invalid zstdLevel: " << zstdLevel
            << " (must be 1–22)\n";
      return 1;
    }
  }

  const bool doCompress   = (mode == "c" || mode == "-c" || mode == "-compress");
  const bool doDecompress = (mode == "d" || mode == "-d" || mode == "-decompress");

  try {
    auto input = readAllBytes(inputFile);

    if (doCompress) {
      auto out = compressZstd(input, zstdLevel);
      writeAllBytes(outputFile, out);
      std::cerr << "  Zstd L" << zstdLevel << ": " << input.size()
            << " -> " << out.size() << " bytes\n";
      return 0;
    }

    if (doDecompress) {
      auto out = decompressZstd(input);
      writeAllBytes(outputFile, out);
      std::cerr << "  Zstd decompress: " << input.size()
            << " -> " << out.size() << " bytes\n";
      return 0;
    }

    std::cerr << "Unknown mode: " << mode << " (use c or d)\n";
    return 1;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 2;
  }
}
