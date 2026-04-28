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

// Zstd C API — declared directly, so no zstd development headers are needed.
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

namespace {

/// Default file extension used for Zstd-compressed output.
constexpr const char* ZST_EXTENSION = ".zst";

/// Returns true if *path* ends with the given suffix (case-sensitive).
bool endsWith(const std::string& path, const std::string& suffix) {
  return path.size() >= suffix.size() &&
       path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/// Derives the default compressed-output path by appending ".zst" to *input*.
std::string defaultCompressedPath(const std::string& input) {
  return input + ZST_EXTENSION;
}

/// Derives the default decompressed-output path by stripping ".zst" from
/// *input*.  If the input does not end in ".zst", appends ".out" instead so
/// the result is never the same as the input file.
std::string defaultDecompressedPath(const std::string& input) {
  if (endsWith(input, ZST_EXTENSION)) {
    return input.substr(0, input.size() - std::string(ZST_EXTENSION).size());
  }
  return input + ".out";
}

}  // namespace

int main(int argc, char* argv[]) {
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
  };

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
    std::cerr << "Zstd codec\n\n"
          << "Usage:\n"
          << "  " << argv[0] << " <mode> <input> [output] [-l N | --level N]\n\n"
          << "Modes:\n"
          << "  c, -c, -compress    Compress (Zstd)\n"
          << "  d, -d, -decompress  Decompress (Zstd)\n\n"
          << "If [output] is omitted:\n"
          << "  - compress writes to <input>.zst\n"
          << "  - decompress strips the trailing .zst (or appends .out)\n\n"
          << "-l, --level: Zstd level 1 (fastest) to 22 (best ratio), default 3.\n"
          << "             Accepts '-l 9', '-l=9', '--level 9', or '--level=9'.\n";
    return 1;
  }

  const std::string mode       = args[1];
  const std::string inputFile  = args[2];

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
    auto input = readAllBytes(inputFile);

    if (doCompress) {
      auto out = compressZstd(input, zstdLevel);
      writeAllBytes(outputFile, out);
      std::cerr << "  Zstd L" << zstdLevel << ": " << input.size()
            << " -> " << out.size() << " bytes  (" << outputFile << ")\n";
      return 0;
    }

    if (doDecompress) {
      auto out = decompressZstd(input);
      writeAllBytes(outputFile, out);
      std::cerr << "  Zstd decompress: " << input.size()
            << " -> " << out.size() << " bytes  (" << outputFile << ")\n";
      return 0;
    }

    std::cerr << "Unknown mode: " << mode << " (use c or d)\n";
    return 1;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 2;
  }
}

