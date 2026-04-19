#include "WavParser.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace {

// WAV format identifiers
constexpr uint16_t WAV_FORMAT_PCM = 0x0001;
constexpr uint16_t WAV_FORMAT_IEEE_FLOAT = 0x0003;
constexpr uint16_t WAV_FORMAT_EXTENSIBLE = 0xFFFE;

// Read 16-bit unsigned integer in little-endian byte order
uint16_t readLE16(std::istream& in) {
  unsigned char b[2] = {};
  in.read(reinterpret_cast<char*>(b), 2);
  if (!in) {
    throw std::runtime_error("Failed to read 16-bit value");
  }
  return static_cast<uint16_t>(b[0] | (static_cast<uint16_t>(b[1]) << 8));
}

// Read 32-bit unsigned integer in little-endian byte order
uint32_t readLE32(std::istream& in) {
  unsigned char b[4] = {};
  in.read(reinterpret_cast<char*>(b), 4);
  if (!in) {
    throw std::runtime_error("Failed to read 32-bit value");
  }
  return static_cast<uint32_t>(b[0]) |
         (static_cast<uint32_t>(b[1]) << 8) |
         (static_cast<uint32_t>(b[2]) << 16) |
         (static_cast<uint32_t>(b[3]) << 24);
}

// Convert raw bytes to float based on format and bit depth
float bytesToFloat(const uint8_t* bytes, std::size_t len, uint16_t wavFormatCode) {
  // Handle PCM formats (8, 16, 24, 32-bit integers)
  if (wavFormatCode == WAV_FORMAT_PCM) {
    switch (len) {
      case 1:
        return static_cast<float>(static_cast<int>(bytes[0]) - 128);
      case 2: {
        const int16_t v = static_cast<int16_t>(
            static_cast<uint16_t>(bytes[0]) | (static_cast<uint16_t>(bytes[1]) << 8));
        return static_cast<float>(v);
      }
      case 3: {
        // 24-bit: sign-extend to 32-bit integer
        int32_t v = (static_cast<int32_t>(bytes[2]) << 16) |
                    (static_cast<int32_t>(bytes[1]) << 8) |
                    static_cast<int32_t>(bytes[0]);
        if (bytes[2] & 0x80) {
          v |= 0xFF000000;  // Sign extension for negative values
        }
        return static_cast<float>(v);
      }
      case 4: {
        const int32_t v = static_cast<int32_t>(
            static_cast<uint32_t>(bytes[0]) |
            (static_cast<uint32_t>(bytes[1]) << 8) |
            (static_cast<uint32_t>(bytes[2]) << 16) |
            (static_cast<uint32_t>(bytes[3]) << 24));
        return static_cast<float>(v);
      }
      default:
        throw std::runtime_error("Unsupported byte count");
    }
  }

  // Handle IEEE 32-bit float format
  if (wavFormatCode == WAV_FORMAT_IEEE_FLOAT) {
    if (len != 4) {
      throw std::runtime_error("Unsupported float byte count");
    }
    // Reconstruct 32-bit little-endian float
    uint32_t raw = static_cast<uint32_t>(bytes[0]) |
                   (static_cast<uint32_t>(bytes[1]) << 8) |
                   (static_cast<uint32_t>(bytes[2]) << 16) |
                   (static_cast<uint32_t>(bytes[3]) << 24);
    float f = 0.0f;
    std::memcpy(&f, &raw, sizeof(float));  // Bitcast to float
    return f;
  }

  throw std::runtime_error("Unsupported format code");
}

// Build a lookup table mapping chunk IDs to file offsets for random access
std::unordered_map<std::string, std::streamoff> createWavChunkLookup(std::istream& in) {
  // Read and validate RIFF header
  char riffId[4] = {};
  in.read(riffId, 4);
  (void)readLE32(in);  // Skip file size

  char waveId[4] = {};
  in.read(waveId, 4);

  const std::string riff(riffId, 4);
  const std::string wave(waveId, 4);
  assert(riff == "RIFF" && wave == "WAVE");
  if (riff != "RIFF" || wave != "WAVE") {
    throw std::runtime_error("Invalid .wav file");
  }

  // Create lookup table with RIFF header at offset 0
  std::unordered_map<std::string, std::streamoff> chunkLookup;
  chunkLookup.emplace(riff, 0);

  // Scan file for all chunks
  while (in && in.peek() != std::char_traits<char>::eof()) {
    std::streamoff chunkOffset = static_cast<std::streamoff>(in.tellg());

    char chunkIdRaw[4] = {};
    in.read(chunkIdRaw, 4);
    if (!in) {
      break;
    }

    // Map chunk ID to its file offset
    const std::string chunkId(chunkIdRaw, 4);
    chunkLookup.emplace(chunkId, chunkOffset);

    // Skip to next chunk (chunks are padded to even boundaries)
    uint32_t chunkSize = readLE32(in);
    in.seekg(static_cast<std::streamoff>(chunkSize), std::ios::cur);
    if (!in) {
      break;
    }

    // Account for RIFF padding
    if (chunkSize % 2 == 1) {
      in.seekg(1, std::ios::cur);
    }
  }

  return chunkLookup;
}

// Parse WAV file structure and convert to Signal object
Signal readWavFile(std::istream& in) {
  // Build chunk offset lookup for random access
  const auto chunkLookup = createWavChunkLookup(in);
  auto fmtIt = chunkLookup.find("fmt ");
  auto dataIt = chunkLookup.find("data");
  if (fmtIt == chunkLookup.end() || dataIt == chunkLookup.end()) {
    throw std::runtime_error("Missing fmt or data chunk");
  }

  // Read format chunk (8 bytes offset: 4 for chunk ID, 4 for size)
  in.clear();
  in.seekg(fmtIt->second + 8, std::ios::beg);

  uint16_t formatCode = readLE16(in);
  int16_t numChannels = static_cast<int16_t>(readLE16(in));
  int32_t sampleRate = static_cast<int32_t>(readLE32(in));

  in.seekg(6, std::ios::cur);  // Skip byteRate and blockAlign
  int16_t bitsPerSample = static_cast<int16_t>(readLE16(in));
  int bytesPerSample = bitsPerSample / 8;

  // Handle extensible WAV format (read actual format code)
  if (formatCode == WAV_FORMAT_EXTENSIBLE) {
    in.seekg(8, std::ios::cur);
    formatCode = readLE16(in);
  }

  // Read data chunk (4 bytes offset: 4 for chunk ID, 4 for size)
  in.clear();
  in.seekg(dataIt->second + 4, std::ios::beg);
  uint32_t numBytesData = readLE32(in);
  std::vector<uint8_t> data(numBytesData);
  in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(numBytesData));
  if (!in) {
    throw std::runtime_error("Failed to read data chunk");
  }

  if (bytesPerSample <= 0 || numChannels <= 0) {
    throw std::runtime_error("Invalid WAV metadata");
  }

  // Calculate total sample count (flattens to mono by using first channel only)
  const std::size_t sampleCount =
      data.size() / static_cast<std::size_t>(bytesPerSample * numChannels);
  std::vector<float> samples(sampleCount, 0.0f);

  // Calculate normalization factor to convert to [-1.0, 1.0] range
  float normFactor = 1.0f / (std::pow(2.0f, static_cast<float>(bitsPerSample - 1)) - 1.0f);
  if (formatCode == WAV_FORMAT_IEEE_FLOAT) {
    normFactor = 1.0f;  // Already normalized
  }

  // Convert raw bytes to float samples
  for (std::size_t i = 0; i < sampleCount; ++i) {
    const std::size_t offset = i * static_cast<std::size_t>(bytesPerSample * numChannels);
    samples[i] = bytesToFloat(data.data() + offset, static_cast<std::size_t>(bytesPerSample), formatCode) * normFactor;
  }

  return Signal(std::move(samples), sampleRate);
}

}  // namespace

Signal Wav::readWav(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Unable to open WAV file: " + path);
  }
  return readWavFile(in);
}

// Write Signal object to a WAV file in 16-bit PCM mono format
void Wav::writeWav(const Signal& signal, const std::string& path) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("Unable to create WAV file: " + path);
  }

  constexpr int kBytesPerSample = 2;  // 16-bit = 2 bytes
  const int numBytes = static_cast<int>(signal.numSamples()) * kBytesPerSample;

  // Write RIFF container header (12 bytes)
  out.write("RIFF", 4);
  const uint32_t fileSize = static_cast<uint32_t>(36 + numBytes);
  out.write(reinterpret_cast<const char*>(&fileSize), sizeof(fileSize));
  out.write("WAVE", 4);

  // Write fmt chunk with audio format metadata (24 bytes)
  out.write("fmt ", 4);
  const uint32_t fmtChunkSize = 16;
  out.write(reinterpret_cast<const char*>(&fmtChunkSize), sizeof(fmtChunkSize));

  const uint16_t formatCode = WAV_FORMAT_PCM;
  const uint16_t numChannels = 1;  // Mono
  const uint32_t sampleRate = static_cast<uint32_t>(signal.getSampleRate());
  const uint32_t bytesPerSecond = sampleRate * kBytesPerSample;
  const uint16_t blockAlign = static_cast<uint16_t>(kBytesPerSample);
  const uint16_t bitsPerSample = static_cast<uint16_t>(kBytesPerSample * 8);

  out.write(reinterpret_cast<const char*>(&formatCode), sizeof(formatCode));
  out.write(reinterpret_cast<const char*>(&numChannels), sizeof(numChannels));
  out.write(reinterpret_cast<const char*>(&sampleRate), sizeof(sampleRate));
  out.write(reinterpret_cast<const char*>(&bytesPerSecond), sizeof(bytesPerSecond));
  out.write(reinterpret_cast<const char*>(&blockAlign), sizeof(blockAlign));
  out.write(reinterpret_cast<const char*>(&bitsPerSample), sizeof(bitsPerSample));

  // Write data chunk header (8 bytes + sample data)
  out.write("data", 4);
  const uint32_t dataSize = static_cast<uint32_t>(numBytes);
  out.write(reinterpret_cast<const char*>(&dataSize), sizeof(dataSize));

  // Convert and write sample data (little-endian 16-bit signed integers)
  for (float s : signal.getSamples()) {
    // Clamp to [-1.0, 1.0] to prevent integer overflow
    const float clamped = std::max(-1.0f, std::min(1.0f, s));
    // Convert float to int16
    const int16_t val16 = static_cast<int16_t>(clamped * std::numeric_limits<int16_t>::max());
    // Write as two bytes (little-endian)
    const uint8_t lo = static_cast<uint8_t>(val16 & 0x00FF);
    const uint8_t hi = static_cast<uint8_t>((static_cast<uint16_t>(val16) >> 8) & 0x00FF);
    out.put(static_cast<char>(lo));
    out.put(static_cast<char>(hi));
  }
}

// Construct WAV file path with directory hierarchy
std::string Wav::getWavPath(const std::string& baseDataPath,
                            const std::string& directory,
                            const std::string& fileName) {
  const auto p = std::filesystem::path(baseDataPath) / "Wav" / directory / (fileName + ".wav");
  return p.string();
}

// Construct WAV file path without subdirectory
std::string Wav::getWavPath(const std::string& baseDataPath, const std::string& fileName) {
  const auto p = std::filesystem::path(baseDataPath) / "Wav" / (fileName + ".wav");
  return p.string();
}
