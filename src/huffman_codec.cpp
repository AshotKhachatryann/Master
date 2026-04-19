/**
 * huffman_codec.cpp
 *
 * Canonical Huffman compression codec.
 *
 * Compress:   raw bytes → Huffman encode → output
 * Decompress: input → Huffman decode → raw bytes
 *
 * Usage:
 *   huffman_codec  c|−c|−compress  <inputFile> <outputFile>
 *   huffman_codec  d|−d|−decompress  <inputFile> <outputFile>
 *
 * Build:
 *   g++ -O2 -std=c++17 -o huffman_codec huffman_codec.cpp
 */

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ── Huffman tree node ────────────────────────────────────────────────────────

/// A node in the Huffman tree. Leaves carry a byte symbol (0–255);
/// internal nodes have symbol == -1 and two children.
struct Node {
  uint64_t freq;
  int symbol;
  std::shared_ptr<Node> left;
  std::shared_ptr<Node> right;

  Node(uint64_t f, int s) : freq(f), symbol(s), left(nullptr), right(nullptr) {}
  Node(std::shared_ptr<Node> l, std::shared_ptr<Node> r)
    : freq(l->freq + r->freq), symbol(-1), left(std::move(l)), right(std::move(r)) {}

  bool isLeaf() const { return symbol >= 0; }
};

/// Min-heap comparator: lowest frequency first; ties broken by symbol.
struct NodeCmp {
  bool operator()(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) const {
    if (a->freq != b->freq) return a->freq > b->freq;
    return a->symbol > b->symbol;
  }
};

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

  /// Writes a variable-length Huffman code (vector of 0/1 bits).
  void writeCode(const std::vector<uint8_t>& code) {
    for (uint8_t b : code) {
      writeBit(static_cast<int>(b));
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
 *    MSB-first. Returns -1 on EOF.
 */
class BitReader {
public:
  explicit BitReader(std::istream& in) : m_in(in), m_buffer(0), m_bitsLeft(0) {}

  /// Returns the next bit (0 or 1), or -1 past end-of-stream.
  int readBit() {
    if (m_bitsLeft == 0) {
      int c = m_in.get();
      if (c == EOF) return -1;
      m_buffer = static_cast<uint8_t>(c);
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

// ── Serialization helpers ────────────────────────────────────────────────────

/// Writes a 64-bit unsigned integer in little-endian byte order.
void writeU64Le(std::ostream& out, uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    out.put(static_cast<char>((v >> (8 * i)) & 0xFF));
  }
}

/// Reads a 64-bit unsigned integer in little-endian byte order.
uint64_t readU64Le(std::istream& in) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) {
    int c = in.get();
    if (c == EOF) throw std::runtime_error("Unexpected EOF while reading header");
    v |= (static_cast<uint64_t>(static_cast<uint8_t>(c)) << (8 * i));
  }
  return v;
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

// ── Huffman tree construction ────────────────────────────────────────────────

/**
 * @brief Builds a Huffman tree from a 256-entry byte frequency table.
 *
 * Uses a min-heap to repeatedly merge the two lowest-frequency nodes.
 * If only one unique symbol exists, a dummy node is added so the tree
 * has at least one internal node (ensuring a 1-bit code).
 *
 * @param freq  Byte frequency table (index = byte value).
 * @return    Root of the Huffman tree, or nullptr if all frequencies are zero.
 */
std::shared_ptr<Node> buildTree(const std::array<uint64_t, 256>& freq) {
  std::priority_queue<std::shared_ptr<Node>,
            std::vector<std::shared_ptr<Node>>, NodeCmp> pq;
  for (int i = 0; i < 256; ++i) {
    if (freq[i] > 0) pq.push(std::make_shared<Node>(freq[i], i));
  }

  if (pq.empty()) return nullptr;

  if (pq.size() == 1) {
    auto only = pq.top();
    auto dummy = std::make_shared<Node>(0, (only->symbol == 0) ? 1 : 0);
    return std::make_shared<Node>(only, dummy);
  }

  while (pq.size() > 1) {
    auto a = pq.top(); pq.pop();
    auto b = pq.top(); pq.pop();
    pq.push(std::make_shared<Node>(a, b));
  }
  return pq.top();
}

/**
 * @brief Recursively builds variable-length codes from the Huffman tree.
 *
 * Traverses left (0) and right (1) to assign each leaf its bit sequence.
 *
 * @param node   Current tree node.
 * @param prefix  Bit path accumulated so far.
 * @param codes  Output table: codes[byte] = vector of 0/1 bits.
 */
void buildCodes(const std::shared_ptr<Node>& node, std::vector<uint8_t>& prefix,
        std::array<std::vector<uint8_t>, 256>& codes) {
  if (!node) return;
  if (node->isLeaf()) {
    if (prefix.empty()) prefix.push_back(0);
    codes[node->symbol] = prefix;
    return;
  }

  prefix.push_back(0);
  buildCodes(node->left, prefix, codes);
  prefix.pop_back();

  prefix.push_back(1);
  buildCodes(node->right, prefix, codes);
  prefix.pop_back();
}

// ── Encoder / Decoder ────────────────────────────────────────────────────────

/**
 * @brief Encodes raw bytes into a Huffman-coded stream.
 *
 * Output format:
 *   [4 bytes]     "HUF1" magic marker
 *   [8 bytes]     original size (uint64, little-endian)
 *   [256 × 8 bytes]  frequency table (one uint64 per byte value)
 *   [variable]    Huffman-coded bitstream (MSB-first packed bytes)
 */
void encode(const std::vector<uint8_t>& input, std::ostream& out) {
  std::array<uint64_t, 256> freq{};
  for (uint8_t b : input) freq[b]++;

  out.write("HUF1", 4);
  writeU64Le(out, static_cast<uint64_t>(input.size()));
  for (int i = 0; i < 256; ++i) writeU64Le(out, freq[i]);

  if (input.empty()) return;

  auto root = buildTree(freq);
  std::array<std::vector<uint8_t>, 256> codes;
  std::vector<uint8_t> prefix;
  buildCodes(root, prefix, codes);

  BitWriter bw(out);
  for (uint8_t b : input) bw.writeCode(codes[b]);
  bw.flush();
}

/**
 * @brief Decodes a Huffman-coded stream back to the original data.
 *
 * Reads the "HUF1" header + frequency table, rebuilds the Huffman tree,
 * and walks the tree bit-by-bit until the original byte count is reached.
 */
std::vector<uint8_t> decode(std::istream& in) {
  char magic[4];
  in.read(magic, 4);
  if (!in.good() || std::string(magic, 4) != "HUF1") {
    throw std::runtime_error("Invalid Huffman stream header");
  }

  uint64_t originalSize = readU64Le(in);
  std::array<uint64_t, 256> freq{};
  for (int i = 0; i < 256; ++i) freq[i] = readU64Le(in);

  std::vector<uint8_t> output;
  output.reserve(static_cast<size_t>(originalSize));
  if (originalSize == 0) return output;

  auto root = buildTree(freq);
  if (!root) throw std::runtime_error("Invalid Huffman frequency table");

  BitReader br(in);
  while (output.size() < originalSize) {
    auto cur = root;
    while (!cur->isLeaf()) {
      int bit = br.readBit();
      if (bit < 0) throw std::runtime_error("Unexpected EOF in Huffman bitstream");
      cur = (bit == 0) ? cur->left : cur->right;
      if (!cur) throw std::runtime_error("Corrupted Huffman tree traversal");
    }
    output.push_back(static_cast<uint8_t>(cur->symbol));
  }

  return output;
}

}  // namespace

// ── Main: Huffman codec CLI ──────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "Huffman codec\n\n"
          << "Usage:\n"
          << "  " << argv[0] << " <mode> <input> <output>\n\n"
          << "Modes:\n"
          << "  c, -c, -compress    Compress (Huffman encode)\n"
          << "  d, -d, -decompress  Decompress (Huffman decode)\n";
    return 1;
  }

  const std::string mode    = argv[1];
  const std::string inputFile  = argv[2];
  const std::string outputFile = argv[3];

  const bool doCompress   = (mode == "c" || mode == "-c" || mode == "-compress");
  const bool doDecompress = (mode == "d" || mode == "-d" || mode == "-decompress");

  try {
    if (doCompress) {
      auto inData = readAllBytes(inputFile);
      std::ofstream out(outputFile, std::ios::binary);
      if (!out) throw std::runtime_error("Cannot open output file: " + outputFile);
      encode(inData, out);
      std::cerr << "  Huffman: " << inData.size() << " -> " << out.tellp()
            << " bytes\n";
      return 0;
    }

    if (doDecompress) {
      std::ifstream in(inputFile, std::ios::binary);
      if (!in) throw std::runtime_error("Cannot open input file: " + inputFile);
      auto outData = decode(in);
      writeAllBytes(outputFile, outData);
      std::cerr << "  Huffman decode: " << outData.size() << " bytes\n";
      return 0;
    }

    std::cerr << "Unknown mode: " << mode << " (use c or d)\n";
    return 1;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 2;
  }
}
