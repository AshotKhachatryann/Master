#ifndef HAR_WAV_PARSER_H
#define HAR_WAV_PARSER_H

#include <cstdint>
#include <string>
#include <vector>

/*!
 * \class Signal
 * \brief Represents an audio signal with sample data and metadata.
 *
 * \details The Signal class encapsulates audio sample data in floating-point format
 * normalized to the range [-1.0, 1.0], along with the sample rate. This class
 * is used as the primary data container for audio I/O operations in the Wav class.
 *
 * \note The samples are stored as 32-bit floating-point values for precision and
 * compatibility with audio processing algorithms. Sample values outside [-1.0, 1.0]
 * may be clipped during WAV file writing.
 */
class Signal {
 public:
  /*!
   * \brief Default constructor.
   * \details Initializes an empty Signal with sampleRate = 0 and no samples.
   */
  Signal() : m_sampleRate(0) {}

  /*!
   * \brief Constructor with sample data and sample rate.
   * \param s The audio samples to store (will be moved, not copied).
   * \param sr The sample rate in Hz for this signal.
   */
  Signal(std::vector<float> s, int sr) : m_samples(std::move(s)), m_sampleRate(sr) {}

  /*!
   * \brief Returns the total number of audio samples.
   * \return The size of the samples vector.
   */
  std::size_t numSamples() const { return m_samples.size(); }

  /*!
   * \brief Returns the audio samples (mutable).
   * \return Reference to the internal samples vector.
   */
  std::vector<float>& getSamples() { return m_samples; }

  /*!
   * \brief Returns the audio samples (const).
   * \return Const reference to the internal samples vector.
   */
  const std::vector<float>& getSamples() const { return m_samples; }

  /*!
   * \brief Replaces the audio samples.
   * \param s The new sample vector (moved in).
   */
  void setSamples(std::vector<float> s) { m_samples = std::move(s); }

  /*!
   * \brief Returns the sample rate in Hz.
   * \return The sample rate.
   */
  int getSampleRate() const { return m_sampleRate; }

  /*!
   * \brief Sets the sample rate in Hz.
   * \param sr The new sample rate.
   */
  void setSampleRate(int sr) { m_sampleRate = sr; }

 private:
  //! Audio sample data: floating-point values in the range [-1.0, 1.0]
  std::vector<float> m_samples;

  //! Sample rate in Hz (e.g., 44100, 48000, 96000)
  int m_sampleRate;
};

/*!
 * \class Wav
 * \brief Static utility class for reading and writing WAV audio files.
 *
 * \details The Wav class provides static methods for:
 * - Reading WAV files from disk and converting to Signal objects
 * - Writing Signal objects to WAV files in 16-bit PCM mono format
 * - Constructing file paths for organized WAV storage
 *
 * The implementation supports:
 * - PCM (0x0001) audio format
 * - IEEE float (0x0003) audio format
 * - Extensible WAV format (0xFFFE) with format detection
 * - Variable bit depths (8, 16, 24, 32 bits)
 * - Multi-channel input (flattened to mono during reading)
 * - Proper little-endian byte handling for binary WAV structure
 *
 * \note All file paths are specified as absolute or relative strings.
 * For Application.dataPath equivalents, pass the base data path explicitly.
 */
class Wav {
 public:
  /*!
   * \brief Reads a WAV file from disk and returns its audio data.
   *
   * \param path The file system path to the WAV file (absolute or relative).
   *
   * \return A Signal object containing:
   *   - Audio samples normalized to [-1.0, 1.0]
   *   - Sample rate extracted from the WAV file header
   *
   * \details This method:
   * 1. Opens the file in binary mode
   * 2. Parses the RIFF/WAVE structure to locate fmt and data chunks
   * 3. Reads format information (channels, bit depth, sample rate)
   * 4. Extracts raw audio data and converts to floating-point
   * 5. Flattens multi-channel audio to mono (takes first channel only)
   * 6. Normalizes samples to [-1.0, 1.0] range
   *
   * \throws std::runtime_error If:
   *   - File cannot be opened
   *   - RIFF/WAVE structure is invalid
   *   - Required chunks (fmt, data) are missing
   *   - File read operations fail
   *
   * \note Supports reading from files with various formats:
   *   - PCM: 8, 16, 24, 32-bit
   *   - IEEE float: 32-bit
   *   - Extensible WAV format with automatic format detection
   */
  static Signal readWav(const std::string& path);

  /*!
   * \brief Writes a Signal object to a WAV file in 16-bit PCM mono format.
   *
   * \param signal The Signal object containing audio samples and sample rate.
   * \param path The destination file path (will be created or overwritten).
   *
   * \details This method:
   * 1. Truncates or creates the output file
   * 2. Writes RIFF container header (file format identification)
   * 3. Writes fmt chunk with format metadata:
   *    - Format code: PCM (0x0001)
   *    - Channels: 1 (mono)
   *    - Sample rate from Signal object
   *    - Bit depth: 16-bit per sample
   * 4. Writes data chunk with audio samples:
   *    - Clamps floating-point values to [-1.0, 1.0]
   *    - Converts to signed 16-bit integers
   *    - Encodes as little-endian byte pairs
   *
   * \throws std::runtime_error If file cannot be created or write operations fail.
   *
   * \note Output is always 16-bit PCM mono. Multi-channel or compressed formats
   * are not supported. Sample values outside [-1.0, 1.0] are automatically clamped
   * to prevent integer overflow.
   */
  static void writeWav(const Signal& signal, const std::string& path);

  /*!
   * \brief Constructs a file path for WAV storage with directory hierarchy.
   *
   * \param baseDataPath The base directory path (e.g., application data directory).
   * \param directory A subdirectory name within the Wav folder (e.g., "speech", "music").
   * \param fileName The base file name without extension (e.g., "recording_001").
   *
   * \return A complete file path: baseDataPath/Wav/directory/fileName.wav
   *
   * \details Constructs paths using platform-independent std::filesystem::path
   * handling, ensuring correct path separators on Windows, macOS, and Linux.
   *
   * \note The .wav extension is automatically appended to the file name.
   */
  static std::string getWavPath(const std::string& baseDataPath,
                                const std::string& directory,
                                const std::string& fileName);

  /*!
   * \brief Constructs a file path for WAV storage without directory hierarchy.
   *
   * \param baseDataPath The base directory path (e.g., application data directory).
   * \param fileName The base file name without extension (e.g., "recording_001").
   *
   * \return A complete file path: baseDataPath/Wav/fileName.wav
   *
   * \details Constructs paths using platform-independent std::filesystem::path
   * handling, ensuring correct path separators on Windows, macOS, and Linux.
   * This overload is useful when subdirectory organization is not needed.
   *
   * \note The .wav extension is automatically appended to the file name.
   */
  static std::string getWavPath(const std::string& baseDataPath,
                                const std::string& fileName);
};

#endif
