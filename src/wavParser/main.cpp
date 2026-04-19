#include "WavParser.h"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>

/*!
 * \brief Displays signal metadata and statistics.
 *
 * \param signal The Signal object to display.
 * \param label Optional label for the signal.
 */
void displaySignalInfo(const Signal& signal, const std::string& label = "Signal") {
  std::cout << "\n" << std::string(60, '=') << "\n";
  std::cout << label << " Information:\n";
  std::cout << std::string(60, '=') << "\n";
  std::cout << "Sample Rate:        " << signal.getSampleRate() << " Hz\n";
  std::cout << "Total Samples:      " << signal.numSamples() << "\n";

  // Guard against division by zero when sample rate is 0 (default-constructed Signal).
  if (signal.getSampleRate() > 0) {
    std::cout << "Duration:           " << std::fixed << std::setprecision(2)
              << (static_cast<double>(signal.numSamples()) / signal.getSampleRate()) << " seconds\n";
  } else {
    std::cout << "Duration:           N/A (sample rate is 0)\n";
  }

  if (!signal.getSamples().empty()) {
    float minVal = signal.getSamples()[0];
    float maxVal = signal.getSamples()[0];
    double sumSquares = 0.0;

    for (float sample : signal.getSamples()) {
      minVal = std::min(minVal, sample);
      maxVal = std::max(maxVal, sample);
      sumSquares += static_cast<double>(sample) * static_cast<double>(sample);
    }

    float rms = std::sqrt(sumSquares / signal.numSamples());
    std::cout << "Min Sample Value:   " << std::setprecision(6) << minVal << "\n";
    std::cout << "Max Sample Value:   " << std::setprecision(6) << maxVal << "\n";
    std::cout << "RMS Level:          " << std::setprecision(6) << rms << "\n";
  }
  std::cout << std::string(60, '=') << "\n\n";
}

/*!
 * \brief Demonstrates reading a WAV file.
 *
 * \param filePath The path to the WAV file to read.
 */
void demoReadWav(const std::string& filePath) {
  std::cout << "\n>>> DEMO: Reading WAV File <<<\n";
  std::cout << "File Path: " << filePath << "\n\n";

  try {
    // Read WAV file
    Signal signal = Wav::readWav(filePath);

    // Display signal information
    displaySignalInfo(signal, "Read Signal");

    std::cout << "Status: Successfully read " << signal.numSamples() << " samples\n";
    std::cout << "        at " << signal.getSampleRate() << " Hz sample rate.\n";

  } catch (const std::exception& e) {
    std::cerr << "Error reading WAV file: " << e.what() << "\n";
  }
}

/*!
 * \brief Demonstrates writing a WAV file.
 *
 * \param outputPath The destination path for the WAV file.
 * \param sampleRate The sample rate for the generated signal.
 * \param durationSeconds The duration of the signal to generate.
 */
void demoWriteWav(const std::string& outputPath, int sampleRate = 44100,
                  double durationSeconds = 2.0) {
  std::cout << "\n>>> DEMO: Writing WAV File <<<\n";
  std::cout << "Output Path:  " << outputPath << "\n";
  std::cout << "Sample Rate:  " << sampleRate << " Hz\n";
  std::cout << "Duration:     " << durationSeconds << " seconds\n\n";

  try {
    // Generate a test signal: 440 Hz sine wave (musical note A4).
    // Amplitude 0.5 keeps headroom below clipping (±1.0).
    const int numSamples = static_cast<int>(sampleRate * durationSeconds);
    const double frequency = 440.0;  // Hz
    const double twoPi = 2.0 * M_PI;

    std::vector<float> samples(numSamples);
    for (int i = 0; i < numSamples; ++i) {
      double t = static_cast<double>(i) / sampleRate;
      samples[i] = static_cast<float>(0.5 * std::sin(twoPi * frequency * t));
    }

    Signal signal(std::move(samples), sampleRate);

    // Display generated signal info
    displaySignalInfo(signal, "Generated Signal (440 Hz sine wave)");

    // Write signal to WAV file
    Wav::writeWav(signal, outputPath);

    std::cout << "Status: Successfully wrote " << signal.numSamples() << " samples\n";
    std::cout << "        to file: " << outputPath << "\n";

  } catch (const std::exception& e) {
    std::cerr << "Error writing WAV file: " << e.what() << "\n";
  }
}

/*!
 * \brief Demonstrates round-trip: write, read, and verify a WAV file.
 *
 * \param filePath The path to use for the round-trip test.
 */
void demoRoundTrip(const std::string& filePath) {
  std::cout << "\n>>> DEMO: Round-Trip Test (Write → Read → Verify) <<<\n";
  std::cout << "File Path: " << filePath << "\n\n";

  try {
    // Step 1: Generate and write
    std::cout << "[1/3] Generating test signal (1 second, 48000 Hz, 880 Hz sine)...\n";
    const int sampleRate = 48000;
    const int numSamples = sampleRate;  // 1 second of audio
    const double frequency = 880.0;     // Hz — one octave above A4
    const double twoPi = 2.0 * M_PI;

    std::vector<float> originalSamples(numSamples);
    for (int i = 0; i < numSamples; ++i) {
      double t = static_cast<double>(i) / sampleRate;
      originalSamples[i] = static_cast<float>(0.3 * std::sin(twoPi * frequency * t));
    }

    Signal originalSignal(std::move(originalSamples), sampleRate);
    displaySignalInfo(originalSignal, "Original Signal");

    std::cout << "[2/3] Writing to WAV file...\n";
    Wav::writeWav(originalSignal, filePath);
    std::cout << "      File written successfully.\n";

    // Step 2: Read back
    std::cout << "[3/3] Reading back from WAV file...\n";
    Signal readSignal = Wav::readWav(filePath);
    displaySignalInfo(readSignal, "Read Signal (from file)");

    // Step 3: Verify metadata.
    // Note: Only sample count and rate are compared here.  Exact sample
    // values are not checked because writeWav() quantises to 16-bit PCM,
    // so the read-back floats will differ from the originals by up to
    // ±1/32768 due to rounding.  A tolerance-based comparison would be
    // needed for full sample-level verification.
    std::cout << "Verification:\n";
    std::cout << "  Original samples:  " << originalSignal.numSamples() << "\n";
    std::cout << "  Read samples:      " << readSignal.numSamples() << "\n";
    std::cout << "  Original rate:     " << originalSignal.getSampleRate() << " Hz\n";
    std::cout << "  Read rate:         " << readSignal.getSampleRate() << " Hz\n";

    if (originalSignal.numSamples() == readSignal.numSamples() &&
      originalSignal.getSampleRate() == readSignal.getSampleRate()) {
      std::cout << "  ✓ Round-trip successful!\n";
    } else {
      std::cout << "  ✗ Mismatch detected!\n";
    }

  } catch (const std::exception& e) {
    std::cerr << "Error during round-trip test: " << e.what() << "\n";
  }
}

/*!
 * \brief Demonstrates path construction utilities.
 *
 * \param baseDataPath The base data directory path.
 */
void demoPathConstruction(const std::string& baseDataPath) {
  std::cout << "\n>>> DEMO: Path Construction <<<\n";
  std::cout << "Base Data Path: " << baseDataPath << "\n\n";

  // Example 1: Path with directory hierarchy
  std::string path1 = Wav::getWavPath(baseDataPath, "speech", "recording_001");
  std::cout << "getWavPath(base, \"speech\", \"recording_001\"):\n";
  std::cout << "  → " << path1 << "\n\n";

  // Example 2: Path without subdirectory
  std::string path2 = Wav::getWavPath(baseDataPath, "audio_output");
  std::cout << "getWavPath(base, \"audio_output\"):\n";
  std::cout << "  → " << path2 << "\n\n";

  // Example 3: Various subdirectories
  std::vector<std::string> categories = {"music", "voice", "effects", "background"};
  std::cout << "Multiple categories:\n";
  for (const auto& category : categories) {
    std::string path = Wav::getWavPath(baseDataPath, category, "sample");
    std::cout << "  " << category << ": " << path << "\n";
  }
}

/*!
 * \brief Prints usage information for command-line arguments.
 *
 * \param programName The name of the executable.
 */
void printUsage(const std::string& programName) {
  std::cout << "\nUsage: " << programName << " [COMMAND] [OPTIONS]\n\n";
  std::cout << "Commands:\n";
  std::cout << "  --read <filepath>          Read and display a WAV file\n";
  std::cout << "  --write <output> [rate]    Generate and write a test WAV file\n";
  std::cout << "                             (optional: sample rate, default 44100)\n";
  std::cout << "  --process <input> <output> Read, apply gain (0.5x), and write\n";
  std::cout << "  --roundtrip <filepath>     Write, read, and verify round-trip\n";
  std::cout << "  --demo                     Run all demonstrations\n\n";
  std::cout << "Examples:\n";
  std::cout << "  " << programName << " --read /path/to/audio.wav\n";
  std::cout << "  " << programName << " --write ./output.wav 48000\n";
  std::cout << "  " << programName << " --process input.wav output.wav\n";
  std::cout << "  " << programName << " --roundtrip test.wav\n";
  std::cout << "  " << programName << " --demo\n\n";
}

/*!
 * \brief Main function demonstrating all WAV parser functionality.
 *
 * Supports command-line arguments for different operations:
 * - read <filepath>: Read a WAV file
 * - write <output> [rate]: Generate and write a WAV file
 * - process <input> <output>: Read, process, and write
 * - roundtrip <filepath>: Test round-trip write/read
 * - demo: Run all demonstrations (default)
 *
 * \param argc Number of command-line arguments.
 * \param argv Array of command-line argument strings.
 * \return 0 on success, non-zero on error.
 */
int main(int argc, char* argv[])
{
  std::string programName = (argc > 0) ? argv[0] : "wav_parser";

  // Parse command-line arguments.
  // When invoked with no arguments the full demo suite runs (5 demos
  // including signal processing).  The --demo flag runs a shorter subset
  // (4 demos, no signal processing step).
  if (argc < 2) {
    std::cout << "\n";
    std::cout << std::string(60, '*') << "\n";
    std::cout << "   WAV Audio File Parser - Comprehensive Usage Demo\n";
    std::cout << std::string(60, '*') << "\n";

    // Example base path (in real usage, this would be your application's data directory)
    std::string baseDataPath = "./audio_data";

    // Demo 1: Path Construction
    demoPathConstruction(baseDataPath);

    // Demo 2: Write a test WAV file
    std::string testFilePath = "./test_signal.wav";
    demoWriteWav(testFilePath, 44100, 3.0);

    // Demo 3: Read the test WAV file
    demoReadWav(testFilePath);

    // Demo 4: Round-trip test
    std::string roundTripPath = "./round_trip_test.wav";
    demoRoundTrip(roundTripPath);

    // Demo 5: Advanced example - Process signal
    std::cout << "\n>>> DEMO: Signal Processing Example <<<\n";
    try {
      // Read a signal
      Signal signal = Wav::readWav(testFilePath);

      // Apply a simple gain adjustment (volume change).
      // Linear gain of 0.5 corresponds to approximately −6 dB.
      const float gainFactor = 0.5f;
      std::cout << "Applying gain factor: " << gainFactor << " (50% volume reduction / -6 dB)\n";
      for (float& sample : signal.getSamples()) {
        sample *= gainFactor;
      }

      // Write the processed signal
      std::string processedPath = "./processed_signal.wav";
      Wav::writeWav(signal, processedPath);

      std::cout << "Processed signal saved to: " << processedPath << "\n";
      displaySignalInfo(signal, "Processed Signal (with gain)");

    } catch (const std::exception& e) {
      std::cerr << "Error in signal processing: " << e.what() << "\n";
    }

    std::cout << std::string(60, '*') << "\n";
    std::cout << "   Demo Complete\n";
    std::cout << std::string(60, '*') << "\n\n";

    return 0;
  }

  // Parse command argument
  std::string command = argv[1];

  if (command == "--help" || command == "-h" || command == "help") {
    printUsage(programName);
    return 0;
  }

  if (command == "--demo") {
    // Run all demonstrations
    std::cout << "\n";
    std::cout << std::string(60, '*') << "\n";
    std::cout << "   WAV Audio File Parser - Comprehensive Usage Demo\n";
    std::cout << std::string(60, '*') << "\n";

    std::string baseDataPath = "./audio_data";
    demoPathConstruction(baseDataPath);

    std::string testFilePath = "./test_signal.wav";
    demoWriteWav(testFilePath, 44100, 3.0);
    demoReadWav(testFilePath);

    std::string roundTripPath = "./round_trip_test.wav";
    demoRoundTrip(roundTripPath);

    std::cout << std::string(60, '*') << "\n";
    std::cout << "   Demo Complete\n";
    std::cout << std::string(60, '*') << "\n\n";

    return 0;
  }

  if (command == "--read") {
    if (argc < 3) {
      std::cerr << "Error: '--read' command requires a file path.\n";
      printUsage(programName);
      return 1;
    }
    demoReadWav(argv[2]);
    return 0;
  }

  if (command == "--write") {
    if (argc < 3) {
      std::cerr << "Error: '--write' command requires an output file path.\n";
      printUsage(programName);
      return 1;
    }
    int sampleRate = 44100;
    if (argc >= 4) {
      try {
        sampleRate = std::stoi(argv[3]);
      } catch (const std::exception& e) {
        std::cerr << "Error: Invalid sample rate: " << argv[3] << "\n";
        return 1;
      }
    }
    demoWriteWav(argv[2], sampleRate, 2.0);
    return 0;
  }

  if (command == "--process") {
    if (argc < 4) {
      std::cerr << "Error: '--process' command requires input and output file paths.\n";
      printUsage(programName);
      return 1;
    }
    std::string inputPath = argv[2];
    std::string outputPath = argv[3];

    std::cout << "\n>>> DEMO: Signal Processing Example <<<\n";
    try {
      Signal signal = Wav::readWav(inputPath);
      displaySignalInfo(signal, "Input Signal");

      const float gainFactor = 0.5f;  // 50% volume reduction
      std::cout << "Applying gain factor: " << gainFactor << " (50% volume reduction)\n";
      for (float& sample : signal.getSamples()) {
        sample *= gainFactor;
      }

      Wav::writeWav(signal, outputPath);
      std::cout << "Processed signal saved to: " << outputPath << "\n";
      displaySignalInfo(signal, "Output Signal (processed)");

    } catch (const std::exception& e) {
      std::cerr << "Error in signal processing: " << e.what() << "\n";
      return 1;
    }
    return 0;
  }

  if (command == "--roundtrip") {
    if (argc < 3) {
      std::cerr << "Error: '--roundtrip' command requires a file path.\n";
      printUsage(programName);
      return 1;
    }
    demoRoundTrip(argv[2]);
    return 0;
  }

  // Unknown command
  std::cerr << "Error: Unknown command '" << command << "'.\n";
  printUsage(programName);
  return 1;
}
