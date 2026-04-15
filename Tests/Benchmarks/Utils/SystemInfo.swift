import Foundation
import MLX
import MLXHuggingFace

/// Hardware detection and memory-aware model variant selection.
enum SystemInfo {

    struct Hardware: CustomStringConvertible {
        let architecture: String
        let systemMemoryGB: Double
        let gpuMemoryLimitGB: Double

        var description: String {
            "\(architecture), \(String(format: "%.0f", systemMemoryGB))GB RAM, \(String(format: "%.0f", gpuMemoryLimitGB))GB GPU limit"
        }
    }

    /// Query current hardware info via Metal / sysctl.
    static func hardware() -> Hardware {
        let info = GPU.deviceInfo()
        return Hardware(
            architecture: info.architecture,
            systemMemoryGB: Double(info.memorySize) / 1_073_741_824,
            gpuMemoryLimitGB: Double(info.maxRecommendedWorkingSetSize) / 1_073_741_824
        )
    }

    /// Estimate model weight size by summing .safetensors file sizes from HuggingFace Hub metadata.
    /// Uses the HuggingFace API to list repo files, then sums safetensors sizes.
    static func estimateModelSize(repo: String) async throws -> Int {
        // Use the HF API to list files and their sizes
        let urlString = "https://huggingface.co/api/models/\(repo)"
        guard let url = URL(string: urlString) else {
            throw URLError(.badURL)
        }
        let (data, _) = try await URLSession.shared.data(from: url)
        // Parse the JSON response to extract sibling file sizes
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let siblings = json["siblings"] as? [[String: Any]]
        else {
            return 0
        }
        // Sum sizes of .safetensors files
        return siblings
            .filter { ($0["rfilename"] as? String)?.hasSuffix(".safetensors") == true }
            .compactMap { $0["size"] as? Int }
            .reduce(0, +)
    }

    /// Check if a model fits in available GPU memory.
    /// Adds a 2GB overhead estimate for KV cache, workspace, and framework buffers.
    static func fitsInMemory(modelSizeBytes: Int, hardware: Hardware) -> Bool {
        let overheadBytes = 2 * 1_073_741_824  // 2GB
        let totalNeeded = Double(modelSizeBytes + overheadBytes)
        return totalNeeded < hardware.gpuMemoryLimitGB * 1_073_741_824
    }

    /// Format bytes as a human-readable string.
    static func formatGB(_ bytes: Int) -> String {
        String(format: "%.1fGB", Double(bytes) / 1_073_741_824)
    }

    /// Print hardware info as [BENCH] lines.
    static func printHardwareInfo() {
        let hw = hardware()
        print("[BENCH] Hardware: \(hw.architecture)")
        print("[BENCH] System RAM: \(String(format: "%.0f", hw.systemMemoryGB))GB")
        print("[BENCH] GPU Memory Limit: \(String(format: "%.0f", hw.gpuMemoryLimitGB))GB")
    }
}
