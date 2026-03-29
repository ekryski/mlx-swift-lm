import Foundation
import Hub
import MLX

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
    /// Uses HTTP HEAD requests — does not download the model.
    static func estimateModelSize(repo: String) async throws -> Int {
        let hub = HubApi()
        let repoObj = Hub.Repo(id: repo)
        let metadata = try await hub.getFileMetadata(
            from: repoObj, matching: ["*.safetensors"]
        )
        return metadata.compactMap(\.size).reduce(0, +)
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
