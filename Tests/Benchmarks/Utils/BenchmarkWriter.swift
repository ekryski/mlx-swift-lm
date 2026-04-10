import Foundation
import MLX

/// Writes benchmark results to markdown files in benchmarks/.
///
/// Each run creates one file per model with context scaling table and special test results.
/// Filename: `YYYY-MM-DD-HHmm-{model-slug}-benchmark.md`
enum BenchmarkWriter {
    private static let lock = NSLock()
    /// Track which files have been initialized (header written) this session
    nonisolated(unsafe) private static var initializedFiles: Set<String> = []

    /// Generation parameters used for this benchmark run.
    struct BenchmarkParameters {
        let temperature: Float
        let topP: Float
        let topK: Int
        let minP: Float
        let maxTokens: Int?
        let thinkingBudget: Int?
        let repetitionPenalty: Float?
        let presencePenalty: Float?
        let reasoningEffort: String?
        /// Effective thinking mode (model supports it and MLX_BENCH_THINK=1).
        let thinkingEnabled: Bool
        /// Generation-time perplexity tracking (MLX_BENCH_PPL).
        let perplexityTrackingEnabled: Bool
        /// KL divergence env + whether it applies to this benchmark method/config.
        let kldSummary: String
        /// MLX_MAX_OPS_PER_BUFFER env value, or "default" when unset.
        let maxOpsPerBuffer: String
        let batchSize: Int
        /// e.g. none, ngram (size=…), draft (repo-id).
        let speculativeDecoding: String
    }

    /// Append a benchmark result row to the model's markdown file.
    static func append(
        model: String,
        repoId: String = "",
        quantization: String,
        kvConfig: String,
        scenario: String,
        contextSize: Int,
        promptTokens: Int,
        prefillTokPerSec: Double,
        genTokPerSec: Double,
        genTokens: Int,
        ttftMs: Double,
        thinkingPerplexity: Double?,
        generationPerplexity: Double?,
        thinkingKLD: Double? = nil,
        generationKLD: Double? = nil,
        baselineGPU: Int,
        peakGPU: Int,
        kvDelta: Int,
        kvCacheBytes: Int = 0,
        outputPreview: String,
        parameters: BenchmarkParameters? = nil
    ) {
        let slug = model
            .replacingOccurrences(of: " ", with: "-")
            .lowercased()
            .replacingOccurrences(of: "/", with: "-")
        let dir = benchmarkDir(modelSlug: slug)
        let dateStr = Self.sessionDateString
        let method = ProcessInfo.processInfo.environment["MLX_BENCH_METHOD"] ?? "simple"
        let filename = "\(slug)-\(quantization)-\(kvConfig)-\(method)-benchmark-\(dateStr).md"
        let path = dir.appendingPathComponent(filename)

        lock.lock()
        defer { lock.unlock() }

        // Write header on first append for this file
        if !initializedFiles.contains(filename) {
            initializedFiles.insert(filename)

            let branch = gitBranch()
            let commit = gitLastCommitLine()
            let hw = hardwareInfo()

            var header = "# Inference Benchmark - \(model)\n\n"
            header += "**Date**: \(Self.humanDateString)\n"
            header += "**Branch**: `\(branch)`\n"
            header += "**Commit**: \(commit)\n"
            header += "**Quantization**: \(quantization)\n"
            if !repoId.isEmpty {
                header += "**Model**: `\(repoId)`\n"
            }
            header += "\n"
            header += "## Hardware\n\n"
            header += "| Property | Value |\n"
            header += "|----------|-------|\n"
            header += "| Chip | \(hw.chip) |\n"
            header += "| System RAM | \(hw.systemRAM) |\n"
            header += "| GPU Memory Limit | \(hw.gpuLimit) |\n"
            header += "| macOS | \(hw.osVersion) |\n"
            header += "\n"
            if let p = parameters {
                header += "## Parameters\n\n"
                header += "| Parameter | Value |\n"
                header += "|-----------|-------|\n"
                header += "| Temperature | \(p.temperature) |\n"
                header += "| Top P | \(p.topP) |\n"
                header += "| Top K | \(p.topK) |\n"
                header += "| Min P | \(p.minP) |\n"
                if let max = p.maxTokens { header += "| Max Tokens | \(max) |\n" }
                if let budget = p.thinkingBudget { header += "| Thinking Budget | \(budget) |\n" }
                if let effort = p.reasoningEffort { header += "| Reasoning Effort | \(effort) |\n" }
                if let rep = p.repetitionPenalty { header += "| Repetition Penalty | \(rep) |\n" }
                if let pres = p.presencePenalty { header += "| Presence Penalty | \(pres) |\n" }
                header += "| Thinking | \(p.thinkingEnabled ? "Yes" : "No") |\n"
                header += "| Perplexity tracking (MLX_BENCH_PPL) | \(p.perplexityTrackingEnabled ? "Yes" : "No") |\n"
                header += "| KL divergence (MLX_BENCH_KLD) | \(p.kldSummary) |\n"
                header += "| Batch size (MLX_BENCH_BATCH) | \(p.batchSize) |\n"
                header += "| Speculative decoding | \(p.speculativeDecoding) |\n"
                header += "| Max ops per buffer (MLX_MAX_OPS_PER_BUFFER) | \(p.maxOpsPerBuffer) |\n"
                header += "\n"
            }
            header += "## Methodology\n\n"
            header += "For details see [here](../README.md#methodology).\n"
            header += "\n"
            header += "## Results\n\n"
            header += "| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |\n"
            header += "|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|\n"

            try? header.write(to: path, atomically: true, encoding: .utf8)
        }

        // Append row + full output in collapsible block
        let tablePreview = String(outputPreview.prefix(60))
            .replacingOccurrences(of: "|", with: "\\|")
            .replacingOccurrences(of: "\n", with: " ")
        let contextStr = contextSize > 0 ? "\(contextSize)" : "—"
        let thinkPplStr = thinkingPerplexity.map { String(format: "%.4f", $0) } ?? "—"
        let genPplStr = generationPerplexity.map { String(format: "%.4f", $0) } ?? "—"
        let thinkKldStr = thinkingKLD.map { String(format: "%.4f", $0) } ?? "—"
        let genKldStr = generationKLD.map { String(format: "%.4f", $0) } ?? "—"
        let kvCacheStr = kvCacheBytes > 0 ? formatBytes(kvCacheBytes) : "—"
        let content = "| \(scenario) | \(contextStr) | \(promptTokens) | \(kvConfig) | \(String(format: "%.1f", prefillTokPerSec)) | \(String(format: "%.1f", genTokPerSec)) | \(genTokens) | \(String(format: "%.0f", ttftMs))ms | \(thinkPplStr) | \(genPplStr) | \(thinkKldStr) | \(genKldStr) | \(formatBytes(baselineGPU)) | \(formatBytes(peakGPU)) | \(formatBytes(kvDelta)) | \(kvCacheStr) | \(tablePreview) |\n"

        if let handle = try? FileHandle(forWritingTo: path) {
            handle.seekToEndOfFile()
            handle.write(content.data(using: .utf8)!)
            handle.closeFile()
        }
    }

    // MARK: - Helpers

    private static func benchmarkDir(modelSlug: String) -> URL {
        // Route to model family subfolder: "qwen3.5-0.8b-4bit" → "qwen3.5-0.8b"
        // Strip the quantization suffix (last hyphenated component like -bf16, -4bit, -8bit)
        let quantSuffixes = ["-bf16", "-8bit", "-4bit", "-nvfp4", "-mxfp4"]
        var familySlug = modelSlug
        for suffix in quantSuffixes {
            if familySlug.hasSuffix(suffix) {
                familySlug = String(familySlug.dropLast(suffix.count))
                break
            }
        }
        let dir = projectRoot()
            .appendingPathComponent("benchmarks")
            .appendingPathComponent(familySlug)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static func projectRoot() -> URL {
        // Walk up from the build directory to find Package.swift
        var dir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        for _ in 0..<5 {
            if FileManager.default.fileExists(atPath: dir.appendingPathComponent("Package.swift").path) {
                return dir
            }
            dir = dir.deletingLastPathComponent()
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    }

    static func gitBranch() -> String {
        let root = projectRoot()
        let headPath = root.appendingPathComponent(".git/HEAD").path
        return (try? String(contentsOfFile: headPath, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "ref: refs/heads/", with: "") ?? "unknown"
    }

    /// Short hash and subject for the current HEAD, for benchmark provenance.
    private static func gitLastCommitLine() -> String {
        let root = projectRoot()
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = ["-C", root.path, "log", "-1", "--format=%h %s"]
        let out = Pipe()
        process.standardOutput = out
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
            process.waitUntilExit()
            let data = out.fileHandleForReading.readDataToEndOfFile()
            guard process.terminationStatus == 0,
                  var line = String(data: data, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                  !line.isEmpty
            else {
                return "`unknown`"
            }
            line = line.replacingOccurrences(of: "`", with: "'")
            line = line.replacingOccurrences(of: "|", with: "\\|")
            return "`\(line)`"
        } catch {
            return "`unknown`"
        }
    }

    /// Session-level date string (same for all results in one test run)
    nonisolated(unsafe) private static var _sessionDate: String?
    private static var sessionDateString: String {
        if let cached = _sessionDate { return cached }
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd-HHmm"
        let s = fmt.string(from: Date())
        _sessionDate = s
        return s
    }

    private static var humanDateString: String {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd HH:mm"
        return fmt.string(from: Date())
    }

    struct HardwareInfo {
        let chip: String
        let systemRAM: String
        let gpuLimit: String
        let osVersion: String
    }

    static func hardwareInfo() -> HardwareInfo {
        let info = GPU.deviceInfo()
        let ramGB = Double(info.memorySize) / 1_073_741_824
        let gpuGB = Double(info.maxRecommendedWorkingSetSize) / 1_073_741_824
        let os = ProcessInfo.processInfo.operatingSystemVersion
        let chipName = humanReadableChipName(gpuArch: info.architecture)
        return HardwareInfo(
            chip: chipName,
            systemRAM: String(format: "%.0fGB", ramGB),
            gpuLimit: String(format: "%.0fGB", gpuGB),
            osVersion: "\(os.majorVersion).\(os.minorVersion).\(os.patchVersion)"
        )
    }

    /// Returns a human-readable chip name combined with the GPU architecture ID.
    /// e.g. "Apple M1 Max (applegpu_g13s)"
    private static func humanReadableChipName(gpuArch: String) -> String {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        var brand = [CChar](repeating: 0, count: size)
        sysctlbyname("machdep.cpu.brand_string", &brand, &size, nil, 0)
        let brandStr = String(cString: brand)

        // On Apple Silicon, brand_string is empty — use hw.model instead
        if !brandStr.isEmpty {
            return "\(brandStr) (\(gpuArch))"
        }

        size = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        var model = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.model", &model, &size, nil, 0)
        let modelStr = String(cString: model)

        // Map hw.model identifiers to marketing names
        let marketingName = appleChipName(from: modelStr, gpuArch: gpuArch)
        return "\(marketingName) (\(gpuArch))"
    }

    /// Derive a marketing-style chip name from hw.model + GPU arch.
    private static func appleChipName(from model: String, gpuArch: String) -> String {
        // GPU arch suffix encodes the die variant:
        // g13  = M1 family, g14 = M2 family, g15 = M3 family, g16 = M4 family
        // s = standard, d = Pro, x = Max, c = Ultra
        let archLower = gpuArch.lowercased()
        let gen: String
        if archLower.contains("g13") { gen = "M1" }
        else if archLower.contains("g14") { gen = "M2" }
        else if archLower.contains("g15") { gen = "M3" }
        else if archLower.contains("g16") { gen = "M4" }
        else { return model.isEmpty ? gpuArch : model }

        let variant: String
        if archLower.hasSuffix("c") { variant = "Ultra" }
        else if archLower.hasSuffix("x") { variant = "Max" }
        else if archLower.hasSuffix("d") { variant = "Pro" }
        else { variant = "" }

        return variant.isEmpty ? "Apple \(gen)" : "Apple \(gen) \(variant)"
    }

    static func formatBytes(_ bytes: Int) -> String {
        if bytes >= 1_073_741_824 {
            return String(format: "%.2fGB", Double(bytes) / 1_073_741_824)
        } else {
            return String(format: "%.0fMB", Double(bytes) / 1_048_576)
        }
    }
}
