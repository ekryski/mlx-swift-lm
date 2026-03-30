import Foundation

/// Writes benchmark results to markdown files in benchmarks/.
///
/// Each run creates one file per model with context scaling table and special test results.
/// Filename: `YYYY-MM-DD-HHmm-{model-slug}-benchmark.md`
enum BenchmarkWriter {
    private static let lock = NSLock()
    /// Track which files have been initialized (header written) this session
    nonisolated(unsafe) private static var initializedFiles: Set<String> = []

    /// Append a benchmark result row to the model's markdown file.
    static func append(
        model: String,
        quantization: String,
        kvConfig: String,
        scenario: String,
        contextTokens: Int,
        prefillTokPerSec: Double,
        genTokPerSec: Double,
        genTokens: Int,
        ttftMs: Double,
        perplexity: Double?,
        baselineGPU: Int,
        peakGPU: Int,
        kvDelta: Int,
        outputPreview: String
    ) {
        let slug = model
            .replacingOccurrences(of: " ", with: "-")
            .lowercased()
            .replacingOccurrences(of: "/", with: "-")
        let dir = benchmarkDir()
        let dateStr = Self.sessionDateString
        let filename = "\(dateStr)-\(slug)-benchmark.md"
        let path = dir.appendingPathComponent(filename)

        lock.lock()
        defer { lock.unlock() }

        // Write header on first append for this file
        if !initializedFiles.contains(filename) {
            initializedFiles.insert(filename)

            let branch = gitBranch()

            var header = "# Inference Benchmark - \(model)\n\n"
            header += "**Date**: \(Self.humanDateString)\n"
            header += "**Branch**: `\(branch)`\n"
            header += "**Quantization**: \(quantization)\n\n"
            header += "## Results\n\n"
            header += "| Scenario | Context | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Perplexity | GPU Baseline | GPU Peak | KV Delta | Output |\n"
            header += "|----------|---------|-----------|---------------|-----------|------------|------|------------|-------------|----------|----------|--------|\n"

            try? header.write(to: path, atomically: true, encoding: .utf8)
        }

        // Append row + full output in collapsible block
        let tablePreview = String(outputPreview.prefix(60))
            .replacingOccurrences(of: "|", with: "\\|")
            .replacingOccurrences(of: "\n", with: " ")
        let pplStr = perplexity.map { String(format: "%.4f", $0) } ?? "—"
        let content = "| \(scenario) | \(contextTokens) | \(kvConfig) | \(String(format: "%.1f", prefillTokPerSec)) | \(String(format: "%.1f", genTokPerSec)) | \(genTokens) | \(String(format: "%.0f", ttftMs))ms | \(pplStr) | \(formatBytes(baselineGPU)) | \(formatBytes(peakGPU)) | \(formatBytes(kvDelta)) | \(tablePreview) |\n"

        if let handle = try? FileHandle(forWritingTo: path) {
            handle.seekToEndOfFile()
            handle.write(content.data(using: .utf8)!)
            handle.closeFile()
        }
    }

    // MARK: - Helpers

    private static func benchmarkDir() -> URL {
        let dir = projectRoot().appendingPathComponent("benchmarks")
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

    static func formatBytes(_ bytes: Int) -> String {
        if bytes >= 1_073_741_824 {
            return String(format: "%.2fGB", Double(bytes) / 1_073_741_824)
        } else {
            return String(format: "%.0fMB", Double(bytes) / 1_048_576)
        }
    }
}
