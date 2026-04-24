import Foundation
import MLX
import MLXLMCommon

/// Writes benchmark results to a single hardware-dated markdown file under `benchmarks/`.
///
/// File naming: `{chip-slug}-{ram}gb-{YYYY-MM-DD}.md`
/// Example: `m1-max-64gb-2026-04-16.md`, `m5-max-128gb-2026-04-16.md`
///
/// Multiple runs (across models, quantizations, KV configs, methods) accumulate into the
/// same file, grouped by model. State lives in a `.{filename}.state.json` sidecar so the
/// markdown can be re-rendered deterministically and runs in separate `swift test`
/// invocations (as the CLI does for multi-model sweeps) continue to append into the
/// same file.
///
/// See `benchmarks/README.md#output` for the on-disk layout.
enum BenchmarkWriter {
    private static let lock = NSLock()

    // MARK: - Public append API (unchanged signature)

    /// Generation + runner metadata for the benchmark markdown header.
    struct BenchmarkParameters {
        let generate: GenerateParameters
        let thinkingEnabled: Bool
        let thinkingTokenBudget: Int?
        let kldSummary: String
        let maxOpsPerBuffer: String
        let batchSize: Int
        let speculativeDecoding: String
        let systemPromptSummary: String
    }

    /// Append a benchmark result row to the session's markdown file.
    ///
    /// Rows are grouped by model → config. The config identity is
    /// `quantization / kvConfig / scenario` plus any non-empty `configKeyExtras`
    /// appended as ` / key=value` pairs (e.g. `ngram=3`) so that two runs which
    /// differ only on a dimension outside (quant, kv, method) — like `--ngram`
    /// — land in separate config blocks with distinct Parameters tables.
    static func append(
        model: String,
        repoId: String = "",
        quantization: String,
        kvConfig: String,
        scenario: String,
        configKeyExtras: [(String, String)] = [],
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
        lock.lock()
        defer { lock.unlock() }

        let hw = hardwareInfo()
        let filename = sessionFilename(hw: hw)
        let projectRoot = Self.projectRoot()
        let markdownURL = projectRoot
            .appendingPathComponent("benchmarks")
            .appendingPathComponent(filename + ".md")
        let sidecarURL = projectRoot
            .appendingPathComponent("benchmarks")
            .appendingPathComponent("." + filename + ".state.json")

        try? FileManager.default.createDirectory(
            at: markdownURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        var state = loadState(from: sidecarURL) ?? SessionState(
            chip: hw.chip,
            gpuArch: GPU.deviceInfo().architecture,
            systemRAM: hw.systemRAM,
            gpuLimit: hw.gpuLimit,
            osVersion: hw.osVersion,
            branch: gitBranch(),
            commit: gitLastCommitLine(),
            naxEnabled: detectNAX(),
            createdAt: isoDateTime(Date()),
            models: []
        )

        let row = ResultRow(
            contextSize: contextSize,
            promptTokens: promptTokens,
            prefillTokPerSec: prefillTokPerSec,
            decodeTokPerSec: genTokPerSec,
            genTokens: genTokens,
            ttftMs: ttftMs,
            thinkPPL: thinkingPerplexity,
            genPPL: generationPerplexity,
            thinkKLD: thinkingKLD,
            genKLD: generationKLD,
            baselineGPU: baselineGPU,
            peakGPU: peakGPU,
            kvDelta: kvDelta,
            kvCacheBytes: kvCacheBytes
        )

        var configKey = "\(quantization) / \(kvConfig) / \(scenario)"
        for (k, v) in configKeyExtras {
            configKey += " / \(k)=\(v)"
        }
        var paramRows: [[String]] = []
        var systemPrompt = ""
        if let p = parameters {
            paramRows = buildParameterRows(p)
            systemPrompt = p.systemPromptSummary
        }

        // Locate or create model entry (preserves insertion order).
        if let mi = state.models.firstIndex(where: { $0.displayName == model }) {
            if let ci = state.models[mi].configs.firstIndex(where: { $0.key == configKey }) {
                state.models[mi].configs[ci].rows.append(row)
                if state.models[mi].configs[ci].outputSample.isEmpty {
                    state.models[mi].configs[ci].outputSample = outputPreview
                }
            } else {
                state.models[mi].configs.append(ConfigEntry(
                    key: configKey,
                    quantization: quantization,
                    kvConfig: kvConfig,
                    method: scenario,
                    parameterRows: paramRows,
                    systemPromptSummary: systemPrompt,
                    rows: [row],
                    outputSample: outputPreview
                ))
            }
        } else {
            state.models.append(ModelEntry(
                displayName: model,
                repoId: repoId,
                configs: [ConfigEntry(
                    key: configKey,
                    quantization: quantization,
                    kvConfig: kvConfig,
                    method: scenario,
                    parameterRows: paramRows,
                    systemPromptSummary: systemPrompt,
                    rows: [row],
                    outputSample: outputPreview
                )]
            ))
        }

        saveState(state, to: sidecarURL)
        let markdown = renderMarkdown(state: state)
        try? markdown.write(to: markdownURL, atomically: true, encoding: .utf8)
    }

    // MARK: - State types (JSON sidecar)

    private struct SessionState: Codable {
        var chip: String
        var gpuArch: String
        var systemRAM: String
        var gpuLimit: String
        var osVersion: String
        var branch: String
        var commit: String
        var naxEnabled: Bool
        var createdAt: String
        var models: [ModelEntry]
    }

    private struct ModelEntry: Codable {
        var displayName: String
        var repoId: String
        var configs: [ConfigEntry]
    }

    private struct ConfigEntry: Codable {
        var key: String
        var quantization: String
        var kvConfig: String
        var method: String
        var parameterRows: [[String]]
        var systemPromptSummary: String
        var rows: [ResultRow]
        var outputSample: String
    }

    private struct ResultRow: Codable {
        var contextSize: Int
        var promptTokens: Int
        var prefillTokPerSec: Double
        var decodeTokPerSec: Double
        var genTokens: Int
        var ttftMs: Double
        var thinkPPL: Double?
        var genPPL: Double?
        var thinkKLD: Double?
        var genKLD: Double?
        var baselineGPU: Int
        var peakGPU: Int
        var kvDelta: Int
        var kvCacheBytes: Int
    }

    // MARK: - Sidecar I/O

    private static func loadState(from url: URL) -> SessionState? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(SessionState.self, from: data)
    }

    private static func saveState(_ state: SessionState, to url: URL) {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? enc.encode(state) else { return }
        try? data.write(to: url, options: .atomic)
    }

    // MARK: - Markdown rendering

    private static func renderMarkdown(state: SessionState) -> String {
        var md = ""
        md += "# Benchmark: \(state.chip) — \(datePortion(state.createdAt))\n\n"

        // Environment block
        md += "**Hardware:** \(state.chip), \(state.systemRAM) unified memory"
        md += " (GPU limit \(state.gpuLimit))\n"
        md += "**OS:** macOS \(state.osVersion)\n"
        md += "**Branch:** `\(state.branch)`\n"
        md += "**Commit:** \(state.commit)\n"
        md += "**NAX:** \(state.naxEnabled ? "ENABLED ✓" : "DISABLED ✗")\n"
        md += "**Created:** \(state.createdAt)\n\n"

        if state.models.isEmpty {
            md += "_No benchmark rows recorded yet._\n"
        } else {
            md += "## Models\n\n"
            for m in state.models {
                md += renderModelSection(m)
            }
        }

        // Methodology link
        md += "## Methodology\n\n"
        md += "See [benchmarks/README.md](README.md#methodology) for method definitions, "
        md += "perplexity / KLD computation, and memory accounting.\n"

        return md
    }

    private static func renderModelSection(_ model: ModelEntry) -> String {
        var md = "### \(model.displayName)\n\n"
        if !model.repoId.isEmpty {
            md += "**Model:** `\(model.repoId)`\n\n"
        }

        // Results table (shared across all configs for this model).
        md += "#### Results\n\n"
        md += "| Config | Ctx | Prompt | Prefill tok/s | Decode tok/s | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Base | GPU Peak | KV Cache |\n"
        md += "|--------|----:|-------:|--------------:|-------------:|-----:|----------:|--------:|----------:|--------:|---------:|---------:|---------:|\n"
        for c in model.configs {
            for r in c.rows {
                let ctx = r.contextSize > 0 ? "\(r.contextSize)" : "—"
                let thinkPPL = r.thinkPPL.map { String(format: "%.4f", $0) } ?? "—"
                let genPPL = r.genPPL.map { String(format: "%.4f", $0) } ?? "—"
                let thinkKLD = r.thinkKLD.map { String(format: "%.4f", $0) } ?? "—"
                let genKLD = r.genKLD.map { String(format: "%.4f", $0) } ?? "—"
                let kvCell = r.kvCacheBytes > 0 ? formatBytes(r.kvCacheBytes) : "—"
                md += "| \(mdTableCell(c.key))"
                md += " | \(ctx)"
                md += " | \(r.promptTokens)"
                md += " | \(String(format: "%.1f", r.prefillTokPerSec))"
                md += " | \(String(format: "%.1f", r.decodeTokPerSec))"
                md += " | \(String(format: "%.0f", r.ttftMs))ms"
                md += " | \(thinkPPL) | \(genPPL)"
                md += " | \(thinkKLD) | \(genKLD)"
                md += " | \(formatBytes(r.baselineGPU))"
                md += " | \(formatBytes(r.peakGPU))"
                md += " | \(kvCell) |\n"
            }
        }
        md += "\n"

        // Output samples (one per config)
        md += "#### Output samples\n\n"
        for c in model.configs {
            md += "**\(c.key)**\n\n"
            let sample = c.outputSample.isEmpty ? "_(no output captured)_"
                : String(c.outputSample.prefix(400))
                    .replacingOccurrences(of: "\n", with: " ")
            md += "```\n\(sample)\n```\n\n"
        }

        // Parameters (one block per config — placed BELOW results per user direction)
        md += "#### Parameters\n\n"
        for c in model.configs {
            md += "**\(c.key)**\n\n"
            md += "| Parameter | Value |\n"
            md += "|-----------|-------|\n"
            for row in c.parameterRows where row.count == 2 {
                md += "| \(row[0]) | \(row[1]) |\n"
            }
            md += "\n"
            if !c.systemPromptSummary.isEmpty {
                md += "_System prompt:_ \(c.systemPromptSummary)\n\n"
            }
        }

        return md
    }

    // MARK: - Parameters serialisation

    private static func buildParameterRows(_ p: BenchmarkParameters) -> [[String]] {
        func optFloat(_ x: Float?) -> String {
            guard let x else { return "nil" }
            return String(format: "%g", x)
        }
        let gp = p.generate
        let strat = mdTableCell(kvCacheStrategyLine(from: gp))
        let maxKV = gp.maxKVSize.map { "\($0) tokens (RotatingKVCache)" } ?? "unbounded (KVCacheSimple)"

        // Parameter table — ordered into semantic groups so readers can scan
        // related knobs together. Groups in order: KV cache → generation →
        // sampling → penalties → thinking → speculative decoding →
        // telemetry/runtime.
        var rows: [[String]] = []

        // KV cache
        rows.append(contentsOf: [
            ["KV cache strategy", strat],
            ["Max KV size", mdTableCell(maxKV)],
            ["KV bits", gp.kvBits.map(String.init) ?? "nil"],
            ["KV scheme", mdTableCell(gp.kvScheme ?? "nil")],
            ["KV group size", "\(gp.kvGroupSize)"],
            ["Quantized KV start", "\(gp.quantizedKVStart)"],
        ])

        // Generation budget
        rows.append(contentsOf: [
            ["Prefill step size", "\(gp.prefillStepSize)"],
            ["Max tokens", gp.maxTokens.map(String.init) ?? "nil"],
        ])

        // Sampling
        rows.append(contentsOf: [
            ["Temperature", "\(gp.temperature)"],
            ["Top P", "\(gp.topP)"],
            ["Top K", "\(gp.topK)"],
            ["Min P", "\(gp.minP)"],
        ])

        // Penalties
        rows.append(contentsOf: [
            ["Repetition penalty", optFloat(gp.repetitionPenalty)],
            ["Repetition context size", "\(gp.repetitionContextSize)"],
            ["Presence penalty", optFloat(gp.presencePenalty)],
            ["Presence context size", "\(gp.presenceContextSize)"],
            ["Frequency penalty", optFloat(gp.frequencyPenalty)],
            ["Frequency context size", "\(gp.frequencyContextSize)"],
        ])

        // Thinking / reasoning
        rows.append(contentsOf: [
            ["Reasoning effort", mdTableCell(gp.reasoningEffort ?? "nil")],
            ["Think start token id", gp.thinkStartTokenId.map { "\($0)" } ?? "nil"],
            ["Think end token id", gp.thinkEndTokenId.map { "\($0)" } ?? "nil"],
            ["Thinking phase prefilled", gp.thinkingPhasePrefilled ? "true" : "false"],
        ])
        if let budget = p.thinkingTokenBudget {
            rows.append(["Thinking token budget (processor)", "\(budget)"])
        }
        rows.append(["Thinking (effective)", p.thinkingEnabled ? "Yes" : "No"])

        // Speculative decoding (n-gram + draft-model fields grouped together).
        // A later row names the active strategy ("none", "ngram (size=N, …)",
        // or "draft (repo-id)") so readers can see the strategy alongside its
        // parameters without hopping around the table.
        rows.append(contentsOf: [
            ["Speculative decoding", mdTableCell(p.speculativeDecoding)],
            ["N-gram size", "\(gp.ngramSize)"],
            ["Max n-gram draft tokens", "\(gp.maxNgramDraftTokens)"],
        ])

        // Telemetry / runtime
        rows.append(contentsOf: [
            ["Collect per-token data", gp.collectPerTokenData ? "true" : "false"],
            ["Track perplexity", gp.trackPerplexity ? "true" : "false"],
            ["Perplexity tracking (MLX_BENCH_PPL)", gp.trackPerplexity ? "Yes" : "No"],
            ["KL divergence (MLX_BENCH_KLD)", mdTableCell(p.kldSummary)],
            ["Batch size (MLX_BENCH_BATCH)", "\(p.batchSize)"],
            ["Additional processors count", "\(gp.additionalProcessors.count)"],
            ["Max ops per buffer (MLX_MAX_OPS_PER_BUFFER)", mdTableCell(p.maxOpsPerBuffer)],
        ])

        return rows
    }

    private static func kvCacheStrategyLine(from p: GenerateParameters) -> String {
        if let s = p.kvScheme, !s.isEmpty {
            return "TurboQuant (\(s))"
        }
        if let bits = p.kvBits {
            return "Affine (\(bits)-bit, group \(p.kvGroupSize), start \(p.quantizedKVStart))"
        }
        return "None (FP16)"
    }

    private static func mdTableCell(_ s: String) -> String {
        s.replacingOccurrences(of: "|", with: "\\|")
    }

    // MARK: - Filename helpers

    /// `{chip-slug}-{ram}gb-{YYYY-MM-DD}` — no extension.
    static func sessionFilename(hw: HardwareInfo) -> String {
        let chipSlug = chipSlug(from: hw.chip)
        let ramSlug = hw.systemRAM.lowercased()  // "64GB" → "64gb"
        let date = sessionDatePortion
        return "\(chipSlug)-\(ramSlug)-\(date)"
    }

    /// "Apple M1 Max (applegpu_g13s)" → "m1-max".
    private static func chipSlug(from chip: String) -> String {
        // Strip trailing "(...)" annotation.
        var s = chip
        if let paren = s.firstIndex(of: "(") {
            s = String(s[..<paren])
        }
        s = s.trimmingCharacters(in: .whitespaces).lowercased()
        // Drop leading "apple " if present.
        if s.hasPrefix("apple ") { s = String(s.dropFirst("apple ".count)) }
        // "m1 max" → "m1-max"
        return s
            .split(separator: " ")
            .joined(separator: "-")
    }

    // MARK: - Session date

    nonisolated(unsafe) private static var _sessionDate: String?
    /// `YYYY-MM-DD` — fixed for the life of the process. Across processes the date comes
    /// from the local clock, so a sweep that spans midnight will split into two files.
    static var sessionDatePortion: String {
        if let cached = _sessionDate { return cached }
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd"
        fmt.timeZone = .current
        let s = fmt.string(from: Date())
        _sessionDate = s
        return s
    }

    /// Pull the date portion back out of an ISO8601 string for display.
    private static func datePortion(_ iso: String) -> String {
        String(iso.prefix(10))
    }

    private static func isoDateTime(_ d: Date) -> String {
        let fmt = ISO8601DateFormatter()
        fmt.formatOptions = [.withInternetDateTime]
        return fmt.string(from: d)
    }

    // MARK: - Project root / git

    static func projectRoot() -> URL {
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

    // MARK: - Hardware

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

    private static func humanReadableChipName(gpuArch: String) -> String {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        var brand = [CChar](repeating: 0, count: size)
        sysctlbyname("machdep.cpu.brand_string", &brand, &size, nil, 0)
        let brandStr = String(cString: brand)
        if !brandStr.isEmpty {
            return "\(brandStr) (\(gpuArch))"
        }
        size = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        var model = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.model", &model, &size, nil, 0)
        let modelStr = String(cString: model)
        let marketingName = appleChipName(from: modelStr, gpuArch: gpuArch)
        return "\(marketingName) (\(gpuArch))"
    }

    private static func appleChipName(from model: String, gpuArch: String) -> String {
        let archLower = gpuArch.lowercased()
        let gen: String
        if archLower.contains("g13") { gen = "M1" }
        else if archLower.contains("g14") { gen = "M2" }
        else if archLower.contains("g15") { gen = "M3" }
        else if archLower.contains("g16") { gen = "M4" }
        else if archLower.contains("g17") { gen = "M5" }
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

    // MARK: - NAX detection (parity with InferenceBenchmark.printBuildEnvironment)

    private static func detectNAX() -> Bool {
        let naxObj = FileManager.default.fileExists(
            atPath: ".build/arm64-apple-macosx/release/Cmlx.build/mlx-generated/steel_attention_nax.cpp.o")
        return naxObj
    }

    // MARK: - MLX max ops per buffer (Metal backend defaults)

    /// Hard-coded fallback values. Only used when the parser at
    /// ``hardwareDefaultMaxOpsPerBuffer(gpuArch:)`` cannot locate
    /// `device.cpp` or the file's format has drifted from what the parser
    /// expects. Kept in sync with the current committed defaults.
    private static func hardcodedFallback(gpuArchSuffix: Character) -> Int {
        switch gpuArchSuffix {
        case "p": return 20
        case "g": return 40
        case "s": return 200
        case "d": return 200
        default: return 40
        }
    }

    /// Hardware default for max ops per command buffer **before**
    /// `MLX_MAX_OPS_PER_BUFFER` is applied.
    ///
    /// Resolved at runtime by parsing `mlx/backend/metal/device.cpp` from the
    /// local mlx-swift dependency (sibling checkout or SPM-fetched path).
    /// Reading the same source the library was compiled from eliminates the
    /// staleness we otherwise get from duplicating the switch statement in
    /// Swift. Falls back to a hardcoded table if the file can't be located or
    /// parsed.
    static func hardwareDefaultMaxOpsPerBuffer(gpuArch: String) -> Int {
        guard let suffix = gpuArch.last else { return 40 }
        if let parsed = parseMaxOpsPerBufferFromDeviceCpp(archSuffix: suffix) {
            return parsed
        }
        return hardcodedFallback(gpuArchSuffix: suffix)
    }

    /// Whether the last call to ``hardwareDefaultMaxOpsPerBuffer(gpuArch:)``
    /// resolved via the `device.cpp` parse (`true`) or fell back to the
    /// hardcoded table (`false`). Surfaced in the Parameters row label so
    /// readers can see which path produced the value.
    nonisolated(unsafe) private static var lastParseSucceeded = false

    /// Parse the `max_ops_per_buffer_ = N;` line for the given GPU arch
    /// suffix from `mlx/backend/metal/device.cpp`. Returns `nil` if the file
    /// can't be found or the case isn't present / doesn't match the expected
    /// shape.
    private static func parseMaxOpsPerBufferFromDeviceCpp(archSuffix: Character) -> Int? {
        guard let source = loadDeviceCppSource() else {
            lastParseSucceeded = false
            return nil
        }

        // State machine: find the Device::Device() arch switch, advance to
        // the case matching our suffix, then grab the first
        // `max_ops_per_buffer_ = N;` inside that case block.
        let caseNeedle = "case '\(archSuffix)'"
        var inTargetCase = false
        for rawLine in source.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if !inTargetCase {
                if line.hasPrefix(caseNeedle) {
                    inTargetCase = true
                }
                continue
            }
            // End of case block without finding the assignment — give up.
            if line.hasPrefix("case ") || line.hasPrefix("default:") {
                break
            }
            if let value = extractMaxOpsAssignment(line) {
                lastParseSucceeded = true
                return value
            }
            if line == "break;" {
                break
            }
        }
        lastParseSucceeded = false
        return nil
    }

    /// Extract `N` from a line like `max_ops_per_buffer_ = 200;`.
    private static func extractMaxOpsAssignment(_ line: String) -> Int? {
        let prefix = "max_ops_per_buffer_"
        guard line.hasPrefix(prefix) else { return nil }
        // Skip past `max_ops_per_buffer_`, whitespace, `=`, whitespace.
        let afterKey = line.dropFirst(prefix.count)
            .drop(while: { $0.isWhitespace })
        guard afterKey.first == "=" else { return nil }
        let afterEq = afterKey.dropFirst().drop(while: { $0.isWhitespace })
        let digits = afterEq.prefix(while: { $0.isNumber })
        return Int(digits)
    }

    /// Locate and read the mlx `device.cpp` source file. Searches, in order:
    ///
    /// 1. `$MLX_SWIFT_PATH/Source/Cmlx/mlx/mlx/backend/metal/device.cpp`
    ///    (explicit override, matches the Makefile's resolution).
    /// 2. Sibling repository at `{projectRoot}/../mlx-swift/...` (our primary
    ///    dev layout; aligned with how `Makefile` finds mlx-swift).
    /// 3. SPM checkout at `{projectRoot}/.build/checkouts/mlx-swift/...`.
    ///
    /// Returns the full file contents, or `nil` if no candidate exists.
    private static func loadDeviceCppSource() -> String? {
        let relative = "Source/Cmlx/mlx/mlx/backend/metal/device.cpp"
        var candidates: [URL] = []
        if let override = ProcessInfo.processInfo.environment["MLX_SWIFT_PATH"], !override.isEmpty {
            candidates.append(URL(fileURLWithPath: override).appendingPathComponent(relative))
        }
        let root = projectRoot()
        candidates.append(root.deletingLastPathComponent()
            .appendingPathComponent("mlx-swift")
            .appendingPathComponent(relative))
        candidates.append(root.appendingPathComponent(".build/checkouts/mlx-swift")
            .appendingPathComponent(relative))

        for url in candidates where FileManager.default.fileExists(atPath: url.path) {
            if let text = try? String(contentsOf: url, encoding: .utf8) {
                return text
            }
        }
        return nil
    }

    /// Value shown in the Parameters table. Three-state label:
    /// - `N (env)` — `MLX_MAX_OPS_PER_BUFFER` was set.
    /// - `N (from device.cpp, <arch>)` — parsed directly from the mlx source
    ///   that was built into the linked library. Authoritative.
    /// - `N (fallback, <arch>)` — `device.cpp` not found or format drifted;
    ///   hardcoded table was used. A warning signal that the benchmark harness
    ///   may report stale values.
    static func resolvedMaxOpsPerBufferReport() -> String {
        let raw = ProcessInfo.processInfo.environment["MLX_MAX_OPS_PER_BUFFER"]?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !raw.isEmpty, let v = Int(raw) {
            return "\(v) (env)"
        }
        let arch = GPU.deviceInfo().architecture
        let d = hardwareDefaultMaxOpsPerBuffer(gpuArch: arch)
        let safeArch = arch.replacingOccurrences(of: "|", with: "\\|")
        let source = lastParseSucceeded ? "from device.cpp" : "fallback"
        return "\(d) (\(source), \(safeArch))"
    }
}
