import Foundation
import MLX
import MLXLLM
@preconcurrency import MLXLMCommon
import MLXNN
import NativeDecodeBridge

func log(_ msg: String) {
    FileHandle.standardError.write(Data("[Bench] \(msg)\n".utf8))
}

func runFixedDecodeBenchmark(
    input: LMInput,
    container: ModelContainer,
    parameters: GenerateParameters,
    totalTokens: Int
) async throws -> (prefillMs: Double, decodeMs: Double, generatedTokens: Int) {
    precondition(totalTokens >= 1, "totalTokens must be at least 1")

    return try await container.perform(nonSendable: input) { context, input in
        let t0 = CFAbsoluteTimeGetCurrent()
        var iterator = try TokenIterator(
            input: input,
            model: context.model,
            parameters: parameters
        )

        guard iterator.next() != nil else {
            Stream().synchronize()
            return (0, 0, 0)
        }

        let firstTokTime = CFAbsoluteTimeGetCurrent()
        var generated = 1

        while generated < totalTokens {
            guard iterator.next() != nil else { break }
            generated += 1
        }

        let totalTime = CFAbsoluteTimeGetCurrent()
        Stream().synchronize()

        return (
            prefillMs: (firstTokTime - t0) * 1000,
            decodeMs: (totalTime - firstTokTime) * 1000,
            generatedTokens: generated
        )
    }
}

@main
struct PrefillBenchmark {
    @MainActor
    static func main() async {
        setlinebuf(stdout)
        log("=== MLX Swift End-to-End Benchmark ===")

        // Optional convenience mode for benchmark runs:
        // - BENCH_MODE=swift  => force pure Swift prefill + decode
        // - BENCH_MODE=bridge => force native prefill + decode bridges on
        if let benchMode = ProcessInfo.processInfo.environment["BENCH_MODE"]?.lowercased() {
            switch benchMode {
            case "swift":
                setenv("NATIVE_PREFILL", "0", 1)
                setenv("NATIVE_DECODE", "0", 1)
            case "bridge":
                setenv("NATIVE_PREFILL", "1", 1)
                setenv("NATIVE_DECODE", "1", 1)
            default:
                log("Unknown BENCH_MODE=\(benchMode), using ambient env")
            }
        }

        let nativePrefillEnabled = ProcessInfo.processInfo.environment["NATIVE_PREFILL"] != "0"
        let nativeDecodeEnabled = ProcessInfo.processInfo.environment["NATIVE_DECODE"] != "0"
        log(
            "Mode: prefill=\(nativePrefillEnabled ? "native" : "swift") decode=\(nativeDecodeEnabled ? "native" : "swift")"
        )

        do {
            let config = ModelConfiguration(id: "mlx-community/gemma-4-e2b-it-4bit")
            let container = try await LLMModelFactory.shared.loadContainer(
                configuration: config) { p in
                if p.fractionCompleted > 0.99 { log("Model loaded") }
            }

            let tokenPath = FileManager.default.fileExists(atPath: "/tmp/bench_tokens_32k.json")
                ? "/tmp/bench_tokens_32k.json" : "/tmp/bench_tokens_1024.json"
            let data = try Data(contentsOf: URL(fileURLWithPath: tokenPath))
            let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
            let allTokens = (json["tokens"] as! [Int]).map { Int32($0) }
            log("\(allTokens.count) frozen tokens from \(tokenPath)")

            // Per-block stub for profiling: DB_STUB=mlp|attn|ple|mlp+ple
            if let stub = ProcessInfo.processInfo.environment["DB_STUB"] {
                let smlp: Int32 = stub.contains("mlp") ? 1 : 0
                let sattn: Int32 = stub.contains("attn") ? 1 : 0
                let sple: Int32 = stub.contains("ple") ? 1 : 0
                db_set_stub(smlp, sattn, sple)
                log("Stubs set: mlp=\(smlp) attn=\(sattn) ple=\(sple)")
            }

            // Correctness check: generate with a short prompt and print tokens
            if ProcessInfo.processInfo.environment["CORRECTNESS"] == "1" {
                let shortTokens = Array(allTokens.prefix(32))
                let shortInput = LMInput(text: LMInput.Text(tokens: MLXArray(shortTokens.map { Int($0) })))
                var genText = ""
                let stream = try await container.generate(
                    input: shortInput,
                    parameters: GenerateParameters(temperature: 0)
                )
                for try await result in stream {
                    if let chunk = result.chunk { genText += chunk }
                    if genText.count >= 100 { break }
                }
                log("CORRECTNESS: \(genText.prefix(200))")
                return
            }

            // Mode C: true native decode via db_generate (no Swift token loop)
            // Uses Swift's generate() for ONE token (triggers prefill + bridge init + KV import),
            // then runs remaining decode tokens entirely in C++ via db_generate.
            if ProcessInfo.processInfo.environment["DECODE_MODE"] == "C" {
                let sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384].filter { $0 <= allTokens.count }
                let decodeCount = 15
                for n in sizes {
                    let tokens = Array(allTokens.prefix(n))
                    let tokenArray = MLXArray(tokens.map { Int($0) })
                    let input = LMInput(text: LMInput.Text(tokens: tokenArray))
                    let params = GenerateParameters(temperature: 0)

                    // Warmup
                    var wc = 0
                    let warmupStream = try await container.generate(
                        input: input,
                        parameters: params
                    )
                    for try await _ in warmupStream { wc += 1; if wc >= decodeCount + 1 { break } }

                    // Timed: one token through Swift (prefill + bridge init), rest via db_generate
                    let t0 = CFAbsoluteTimeGetCurrent()
                    var gotFirst = false
                    let timedStream = try await container.generate(
                        input: input,
                        parameters: params
                    )
                    for try await result in timedStream {
                        if !gotFirst, let _ = result.chunk {
                            gotFirst = true
                            break
                        }
                    }
                    let prefillTime = CFAbsoluteTimeGetCurrent()
                    let prefillMs = (prefillTime - t0) * 1000

                    // The bridge is now active with imported KV.
                    // db_step returns next token given current token.
                    // We need the last token — use db_step(0) as a probe or get it from context.
                    // Actually, the first generate() call already ran one decode step through the bridge.
                    // db_get_cache_offset() tells us where we are.
                    let offset = db_get_cache_offset()
                    if offset <= 0 {
                        log("MODE_C %5d tok | bridge not active (offset=\(offset))")
                        continue
                    }

                    // Run remaining decode tokens in pure C++
                    // Use token ID 0 as placeholder — the bridge's KV context determines output
                    var outTokens = [Int32](repeating: 0, count: decodeCount)
                    var elapsed: Double = 0
                    // First: get the token that was generated by the first step
                    // It was returned as a chunk but we don't have the raw ID.
                    // Use db_step with a dummy to see what comes next.
                    let tok = db_step(0)  // not ideal, but tests the C++ decode path speed
                    let decodeT0 = CFAbsoluteTimeGetCurrent()
                    let count = outTokens.withUnsafeMutableBufferPointer { buf in
                        db_generate(tok, Int32(decodeCount), buf.baseAddress!, 1, &elapsed)
                    }
                    let decodeT1 = CFAbsoluteTimeGetCurrent()
                    let decodeMs = (decodeT1 - decodeT0) * 1000
                    let decToks = Int(count)

                    log(String(format: "MODE_C %5d tok | prefill %7.1fms (%6.0f t/s) | decode %7.1fms (%5.1f t/s) [%d tokens in C++]",
                        n, prefillMs, Double(n) / (prefillMs / 1000),
                        decodeMs, Double(decToks) / (decodeMs / 1000), decToks))
                    MLX.Memory.clearCache()
                }
                log("Mode C done.")
                return
            }

            // Locked benchmark harness:
            // - Pre-warmup: 50 tokens at 512 ctx to cache all compile traces
            // - Per context: warmup run (same token count), then timed run
            // - All runs use same GenerateParameters(temperature: 0)
            // - Decode measured with an exact fixed token count (ignores EOS)
            let sizes = [512, 2048, 8192, 16384, 32768].filter { $0 <= allTokens.count }
            let totalDecodeTokens = 16

            // Pre-warmup: cache compile traces and Metal kernels
            do {
                let warmInput = LMInput(text: LMInput.Text(tokens: MLXArray(Array(allTokens.prefix(512)).map { Int($0) })))
                let warmResult = try await runFixedDecodeBenchmark(
                    input: warmInput,
                    container: container,
                    parameters: GenerateParameters(temperature: 0),
                    totalTokens: 50
                )
                log("Pre-warmup: \(warmResult.generatedTokens) tokens at 512 ctx")
            }

            for n in sizes {
                let tokens = Array(allTokens.prefix(n))
                let tokenArray = MLXArray(tokens.map { Int($0) })
                let input = LMInput(text: LMInput.Text(tokens: tokenArray))
                let params = GenerateParameters(temperature: 0)

                // Warmup run (same structure as timed, exact fixed token count)
                _ = try await runFixedDecodeBenchmark(
                    input: input,
                    container: container,
                    parameters: params,
                    totalTokens: totalDecodeTokens
                )

                // Timed run
                let result = try await runFixedDecodeBenchmark(
                    input: input,
                    container: container,
                    parameters: params,
                    totalTokens: totalDecodeTokens
                )
                let decToks = max(0, result.generatedTokens - 1)

                log(String(format: "%5d tok | prefill %7.1fms (%6.0f t/s) | decode %7.1fms (%5.1f t/s) [%d/%d toks]",
                    n, result.prefillMs, Double(n) / (result.prefillMs / 1000),
                    result.decodeMs, Double(decToks) / (result.decodeMs / 1000),
                    decToks, totalDecodeTokens - 1))
                MLX.Memory.clearCache()
            }

            log("Done.")
        } catch {
            log("ERROR: \(error)")
        }
    }
}
