import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import NativeDecodeBridge

func log(_ msg: String) {
    FileHandle.standardError.write(Data("[Bench] \(msg)\n".utf8))
}

@main
struct PrefillBenchmark {
    @MainActor
    static func main() async {
        setlinebuf(stdout)
        log("=== MLX Swift End-to-End Benchmark ===")

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
                let ctx = try await container.perform { ctx in ctx }
                var genText = ""
                for try await result in try generate(
                    input: shortInput, parameters: GenerateParameters(temperature: 0), context: ctx
                ) {
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
                    let ctx0 = try await container.perform { ctx in ctx }
                    var wc = 0
                    for try await _ in try generate(
                        input: input, parameters: params, context: ctx0
                    ) { wc += 1; if wc >= decodeCount + 1 { break } }

                    // Timed: one token through Swift (prefill + bridge init), rest via db_generate
                    let ctx1 = try await container.perform { ctx in ctx }
                    let t0 = CFAbsoluteTimeGetCurrent()
                    var firstTokenId: Int32 = 0
                    var gotFirst = false
                    for try await result in try generate(
                        input: input, parameters: params, context: ctx1
                    ) {
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

            let sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384].filter { $0 <= allTokens.count }
            let decodeCount = 16

            for n in sizes {
                let tokens = Array(allTokens.prefix(n))
                let tokenArray = MLXArray(tokens.map { Int($0) })
                let input = LMInput(text: LMInput.Text(tokens: tokenArray))
                let params = GenerateParameters(temperature: 0)

                // Warmup
                let ctx0 = try await container.perform { ctx in ctx }
                var wc = 0
                for try await _ in try generate(
                    input: input, parameters: params, context: ctx0
                ) { wc += 1; if wc >= decodeCount { break } }

                // Timed
                let ctx1 = try await container.perform { ctx in ctx }
                let t0 = CFAbsoluteTimeGetCurrent()
                var firstTokTime: Double = 0
                var count = 0
                for try await _ in try generate(
                    input: input, parameters: params, context: ctx1
                ) {
                    count += 1
                    if count == 1 { firstTokTime = CFAbsoluteTimeGetCurrent() }
                    if count >= decodeCount { break }
                }
                let totalTime = CFAbsoluteTimeGetCurrent()
                let prefillMs = (firstTokTime - t0) * 1000
                let decodeMs = (totalTime - firstTokTime) * 1000
                let decToks = count - 1

                log(String(format: "%5d tok | prefill %7.1fms (%6.0f t/s) | decode %7.1fms (%5.1f t/s)",
                    n, prefillMs, Double(n) / (prefillMs / 1000),
                    decodeMs, Double(decToks) / (decodeMs / 1000)))
                MLX.Memory.clearCache()
            }

            log("Done.")
        } catch {
            log("ERROR: \(error)")
        }
    }
}
