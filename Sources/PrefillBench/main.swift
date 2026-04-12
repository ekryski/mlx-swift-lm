import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

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
